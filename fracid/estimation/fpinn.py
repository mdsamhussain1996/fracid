r"""Physics-Informed Neural Network (fPINN) for fractional dynamical systems.

In this module:
1. A multi-layer perceptron :math:`\mathbf{x}_{NN}(t; \mathbf{w})` approximates the system trajectory.
2. The Caputo fractional derivative :math:`{}^C D^\alpha \mathbf{x}(t)` is computed on a uniform
   collocation grid using the L1 scheme:
   .. math::
       {}^C D^\alpha \mathbf{x}(t_n) \approx \frac{h^{-\alpha}}{\Gamma(2-\alpha)}
       \sum_{k=0}^{n-1} b_k \bigl(\mathbf{x}(t_{n-k}) - \mathbf{x}(t_{n-k-1})\bigr),
       \quad b_k = (k+1)^{1-\alpha} - k^{1-\alpha}.
3. The fractional order :math:`\alpha` and model parameters :math:`\theta` are represented
   as trainable tensors with smooth sigmoid parameterization bounded in :math:`[\ell, u]`.
4. The composite loss is:
   .. math::
       \mathcal{L} = \mathcal{L}_{\text{data}} + \lambda_{\text{phys}} \mathcal{L}_{\text{physics}} + \lambda_{\text{IC}} \mathcal{L}_{\text{IC}}.
"""
from __future__ import annotations

import time
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..data.dataset import Dataset
from ..models.base import FractionalModel
from .least_squares import FitResult, make_result
from .problem import FitProblem


def l1_caputo_derivative_torch(X: torch.Tensor, alpha: torch.Tensor, h: float) -> torch.Tensor:
    r"""Compute Caputo derivative :math:`{}^C D^\alpha \mathbf{X}` along time axis (dim 0) using L1 scheme.

    Parameters
    ----------
    X : torch.Tensor of shape (N + 1, d)
        Trajectory evaluated at uniform times :math:`t_n = n h`.
    alpha : torch.Tensor scalar
        Fractional order :math:`0 < \alpha \le 1`.
    h : float
        Uniform grid step size.

    Returns
    -------
    D_alpha : torch.Tensor of shape (N, d)
        Caputo derivative at times :math:`t_1, \dots, t_N`.
    """
    N, d = X.shape[0] - 1, X.shape[1]
    if N < 1:
        raise ValueError("Need at least 2 points for derivative")

    # k = 0, ..., N-1
    k = torch.arange(N, device=X.device, dtype=X.dtype)
    # b_k = (k+1)**(1-alpha) - k**(1-alpha)
    one_minus_alpha = 1.0 - alpha
    # (k+1)**(1-alpha)
    kp1_pow = torch.pow(k + 1.0, one_minus_alpha)
    k_pow = torch.where(k > 0, torch.pow(k.clamp(min=1e-8), one_minus_alpha), torch.zeros_like(k))
    b = kp1_pow - k_pow  # shape (N,)

    # Differences: Delta_j = X[j+1] - X[j] for j=0..N-1
    diffs = X[1:] - X[:-1]  # shape (N, d)

    # For each n from 1..N:
    # sum_{j=0}^{n-1} b_{n-1-j} * diffs[j]
    # This is a 1D convolution of diffs with b!
    # Let's perform using causal convolution or matrix multiplication:
    # Construct lower triangular Toeplitz matrix W where W[n-1, j] = b[n-1-j]
    # Since N is typically modest (100-500 points), matrix multiply is super fast and autograd friendly:
    row_idx = torch.arange(N, device=X.device)[:, None]
    col_idx = torch.arange(N, device=X.device)[None, :]
    diff_idx = row_idx - col_idx
    mask = diff_idx >= 0
    weights = torch.where(mask, b[diff_idx.clamp(min=0)], torch.zeros(1, device=X.device, dtype=X.dtype))

    # Conv result: (N, N) @ (N, d) -> (N, d)
    sum_terms = torch.matmul(weights, diffs)

    # Prefactor: h**(-alpha) / Gamma(2 - alpha)
    # torch.lgamma for Gamma(2 - alpha)
    gamma_2_minus_alpha = torch.exp(torch.lgamma(2.0 - alpha))
    prefactor = torch.pow(torch.tensor(h, device=X.device, dtype=X.dtype), -alpha) / gamma_2_minus_alpha
    return prefactor * sum_terms


class TrajectoryMLP(nn.Module):
    """Multi-Layer Perceptron predicting system state x(t) from scalar time t."""

    def __init__(self, out_dim: int, hidden_dim: int = 64, num_layers: int = 3, t_max: float = 1.0):
        super().__init__()
        self.t_max = max(t_max, 1e-6)
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Normalize time to [-1, 1] for stable MLP training
        t_norm = 2.0 * (t / self.t_max) - 1.0
        return self.net(t_norm)


class FPINNModel(nn.Module):
    """Encapsulates the neural network and trainable physical parameters."""

    def __init__(self, problem: FitProblem, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.problem = problem
        self.model = problem.model
        self.dim = self.model.dim
        self.T = float(problem.data.t[-1])

        # State neural network
        self.net = TrajectoryMLP(out_dim=self.dim, hidden_dim=hidden_dim,
                                 num_layers=num_layers, t_max=self.T)

        # Trainable alpha
        if problem.alpha_fixed is not None:
            self.register_buffer("alpha_val", torch.tensor(float(problem.alpha_fixed), dtype=torch.float32))
            self.train_alpha = False
        else:
            self.train_alpha = True
            a_lo, a_hi = problem.alpha_bounds
            self.register_buffer("alpha_lo", torch.tensor(float(a_lo), dtype=torch.float32))
            self.register_buffer("alpha_hi", torch.tensor(float(a_hi), dtype=torch.float32))
            # Start at center in unconstrained logit space
            self.raw_alpha = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # Trainable parameters
        self.raw_params = nn.ParameterDict()
        self.param_bounds = {}
        effective_bounds = {**self.model.default_bounds, **(problem.bounds or {})}
        for k in problem.free:
            lo, hi = effective_bounds[k]
            self.param_bounds[k] = (float(lo), float(hi))
            # Initial guess: default template or mid point
            init_val = float(self.model.params.get(k, 0.5 * (lo + hi)))
            init_val = np.clip(init_val, lo + 1e-4, hi - 1e-4)
            # Inverse sigmoid (logit)
            u = (init_val - lo) / (hi - lo)
            logit = np.log(u / (1.0 - u))
            self.raw_params[k] = nn.Parameter(torch.tensor(logit, dtype=torch.float32))

    @property
    def alpha(self) -> torch.Tensor:
        if not self.train_alpha:
            return self.alpha_val
        return self.alpha_lo + torch.sigmoid(self.raw_alpha) * (self.alpha_hi - self.alpha_lo)

    def get_params_dict(self) -> dict[str, torch.Tensor]:
        p = {}
        # Fixed parameters from template
        for k, v in self.model.params.items():
            p[k] = torch.tensor(float(v), dtype=torch.float32)
        # Overwrite free parameters
        for k, param in self.raw_params.items():
            lo, hi = self.param_bounds[k]
            p[k] = lo + torch.sigmoid(param) * (hi - lo)
        return p

    def get_params_numpy(self) -> dict[str, float]:
        p_dict = self.get_params_dict()
        return {k: float(v.detach().cpu().item()) for k, v in p_dict.items()}


def fit_fpinn(problem: FitProblem, *, epochs: int = 1500, lr: float = 2e-3,
              lambda_phys: float = 1.0, lambda_ic: float = 10.0,
              n_colloc: int = 150, hidden_dim: int = 64, num_layers: int = 3,
              seed: int = 0, callback: Callable[[float, int], None] | None = None) -> FitResult:
    r"""Train a Physics-Informed Neural Network to estimate :math:`(\alpha, \theta)`.

    Parameters
    ----------
    problem : FitProblem
        Inverse problem specification.
    epochs : int
        Number of Adam training steps.
    lr : float
        Initial learning rate.
    lambda_phys : float
        Weight of fractional ODE residual loss.
    lambda_ic : float
        Weight of initial condition loss.
    n_colloc : int
        Number of uniform collocation points for L1 fractional derivative.
    hidden_dim, num_layers : int
        Architecture of the trajectory MLP.
    seed : int
        PyTorch and NumPy random seed.
    callback : callable, optional
        ``callback(loss, epoch)`` periodically for UI monitoring.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    t_start = time.perf_counter()

    ds = problem.data
    t_obs = torch.tensor(ds.t, dtype=torch.float32).unsqueeze(1)  # (M, 1)
    y_obs = torch.tensor(ds.y, dtype=torch.float32)               # (M, K)
    scale = torch.tensor(problem.scale, dtype=torch.float32)      # (K,)
    obs_cols = list(ds.observed)

    # Collocation grid: uniform on [0, T]
    T = float(ds.t[-1])
    h_colloc = T / max(n_colloc, 10)
    t_colloc = torch.linspace(0.0, T, n_colloc + 1, dtype=torch.float32).unsqueeze(1)

    fpinn = FPINNModel(problem, hidden_dim=hidden_dim, num_layers=num_layers)
    optimizer = torch.optim.Adam(fpinn.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = []
    best_loss = float("inf")
    best_state = None

    x0_target = torch.tensor(problem._x0_base, dtype=torch.float32)

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # 1. Data loss
        pred_obs_all = fpinn.net(t_obs)                      # (M, d)
        pred_obs = pred_obs_all[:, obs_cols]                 # (M, K)
        data_loss = torch.mean(((pred_obs - y_obs) / scale) ** 2)

        # 2. Initial condition loss
        pred_ic = fpinn.net(torch.zeros(1, 1))
        ic_loss = torch.mean((pred_ic[0] - x0_target) ** 2)

        # 3. Physics residual loss on collocation grid
        X_colloc = fpinn.net(t_colloc)                       # (N+1, d)
        alpha = fpinn.alpha
        D_alpha = l1_caputo_derivative_torch(X_colloc, alpha, h_colloc)  # (N, d) at t_1..t_N

        # ODE RHS on t_1..t_N
        # Note: For delayed systems like Hopfield, evaluate delayed state with time offset
        params_t = fpinn.get_params_dict()
        if problem.model.delayed:
            tau = float(params_t["tau"].detach().cpu().item())
            t_colloc_d = (t_colloc[1:] - tau).clamp(min=0.0)
            X_delayed = fpinn.net(t_colloc_d)
            rhs_vals = problem.model._rhs_xp(X_colloc[1:], X_delayed, params_t, torch)
        else:
            rhs_vals = problem.model._rhs_xp(X_colloc[1:], None, params_t, torch)

        phys_loss = torch.mean((D_alpha - rhs_vals) ** 2)

        total_loss = data_loss + lambda_phys * phys_loss + lambda_ic * ic_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(fpinn.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        loss_val = float(total_loss.item())
        history.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_alpha = float(alpha.detach().cpu().item())
            best_params = fpinn.get_params_numpy()

        if callback is not None and (epoch % 50 == 0 or epoch == epochs):
            callback(loss_val, epoch)

    # Reconstruct optimal theta
    final_dict = {"alpha": best_alpha, **best_params}
    final_theta = problem.theta_from_dict(final_dict)

    res = make_result(
        problem=problem,
        method="fpinn",
        theta=final_theta,
        nfev=epochs,
        history=history,
        success=True,
        message=f"fPINN completed {epochs} epochs with final loss {best_loss:.4e}",
        runtime=time.perf_counter() - t_start,
        extra={
            "final_loss": best_loss,
            "data_loss": float(data_loss.item()),
            "phys_loss": float(phys_loss.item()),
            "epochs": epochs
        }
    )
    return res
