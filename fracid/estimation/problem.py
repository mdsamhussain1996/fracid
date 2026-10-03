r"""Parameter-estimation problem definition (shared by the least-squares and fPINN modules).

Unknowns
--------
The parameter vector is :math:`\vartheta = (\alpha,\ \theta_{free},\ x_{0,free})`:

* :math:`\alpha\in[\alpha_{lo},\alpha_{hi}]` (or fixed, e.g. :math:`\alpha=1` for the integer-order fit),
* the *free* model parameters :math:`\theta_{free}\subseteq\theta` (others stay at the template value),
* optionally unknown initial conditions (``x0_mode='estimate'``).

Objective
---------
Weighted sum of squared errors between the simulated and observed trajectories

.. math::
    J(\vartheta) = \sum_{i=1}^{M}\sum_{k=1}^{K}
    \Bigl(\frac{y_k(t_i) - \hat x_{c_k}(t_i;\vartheta)}{s_k}\Bigr)^2 ,

where :math:`c_k` is the observed state of column :math:`k` and :math:`s_k=\mathrm{rms}(y_k)` if
``normalize=True`` (this equalises channels and is the maximum-likelihood weighting for the
noise model of :mod:`fracid.data.generator`, whose :math:`\sigma_k\propto \mathrm{rms}(y_k)`), else 1.

Numerical choices
-----------------
* **Step size**: if the data are uniformly sampled with spacing :math:`\Delta t`, the solver
  step is :math:`h=\Delta t/m` with :math:`m=\lceil \Delta t/h_{target}\rceil`, so every
  observation lies on the grid (no interpolation error). For irregular data linear
  interpolation of the simulated trajectory is used.
* **Optimisation variables** are rescaled to the unit box :math:`u=(\vartheta-\ell)/(u_b-\ell)\in[0,1]^p`;
  this makes Nelder–Mead, L-BFGS-B (finite-difference gradients) and DE well conditioned
  even when parameters differ by orders of magnitude.
* **Divergent simulations** (blow-up or NaN) are penalised sample-wise (``PENALTY`` per
  diverged normalised residual) instead of raising, see :meth:`FitProblem.residuals`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..data.dataset import Dataset
from ..models.base import FractionalModel
from ..solvers import FDESolution, solve_fde

PENALTY = 5.0          # normalised residual assigned to samples where the simulation diverged


def choose_step(t_obs: np.ndarray, h_target: float) -> float:
    """Solver step compatible with the observation grid (see module docstring)."""
    dt = np.diff(t_obs)
    if np.allclose(dt, dt[0], rtol=1e-6, atol=1e-12):
        m = max(1, int(np.ceil(dt[0] / h_target - 1e-9)))
        return float(dt[0] / m)
    return float(min(h_target, dt.min()))


@dataclass
class FitProblem:
    """Inverse problem: identify :math:`(\\alpha,\\theta)` of ``model`` from ``data``.

    Parameters
    ----------
    model : FractionalModel
        Template instance. Its ``params`` provide the values of non-free parameters (and the
        structural ones such as the delay ``tau`` or population ``N``). Its ``x0`` is used when
        ``x0_mode='model'`` and as default for unobserved states when ``x0_mode='data'``.
    data : Dataset
        Observations (``data.observed`` maps columns to states).
    free : sequence of str, optional
        Names of the parameters to estimate (default ``model.default_free``).
    bounds : dict, optional
        ``{name: (lo, hi)}`` search box; missing entries use ``model.default_bounds``.
    alpha_bounds : (float, float)
        Search interval for :math:`\\alpha` (upper end must be :math:`\\le 1`).
    alpha_fixed : float, optional
        If given, :math:`\\alpha` is not estimated (``alpha_fixed=1`` = integer-order model).
    x0_mode : {"model", "data", "estimate"}
        Source of the initial condition.
    h : float, optional
        Target solver step (default ``data.meta['h_fit']`` or ``model.default_h``).
    solver : str
        Forward scheme (``"abm"``, ``"gl"``, ``"gl-implicit"``).
    memory_steps : int, optional
        Short-memory window for the solver.
    normalize : bool
        Weight each channel by ``1/rms(y_k)``.
    """

    model: FractionalModel
    data: Dataset
    free: Sequence[str] | None = None
    bounds: Mapping[str, tuple] | None = None
    alpha_bounds: tuple = (0.3, 1.0)
    alpha_fixed: float | None = None
    x0_mode: str = "model"
    x0_bounds: Mapping[str, tuple] | None = None
    h: float | None = None
    solver: str = "abm"
    memory_steps: int | None = None
    normalize: bool = True

    # -- derived (filled in __post_init__)
    names: list = field(init=False, default_factory=list)
    lower: np.ndarray = field(init=False, default=None)
    upper: np.ndarray = field(init=False, default=None)

    def __post_init__(self):
        m, ds = self.model, self.data
        self.data = ds = ds.shifted()
        if self.x0_mode not in ("model", "data", "estimate"):
            raise ValueError("x0_mode must be 'model', 'data' or 'estimate'")
        self.free = tuple(self.free if self.free is not None else m.default_free)
        for k in self.free:
            if k not in m.param_names:
                raise KeyError(f"unknown parameter {k!r}")
        bnds = {**m.default_bounds, **(self.bounds or {})}
        for k in self.free:
            if k not in bnds:
                raise ValueError(f"no bounds for free parameter {k!r}")
            if not bnds[k][0] < bnds[k][1]:
                raise ValueError(f"invalid bounds for {k}: {bnds[k]}")
        if self.alpha_fixed is None and not (0 < self.alpha_bounds[0] < self.alpha_bounds[1] <= 1.0):
            raise ValueError("alpha_bounds must satisfy 0 < lo < hi <= 1")
        # step & grid
        h_target = self.h or ds.meta.get("h_fit") or m.default_h
        self.h_sim = choose_step(ds.t, h_target)
        self.T = float(ds.t[-1])
        self.n_steps = max(1, int(round(self.T / self.h_sim)))
        ratio = ds.t / self.h_sim
        self._on_grid = bool(np.allclose(ratio, np.round(ratio), atol=1e-6))
        self._idx = np.round(ratio).astype(int) if self._on_grid else None
        self._grid = self.h_sim * np.arange(self.n_steps + 1)
        # initial condition handling
        x0 = np.array(m.x0, float)
        if self.x0_mode == "data":
            for col, st in enumerate(ds.observed):
                x0[st] = ds.y[0, col]
        self._x0_base = x0
        # parameter layout
        names, lo, hi = [], [], []
        if self.alpha_fixed is None:
            names.append("alpha"); lo.append(self.alpha_bounds[0]); hi.append(self.alpha_bounds[1])
        for k in self.free:
            names.append(k); lo.append(bnds[k][0]); hi.append(bnds[k][1])
        self._x0_free = []
        if self.x0_mode == "estimate":
            xb = self.x0_bounds or {}
            for i, s in enumerate(m.state_names):
                names.append(f"x0_{s}")
                if f"x0_{s}" in xb:
                    a, b = xb[f"x0_{s}"]
                else:
                    r = max(abs(x0[i]), 1.0)
                    a, b = x0[i] - r, x0[i] + r
                lo.append(a); hi.append(b); self._x0_free.append(i)
        self.names, self.lower, self.upper = names, np.array(lo, float), np.array(hi, float)
        # weights
        rms = np.sqrt(np.mean(ds.y ** 2, axis=0))
        self.scale = np.where(rms > 0, rms, 1.0) if self.normalize else np.ones(ds.y.shape[1])
        self.nfev = 0

    # ------------------------------------------------------------------ layout helpers
    @property
    def n_par(self) -> int:
        return len(self.names)

    @property
    def n_obs(self) -> int:
        return int(self.data.y.size)

    def u_to_theta(self, u) -> np.ndarray:
        return self.lower + np.asarray(u, float) * (self.upper - self.lower)

    def theta_to_u(self, theta) -> np.ndarray:
        return (np.asarray(theta, float) - self.lower) / (self.upper - self.lower)

    def theta_from_dict(self, d: Mapping[str, float]) -> np.ndarray:
        """Vector from ``{name: value}`` (missing -> template/default/centre)."""
        base = self.default_theta()
        return np.array([d.get(n, b) for n, b in zip(self.names, base)], float)

    def default_theta(self) -> np.ndarray:
        """Template values clipped into the box; :math:`\\alpha` -> mid-point of its interval."""
        out = []
        for n, lo, hi in zip(self.names, self.lower, self.upper):
            if n == "alpha":
                v = 0.5 * (lo + hi)
            elif n.startswith("x0_"):
                v = self._x0_base[self.model.state_names.index(n[3:])]
            else:
                v = self.model.params[n]
            out.append(float(np.clip(v, lo, hi)))
        return np.array(out)

    def unpack(self, theta) -> tuple[float, dict, np.ndarray]:
        """Split a parameter vector into ``(alpha, params, x0)``."""
        theta = np.asarray(theta, float)
        params = dict(self.model.params)
        x0 = self._x0_base.copy()
        alpha = float(self.alpha_fixed) if self.alpha_fixed is not None else None
        j = 0
        if alpha is None:
            alpha = float(theta[0]); j = 1
        for k in self.free:
            params[k] = float(theta[j]); j += 1
        for i in self._x0_free:
            x0[i] = float(theta[j]); j += 1
        return alpha, params, x0

    def estimates(self, theta) -> dict:
        """``{name: value}`` for the estimated quantities only."""
        return dict(zip(self.names, map(float, theta)))

    # ------------------------------------------------------------------ forward map
    def simulate_full(self, theta) -> FDESolution:
        """Simulate on the solver grid for the parameter vector ``theta``."""
        alpha, params, x0 = self.unpack(theta)
        self.nfev += 1
        kw = dict(method=self.solver, memory_steps=self.memory_steps)
        if self.model.delayed:
            kw["tau"] = params["tau"]
        with np.errstate(all="ignore"):
            return solve_fde(self.model.solver_function(params), alpha, x0, self.T,
                             n_steps=self.n_steps, **kw)

    def observe(self, X: np.ndarray) -> np.ndarray:
        """Observation operator: model states on the solver grid -> data columns at ``t_obs``."""
        cols = list(self.data.observed)
        if self._on_grid:
            return X[self._idx][:, cols]
        return np.column_stack([np.interp(self.data.t, self._grid, X[:, c]) for c in cols])

    def predict(self, theta) -> np.ndarray:
        """Simulated observables, shape (M, K). Samples after a blow-up are ``NaN``."""
        with np.errstate(all="ignore"):
            return self.observe(self.simulate_full(theta).x)

    def residuals(self, theta) -> np.ndarray:
        r"""Flattened *normalised* residuals :math:`r_{ik}=(y_{ik}-\hat y_{ik})/s_k`.

        Samples where the simulation diverged (NaN/inf, or :math:`|r|>10^3`) are replaced by
        ``PENALTY``. The penalty is *graded*: the earlier the blow-up, the more samples are
        affected, so the objective still ranks diverging parameter sets and optimisers keep a
        usable direction instead of a flat plateau.
        """
        pred = self.predict(theta)
        with np.errstate(all="ignore"):
            r = (self.data.y - pred) / self.scale
        bad = ~np.isfinite(r) | (np.abs(r) > 1e3)
        r[bad] = PENALTY
        return r.ravel()

    def sse(self, theta) -> float:
        r = self.residuals(theta)
        return float(r @ r)

    def objective_u(self, u) -> float:
        """SSE as a function of the unit-box variable (used by all optimisers)."""
        return self.sse(self.u_to_theta(np.clip(u, 0.0, 1.0)))

    # ------------------------------------------------------------------ variants
    def replace(self, **changes) -> "FitProblem":
        """Copy with modified options (e.g. ``alpha_fixed=1.0`` or ``data=...``)."""
        from dataclasses import fields
        kw = {f.name: getattr(self, f.name) for f in fields(self) if f.init}
        kw.update(changes)
        return FitProblem(**kw)

    def integer_order(self) -> "FitProblem":
        """Same problem with :math:`\\alpha=1` fixed (the integer-order competitor)."""
        return self.replace(alpha_fixed=1.0)
