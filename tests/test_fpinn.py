"""Tests for Physics-Informed Neural Network (fPINN) estimation module."""
import numpy as np
import pytest
import torch

from fracid.data import generate
from fracid.estimation.fpinn import fit_fpinn, l1_caputo_derivative_torch
from fracid.estimation.problem import FitProblem
from fracid.models import FractionalLinear


def test_l1_caputo_derivative_torch_matches_analytic():
    # Test on x(t) = t^2.
    # D^alpha t^2 = 2 * t^(2 - alpha) / Gamma(3 - alpha)
    alpha_val = 0.5
    h = 0.01
    t = torch.arange(0, 1.0 + h, h, dtype=torch.float32).unsqueeze(1)
    X = t ** 2

    alpha = torch.tensor(alpha_val, dtype=torch.float32)
    D_alpha = l1_caputo_derivative_torch(X, alpha, h)

    # Analytic value at t_1..t_N
    from scipy.special import gamma
    exact = 2.0 * (t[1:].squeeze().numpy() ** (2.0 - alpha_val)) / gamma(3.0 - alpha_val)

    num_D = D_alpha.squeeze().detach().numpy()
    # L1 scheme has order O(h^(2-alpha)) = O(h^1.5)
    assert np.allclose(num_D[10:], exact[10:], rtol=0.08)


def test_fpinn_linear_system_recovery():
    # Linear system D^alpha x = A x
    m = FractionalLinear(alpha=0.85)
    ds = generate(m, T=2.0, h=0.02, noise_percent=1.0, seed=42)

    prob = FitProblem(m, ds, free=["a11", "a12"], alpha_bounds=(0.7, 1.0))
    res = fit_fpinn(
        prob,
        epochs=300,
        lr=5e-3,
        lambda_phys=0.5,
        lambda_ic=10.0,
        n_colloc=100,
        hidden_dim=32,
        num_layers=2,
        seed=42
    )

    assert res.alpha == pytest.approx(0.85, abs=0.08)
    assert res.params["a11"] == pytest.approx(-1.0, abs=0.25)
    assert res.rmse < 0.1
    assert "data_loss" in res.extra
