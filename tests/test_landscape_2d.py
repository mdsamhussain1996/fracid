"""Tests for Step 4: 2-D objective loss landscape."""
import numpy as np
import pytest

from fracid.data.generator import generate
from fracid.diagnostics.sensitivity import compute_loss_surface_2d
from fracid.estimation import FitProblem, fit_least_squares
from fracid.models import FractionalLinear


@pytest.fixture(scope="module")
def linear_fit():
    model = FractionalLinear(alpha=0.90)
    ds = generate(model, T=1.5, h=0.01, noise_percent=2.0, seed=42)
    prob = FitProblem(model, ds, free=["a11", "a12", "a21", "a22"])
    res = fit_least_squares(prob, method="differential-evolution", max_iter=25, popsize=8, seed=42)
    return res


def test_compute_loss_surface_2d(linear_fit):
    callback_calls = []
    alphas, p_vals, grid = compute_loss_surface_2d(
        linear_fit,
        param_name="a11",
        alpha_range=(0.8, 1.0),
        param_range=(-2.0, 0.0),
        n_alpha=6,
        n_param=5,
        callback=lambda r, tot: callback_calls.append((r, tot))
    )

    assert alphas.shape == (6,)
    assert p_vals.shape == (5,)
    assert grid.shape == (6, 5)
    assert len(callback_calls) == 6
    assert np.all(np.isfinite(grid))
    assert np.all(grid > 0.0)
