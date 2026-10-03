"""Tests for Step 5: Partial observations, subsampling, and initial condition estimation."""
import numpy as np
import pytest

from fracid.data.generator import generate
from fracid.estimation import FitProblem, fit_least_squares
from fracid.models import FractionalChen, FractionalLinear


def test_partial_observations_generation():
    model = FractionalChen()
    ds_full = generate(model, T=1.0, h=0.01, subsample=1)
    assert len(ds_full.names) == 3

    # Only observe x and z (unobserved: y)
    ds_partial = generate(model, T=1.0, h=0.01, observed=["x", "z"], subsample=2)
    assert len(ds_partial.names) == 2
    assert ds_partial.names == ("x", "z")
    assert ds_partial.observed == (0, 2)
    assert len(ds_partial.t) == (len(ds_full.t) + 1) // 2


def test_fit_problem_with_x0_estimation():
    model = FractionalLinear(alpha=0.90)
    # Observe only state x (index 0)
    ds = generate(model, T=1.5, h=0.01, observed=["x"], noise_percent=1.0, seed=42)
    assert len(ds.names) == 1

    prob = FitProblem(
        model=model,
        data=ds,
        free=["a11", "a12", "a21", "a22"],
        x0_mode="estimate",
        alpha_bounds=(0.6, 1.0)
    )

    # Should have alpha + 4 system parameters + 2 initial states = 7 parameters
    assert prob.n_par == 7
    assert "x0_x" in prob.names
    assert "x0_y" in prob.names

    res = fit_least_squares(prob, method="differential-evolution", max_iter=30, popsize=8, seed=42)
    assert abs(res.alpha - 0.90) < 0.1
    assert "x0_x" in res.estimates
    assert "x0_y" in res.estimates
