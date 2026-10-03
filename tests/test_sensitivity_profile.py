"""Tests for Step 2: Batched slice, continuation profile, and profile likelihood CIs."""
import numpy as np
import pandas as pd
import pytest

from fracid.data.generator import generate
from fracid.diagnostics.sensitivity import compute_alpha_sensitivity, compute_profile_ci
from fracid.estimation import FitProblem, fit_least_squares
from fracid.models import FractionalLinear


@pytest.fixture(scope="module")
def linear_fit():
    model = FractionalLinear(alpha=0.85)
    ds = generate(model, T=2.0, h=0.01, noise_percent=2.0, seed=42)
    prob = FitProblem(model, ds, free=["a11", "a12", "a21", "a22"], alpha_bounds=(0.6, 1.0))
    res = fit_least_squares(prob, method="differential-evolution", max_iter=30, popsize=8, seed=42)
    return res


def test_slice_mode_batched(linear_fit):
    df = compute_alpha_sensitivity(linear_fit, n_points=11, mode="slice")
    assert len(df) == 11
    assert "alpha" in df.columns
    assert "sse" in df.columns
    assert "rmse" in df.columns
    assert "ci" in df.attrs
    ci_lo, ci_hi, thr = df.attrs["ci"]
    assert ci_lo <= linear_fit.alpha <= ci_hi
    assert thr >= linear_fit.sse


def test_profile_mode_continuation(linear_fit):
    progress_records = []
    df = compute_alpha_sensitivity(
        linear_fit,
        alpha_range=(0.75, 0.95),
        n_points=7,
        mode="profile",
        callback=lambda s, t: progress_records.append((s, t))
    )
    assert len(df) == 7
    assert len(progress_records) == 7
    assert progress_records[-1] == (7, 7)
    assert "ci" in df.attrs
    ci_lo, ci_hi, thr = df.attrs["ci"]
    assert ci_lo <= ci_hi


def test_compute_profile_ci_math(linear_fit):
    # Parabolic mock profile
    a_grid = np.linspace(0.6, 1.0, 41)
    sse_grid = linear_fit.sse + 50.0 * (a_grid - linear_fit.alpha) ** 2
    mock_df = pd.DataFrame({"alpha": a_grid, "sse": sse_grid})

    ci_lo, ci_hi, thr = compute_profile_ci(linear_fit, mock_df, confidence=0.95)
    assert ci_lo < linear_fit.alpha < ci_hi
    assert ci_lo > 0.6
    assert ci_hi < 1.0
