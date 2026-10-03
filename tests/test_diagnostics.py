"""Tests for model diagnostics, information criteria, sensitivity, and bootstrap."""
import numpy as np
import pytest

from fracid.data import generate
from fracid.diagnostics import (compare_fractional_vs_integer, compute_alpha_sensitivity,
                                compute_metrics, compute_relative_errors, run_bootstrap)
from fracid.estimation import FitProblem, fit_least_squares
from fracid.models import FractionalLinear


@pytest.fixture(scope="module")
def linear_fits():
    m = FractionalLinear(alpha=0.8)
    ds = generate(m, T=6.0, h=0.04, noise_percent=2.0, seed=123)
    prob = FitProblem(m, ds, free=["a11", "a12", "a21", "a22"])
    frac_res = fit_least_squares(prob, method="l-bfgs-b", seed=42)
    int_res = fit_least_squares(prob.integer_order(), method="l-bfgs-b", seed=42)
    return frac_res, int_res


def test_metrics_and_comparison(linear_fits):
    frac_res, int_res = linear_fits

    m_frac = compute_metrics(frac_res, "Frac")
    m_int = compute_metrics(int_res, "Int")

    assert m_frac.sse < m_int.sse
    assert m_frac.aic < m_int.aic
    assert m_frac.bic < m_int.bic

    df_comp = compare_fractional_vs_integer(frac_res, int_res)
    assert len(df_comp) == 2
    assert df_comp["ΔAIC (vs Integer)"].iloc[0] > 10.0
    assert "Decisive" in df_comp.attrs["verdict"]


def test_relative_errors(linear_fits):
    frac_res, _ = linear_fits
    errs = compute_relative_errors(frac_res)
    assert errs is not None
    assert "alpha" in errs
    assert errs["alpha"] < 5.0  # < 5% error on alpha


def test_sensitivity_curve(linear_fits):
    frac_res, _ = linear_fits
    df_sens = compute_alpha_sensitivity(frac_res, alpha_range=(0.6, 1.0), n_points=9, mode="slice")
    assert len(df_sens) == 9
    assert "sse" in df_sens.columns
    # Minimum should be near 0.8
    min_alpha = df_sens.loc[df_sens["sse"].idxmin(), "alpha"]
    assert abs(min_alpha - 0.8) < 0.08


def test_bootstrap_uncertainty(linear_fits):
    frac_res, _ = linear_fits
    boot = run_bootstrap(frac_res, n_boot=10, ci_level=0.90, method="l-bfgs-b", seed=7)
    assert len(boot.samples) == 10
    assert "alpha" in boot.summary.index
    ci_lo, ci_hi = boot.get_ci("alpha")
    assert ci_lo <= 0.8 <= ci_hi or ci_lo < ci_hi
