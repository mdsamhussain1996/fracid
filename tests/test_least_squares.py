"""Tests of the optimisation-based estimation: problem set-up and recovery of known parameters."""
import numpy as np
import pytest

from fracid.data import Dataset, generate
from fracid.estimation import FitProblem, choose_step, fit_least_squares
from fracid.models import (FractionalChen, FractionalHopfield, FractionalLinear,
                           FractionalSIR)


# --------------------------------------------------------------------- FitProblem
def test_choose_step_aligns_with_uniform_data():
    t = np.arange(0, 5.01, 0.05)
    h = choose_step(t, 0.02)
    assert h == pytest.approx(0.05 / 3) and 0.05 / h == pytest.approx(3.0)
    assert choose_step(np.array([0, 0.1, 0.35, 0.4]), 0.2) == pytest.approx(0.05)


def test_parameter_packing_roundtrip():
    m = FractionalChen()
    ds = generate(m, T=1.0, h=0.01)
    p = FitProblem(m, ds, free=["a", "c"], bounds={"a": (20, 50)})
    assert p.names == ["alpha", "a", "c"]
    theta = np.array([0.8, 33.0, 27.0])
    alpha, params, x0 = p.unpack(theta)
    assert alpha == 0.8 and params["a"] == 33.0 and params["b"] == 3.0
    assert np.allclose(p.u_to_theta(p.theta_to_u(theta)), theta)
    q = p.integer_order()
    assert q.names == ["a", "c"] and q.unpack(np.array([33.0, 27.0]))[0] == 1.0


def test_residuals_zero_at_truth_for_noise_free_data():
    m = FractionalChen(alpha=0.9)
    ds = generate(m, T=1.0, h=0.005)
    p = FitProblem(m, ds)
    theta = p.theta_from_dict({"alpha": 0.9, "a": 35.0, "b": 3.0, "c": 28.0})
    assert p.sse(theta) < 1e-12


def test_divergence_is_penalised_not_raised():
    m = FractionalChen()
    ds = generate(m, T=1.0, h=0.005)
    p = FitProblem(m, ds, bounds={"a": (1, 1e4)})
    bad = p.theta_from_dict({"alpha": 0.9, "a": 1e4, "b": 3.0, "c": 28.0})
    v = p.sse(bad)
    assert np.isfinite(v) and v > 1.0


def test_irregular_sampling_uses_interpolation():
    from fracid.solvers import ml_linear_solution
    m = FractionalLinear(alpha=0.8)
    rng = np.random.default_rng(0)
    t = np.r_[0.0, np.sort(rng.uniform(0.05, 6.0, 60))]
    y = ml_linear_solution(t, m.matrix(), m.x0, 0.8)          # exact values at irregular times
    p = FitProblem(m, Dataset(t, y), h=0.01)
    assert not p._on_grid
    theta = p.theta_from_dict({"alpha": 0.8, "a11": -1, "a12": 2, "a21": -2, "a22": -1})
    assert p.sse(theta) < 1e-3


def test_validation_errors():
    m = FractionalChen()
    ds = generate(m, T=1.0, h=0.01)
    with pytest.raises(KeyError):
        FitProblem(m, ds, free=["nope"])
    with pytest.raises(ValueError):
        FitProblem(m, ds, bounds={"a": (5, 1)})
    with pytest.raises(ValueError):
        FitProblem(m, ds, alpha_bounds=(0.5, 1.5))
    with pytest.raises(ValueError):
        FitProblem(m, ds, x0_mode="magic")


# --------------------------------------------------------------------- recovery
@pytest.fixture(scope="module")
def linear_case():
    m = FractionalLinear(alpha=0.8)
    ds = generate(m, T=10.0, h=0.05, noise_percent=2.0, seed=3, fine_factor=2)
    return m, ds


@pytest.mark.parametrize("method", ["nelder-mead", "l-bfgs-b"])
def test_linear_system_recovery_local(linear_case, method):
    m, ds = linear_case
    r = fit_least_squares(FitProblem(m, ds), method, seed=1)
    assert r.alpha == pytest.approx(0.8, abs=0.03)
    for k, v in m.params.items():
        assert r.params[k] == pytest.approx(v, abs=0.06 * abs(v) + 0.02)
    assert r.rmse < 0.02 and r.success is not None and r.history[-1] == pytest.approx(r.sse, rel=1e-6)


def test_sir_recovery_differential_evolution():
    m = FractionalSIR(alpha=0.85)
    ds = generate(m, T=40.0, h=0.2, noise_percent=3.0, seed=3, fine_factor=2)
    r = fit_least_squares(FitProblem(m, ds), "differential-evolution", seed=0, popsize=8, max_iter=60)
    assert r.alpha == pytest.approx(0.85, abs=0.03)
    assert r.params["beta"] == pytest.approx(0.5, rel=0.05)
    assert r.params["gamma"] == pytest.approx(0.2, rel=0.05)


def test_sir_partial_observation_only_infected():
    m = FractionalSIR(alpha=0.85)
    ds = generate(m, T=40.0, h=0.2, noise_percent=3.0, seed=3, fine_factor=2, observed=["I"])
    r = fit_least_squares(FitProblem(m, ds), "nelder-mead")
    assert r.alpha == pytest.approx(0.85, abs=0.04)
    assert r.params["beta"] == pytest.approx(0.5, rel=0.08)


def test_chen_chaotic_recovery_with_5pct_noise():
    m = FractionalChen(alpha=0.9)
    ds = generate(m, T=1.5, h=0.005, noise_percent=5.0, seed=3, fine_factor=2)
    r = fit_least_squares(FitProblem(m, ds), "l-bfgs-b")
    assert r.alpha == pytest.approx(0.9, abs=0.03)
    assert r.params["a"] == pytest.approx(35.0, rel=0.05)
    assert r.params["c"] == pytest.approx(28.0, rel=0.05)


def test_hopfield_delay_system_recovery():
    m = FractionalHopfield(alpha=0.9)
    ds = generate(m, T=10.0, h=0.025, noise_percent=2.0, seed=3, fine_factor=2)
    r = fit_least_squares(FitProblem(m, ds), "l-bfgs-b")
    assert r.alpha == pytest.approx(0.9, abs=0.03)
    assert r.params["a22"] == pytest.approx(3.0, rel=0.06)


def test_initial_condition_estimation():
    m = FractionalLinear(alpha=0.8)
    ds = generate(m, T=8.0, h=0.05, noise_percent=1.0, seed=2, fine_factor=2)
    p = FitProblem(m.with_(x0=[0.5, 0.0]), ds, x0_mode="estimate")   # wrong template x0
    r = fit_least_squares(p, "l-bfgs-b", seed=0, n_starts=1)
    assert r.x0 == pytest.approx([1.0, 0.5], abs=0.06)


def test_integer_order_fit_has_alpha_one_and_worse_sse(linear_case):
    m, ds = linear_case
    p = FitProblem(m, ds)
    frac = fit_least_squares(p, "l-bfgs-b")
    integer = fit_least_squares(p.integer_order(), "l-bfgs-b")
    assert integer.alpha == 1.0 and integer.n_par == frac.n_par - 1
    assert integer.sse > 5 * frac.sse


def test_progress_callback_is_called(linear_case):
    m, ds = linear_case
    calls = []
    fit_least_squares(FitProblem(m, ds, free=["a11"]), "nelder-mead", max_iter=60,
                      callback=lambda best, n: calls.append(n))
    assert calls and calls[0] % 25 == 0
