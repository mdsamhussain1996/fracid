"""Tests of the forward solvers: analytic references and convergence orders."""
import numpy as np
import pytest
from scipy.special import erfcx, gamma

from fracid.solvers import (abm_solve, gl_solve, memory_length_for_tolerance, mittag_leffler,
                            ml_linear_solution, ml_scalar_solution, short_memory_error_bound,
                            solve_fde)
from fracid.solvers.convergence import convergence_study, mittag_leffler_validation


# ---------------------------------------------------------------- Mittag-Leffler
class TestMittagLeffler:
    def test_alpha1_is_exp(self):
        assert mittag_leffler(2.0, 1.0) == pytest.approx(np.exp(2.0), rel=1e-12)

    def test_alpha_half_closed_form(self):
        # E_{1/2}(-x) = exp(x^2) erfc(x)
        for x in [0.1, 1.0, 2.0, 5.0, 20.0]:
            assert mittag_leffler(-x, 0.5) == pytest.approx(erfcx(x), rel=1e-9)

    def test_alpha2_cos(self):
        assert mittag_leffler(-4.0, 2.0) == pytest.approx(np.cos(2.0), rel=1e-12)

    def test_array_input_and_zero(self):
        out = mittag_leffler(np.array([0.0, -0.5, -1.0]), 0.7)
        assert out.shape == (3,) and out[0] == pytest.approx(1.0)
        assert np.all(np.diff(out) < 0)  # completely monotone decay


# ---------------------------------------------------------------- ABM
@pytest.mark.parametrize("alpha", [0.5, 0.8, 1.0])
def test_abm_smooth_solution_convergence_order(alpha):
    """Manufactured solution y=t^2:  D^a y = -y + g(t)  with g = D^a t^2 + t^2.

    For y in C^2 the Diethelm-Ford-Freed theorem gives order min(2, 1+alpha).
    """
    f = lambda t, x: -x + 2 * t ** (2 - alpha) / gamma(3 - alpha) + t ** 2
    df = convergence_study(f, lambda t: t ** 2, alpha, [0.0], 2.0, [0.1, 0.05, 0.025, 0.0125])
    expected = min(2.0, 1.0 + alpha)
    assert df["order"].iloc[-1] == pytest.approx(expected, abs=0.15)
    assert df["max_error"].iloc[-1] < 2e-3


@pytest.mark.parametrize("alpha", [0.5, 0.8, 0.95])
def test_abm_mittag_leffler_order_away_from_origin(alpha):
    """Exact solution x0*E_a(lam t^a); error measured for t>=1 (see convergence_study docstring)."""
    df = mittag_leffler_validation(alpha, -1.0, 5.0, hs=(0.1, 0.05, 0.025, 0.0125), t_min=1.0)
    assert df["order"].iloc[-1] >= min(2.0, 1.0 + alpha) - 0.2
    assert df["max_error"].is_monotonic_decreasing


def test_abm_mittag_leffler_global_accuracy():
    """Even including the initial layer, the error on the whole interval is small and shrinks."""
    for alpha in (0.5, 0.9):
        df = mittag_leffler_validation(alpha, -1.0, 5.0, hs=(0.05, 0.0125))
        assert df["max_error"].iloc[-1] < 1.5e-3


def test_abm_alpha_one_matches_exp():
    sol = abm_solve(lambda t, x: -x, 1.0, [1.0], 3.0, h=0.01)
    assert np.max(np.abs(sol.x[:, 0] - np.exp(-sol.t))) < 1e-4


def test_abm_linear_system_vs_mittag_leffler():
    A = np.array([[-1.0, 2.0], [-2.0, -1.0]])    # complex eigenvalues -1 +/- 2i
    x0 = np.array([1.0, 0.5])
    alpha = 0.9
    sol = abm_solve(lambda t, x: A @ x, alpha, x0, 3.0, h=0.01)
    exact = ml_linear_solution(sol.t, A, x0, alpha)
    mask = sol.t >= 0.5
    assert np.max(np.abs(sol.x[mask] - exact[mask])) < 2e-3


def test_ml_scalar_solution_positive_lambda():
    sol = abm_solve(lambda t, x: 0.5 * x, 0.7, [1.0], 2.0, h=0.005)
    assert sol.x[-1, 0] == pytest.approx(float(ml_scalar_solution(2.0, 0.5, 0.7)), rel=2e-3)


# ---------------------------------------------------------------- GL
@pytest.mark.parametrize("method", ["gl", "gl-implicit"])
@pytest.mark.parametrize("alpha", [0.5, 0.9])
def test_gl_first_order(alpha, method):
    f = lambda t, x: -x + 2 * t ** (2 - alpha) / gamma(3 - alpha) + t ** 2
    df = convergence_study(f, lambda t: t ** 2, alpha, [0.0], 2.0, [0.04, 0.02, 0.01, 0.005],
                           method=method)
    assert df["order"].iloc[-1] == pytest.approx(1.0, abs=0.12)


def test_gl_and_abm_agree_on_chen_like_system():
    f = lambda t, x: np.array([-x[0] + 0.5 * x[1], -0.5 * x[0] - x[1]])
    a = abm_solve(f, 0.8, [1.0, 0.0], 5.0, h=0.01).x
    g = gl_solve(f, 0.8, [1.0, 0.0], 5.0, h=0.01).x
    gi = solve_fde(f, 0.8, [1.0, 0.0], 5.0, h=0.01, method="gl-implicit").x
    assert np.max(np.abs(a - g)) < 2e-2
    assert np.max(np.abs(a - gi)) < 2e-2


def test_gl_implicit_is_stable_for_stiff_problem():
    f = lambda t, x: -200.0 * x
    expl = gl_solve(f, 0.9, [1.0], 2.0, h=0.05)
    impl = gl_solve(f, 0.9, [1.0], 2.0, h=0.05, implicit=True)
    assert (not expl.finite) or np.max(np.abs(expl.x)) > 10     # explicit scheme blows up
    assert impl.finite and np.max(np.abs(impl.x)) <= 1.0 + 1e-9


# ---------------------------------------------------------------- memory & delay
def test_short_memory_error_within_bound():
    alpha, h, T = 0.7, 0.01, 10.0
    f = lambda t, x: -x
    full = abm_solve(f, alpha, [1.0], T, h=h)
    L_time = memory_length_for_tolerance(M=1.0, alpha=alpha, tol=2e-3)
    short = abm_solve(f, alpha, [1.0], T, h=h, memory_steps=int(L_time / h))
    err = np.max(np.abs(full.x - short.x))
    assert err < 20 * 2e-3                      # bound controls the derivative; solution error is same order
    assert short_memory_error_bound(1.0, alpha, L_time) == pytest.approx(2e-3, rel=1e-6)
    coarse = abm_solve(f, alpha, [1.0], T, h=h, memory_steps=50)
    assert np.max(np.abs(full.x - coarse.x)) > err   # shorter memory => larger deviation


def test_delay_exact_on_first_interval():
    """D^a x = -x(t-tau), constant history x0 => on [0,tau]: x = x0 (1 - t^a/Gamma(a+1))."""
    alpha, tau = 0.8, 1.0
    for method in ("abm", "gl"):
        sol = solve_fde(lambda t, x, xd: -xd, alpha, [2.0], 1.0, h=0.01, method=method, tau=tau)
        exact = 2.0 * (1 - sol.t ** alpha / gamma(alpha + 1))
        tol = 1e-3 if method == "abm" else 1.5e-2
        assert np.max(np.abs(sol.x[:, 0] - exact)) < tol


def test_delay_requires_small_step():
    with pytest.raises(ValueError):
        abm_solve(lambda t, x, xd: -xd, 0.9, [1.0], 2.0, h=0.5, tau=0.1)


def test_input_validation():
    with pytest.raises(ValueError):
        abm_solve(lambda t, x: x, 1.5, [1.0], 1.0, h=0.1)
    with pytest.raises(ValueError):
        abm_solve(lambda t, x: x, 0.5, [1.0], 1.0)
    with pytest.raises(ValueError):
        solve_fde(lambda t, x: x, 0.5, [1.0], 1.0, h=0.1, method="rk4")
