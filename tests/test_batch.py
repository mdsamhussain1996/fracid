"""The population-batched ABM solver must reproduce the serial solver member by member."""
import numpy as np
import pytest

from fracid.data.generator import generate
from fracid.estimation import FitProblem, fit_least_squares
from fracid.models import MODELS
from fracid.solvers import abm_solve, abm_solve_batch


def test_batch_matches_serial_linear():
    A = np.array([[-1.0, 2.0], [-2.0, -1.0]])
    alphas = np.array([0.3, 0.6, 0.95, 1.0])
    x0 = np.array([[1.0, 0.5], [0.2, -1.0], [1.0, 0.0], [0.5, 0.5]])
    _, X = abm_solve_batch(lambda t, X: X @ A.T, alphas, x0, 5.0, h=0.01)
    for s in range(alphas.size):
        ref = abm_solve(lambda t, x: A @ x, alphas[s], x0[s], 5.0, h=0.01).x
        np.testing.assert_allclose(X[:, s], ref, rtol=1e-12, atol=1e-12)


def test_batch_short_memory_matches_serial():
    f = lambda t, X: -X ** 3 + np.sin(t)                     # noqa: E731
    alphas = np.array([0.5, 0.8])
    x0 = np.array([[1.0], [2.0]])
    _, X = abm_solve_batch(f, alphas, x0, 4.0, h=0.02, memory_steps=50)
    for s in range(2):
        ref = abm_solve(lambda t, x: -x ** 3 + np.sin(t), alphas[s], x0[s], 4.0, h=0.02,
                        memory_steps=50).x
        np.testing.assert_allclose(X[:, s], ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name", list(MODELS))
def test_problem_sse_batch_equals_serial(name):
    m = MODELS[name]()
    ds = generate(m, noise_percent=5, seed=3, fine_factor=2)
    prob = FitProblem(model=m, data=ds)
    U = np.random.default_rng(1).random((6, prob.n_par))
    serial = np.array([prob.objective_u(u) for u in U])
    np.testing.assert_allclose(prob.objective_u_batch(U), serial, rtol=1e-9)


def test_vectorised_de_recovers_chen():
    m = MODELS["Fractional Chen"]()
    ds = generate(m, T=3.0, h=0.005, noise_percent=2, seed=42, fine_factor=2)
    prob = FitProblem(model=m, data=ds, alpha_bounds=(0.4, 1.0))
    res = fit_least_squares(prob, method="differential-evolution", max_iter=80, popsize=12, seed=1)
    assert abs(res.alpha - 0.9) < 0.02
    for k, v in m.params.items():
        assert abs(res.params[k] - v) / v < 0.06
