r"""Optimisation-based identification of :math:`(\alpha,\theta)`.

Solves the bound-constrained nonlinear least-squares problem

.. math::
    \min_{\vartheta\in[\ell,u]}\ J(\vartheta)=\sum_{i,k}\Bigl(\frac{y_k(t_i)-\hat x_{c_k}(t_i;\vartheta)}{s_k}\Bigr)^2

(see :mod:`fracid.estimation.problem`) with one of

* ``"nelder-mead"`` – derivative-free simplex search (local, fast, robust to the
  non-smoothness caused by blow-up penalties); bounds are enforced by clipping.
* ``"l-bfgs-b"`` – quasi-Newton with box constraints and finite-difference gradients
  (the unit-box scaling makes a single step ``eps`` meaningful for every variable).
* ``"differential-evolution"`` – global population method (``best1bin``), recommended for
  chaotic systems and wide bounds; the best member is optionally polished by Nelder–Mead.

Local methods can be restarted from several random points (``n_starts``).  The returned
:class:`FitResult` stores the optimum, goodness-of-fit numbers and the optimisation history.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from scipy import optimize

from .problem import FitProblem

LS_METHODS = ("nelder-mead", "l-bfgs-b", "differential-evolution")


@dataclass
class FitResult:
    """Outcome of an identification run.

    Attributes
    ----------
    problem : FitProblem
    method : str
    theta : ndarray
        Optimal parameter vector in the layout of ``problem.names``.
    alpha : float
        Fractional order (the fixed value for integer-order fits).
    params : dict
        Complete model parameter dictionary (estimated + fixed).
    x0 : ndarray
        Initial condition used.
    sse : float
        Normalised sum of squared residuals :math:`J(\\hat\\vartheta)`.
    rmse : float
        Root mean squared error in *data units*, over all observed values.
    n_par : int
        Number of estimated scalars (``alpha`` included when estimated).
    nfev : int
        Number of forward simulations.
    history : ndarray
        Best-so-far objective vs. evaluation count (``(nfev, )``) or per epoch for fPINN.
    extra : dict
        Method specific data (e.g. fPINN loss components).
    """

    problem: FitProblem
    method: str
    theta: np.ndarray
    alpha: float
    params: dict
    x0: np.ndarray
    sse: float
    rmse: float
    n_par: int
    nfev: int
    success: bool = True
    message: str = ""
    runtime: float = 0.0
    history: np.ndarray = field(default_factory=lambda: np.zeros(0))
    extra: dict = field(default_factory=dict)

    @property
    def estimates(self) -> dict:
        return self.problem.estimates(self.theta)

    def simulate(self, T: float | None = None, h: float | None = None):
        """Re-simulate the fitted model (e.g. on a longer horizon for forecasting)."""
        p = self.problem
        T = p.T if T is None else T
        h = p.h_sim if h is None else h
        return p.model.simulate(T=T, h=h, alpha=self.alpha, params=self.params, x0=self.x0,
                                method=p.solver, memory_steps=p.memory_steps)

    def prediction(self) -> np.ndarray:
        """Fitted observables at the data times."""
        return self.problem.predict(self.theta)

    def residuals(self, normalised: bool = False) -> np.ndarray:
        """Residual matrix :math:`y-\\hat y`, shape (M, K) (data units unless ``normalised``)."""
        r = self.problem.data.y - self.prediction()
        return r / self.problem.scale if normalised else r


def make_result(problem: FitProblem, method: str, theta, nfev: int, history, success=True,
                message="", runtime=0.0, extra=None) -> FitResult:
    """Assemble a :class:`FitResult`, recomputing the diagnostics from ``theta``."""
    theta = np.asarray(theta, float)
    alpha, params, x0 = problem.unpack(theta)
    pred = problem.predict(theta)
    if not np.all(np.isfinite(pred)):
        sse, rmse = problem.sse(theta), np.inf        # diverged: report penalised objective
    else:
        r = (problem.data.y - pred)
        sse = float(np.sum((r / problem.scale) ** 2))
        rmse = float(np.sqrt(np.mean(r ** 2)))
    return FitResult(problem, method, theta, alpha, params, x0, sse, rmse, problem.n_par, nfev,
                     success, message, runtime, np.asarray(history, float), extra or {})


class _Tracker:
    """Wraps the objective to count evaluations and record best-so-far values."""

    def __init__(self, problem: FitProblem):
        self.p, self.best, self.hist, self.n = problem, np.inf, [], 0
        self.best_u = None

    def __call__(self, u):
        u = np.clip(np.asarray(u, float), 0.0, 1.0)
        v = self.p.objective_u(u)
        self.n += 1
        if v < self.best:
            self.best, self.best_u = v, u.copy()
        self.hist.append(self.best)
        return v

    def batch(self, U):
        """Evaluate a population ``U`` of shape ``(S, p)`` in one batched sweep."""
        U = np.clip(np.atleast_2d(np.asarray(U, float)), 0.0, 1.0)
        V = self.p.objective_u_batch(U)
        for u, v in zip(U, V):
            self.n += 1
            if v < self.best:
                self.best, self.best_u = float(v), u.copy()
            self.hist.append(self.best)
        return V


def _fd_value_and_grad(trk: "_Tracker", u: np.ndarray, eps: float = 1e-6):
    """Objective and forward-difference gradient from ONE batched sweep of ``p+1`` points."""
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    p = u.size
    steps = np.where(u + eps <= 1.0, eps, -eps)           # stay inside the unit box
    U = np.tile(u, (p + 1, 1))
    U[1:][np.arange(p), np.arange(p)] += steps
    V = trk.batch(U)
    return float(V[0]), (V[1:] - V[0]) / steps


def alpha_scan_start(problem: FitProblem, u0: np.ndarray, n: int = 9) -> np.ndarray:
    """Warm start: scan :math:`\\alpha` over ``n`` points with the other variables fixed at ``u0``.

    Returns the unit-box point with the lowest SSE (``u0`` itself if :math:`\\alpha` is fixed).
    """
    if problem.alpha_fixed is not None:
        return u0
    U = np.tile(u0, (n + 1, 1))
    U[1:, 0] = np.linspace(0.0, 1.0, n)
    V = problem.objective_u_batch(U)
    return U[int(np.argmin(V))].copy()


def fit_least_squares(problem: FitProblem, method: str = "differential-evolution", *,
                      theta0: Mapping[str, float] | None = None, n_starts: int = 1,
                      max_iter: int | None = None, tol: float = 1e-10, seed: int = 0,
                      popsize: int = 12, polish: bool = True, polish_iter: int = 100,
                      callback=None) -> FitResult:
    """Estimate :math:`(\\alpha,\\theta)` by minimising the SSE.

    Parameters
    ----------
    problem : FitProblem
    method : {"nelder-mead", "l-bfgs-b", "differential-evolution"}
    theta0 : dict, optional
        Starting values ``{name: value}`` (local methods; also seeds the DE population).
    n_starts : int
        Number of starts for local methods (first = ``theta0``/template, others random in the box).
    max_iter : int, optional
        Iteration limit (method-specific defaults: NM 2000, L-BFGS-B 200, DE 150 generations).
    tol : float
        Convergence tolerance on the objective.
    seed : int
        RNG seed (DE and random restarts).
    popsize : int
        DE population multiplier (population = ``popsize * n_par``).
    polish : bool
        After DE, refine locally (batched-gradient L-BFGS-B when the problem supports
        population batching, otherwise Nelder–Mead).
    polish_iter : int
        Iteration budget of the polishing step.
    callback : callable, optional
        ``callback(best_objective, n_evals)`` for UI progress: every 25 serial evaluations,
        once per DE generation, once per batched L-BFGS-B iteration.

    Notes
    -----
    When ``problem.can_batch`` (ABM solver), differential evolution evaluates its whole
    population with :func:`fracid.solvers.abm_solve_batch` and L-BFGS-B obtains its
    finite-difference gradient from one batched sweep of ``p+1`` points. This is 10-20x
    faster than one-by-one simulation and gives identical objective values.
    """
    method = method.lower()
    if method not in LS_METHODS:
        raise ValueError(f"method must be one of {LS_METHODS}")
    t_start = time.perf_counter()
    rng = np.random.default_rng(seed)
    trk = _Tracker(problem)
    objective = trk
    if callback is not None:
        def objective(u, _trk=trk):
            v = _trk(u)
            if _trk.n % 25 == 0:
                callback(_trk.best, _trk.n)
            return v
    p = problem.n_par
    base = problem.theta_from_dict(theta0 or {})
    u0 = np.clip(problem.theta_to_u(base), 0.0, 1.0)
    if method != "differential-evolution" and not theta0:
        u0 = alpha_scan_start(problem, u0)
    bounds01 = [(0.0, 1.0)] * p
    msg, ok = "", True

    batched = problem.can_batch
    if method == "differential-evolution":
        init = rng.random((popsize * p, p))
        init[0] = u0                                          # seed with the initial guess
        gen = {"k": 0}

        def de_cb(*_a, **_k):
            gen["k"] += 1
            if callback is not None:
                callback(trk.best, trk.n)
            return False

        if batched:
            # scipy passes the whole population as an array of shape (p, S)
            de_obj = lambda U: trk.batch(np.asarray(U).T)       # noqa: E731
            de_kw = dict(vectorized=True, updating="deferred")
        else:
            de_obj, de_kw = objective, dict(updating="immediate")
        res = optimize.differential_evolution(
            de_obj, bounds01, maxiter=max_iter or 150, popsize=popsize, tol=tol, atol=0,
            seed=int(rng.integers(2 ** 31)), polish=False, init=init,
            mutation=(0.5, 1.0), recombination=0.8, callback=de_cb, **de_kw)
        msg, ok = res.message, bool(res.success)
        if polish:
            if batched:
                # gradient polish: each iteration costs one batched sweep of p+1 members
                optimize.minimize(lambda u: _fd_value_and_grad(trk, u), trk.best_u, jac=True,
                                  method="L-BFGS-B", bounds=bounds01,
                                  options=dict(maxiter=polish_iter, ftol=1e-12, gtol=1e-10))
            else:
                optimize.minimize(objective, trk.best_u, method="Nelder-Mead", bounds=bounds01,
                                  options=dict(maxiter=polish_iter * p, xatol=1e-8, fatol=tol))
    else:
        for s in range(max(1, n_starts)):
            start = u0 if s == 0 else rng.random(p)
            if method == "nelder-mead":
                res = optimize.minimize(objective, start, method="Nelder-Mead", bounds=bounds01,
                                        options=dict(maxiter=max_iter or 2000 * p, xatol=1e-9,
                                                     fatol=tol, adaptive=True))
            elif batched:
                def vg(u):
                    out = _fd_value_and_grad(trk, u)
                    if callback is not None:
                        callback(trk.best, trk.n)
                    return out
                res = optimize.minimize(vg, start, jac=True, method="L-BFGS-B", bounds=bounds01,
                                        options=dict(maxiter=max_iter or 200, ftol=tol, gtol=1e-9))
            else:
                res = optimize.minimize(objective, start, method="L-BFGS-B", bounds=bounds01,
                                        options=dict(maxiter=max_iter or 200, ftol=tol, gtol=1e-9,
                                                     eps=1e-6))
            msg, ok = str(res.message), bool(res.success)

    theta = problem.u_to_theta(trk.best_u)
    return make_result(problem, method, theta, trk.n, trk.hist, ok, msg,
                       time.perf_counter() - t_start)
