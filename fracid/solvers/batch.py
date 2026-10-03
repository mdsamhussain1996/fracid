r"""Population-batched ABM predictor–corrector (many parameter sets in one sweep).

Identification methods such as differential evolution evaluate a whole *population* of
candidate parameter vectors :math:`\vartheta^{(1)},\dots,\vartheta^{(S)}` per generation.
Integrating them one by one costs :math:`S` Python time-loops; here all :math:`S` systems
are advanced **together**, each with its own order :math:`\alpha_s` and parameters, so the
Python loop runs once and every convolution is a single batched BLAS ``matmul``.

The scheme is identical to :func:`fracid.solvers.abm.abm_solve` (Diethelm's ABM method)
member by member, so results agree to round-off (see ``tests/test_batch.py``).

Shapes
------
* ``alphas``: ``(S,)``
* ``x0``: ``(S, d)``
* ``f(t, X[, Xd])``: ``X`` of shape ``(S, d)`` -> ``(S, d)``
* returned trajectory: ``(N+1, S, d)`` (time first).

A member whose trajectory blows up simply carries ``inf``/``NaN`` from then on; the other
members are unaffected because every operation acts row-wise.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.special import gamma

from ._common import DelayLookup, make_grid


def abm_solve_batch(f: Callable, alphas, x0, T: float, *, t0: float = 0.0,
                    h: float | None = None, n_steps: int | None = None,
                    tau: float | None = None, history: Callable | None = None,
                    memory_steps: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    r"""Integrate :math:`S` Caputo systems :math:`{}^C D^{\alpha_s} x_s = f_s(t, x_s)` at once.

    Parameters
    ----------
    f : callable
        Batched right-hand side ``f(t, X)`` (or ``f(t, X, Xd)`` when ``tau`` is given),
        ``X`` of shape ``(S, d)``.
    alphas : array_like, shape (S,)
        Orders :math:`0<\alpha_s\le 1`.
    x0 : array_like, shape (S, d)
        Initial states.
    T, t0, h, n_steps, tau, history, memory_steps :
        As in :func:`fracid.solvers.abm.abm_solve` (``tau`` common to all members).

    Returns
    -------
    t : ndarray, shape (N+1,)
    X : ndarray, shape (N+1, S, d)
    """
    alphas = np.atleast_1d(np.asarray(alphas, float))
    if np.any(alphas <= 0) or np.any(alphas > 1):
        raise ValueError("all alphas must satisfy 0 < alpha <= 1")
    x0 = np.atleast_2d(np.asarray(x0, float))
    S, d = x0.shape
    if alphas.size != S:
        raise ValueError("alphas and x0 must have the same number of members")
    t, h = make_grid(t0, T, h, n_steps)
    N = t.size - 1

    # weights per member: b[s,k] (predictor), c[s,k] (corrector interior)
    k = np.arange(N + 2, dtype=float)[None, :]
    a = alphas[:, None]
    b = (k + 1) ** a - k ** a
    c = (k + 2) ** (a + 1) + k ** (a + 1) - 2 * (k + 1) ** (a + 1)
    # reversed copies so that history sums are contiguous slices: brev[:, -m:] = b[:, m-1..0]
    brev, crev = b[:, ::-1].copy(), c[:, ::-1].copy()
    W = N + 2
    ca = (h ** alphas / gamma(alphas + 1))[:, None]
    cb = (h ** alphas / gamma(alphas + 2))[:, None]
    L = N + 1 if memory_steps is None else max(int(memory_steps), 1)

    delay = DelayLookup(tau, h, t0, x0, history) if tau is not None else None

    def conv(w, Fh):
        """Row-wise history sum ``sum_k w[s,k] F[s,k,:]`` via batched BLAS matmul."""
        return np.matmul(w[:, None, :], Fh)[:, 0]

    def rhs(tt, X, n):
        if delay is None:
            return f(tt, X)
        return f(tt, X, delay(n, Xs))

    Xs = np.empty((N + 1, S, d))
    F = np.empty((S, N + 1, d))          # member-first for contiguous einsum
    Xs[0] = x0
    F[:, 0] = rhs(t[0], x0, 0)

    with np.errstate(all="ignore"):
        for n in range(N):
            s0 = max(0, n + 1 - L)
            m = n + 1 - s0                                    # history length
            P = x0 + ca * conv(brev[:, W - m:], F[:, s0:n + 1])
            fP = rhs(t[n + 1], P, n + 1)
            if s0 == 0:
                a0 = n ** (alphas + 1) - (n - alphas) * (n + 1) ** alphas
                hist = a0[:, None] * F[:, 0]
                if n >= 1:
                    hist = hist + conv(crev[:, W - n:], F[:, 1:n + 1])
            else:
                hist = conv(crev[:, W - m:], F[:, s0:n + 1])
            Xs[n + 1] = x0 + cb * (fP + hist)
            F[:, n + 1] = rhs(t[n + 1], Xs[n + 1], n + 1)
    return t, Xs
