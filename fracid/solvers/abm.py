r"""Adams–Bashforth–Moulton predictor–corrector for Caputo derivatives (Diethelm).

The Caputo IVP :math:`{}^C D^\alpha x = f(t,x)`, :math:`x(0)=x_0`, is equivalent to the
Volterra integral equation

.. math::
    x(t) = x_0 + \frac{1}{\Gamma(\alpha)}\int_0^t (t-s)^{\alpha-1} f(s,x(s))\,ds .

*Predictor* (product rectangle rule, Diethelm–Ford–Freed 2002):

.. math::
    x^P_{n+1} = x_0 + \frac{h^\alpha}{\Gamma(\alpha+1)}\sum_{j=0}^{n} b_{j,n+1} f_j,
    \qquad b_{j,n+1} = (n+1-j)^\alpha-(n-j)^\alpha .

*Corrector* (product trapezoidal rule):

.. math::
    x_{n+1} = x_0 + \frac{h^\alpha}{\Gamma(\alpha+2)}\Bigl[f(t_{n+1},x^P_{n+1})
    + \sum_{j=0}^{n} a_{j,n+1} f_j\Bigr],

.. math::
    a_{0,n+1} = n^{\alpha+1}-(n-\alpha)(n+1)^\alpha,\qquad
    a_{j,n+1} = (n-j+2)^{\alpha+1}+(n-j)^{\alpha+1}-2(n-j+1)^{\alpha+1}\ (1\le j\le n).

Convergence: :math:`\max_n |x(t_n)-x_n| = O(h^{p})`, :math:`p=\min(2,1+\alpha)`
when the exact solution is :math:`C^2` (Diethelm, Ford & Freed, *Nonlinear Dynamics* 29, 2002).
For :math:`\alpha=1` the scheme collapses to Heun's method (explicit trapezoid PECE).
For solutions with the typical :math:`t^\alpha` singularity at :math:`t=0` (e.g. the
Mittag-Leffler solution) the observed order is lower, :math:`\approx 1+\alpha-\ldots`
near the origin; see ``tests/test_solvers.py`` and the README.

Numerical notes
---------------
* Cost is :math:`O(N^2 d)` (full memory). ``memory_steps=L`` keeps only the last
  :math:`L` history points (short-memory principle), giving :math:`O(N L d)` at an
  extra error of order :math:`M (Lh)^{-\alpha}/|\Gamma(1-\alpha)|`.
* The weights :math:`b_k`, :math:`c_k` are pre-computed once (vectorised) and every
  convolution is a single BLAS ``dot`` over the (reversed) history.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.special import gamma

from ._common import (DelayLookup, FDESolution, as_state, call_rhs, check_alpha,
                      make_grid)


def abm_weights(alpha: float, n_max: int) -> tuple[np.ndarray, np.ndarray]:
    r"""Pre-compute the convolution weights of the ABM scheme.

    Returns
    -------
    b : ndarray
        :math:`b_k = (k+1)^\alpha - k^\alpha`, :math:`k=0..n_{max}` (predictor).
    c : ndarray
        :math:`c_k = (k+2)^{\alpha+1} + k^{\alpha+1} - 2(k+1)^{\alpha+1}` (corrector, interior).
    """
    k = np.arange(n_max + 1, dtype=float)
    b = (k + 1) ** alpha - k ** alpha
    c = (k + 2) ** (alpha + 1) + k ** (alpha + 1) - 2 * (k + 1) ** (alpha + 1)
    return b, c


def abm_solve(f: Callable, alpha: float, x0, T: float, *, t0: float = 0.0,
              h: float | None = None, n_steps: int | None = None,
              tau: float | None = None, history: Callable | None = None,
              memory_steps: int | None = None) -> FDESolution:
    r"""Integrate :math:`{}^C D^\alpha x = f(t,x)` with the Diethelm ABM scheme.

    Parameters
    ----------
    f : callable
        ``f(t, x)`` or, when ``tau`` is given, ``f(t, x, x_delayed)``; returns shape (d,).
    alpha : float
        Fractional order, :math:`0<\alpha\le1`.
    x0 : array_like
        Initial state.
    T : float
        Final time.
    h, n_steps : float, int
        Step size or number of steps (exactly one).
    tau, history :
        Constant delay and optional history function ``phi(t)`` for :math:`t\le t_0`
        (default: constant ``x0``).
    memory_steps : int, optional
        Short-memory window length (number of steps). ``None`` = full memory.
    """
    alpha = check_alpha(alpha)
    x0 = as_state(x0)
    d = x0.size
    t, h = make_grid(t0, T, h, n_steps)
    N = t.size - 1
    b, c = abm_weights(alpha, N + 1)
    ca = h ** alpha / gamma(alpha + 1)      # predictor prefactor
    cb = h ** alpha / gamma(alpha + 2)      # corrector prefactor
    L = N + 1 if memory_steps is None else max(int(memory_steps), 1)

    delay = DelayLookup(tau, h, t0, x0, history) if tau is not None else None
    X = np.empty((N + 1, d))
    F = np.empty((N + 1, d))
    X[0] = x0
    F[0] = call_rhs(f, t[0], x0, delay(0, X) if delay else None)
    nfev = 1

    for n in range(N):
        s = max(0, n + 1 - L)               # first index inside the memory window
        # ---- predictor: sum_j b_{n-j} f_j,  j = s..n
        P = x0 + ca * (b[: n + 1 - s][::-1] @ F[s: n + 1])
        # ---- corrector
        xd = delay(n + 1, X) if delay else None
        fP = call_rhs(f, t[n + 1], P, xd)
        if s == 0:
            a0 = n ** (alpha + 1) - (n - alpha) * (n + 1) ** alpha
            hist = a0 * F[0]
            if n >= 1:
                hist = hist + c[:n][::-1] @ F[1: n + 1]
        else:
            hist = c[: n + 1 - s][::-1] @ F[s: n + 1]
        X[n + 1] = x0 + cb * (fP + hist)
        F[n + 1] = call_rhs(f, t[n + 1], X[n + 1], xd)
        nfev += 2
        if not np.isfinite(X[n + 1]).all():          # blow-up: stop early, pad with NaN
            X[n + 2:] = np.nan
            break

    return FDESolution(t=t, x=X, alpha=alpha, method="abm", h=h,
                       info={"memory_steps": None if memory_steps is None else L, "nfev": nfev,
                             "order_expected": min(2.0, 1.0 + alpha)})
