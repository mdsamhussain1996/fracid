r"""Grünwald–Letnikov scheme for Caputo initial-value problems.

For a function with :math:`x(0)=x_0` the Caputo derivative equals the
Grünwald–Letnikov derivative of :math:`x-x_0`:

.. math::
    {}^C D^\alpha x(t_n) \approx h^{-\alpha}\sum_{j=0}^{n} w_j\,(x_{n-j}-x_0),
    \qquad w_0=1,\quad w_j = \Bigl(1-\frac{\alpha+1}{j}\Bigr)w_{j-1}
    \;\;\bigl(w_j=(-1)^j\tbinom{\alpha}{j}\bigr).

*Explicit* variant (Petráš' "fractional Euler", widely used for fractional chaos):

.. math::
    x_{n+1} = x_0 + h^\alpha f(t_n,x_n) - \sum_{j=1}^{n+1} w_j\,(x_{n+1-j}-x_0).

*Implicit* variant (unconditionally stable for linear dissipative problems, solved
with Newton's method and a finite-difference Jacobian):

.. math::
    x_{n+1} - h^\alpha f(t_{n+1},x_{n+1}) = x_0 - \sum_{j=1}^{n+1} w_j\,(x_{n+1-j}-x_0).

Both are first-order accurate, :math:`O(h)`, and :math:`O(N^2)` in cost; the
``memory_steps`` argument implements the short-memory principle.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from ._common import (DelayLookup, FDESolution, as_state, call_rhs, check_alpha,
                      make_grid)


def gl_weights(alpha: float, n: int) -> np.ndarray:
    r"""Binomial weights :math:`w_j=(-1)^j\binom{\alpha}{j}`, :math:`j=0..n` (stable recursion)."""
    w = np.empty(n + 1)
    w[0] = 1.0
    for j in range(1, n + 1):
        w[j] = w[j - 1] * (1.0 - (alpha + 1.0) / j)
    return w


def gl_solve(f: Callable, alpha: float, x0, T: float, *, t0: float = 0.0,
             h: float | None = None, n_steps: int | None = None,
             implicit: bool = False, tau: float | None = None,
             history: Callable | None = None, memory_steps: int | None = None,
             newton_tol: float = 1e-12, newton_maxit: int = 25) -> FDESolution:
    r"""Integrate :math:`{}^C D^\alpha x = f(t,x)` with the Grünwald–Letnikov scheme.

    Parameters are as in :func:`fracid.solvers.abm.abm_solve`; ``implicit=True`` selects the
    implicit (backward) variant.
    """
    alpha = check_alpha(alpha)
    x0 = as_state(x0)
    d = x0.size
    t, h = make_grid(t0, T, h, n_steps)
    N = t.size - 1
    w = gl_weights(alpha, N + 1)
    ha = h ** alpha
    L = N + 1 if memory_steps is None else max(int(memory_steps), 1)
    delay = DelayLookup(tau, h, t0, x0, history) if tau is not None else None

    X = np.empty((N + 1, d))
    X[0] = x0
    nfev = 0
    for n in range(N):
        m = min(n + 1, L)                       # number of memory terms j = 1..m
        # sum_{j=1}^{m} w_j (x_{n+1-j} - x0)  ->  X[n], X[n-1], ..., X[n+1-m]
        S = w[1: m + 1] @ (X[n + 1 - m: n + 1][::-1] - x0)
        if not implicit:
            xd = delay(n, X) if delay else None
            X[n + 1] = x0 + ha * call_rhs(f, t[n], X[n], xd) - S
            nfev += 1
        else:
            xd = delay(n + 1, X) if delay else None
            rhs_const = x0 - S
            y = X[n].copy()
            for _ in range(newton_maxit):
                fy = call_rhs(f, t[n + 1], y, xd)
                G = y - ha * fy - rhs_const
                nfev += 1
                if np.max(np.abs(G)) < newton_tol * (1.0 + np.max(np.abs(y))):
                    break
                J = np.empty((d, d))
                eps = 1e-7
                for k in range(d):
                    yk = y.copy()
                    dy = eps * (1.0 + abs(y[k]))
                    yk[k] += dy
                    J[:, k] = (call_rhs(f, t[n + 1], yk, xd) - fy) / dy
                nfev += d
                y = y - np.linalg.solve(np.eye(d) - ha * J, G)
            X[n + 1] = y
        if not np.isfinite(X[n + 1]).all():
            X[n + 2:] = np.nan
            break

    return FDESolution(t=t, x=X, alpha=alpha, method="gl-implicit" if implicit else "gl", h=h,
                       info={"memory_steps": None if memory_steps is None else L, "nfev": nfev,
                             "order_expected": 1.0})
