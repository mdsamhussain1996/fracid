r"""Shared helpers for the fractional ODE solvers.

All solvers integrate the Caputo initial-value problem (commensurate order)

.. math::
    {}^C D^\alpha \mathbf{x}(t) = \mathbf{f}\bigl(t,\mathbf{x}(t)\bigr), \qquad
    \mathbf{x}(t_0)=\mathbf{x}_0,\qquad 0<\alpha\le 1,

on the uniform grid :math:`t_n = t_0 + n h`, :math:`n=0,\dots,N`.  For delayed
problems the right-hand side is :math:`\mathbf{f}(t,\mathbf{x}(t),\mathbf{x}(t-\tau))`
with a prescribed history :math:`\mathbf{x}(t)=\boldsymbol\varphi(t)` for
:math:`t\in[t_0-\tau,t_0]`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class FDESolution:
    """Container returned by every forward solver.

    Attributes
    ----------
    t : ndarray, shape (N+1,)
        Time grid.
    x : ndarray, shape (N+1, d)
        State trajectory.
    alpha : float
        Fractional order used.
    method : str
        Name of the scheme.
    h : float
        Step size.
    info : dict
        Extra diagnostics (memory window, number of f-evaluations, ...).
    """

    t: np.ndarray
    x: np.ndarray
    alpha: float
    method: str
    h: float
    info: dict = field(default_factory=dict)

    @property
    def finite(self) -> bool:
        """``True`` if the trajectory contains no NaN/inf (i.e. did not blow up)."""
        return bool(np.all(np.isfinite(self.x)))


def check_alpha(alpha: float) -> float:
    """Validate :math:`0<\\alpha\\le 1` and return it as ``float``."""
    alpha = float(alpha)
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must satisfy 0 < alpha <= 1, got {alpha}")
    return alpha


def make_grid(t0: float, T: float, h: float | None, n_steps: int | None) -> tuple[np.ndarray, float]:
    """Uniform grid on ``[t0, T]`` from either ``h`` or ``n_steps``."""
    if (h is None) == (n_steps is None):
        raise ValueError("Specify exactly one of h or n_steps")
    if T <= t0:
        raise ValueError("T must be larger than t0")
    if n_steps is None:
        n_steps = int(round((T - t0) / h))
    n_steps = max(int(n_steps), 1)
    h = (T - t0) / n_steps
    return t0 + h * np.arange(n_steps + 1), h


class DelayLookup:
    r"""Evaluate :math:`\mathbf{x}(t_{n}-\tau)` on a uniform grid.

    * If :math:`t_n-\tau \le t_0` the user-supplied history :math:`\varphi` is used
      (default: the constant initial condition).
    * Otherwise the stored trajectory is interpolated linearly. Because linear
      interpolation is :math:`O(h^2)` and fractional schemes are at most
      :math:`O(h^{2})`, this does not reduce the global order.

    The delay must satisfy :math:`\tau \ge h` so that the delayed value is always
    already known when stepping explicitly in time.
    """

    def __init__(self, tau: float, h: float, t0: float, x0: np.ndarray,
                 history: Callable[[float], np.ndarray] | None):
        if tau < h * (1 - 1e-9):
            raise ValueError(f"delay tau={tau} must be >= step size h={h}; decrease h")
        self.lag = tau / h
        self.h, self.t0, self.tau = h, t0, tau
        self.x0 = np.asarray(x0, dtype=float)
        self.history = history
        r = round(self.lag)
        self._int_lag = int(r) if abs(self.lag - r) < 1e-9 else None

    def __call__(self, n: int, X: np.ndarray) -> np.ndarray:
        pos = n - self.lag  # fractional grid index of t_n - tau
        if pos <= 1e-12:
            if pos > -1e-12:
                return X[0]
            if self.history is None:
                return self.x0
            return np.asarray(self.history(self.t0 + pos * self.h), dtype=float)
        if self._int_lag is not None:
            return X[n - self._int_lag]
        i0 = int(np.floor(pos))
        w = pos - i0
        return (1.0 - w) * X[i0] + w * X[i0 + 1]


def as_state(x0) -> np.ndarray:
    """Return ``x0`` as a 1-D float array."""
    return np.atleast_1d(np.asarray(x0, dtype=float)).copy()


def call_rhs(f, t, x, xd):
    """Call the right-hand side with or without the delayed state."""
    if xd is None:
        return np.asarray(f(t, x), dtype=float)
    return np.asarray(f(t, x, xd), dtype=float)


def short_memory_error_bound(M: float, alpha: float, memory_length: float) -> float:
    r"""Bound for the error of the *short-memory principle* (Podlubny, 1999).

    If :math:`\|\mathbf{f}\|\le M` and only the last :math:`L_m` time units of
    history are used, the Caputo derivative is perturbed by at most

    .. math::
        \Delta \le \frac{M\, L_m^{-\alpha}}{|\Gamma(1-\alpha)|}.

    Useful to choose ``memory_steps = L_m / h`` for a target tolerance.
    """
    from scipy.special import gamma

    if alpha >= 1.0:
        return 0.0
    return M * memory_length ** (-alpha) / abs(gamma(1.0 - alpha))


def memory_length_for_tolerance(M: float, alpha: float, tol: float) -> float:
    r"""Smallest :math:`L_m` such that :func:`short_memory_error_bound` :math:`\le` ``tol``."""
    from scipy.special import gamma

    if alpha >= 1.0:
        return 0.0
    return (M / (tol * abs(gamma(1.0 - alpha)))) ** (1.0 / alpha)
