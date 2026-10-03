r"""Convergence-study utilities for the forward solvers."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .api import solve_fde
from .mittag_leffler import ml_scalar_solution


def observed_orders(hs, errors) -> np.ndarray:
    r"""Experimental order of convergence between consecutive refinements.

    .. math:: p_i = \frac{\log(e_i/e_{i+1})}{\log(h_i/h_{i+1})}.
    """
    hs, errors = np.asarray(hs, float), np.asarray(errors, float)
    return np.log(errors[:-1] / errors[1:]) / np.log(hs[:-1] / hs[1:])


def convergence_study(f: Callable, exact: Callable, alpha: float, x0, T: float, hs,
                      method: str = "abm", t_min: float = 0.0, **kw) -> pd.DataFrame:
    """Run ``method`` for each step size and tabulate the max-norm error and observed order.

    ``exact(t)`` must return an array of shape (len(t),) or (len(t), d).

    ``t_min`` restricts the error norm to :math:`t\\ge t_{min}`. Solutions that behave
    like :math:`t^{\\alpha}` at the origin (e.g. Mittag-Leffler) produce a local error of
    size :math:`O(h^{2\\alpha})` in the first steps, which dominates the sup norm for
    small :math:`\\alpha`; excluding that initial layer reveals the asymptotic order.
    """
    rows, errs = [], []
    for h in hs:
        sol = solve_fde(f, alpha, x0, T, h=h, method=method, **kw)
        ex = np.asarray(exact(sol.t)).reshape(sol.x.shape)
        mask = sol.t >= t_min - 1e-12
        e = float(np.max(np.abs(sol.x[mask] - ex[mask])))
        errs.append(e)
        rows.append({"h": sol.h, "max_error": e})
    df = pd.DataFrame(rows)
    df["order"] = [np.nan, *observed_orders(df["h"], df["max_error"])]
    return df


def mittag_leffler_validation(alpha: float, lam: float = -1.0, T: float = 5.0,
                              hs=(0.1, 0.05, 0.025, 0.0125), method: str = "abm",
                              t_min: float = 0.0, **kw) -> pd.DataFrame:
    r"""Compare a scheme with the exact Mittag-Leffler solution of :math:`D^\alpha x=\lambda x`, :math:`x(0)=1`."""
    return convergence_study(lambda t, x: lam * x,
                             lambda t: ml_scalar_solution(t, lam, alpha),
                             alpha, [1.0], T, hs, method=method, t_min=t_min, **kw)
