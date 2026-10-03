r"""Sensitivity and profile analysis of objective loss with respect to fractional order α.

Computes the objective curve :math:`J(\alpha)` over an interval :math:`[\alpha_{\min}, \alpha_{\max}]`:
- **Slice**: Evaluates loss along :math:`\alpha` with other parameters fixed at their optimal values.
- **Profile**: Optimizes free parameters :math:`\theta` for each candidate :math:`\alpha` (profile likelihood).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..estimation.least_squares import FitResult, fit_least_squares
from ..estimation.problem import FitProblem


def compute_alpha_sensitivity(
    result: FitResult,
    alpha_range: tuple[float, float] | None = None,
    n_points: int = 31,
    mode: str = "slice",
    callback: Callable[[int, int], None] | None = None
) -> pd.DataFrame:
    r"""Compute loss vs fractional order :math:`\alpha` sensitivity curve.

    Parameters
    ----------
    result : FitResult
        Converged optimization result.
    alpha_range : (float, float), optional
        Range of :math:`\alpha` to sweep (defaults to problem's alpha_bounds or [0.4, 1.0]).
    n_points : int
        Number of grid points in the sweep.
    mode : {"slice", "profile"}
        "slice": evaluate with other parameters frozen (fast, < 0.5s).
        "profile": re-optimize free parameters at each alpha.
    callback : callable, optional
        Progress callback ``callback(step, total)``.
    """
    prob = result.problem
    if alpha_range is None:
        alpha_range = prob.alpha_bounds if prob.alpha_bounds else (0.4, 1.0)

    alphas = np.linspace(alpha_range[0], alpha_range[1], n_points)
    rows = []

    best_theta_dict = result.estimates

    for i, a in enumerate(alphas):
        a_val = float(a)
        if mode == "slice":
            # Form vector with alpha = a_val and other parameters at their estimates
            vec_dict = dict(best_theta_dict)
            vec_dict["alpha"] = a_val
            theta_vec = prob.theta_from_dict(vec_dict)
            sse = prob.sse(theta_vec)
            N = prob.n_obs
            rmse = float(np.sqrt(sse / N)) if np.isfinite(sse) else float("inf")
            rows.append({"alpha": a_val, "sse": sse, "rmse": rmse})
        else:
            # Profile: fix alpha = a_val and optimize remaining parameters
            sub_prob = prob.replace(alpha_fixed=a_val)
            fit_a = fit_least_squares(sub_prob, method="l-bfgs-b", theta0=best_theta_dict, max_iter=80)
            rows.append({"alpha": a_val, "sse": fit_a.sse, "rmse": fit_a.rmse})

        if callback is not None:
            callback(i + 1, n_points)

    df = pd.DataFrame(rows)
    return df
