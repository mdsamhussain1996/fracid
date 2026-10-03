r"""Sensitivity and profile analysis of objective loss with respect to fractional order α.

Computes the objective curve :math:`J(\alpha)` over an interval :math:`[\alpha_{\min}, \alpha_{\max}]`:
- **Slice**: Evaluates loss along :math:`\alpha` with other parameters fixed at their optimal values.
  Batched across all grid points in a single forward pass when supported.
- **Profile**: Optimizes free parameters :math:`\theta` for each candidate :math:`\alpha` (profile likelihood)
  using continuation starting outward from :math:`\hat\alpha`.
- **Confidence Intervals**: Computes profile-likelihood intervals
  :math:`\{\alpha : J(\alpha) - J(\hat\alpha) \le \chi^2_1(0.95) \cdot J(\hat\alpha)/(N - p)\}`.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from ..estimation.least_squares import FitResult, fit_least_squares
from ..estimation.problem import FitProblem


def compute_profile_ci(
    result: FitResult,
    sens_df: pd.DataFrame,
    confidence: float = 0.95
) -> tuple[float, float, float]:
    r"""Compute profile-likelihood confidence interval for :math:`\alpha`.

    The interval is defined by the likelihood-ratio cut-off:

    .. math::
        \text{CI} = \Bigl\{ \alpha : J(\alpha) - J(\hat\alpha) \le \chi^2_1(c) \cdot \frac{J(\hat\alpha)}{N - p} \Bigr\}

    Parameters
    ----------
    result : FitResult
        Converged optimization result.
    sens_df : pd.DataFrame
        DataFrame with columns 'alpha' and 'sse' from :func:`compute_alpha_sensitivity`.
    confidence : float, default 0.95
        Confidence level (e.g. 0.95 for 95% interval).

    Returns
    -------
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    threshold : float
        Critical cut-off level :math:`J_{\text{crit}} = J(\hat\alpha) + \chi^2_1(c) \cdot \frac{J(\hat\alpha)}{N - p}`.
    """
    prob = result.problem
    chi2_val = float(stats.chi2.ppf(confidence, df=1))
    dof = max(prob.n_obs - prob.n_par, 1)
    threshold = float(result.sse + chi2_val * result.sse / dof)

    alphas = sens_df["alpha"].to_numpy(dtype=float)
    sses = sens_df["sse"].to_numpy(dtype=float)

    if len(alphas) == 0:
        return result.alpha, result.alpha, threshold

    # Index closest to estimated alpha
    min_idx = int(np.argmin(np.abs(alphas - result.alpha)))

    # Left search (min_idx down to 0)
    ci_lo = alphas[0]
    for i in range(min_idx, 0, -1):
        if (sses[i] <= threshold < sses[i - 1]) or (sses[i - 1] <= threshold < sses[i]):
            denom = sses[i - 1] - sses[i]
            frac = (threshold - sses[i]) / denom if abs(denom) > 1e-12 else 0.5
            ci_lo = alphas[i] + frac * (alphas[i - 1] - alphas[i])
            break

    # Right search (min_idx up to len(alphas) - 1)
    ci_hi = alphas[-1]
    for j in range(min_idx, len(alphas) - 1):
        if (sses[j] <= threshold < sses[j + 1]) or (sses[j + 1] <= threshold < sses[j]):
            denom = sses[j + 1] - sses[j]
            frac = (threshold - sses[j]) / denom if abs(denom) > 1e-12 else 0.5
            ci_hi = alphas[j] + frac * (alphas[j + 1] - alphas[j])
            break

    ci_lo = float(np.clip(ci_lo, alphas[0], alphas[-1]))
    ci_hi = float(np.clip(ci_hi, alphas[0], alphas[-1]))
    if ci_lo > ci_hi:
        ci_lo, ci_hi = ci_hi, ci_lo

    return ci_lo, ci_hi, threshold


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
        "slice": evaluate with other parameters frozen (batched across all grid points).
        "profile": re-optimize free parameters at each alpha using outward continuation from :math:`\hat\alpha`.
    callback : callable, optional
        Progress callback ``callback(step, total)``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``['alpha', 'sse', 'rmse']``. Profile CI metadata is attached
        under ``df.attrs['ci'] = (ci_lower, ci_upper, threshold)``.
    """
    prob = result.problem
    if alpha_range is None:
        alpha_range = prob.alpha_bounds if prob.alpha_bounds else (0.4, 1.0)

    alphas = np.linspace(alpha_range[0], alpha_range[1], n_points)
    best_theta_dict = result.estimates

    if mode == "slice":
        # Form parameter matrix for all candidate alphas
        thetas = np.array([prob.theta_from_dict({**best_theta_dict, "alpha": float(a)}) for a in alphas])
        if prob.can_batch:
            sses = prob.sse_batch(thetas)
        else:
            sses = np.array([prob.sse(th) for th in thetas])
        N = prob.n_obs
        rmses = np.sqrt(np.maximum(sses, 0.0) / max(N, 1))
        rows = [{"alpha": float(a), "sse": float(s), "rmse": float(r)}
                for a, s, r in zip(alphas, sses, rmses)]
        if callback is not None:
            callback(n_points, n_points)
    else:
        # Profile mode with continuation:
        # Sweep outward from alpha_hat in both directions and warm-start each fit
        # from the neighbouring alpha's optimum.
        alpha_hat = result.alpha
        right_indices = [i for i, a in enumerate(alphas) if a >= alpha_hat]
        left_indices = [i for i, a in enumerate(alphas) if a < alpha_hat][::-1]

        results_dict: dict[int, dict[str, float]] = {}
        steps_done = 0

        # Rightward continuation (alpha >= alpha_hat)
        curr_theta = dict(best_theta_dict)
        for i in right_indices:
            a_val = float(alphas[i])
            sub_prob = prob.replace(alpha_fixed=a_val)
            fit_a = fit_least_squares(sub_prob, method="l-bfgs-b", theta0=curr_theta, max_iter=80)
            results_dict[i] = {"alpha": a_val, "sse": fit_a.sse, "rmse": fit_a.rmse}
            curr_theta = dict(fit_a.estimates)
            steps_done += 1
            if callback is not None:
                callback(steps_done, n_points)

        # Leftward continuation (alpha < alpha_hat)
        curr_theta = dict(best_theta_dict)
        for i in left_indices:
            a_val = float(alphas[i])
            sub_prob = prob.replace(alpha_fixed=a_val)
            fit_a = fit_least_squares(sub_prob, method="l-bfgs-b", theta0=curr_theta, max_iter=80)
            results_dict[i] = {"alpha": a_val, "sse": fit_a.sse, "rmse": fit_a.rmse}
            curr_theta = dict(fit_a.estimates)
            steps_done += 1
            if callback is not None:
                callback(steps_done, n_points)

        rows = [results_dict[i] for i in range(n_points)]

    df = pd.DataFrame(rows)
    ci_lo, ci_hi, threshold = compute_profile_ci(result, df, confidence=0.95)
    df.attrs["ci"] = (ci_lo, ci_hi, threshold)
    return df
