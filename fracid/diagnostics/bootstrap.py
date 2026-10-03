r"""Bootstrap confidence intervals for estimated parameters and fractional order α.

Performs parametric or residual bootstrap:
1. Residuals :math:`\mathbf{r}_i = \mathbf{y}(t_i) - \hat{\mathbf{y}}(t_i)` are centered.
2. For each replicate :math:`b = 1, \dots, B`, resample residuals with replacement:
   .. math::
       \mathbf{y}^{*(b)}(t_i) = \hat{\mathbf{y}}(t_i) + \mathbf{r}_i^{*(b)}.
3. Refit model on synthetic bootstrap data to generate parameter distribution.
4. Report mean, standard error, and empirical percentile confidence intervals (e.g., 95%).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ..data.dataset import Dataset
from ..estimation.least_squares import FitResult, fit_least_squares
from ..estimation.problem import FitProblem


@dataclass
class BootstrapResult:
    """Summary of parameter uncertainty from bootstrap resampling."""
    samples: pd.DataFrame
    summary: pd.DataFrame
    ci_level: float = 0.95

    def get_ci(self, param_name: str) -> tuple[float, float]:
        row = self.summary.loc[param_name]
        return float(row["CI_Lower"]), float(row["CI_Upper"])


def run_bootstrap(
    result: FitResult,
    n_boot: int = 40,
    ci_level: float = 0.95,
    method: str | None = None,
    seed: int = 42,
    callback: Callable[[int, int], None] | None = None
) -> BootstrapResult:
    """Run residual bootstrap uncertainty quantification.

    Parameters
    ----------
    result : FitResult
        Initial converged identification result.
    n_boot : int
        Number of bootstrap iterations (e.g., 30-100).
    ci_level : float
        Confidence level (default 0.95 for 95% CI).
    method : str, optional
        Optimization method for refitting (defaults to fast Nelder-Mead or L-BFGS-B).
    seed : int
        RNG seed for reproducibility.
    callback : callable, optional
        ``callback(current_iter, total_iters)`` for UI progress bar.
    """
    prob = result.problem
    ds = prob.data
    pred = result.prediction()
    if pred is None:
        raise ValueError("Cannot bootstrap from an invalid prediction")

    residuals = ds.y - pred
    # Center residuals to ensure zero-mean noise
    centered_resid = residuals - np.mean(residuals, axis=0, keepdims=True)

    opt_method = method or ("nelder-mead" if result.method in ("nelder-mead", "l-bfgs-b") else "l-bfgs-b")
    rng = np.random.default_rng(seed)

    records = []
    # Starting guess is the optimum from the original fit
    theta0_dict = result.estimates

    alpha_half = (1.0 - ci_level) / 2.0
    lower_pct = alpha_half * 100.0
    upper_pct = (1.0 - alpha_half) * 100.0

    for b in range(1, n_boot + 1):
        # Sample residuals with replacement across time steps
        idx = rng.choice(len(centered_resid), size=len(centered_resid), replace=True)
        boot_y = pred + centered_resid[idx]

        # Construct perturbed dataset
        boot_ds = Dataset(
            t=ds.t,
            y=boot_y,
            observed=ds.observed,
            names=ds.names,
            clean=pred,
            meta=dict(ds.meta)
        )

        boot_prob = prob.replace(data=boot_ds)
        # Fast local fit starting right at the optimal solution
        fit_b = fit_least_squares(
            boot_prob,
            method=opt_method,
            theta0=theta0_dict,
            max_iter=300,
            seed=int(rng.integers(1e8))
        )

        row = dict(fit_b.estimates)
        records.append(row)

        if callback is not None:
            callback(b, n_boot)

    df_samples = pd.DataFrame(records)

    # Compute summary statistics
    summary_rows = []
    for col in df_samples.columns:
        vals = df_samples[col].values
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ci_lo = float(np.percentile(vals, lower_pct))
        ci_hi = float(np.percentile(vals, upper_pct))
        orig_val = float(result.estimates.get(col, np.nan))

        summary_rows.append({
            "Parameter": col,
            "Estimate": orig_val,
            "Mean": mean_val,
            "Std_Error": std_val,
            "CI_Lower": ci_lo,
            "CI_Upper": ci_hi
        })

    df_summary = pd.DataFrame(summary_rows).set_index("Parameter")
    return BootstrapResult(samples=df_samples, summary=df_summary, ci_level=ci_level)
