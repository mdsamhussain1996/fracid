r"""Model comparison metrics and statistical diagnostics.

Computes:
- Sum of Squared Errors (SSE) and Root Mean Squared Error (RMSE)
- Akaike Information Criterion (AIC, AICc)
- Bayesian Information Criterion (BIC)
- Comparison table between Fractional (estimated :math:`\alpha`) and Integer-order (:math:`\alpha=1`) models
- Relative recovery error when ground truth is known
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from ..estimation.least_squares import FitResult


@dataclass
class ModelMetrics:
    """Statistical summary of a fitted dynamical model."""
    name: str
    alpha: float
    p: int          # Number of estimated parameters
    N: int          # Total number of scalar observations
    sse: float
    rmse: float
    aic: float
    aicc: float
    bic: float
    estimates: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "Model": self.name,
            "alpha": self.alpha,
            "Parameters (p)": self.p,
            "Observations (N)": self.N,
            "SSE": self.sse,
            "RMSE": self.rmse,
            "AIC": self.aic,
            "AICc": self.aicc,
            "BIC": self.bic,
        }


def compute_metrics(result: FitResult, label: str = "Model") -> ModelMetrics:
    r"""Calculate goodness-of-fit and information criteria for a FitResult.

    Formulas
    --------
    .. math::
        \text{RMSE} = \sqrt{\frac{1}{N}\sum (y - \hat{y})^2}, \quad
        \text{AIC} = N \ln\left(\frac{\text{SSE}}{N}\right) + 2p, \quad
        \text{BIC} = N \ln\left(\frac{\text{SSE}}{N}\right) + p \ln(N).
    """
    prob = result.problem
    ds = prob.data
    pred = result.prediction()
    if pred is None or not np.all(np.isfinite(pred)):
        sse = float("inf")
        rmse = float("inf")
    else:
        diff = ds.y - pred
        sse = float(np.sum(diff ** 2))
        rmse = float(np.sqrt(np.mean(diff ** 2)))

    N = ds.y.size
    p = result.n_par
    safe_sse = max(sse, 1e-15)

    aic = float(N * np.log(safe_sse / N) + 2 * p)
    bic = float(N * np.log(safe_sse / N) + p * np.log(N))
    if N - p - 1 > 0:
        aicc = aic + (2 * p * (p + 1)) / (N - p - 1)
    else:
        aicc = aic

    return ModelMetrics(
        name=label,
        alpha=result.alpha,
        p=p,
        N=N,
        sse=sse,
        rmse=rmse,
        aic=aic,
        aicc=aicc,
        bic=bic,
        estimates=result.estimates
    )


def compare_fractional_vs_integer(
    frac_result: FitResult,
    int_result: FitResult
) -> pd.DataFrame:
    r"""Generate a side-by-side comparison table between fractional and integer-order models.

    Includes :math:`\Delta\text{AIC} = \text{AIC}_{\text{int}} - \text{AIC}_{\text{frac}}` and
    relative evidence weights (Akaike weights).
    """
    m_frac = compute_metrics(frac_result, label="Fractional (estimated α)")
    m_int = compute_metrics(int_result, label="Integer-order (α = 1)")

    rows = [m_frac.to_dict(), m_int.to_dict()]
    df = pd.DataFrame(rows)

    # Compute delta AIC & BIC
    delta_aic = m_int.aic - m_frac.aic
    delta_bic = m_int.bic - m_frac.bic

    # Add comparison annotations
    df["ΔAIC (vs Integer)"] = [delta_aic, 0.0]
    df["ΔBIC (vs Integer)"] = [delta_bic, 0.0]

    # Model preference interpretation
    if delta_aic > 10.0:
        verdict = "Decisive support for Fractional model (ΔAIC > 10)"
    elif delta_aic > 4.0:
        verdict = "Substantial support for Fractional model (ΔAIC > 4)"
    elif delta_aic < -4.0:
        verdict = "Support for Integer-order model (ΔAIC < -4)"
    else:
        verdict = "Inconclusive evidence between models (|ΔAIC| ≤ 4)"

    df.attrs["verdict"] = verdict
    df.attrs["delta_aic"] = delta_aic
    df.attrs["delta_bic"] = delta_bic
    return df


def compute_relative_errors(result: FitResult) -> dict[str, float] | None:
    r"""Compute percentage relative errors against known ground truth for synthetic data.

    .. math::
        e_{\text{rel}} = \frac{|\hat{\theta} - \theta^*|}{|\theta^*|} \times 100\%
    """
    meta = result.problem.data.meta
    if "true_alpha" not in meta:
        return None

    true_alpha = meta["true_alpha"]
    true_params = meta.get("true_params", {})
    estimates = result.estimates

    errors = {}
    if "alpha" in estimates:
        errors["alpha"] = abs(estimates["alpha"] - true_alpha) / abs(true_alpha) * 100.0

    for k in result.problem.free:
        if k in estimates and k in true_params:
            tv = true_params[k]
            denom = abs(tv) if abs(tv) > 1e-6 else 1.0
            errors[k] = abs(estimates[k] - tv) / denom * 100.0

    return errors
