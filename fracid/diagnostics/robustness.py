r"""Identifiability and noise-robustness Monte-Carlo study across noise levels and random seeds."""
from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data.generator import generate
from ..estimation.least_squares import fit_least_squares
from ..estimation.problem import FitProblem
from ..models.base import FractionalModel


def run_robustness_study(
    model: FractionalModel,
    noise_levels: list[float] | None = None,
    n_seeds: int = 5,
    T: float | None = None,
    h: float | None = None,
    alpha_bounds: tuple[float, float] = (0.4, 1.0),
    max_iter: int = 60,
    popsize: int = 12,
    polish_iter: int = 80,
    callback: Callable[[int, int, float, int], None] | None = None,
) -> pd.DataFrame:
    r"""Run a Monte-Carlo noise robustness study across noise levels and random seeds.

    Parameters
    ----------
    model : FractionalModel
        The benchmark model instance with ground-truth parameters.
    noise_levels : list of float, optional
        Noise percentages to evaluate (default: ``[0.0, 1.0, 2.0, 5.0, 10.0, 15.0]``).
    n_seeds : int, default 5
        Number of random dataset seeds per noise level.
    T : float, optional
        Simulation time horizon (defaults to ``model.default_T``).
    h : float, optional
        Sampling step size (defaults to ``model.default_h``).
    alpha_bounds : (float, float), default (0.4, 1.0)
        Optimization search bounds for fractional order :math:`\alpha`.
    max_iter : int, default 60
        Differential evolution maximum generations.
    popsize : int, default 12
        Population multiplier.
    polish_iter : int, default 80
        L-BFGS-B polishing iterations.
    callback : callable, optional
        ``callback(completed_runs, total_runs, current_noise, current_seed)``.

    Returns
    -------
    pd.DataFrame
        Table of simulation outcomes with columns for estimates, relative errors, and runtimes.
    """
    if noise_levels is None:
        noise_levels = [0.0, 1.0, 2.0, 5.0, 10.0, 15.0]

    T_sim = float(model.default_T if T is None else T)
    h_sim = float(model.default_h if h is None else h)

    true_alpha = float(model.default_alpha)
    true_params = dict(model.default_params)
    free_params = list(model.default_free)

    total_runs = len(noise_levels) * n_seeds
    completed = 0
    records = []

    for noise in noise_levels:
        for s_idx in range(n_seeds):
            seed = int(42 + s_idx * 17)
            # Generate synthetic observation
            ds = generate(
                model=model,
                T=T_sim,
                h=h_sim,
                noise_percent=float(noise) if noise > 0.0 else None,
                seed=seed,
                fine_factor=2,
            )

            prob = FitProblem(
                model=model,
                data=ds,
                free=free_params,
                alpha_bounds=alpha_bounds,
            )

            fit = fit_least_squares(
                prob,
                method="differential-evolution",
                max_iter=max_iter,
                popsize=popsize,
                polish_iter=polish_iter,
                seed=seed,
            )

            alpha_err = abs(fit.alpha - true_alpha) / true_alpha * 100.0

            row = {
                "noise_pct": float(noise),
                "seed": seed,
                "alpha_true": true_alpha,
                "alpha_est": fit.alpha,
                "alpha_rel_err_pct": alpha_err,
                "runtime_s": fit.runtime,
                "sse": fit.sse,
                "rmse": fit.rmse,
            }

            for p_name in free_params:
                p_true = float(true_params[p_name])
                p_est = float(fit.params.get(p_name, np.nan))
                p_err = abs(p_est - p_true) / abs(p_true) * 100.0 if abs(p_true) > 1e-12 else 0.0
                row[f"{p_name}_true"] = p_true
                row[f"{p_name}_est"] = p_est
                row[f"{p_name}_rel_err_pct"] = p_err

            records.append(row)
            completed += 1
            if callback is not None:
                callback(completed, total_runs, float(noise), seed)

    return pd.DataFrame(records)


def plot_robustness_matplotlib(df: pd.DataFrame, model_name: str = "") -> plt.Figure:
    r"""Generate publication-ready 2-panel figure of noise robustness.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`run_robustness_study`.
    model_name : str, optional
        Name of model for plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    # 1. Alpha error vs noise
    grp = df.groupby("noise_pct")["alpha_rel_err_pct"]
    means = grp.mean()
    stds = grp.std().fillna(0.0)
    noises = means.index.to_numpy(dtype=float)

    ax1.plot(noises, means.to_numpy(), "o-", color="#1e3c72", lw=2, ms=6, label=r"Mean $\hat\alpha$ Error")
    ax1.fill_between(
        noises,
        np.maximum(0, means - stds),
        means + stds,
        color="#1e3c72",
        alpha=0.2,
        label=r"$\pm 1\ \sigma$ band",
    )
    ax1.set_xlabel("Noise Level (%)", fontsize=11, fontweight="bold")
    ax1.set_ylabel(r"Order $\alpha$ Relative Error (%)", fontsize=11, fontweight="bold")
    ax1.set_title(r"Fractional Order Identifiability $\hat\alpha$", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")

    # 2. Parameter recovery boxplots or mean errors
    param_cols = [c for c in df.columns if c.endswith("_rel_err_pct") and c != "alpha_rel_err_pct"]
    colors = ["#e74c3c", "#2ecc71", "#9b59b6", "#e67e22", "#1abc9c"]

    for idx, col in enumerate(param_cols):
        p_name = col.replace("_rel_err_pct", "")
        p_grp = df.groupby("noise_pct")[col]
        p_mean = p_grp.mean()
        p_std = p_grp.std().fillna(0.0)
        c = colors[idx % len(colors)]
        ax2.plot(noises, p_mean.to_numpy(), "s--", color=c, lw=1.8, ms=5, label=f"Param {p_name}")
        ax2.fill_between(
            noises,
            np.maximum(0, p_mean - p_std),
            p_mean + p_std,
            color=c,
            alpha=0.12,
        )

    ax2.set_xlabel("Noise Level (%)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Parameter Relative Error (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Physical Parameter Recovery", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left")

    title = f"Noise Robustness & Identifiability Study ({model_name})" if model_name else "Noise Robustness Study"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig
