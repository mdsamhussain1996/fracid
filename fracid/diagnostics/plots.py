r"""Publication-quality plotting functions (Matplotlib 300 DPI and Plotly)."""
from __future__ import annotations

import io
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..estimation.least_squares import FitResult


def plot_trajectory_matplotlib(result: FitResult, figsize=(9, 5), dpi=300) -> plt.Figure:
    """Publication-quality matplotlib plot of fitted vs observed trajectories."""
    ds = result.problem.data
    pred = result.prediction()
    t = ds.t

    fig, axes = plt.subplots(len(ds.observed), 1, figsize=figsize, sharex=True, dpi=dpi)
    if len(ds.observed) == 1:
        axes = [axes]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, (col_name, ax) in enumerate(zip(ds.names, axes)):
        c = colors[idx % len(colors)]
        ax.scatter(t, ds.y[:, idx], s=12, color=c, alpha=0.6, label=f"Observed {col_name}")
        if pred is not None and np.all(np.isfinite(pred)):
            ax.plot(t, pred[:, idx], color="black", lw=1.8, label=f"Fit (α={result.alpha:.3f})")
        ax.set_ylabel(col_name, fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", frameon=True, fontsize=9)

    axes[-1].set_xlabel("Time t", fontsize=11, fontweight="bold")
    fig.suptitle(f"{result.problem.model.name} — Trajectory Fit", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_phase_portrait_matplotlib(result: FitResult, figsize=(7, 6), dpi=300) -> plt.Figure:
    """2D or 3D phase portrait comparing observed and fitted trajectories."""
    ds = result.problem.data
    pred = result.prediction()
    n_dim = ds.y.shape[1]

    fig = plt.figure(figsize=figsize, dpi=dpi)
    if n_dim >= 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(ds.y[:, 0], ds.y[:, 1], ds.y[:, 2], s=8, alpha=0.4, color="crimson", label="Observed")
        if pred is not None:
            ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], color="black", lw=1.8, label="Fitted")
        ax.set_xlabel(ds.names[0])
        ax.set_ylabel(ds.names[1])
        ax.set_zlabel(ds.names[2])
    elif n_dim == 2:
        ax = fig.add_subplot(111)
        ax.scatter(ds.y[:, 0], ds.y[:, 1], s=14, alpha=0.5, color="crimson", label="Observed")
        if pred is not None:
            ax.plot(pred[:, 0], pred[:, 1], color="black", lw=1.8, label="Fitted")
        ax.set_xlabel(ds.names[0])
        ax.set_ylabel(ds.names[1])
        ax.grid(True, linestyle="--", alpha=0.5)
    else:
        # 1D: plot state vs time
        ax = fig.add_subplot(111)
        ax.scatter(ds.t, ds.y[:, 0], s=14, alpha=0.5, color="crimson", label="Observed")
        if pred is not None:
            ax.plot(ds.t, pred[:, 0], color="black", lw=1.8, label="Fitted")
        ax.set_xlabel("Time t")
        ax.set_ylabel(ds.names[0])
        ax.grid(True, linestyle="--", alpha=0.5)

    ax.legend(frameon=True)
    fig.suptitle(f"{result.problem.model.name} — Phase Portrait", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_residuals_matplotlib(result: FitResult, figsize=(9, 4), dpi=300) -> plt.Figure:
    """Residual plot over time and residual distributions."""
    ds = result.problem.data
    resid = result.residuals()
    t = ds.t

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi, gridspec_kw={"width_ratios": [2.5, 1]})

    for idx, name in enumerate(ds.names):
        ax1.plot(t, resid[:, idx], "o-", markersize=3, alpha=0.7, label=name)
        ax2.hist(resid[:, idx], bins=20, alpha=0.5, orientation="horizontal", label=name)

    ax1.axhline(0, color="black", linestyle="--", lw=1)
    ax1.set_xlabel("Time t")
    ax1.set_ylabel("Residual (y - y_hat)")
    ax1.set_title("Residuals over Time")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.axhline(0, color="black", linestyle="--", lw=1)
    ax2.set_xlabel("Frequency")
    ax2.set_title("Distribution")
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    return fig


def plot_trajectory_plotly(result: FitResult) -> go.Figure:
    """Interactive Plotly figure for Streamlit."""
    ds = result.problem.data
    pred = result.prediction()
    t = ds.t

    fig = make_subplots(rows=len(ds.observed), cols=1, shared_xaxes=True,
                        subplot_titles=[f"State: {n}" for n in ds.names])

    for idx, name in enumerate(ds.names):
        # Observed
        fig.add_trace(
            go.Scatter(x=t, y=ds.y[:, idx], mode="markers", name=f"{name} (Obs)",
                       marker=dict(size=5, opacity=0.7)),
            row=idx + 1, col=1
        )
        # Fitted
        if pred is not None and np.all(np.isfinite(pred)):
            fig.add_trace(
                go.Scatter(x=t, y=pred[:, idx], mode="lines", name=f"{name} (Fit α={result.alpha:.3f})",
                           line=dict(color="black", width=2)),
                row=idx + 1, col=1
            )

    fig.update_layout(height=260 * len(ds.observed), template="plotly_white",
                      margin=dict(l=40, r=40, t=40, b=40))
    return fig


def plot_phase_portrait_plotly(result: FitResult) -> go.Figure:
    """Interactive 2D or 3D phase portrait in Plotly."""
    ds = result.problem.data
    pred = result.prediction()
    n_dim = ds.y.shape[1]

    if n_dim >= 3:
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=ds.y[:, 0], y=ds.y[:, 1], z=ds.y[:, 2],
            mode="markers", marker=dict(size=3, color="crimson", opacity=0.6),
            name="Observed"
        ))
        if pred is not None:
            fig.add_trace(go.Scatter3d(
                x=pred[:, 0], y=pred[:, 1], z=pred[:, 2],
                mode="lines", line=dict(color="black", width=4),
                name=f"Fitted (α={result.alpha:.3f})"
            ))
        fig.update_layout(
            scene=dict(xaxis_title=ds.names[0], yaxis_title=ds.names[1], zaxis_title=ds.names[2]),
            margin=dict(l=0, r=0, b=0, t=30), height=550
        )
    elif n_dim == 2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ds.y[:, 0], y=ds.y[:, 1],
            mode="markers", marker=dict(size=6, color="crimson", opacity=0.6),
            name="Observed"
        ))
        if pred is not None:
            fig.add_trace(go.Scatter(
                x=pred[:, 0], y=pred[:, 1],
                mode="lines", line=dict(color="black", width=2.5),
                name=f"Fitted (α={result.alpha:.3f})"
            ))
        fig.update_layout(
            xaxis_title=ds.names[0], yaxis_title=ds.names[1],
            template="plotly_white", margin=dict(l=40, r=40, t=40, b=40), height=450
        )
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ds.t, y=ds.y[:, 0], mode="markers", name="Observed"))
        if pred is not None:
            fig.add_trace(go.Scatter(x=ds.t, y=pred[:, 0], mode="lines", name="Fitted"))
        fig.update_layout(template="plotly_white", height=400)

    return fig


def fig_to_bytes(fig: plt.Figure, fmt: str = "png", dpi: int = 300) -> bytes:
    """Export Matplotlib figure to PNG or PDF bytes for downloading."""
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()
