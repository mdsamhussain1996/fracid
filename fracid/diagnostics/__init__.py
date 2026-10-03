"""Model diagnostics, comparison metrics, bootstrap uncertainty, and plotting."""
from .bootstrap import BootstrapResult, run_bootstrap
from .comparison import (ModelMetrics, compare_fractional_vs_integer,
                         compute_metrics, compute_relative_errors)
from .plots import (fig_to_bytes, plot_phase_portrait_matplotlib,
                    plot_phase_portrait_plotly, plot_residuals_matplotlib,
                    plot_trajectory_matplotlib, plot_trajectory_plotly)
from .sensitivity import compute_alpha_sensitivity

__all__ = [
    "ModelMetrics",
    "compute_metrics",
    "compare_fractional_vs_integer",
    "compute_relative_errors",
    "BootstrapResult",
    "run_bootstrap",
    "compute_alpha_sensitivity",
    "plot_trajectory_matplotlib",
    "plot_phase_portrait_matplotlib",
    "plot_residuals_matplotlib",
    "plot_trajectory_plotly",
    "plot_phase_portrait_plotly",
    "fig_to_bytes"
]
