"""Tests for Step 3: Noise robustness and identifiability Monte-Carlo study."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from fracid.diagnostics.robustness import plot_robustness_matplotlib, run_robustness_study
from fracid.models import FractionalLinear


def test_run_robustness_study():
    model = FractionalLinear(alpha=0.90)
    records = []
    df = run_robustness_study(
        model=model,
        noise_levels=[0.0, 5.0],
        n_seeds=2,
        T=1.0,
        h=0.01,
        max_iter=15,
        popsize=6,
        polish_iter=15,
        callback=lambda d, tot, n, s: records.append((d, tot, n, s))
    )

    assert len(df) == 4
    assert len(records) == 4
    assert "noise_pct" in df.columns
    assert "alpha_rel_err_pct" in df.columns
    assert "a11_rel_err_pct" in df.columns
    assert np.all(np.isfinite(df["alpha_est"]))
    assert np.all(df["alpha_rel_err_pct"] >= 0.0)

    fig = plot_robustness_matplotlib(df, model_name=model.name)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
