"""Tests for Step 6: Real-data benchmark dataset and case study."""
import os
import pandas as pd
import pytest

from fracid.data import Dataset
from fracid.estimation import FitProblem, fit_least_squares
from fracid.models import FractionalSIR


def test_real_dataset_file_exists():
    pkg_dir = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(pkg_dir, "data", "boarding_school_influenza_1978.csv")
    readme_path = os.path.join(pkg_dir, "data", "README.md")
    notebook_path = os.path.join(pkg_dir, "notebooks", "case_study_real_data.ipynb")

    assert os.path.exists(csv_path), f"CSV missing at {csv_path}"
    assert os.path.exists(readme_path), f"README missing at {readme_path}"
    assert os.path.exists(notebook_path), f"Notebook missing at {notebook_path}"

    df = pd.read_csv(csv_path)
    assert len(df) == 15
    assert set(df.columns) == {"t", "S", "I", "R"}
    assert df["t"].iloc[0] == 0.0
    assert df["t"].iloc[-1] == 14.0


def test_real_dataset_sir_fit():
    pkg_dir = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(pkg_dir, "data", "boarding_school_influenza_1978.csv")
    df = pd.read_csv(csv_path)

    ds = Dataset.from_frame(df, time_col="t", value_cols=["S", "I", "R"])
    model = FractionalSIR(params=dict(beta=1.6, gamma=0.45, N=763.0), x0=[760.0, 3.0, 0.0])

    prob = FitProblem(
        model=model,
        data=ds,
        free=["beta", "gamma"],
        alpha_bounds=(0.7, 1.0),
        bounds=dict(beta=(0.5, 3.0), gamma=(0.1, 1.5))
    )

    assert prob.n_par == 3
    res = fit_least_squares(prob, method="differential-evolution", max_iter=20, popsize=8, seed=42)
    assert 0.7 <= res.alpha <= 1.0
    assert 0.5 <= res.params["beta"] <= 3.0
    assert 0.1 <= res.params["gamma"] <= 1.5
    assert res.rmse > 0.0
