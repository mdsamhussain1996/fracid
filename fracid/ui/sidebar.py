"""Sidebar configuration for FracID Streamlit app."""
from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from ..data.dataset import Dataset
from ..data.generator import generate
from ..models import MODELS, FractionalModel, get_model


def render_sidebar() -> dict[str, Any]:
    """Render sidebar controls and return a dictionary of run configurations."""
    st.sidebar.title("🎛️ FracID Configuration")

    # 1. Data Source selection
    st.sidebar.header("1. Data Source")
    data_source = st.sidebar.radio("Data Source", ["Synthetic Generator", "Upload CSV"], index=0)

    model_name = "Fractional Chen"
    model: FractionalModel | None = None
    dataset: Dataset | None = None

    if data_source == "Synthetic Generator":
        model_name = st.sidebar.selectbox("Dynamical Model", list(MODELS.keys()), index=2)
        model_cls = MODELS[model_name]
        default_model = model_cls()

        with st.sidebar.expander("Ground Truth Parameters & IC", expanded=False):
            true_alpha = st.slider("True Order α", min_value=0.1, max_value=1.0,
                                   value=float(default_model.default_alpha), step=0.01)

            param_inputs = {}
            for p_name in default_model.param_names:
                def_val = float(default_model.default_params[p_name])
                param_inputs[p_name] = st.number_input(f"Param: {p_name}", value=def_val)

            x0_inputs = []
            for idx, s_name in enumerate(default_model.state_names):
                def_x0 = float(default_model.default_x0[idx])
                x0_inputs.append(st.number_input(f"x0 [{s_name}]", value=def_x0))

        # Simulation horizon & noise
        col1, col2 = st.sidebar.columns(2)
        with col1:
            T_sim = st.number_input("Horizon T", value=float(default_model.default_T), min_value=0.1)
        with col2:
            h_sim = st.number_input("Step h", value=float(default_model.default_h), min_value=0.0005, format="%.4f")

        noise_pct = st.sidebar.slider("Noise level (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.5)
        seed = st.sidebar.number_input("Random Seed", value=42, step=1)

        # Build model and generate synthetic dataset
        model = model_cls(alpha=true_alpha, params=param_inputs, x0=x0_inputs)
        dataset = generate(
            model=model,
            T=T_sim,
            h=h_sim,
            noise_percent=noise_pct if noise_pct > 0 else None,
            seed=int(seed),
            fine_factor=2
        )
        st.sidebar.success(f"Generated {dataset.n} samples ({', '.join(dataset.names)})")

    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV (time in col 0)", type=["csv"])
        model_name = st.sidebar.selectbox("Candidate Model Template", list(MODELS.keys()), index=2)
        model = MODELS[model_name]()

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.sidebar.write("Columns found:", list(df.columns))
                time_col = st.sidebar.selectbox("Time column", list(df.columns), index=0)
                val_cols = [c for c in df.columns if c != time_col]
                sel_cols = st.sidebar.multiselect("State columns to use", val_cols, default=val_cols[:model.dim])

                if sel_cols:
                    dataset = Dataset.from_frame(df, time_col=time_col, value_cols=sel_cols)
                    st.sidebar.success(f"Loaded {dataset.n} observations")
            except Exception as e:
                st.sidebar.error(f"Error loading CSV: {e}")

    # 2. Estimation Setup
    st.sidebar.header("2. Estimation Method")
    method = st.sidebar.selectbox(
        "Algorithm",
        ["differential-evolution", "l-bfgs-b", "nelder-mead", "fPINN (Neural Network)"],
        index=0
    )

    compare_int = st.sidebar.checkbox("Compare with Integer-order (α=1)", value=True)

    with st.sidebar.expander("Search Bounds for α and Parameters", expanded=False):
        alpha_min = st.slider("Min α", min_value=0.1, max_value=0.9, value=0.4, step=0.05)
        alpha_max = st.slider("Max α", min_value=alpha_min + 0.05, max_value=1.0, value=1.0, step=0.05)
        alpha_bounds = (alpha_min, alpha_max)

        bounds = {}
        if model is not None:
            for p_name in model.default_free:
                def_lo, def_hi = model.default_bounds.get(p_name, (0.1, 10.0))
                c1, c2 = st.columns(2)
                with c1:
                    lo = st.number_input(f"{p_name} min", value=float(def_lo), key=f"lo_{p_name}")
                with c2:
                    hi = st.number_input(f"{p_name} max", value=float(def_hi), key=f"hi_{p_name}")
                bounds[p_name] = (lo, hi)

    return {
        "data_source": data_source,
        "model_name": model_name,
        "model": model,
        "dataset": dataset,
        "method": method,
        "compare_int": compare_int,
        "alpha_bounds": alpha_bounds,
        "bounds": bounds,
    }
