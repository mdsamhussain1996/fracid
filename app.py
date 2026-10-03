"""FracID: Identification of Fractional-Order Dynamical Systems from Time-Series Data."""
from __future__ import annotations

import streamlit as st

from fracid.estimation import FitProblem, fit_fpinn, fit_least_squares
from fracid.ui import render_main_tabs, render_overview_cards, render_sidebar

# Configure page
st.set_page_config(
    page_title="FracID — Fractional System Identification",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for rich UI
st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #1e3c72, #2a5298, #ff4b4b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .badge-bar {
        margin-bottom: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">⚡ FracID: Fractional-Order System Identification</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">'
    'Identify fractional order <b>α</b> (0 &lt; α ≤ 1) and system parameters <b>θ</b> '
    'from measured or synthetic time series, compare fractional vs integer dynamics (AIC/BIC), '
    'and quantify uncertainty.</div>',
    unsafe_allow_html=True
)

# Render Sidebar
cfg = render_sidebar()

model = cfg["model"]
dataset = cfg["dataset"]
method = cfg["method"]
compare_int = cfg["compare_int"]
alpha_bounds = cfg["alpha_bounds"]
bounds = cfg["bounds"]

if dataset is None or model is None:
    st.info("👈 Please select or configure a dataset in the sidebar to begin.")
    st.stop()

# Action button
st.sidebar.markdown("---")
run_clicked = st.sidebar.button("🚀 Run System Identification", type="primary", use_container_width=True)

# Session state caching
if "frac_res" not in st.session_state:
    st.session_state["frac_res"] = None
if "int_res" not in st.session_state:
    st.session_state["int_res"] = None

if run_clicked:
    prob = FitProblem(
        model=model,
        data=dataset,
        alpha_bounds=alpha_bounds,
        bounds=bounds
    )

    with st.spinner(f"Identifying fractional system using {method}..."):
        if method == "fPINN (Neural Network)":
            frac_res = fit_fpinn(prob, epochs=600, lr=3e-3, n_colloc=120)
        else:
            frac_res = fit_least_squares(prob, method=method, max_iter=150)

        st.session_state["frac_res"] = frac_res

    # Integer model if requested
    if compare_int:
        with st.spinner("Fitting integer-order competitor (α = 1.0)..."):
            prob_int = prob.integer_order()
            if method == "fPINN (Neural Network)":
                int_res = fit_fpinn(prob_int, epochs=400, lr=3e-3, n_colloc=100)
            else:
                int_res = fit_least_squares(prob_int, method=method, max_iter=100)
            st.session_state["int_res"] = int_res
    else:
        st.session_state["int_res"] = None

    st.success("Identification finished successfully!")

# Display results
frac_res = st.session_state.get("frac_res")
int_res = st.session_state.get("int_res")

if frac_res is not None:
    render_overview_cards(frac_res, int_res)
    st.divider()
    render_main_tabs(frac_res, int_res)
else:
    st.markdown("""
    ### 📌 Quickstart Guide
    1. Select a benchmark model (e.g. **Fractional Chen**, **Fractional Lorenz**, **Fractional SIR**) or upload a CSV dataset.
    2. Choose an optimization algorithm:
       - **Differential Evolution**: Global search for complex and chaotic landscapes.
       - **L-BFGS-B / Nelder-Mead**: Fast local gradients / simplex search.
       - **fPINN**: Physics-informed neural network combining data misfit with L1 Caputo derivative residual.
    3. Click **🚀 Run System Identification** in the sidebar.
    """)
    # Preview generated data
    st.subheader(f"📊 Preview: {model.name} Data")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.line_chart(dataset.to_frame().set_index("t"))
    with c2:
        st.write("Summary Statistics:")
        st.dataframe(dataset.to_frame().describe().T)
