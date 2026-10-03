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
    initial_sidebar_state="auto"
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

import os
import pickle

@st.cache_resource
def load_example_result():
    asset_path = os.path.join(os.path.dirname(__file__), "fracid", "assets", "example_chen.pkl")
    with open(asset_path, "rb") as f:
        return pickle.load(f)

# Action buttons at top of main page (accessible even when sidebar is collapsed)
col_btn1, col_btn2 = st.columns([1, 1])
with col_btn1:
    run_main_clicked = st.button("🚀 Run System Identification", type="primary", width="stretch", key="run_main")
with col_btn2:
    load_example_clicked = st.button("✨ Load Example Result (Fractional Chen)", width="stretch", key="load_example")

if load_example_clicked:
    ex = load_example_result()
    st.session_state["frac_res"] = ex["frac_res"]
    st.session_state["int_res"] = ex["int_res"]
    for k in ("boot_res", "sens_df"):
        st.session_state.pop(k, None)
    st.success("Loaded pre-computed Fractional Chen benchmark result!")
    st.rerun()

# Render Sidebar
cfg = render_sidebar()

model = cfg["model"]
dataset = cfg["dataset"]
method = cfg["method"]
compare_int = cfg["compare_int"]
alpha_bounds = cfg["alpha_bounds"]
acc = cfg["accuracy"]
bounds = cfg["bounds"]

# Sidebar run button
st.sidebar.markdown("---")
run_sidebar_clicked = st.sidebar.button("🚀 Run System Identification", type="primary", width="stretch", key="run_sidebar")
run_clicked = run_main_clicked or run_sidebar_clicked

if dataset is None or model is None:
    st.info("👈 Please select or configure a dataset in the sidebar to begin.")
    st.stop()

# Session state caching
if "frac_res" not in st.session_state:
    st.session_state["frac_res"] = None
if "int_res" not in st.session_state:
    st.session_state["int_res"] = None

def _run_fit(prob, label: str, progress, budget: int):
    """Run the selected estimator with a live progress bar."""
    def cb(best, n):
        frac = min(n / budget, 0.99)
        progress.progress(frac, text=f"{label}: {n} evaluations · best objective {best:.4g}")

    if method == "fPINN (Neural Network)":
        epochs = acc["epochs"]
        return fit_fpinn(prob, epochs=epochs, lr=3e-3, n_colloc=120,
                         callback=lambda loss, ep: progress.progress(
                             min(ep / epochs, 0.99), text=f"{label}: epoch {ep}/{epochs} · loss {loss:.4g}"))
    return fit_least_squares(prob, method=method, max_iter=acc["max_iter"], popsize=acc["popsize"],
                             polish_iter=acc["polish_iter"], callback=cb)


def execute_identification():
    try:
        x0_mode = cfg.get("x0_mode", "model")
        prob = FitProblem(model=model, data=dataset, alpha_bounds=alpha_bounds, bounds=bounds, x0_mode=x0_mode)
    except (ValueError, KeyError) as e:
        st.error(f"Invalid problem set-up: {e}")
        st.stop()

    p = prob.n_par
    if method == "differential-evolution":
        budget = (acc["max_iter"] + 1) * acc["popsize"] * p + acc["polish_iter"] * (p + 1)
    elif method == "l-bfgs-b":
        budget = acc["max_iter"] * (p + 1) // 3
    else:
        budget = 400 * p

    with st.status(f"Identifying the fractional system with **{method}** …", expanded=True) as status:
        bar = st.progress(0.0, text="Starting…")
        frac_res = _run_fit(prob, "Fractional model", bar, budget)
        bar.progress(1.0, text=f"Fractional model done in {frac_res.runtime:.1f} s")
        st.session_state["frac_res"] = frac_res

        st.session_state["int_res"] = None
        if compare_int:
            bar2 = st.progress(0.0, text="Fitting integer-order competitor (α = 1)…")
            prob_int = prob.integer_order()
            int_res = _run_fit(prob_int, "Integer model", bar2, budget)
            bar2.progress(1.0, text=f"Integer model done in {int_res.runtime:.1f} s")
            st.session_state["int_res"] = int_res
        for k in ("boot_res", "sens_df"):
            st.session_state.pop(k, None)
        status.update(label="Identification finished ✅", state="complete", expanded=False)


if run_clicked:
    execute_identification()

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

# Footer: authorship & citation
st.divider()
st.markdown(
    "<div style='font-size:0.9rem;color:#666'>"
    "Developed by <b>Dr. Md Samshad Hussain Ansari</b> · "
    "<a href='https://orcid.org/0000-0002-7757-3216' target='_blank'>ORCID 0000-0002-7757-3216</a> · "
    "<a href='https://github.com/mdsamhussain1996/fracid' target='_blank'>Source on GitHub</a>"
    "</div>", unsafe_allow_html=True)
with st.expander("📚 How to cite FracID"):
    st.code("""@software{ansari_fracid,
  author = {Ansari, Md Samshad Hussain},
  title  = {{FracID}: Identification of Fractional-Order Dynamical Systems from Time-Series Data},
  url    = {https://github.com/mdsamhussain1996/fracid},
  year   = {2026}
}""", language="bibtex")
