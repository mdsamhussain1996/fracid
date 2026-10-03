"""UI Presentation views and visualization tabs for FracID."""
from __future__ import annotations

import io
import pandas as pd
import streamlit as st

from ..diagnostics import (BootstrapResult, ModelMetrics,
                           compare_fractional_vs_integer,
                           compute_alpha_sensitivity, compute_metrics,
                           compute_relative_errors, fig_to_bytes,
                           plot_phase_portrait_matplotlib,
                           plot_phase_portrait_plotly,
                           plot_residuals_matplotlib,
                           plot_trajectory_matplotlib, plot_trajectory_plotly,
                           run_bootstrap)
from ..estimation.least_squares import FitResult


def render_overview_cards(frac_res: FitResult, int_res: FitResult | None = None):
    """Render top metric cards comparing fractional and integer models."""
    cols = st.columns(4)

    # 1. Estimated alpha
    cols[0].metric(
        label="Estimated Fractional Order α",
        value=f"{frac_res.alpha:.4f}",
        delta=None if "true_alpha" not in frac_res.problem.data.meta else
        f"True: {frac_res.problem.data.meta['true_alpha']:.3f}"
    )

    # 2. RMSE
    cols[1].metric(
        label="RMSE (Data units)",
        value=f"{frac_res.rmse:.4e}",
        delta=f"Int: {int_res.rmse:.4e}" if int_res else None,
        delta_color="inverse"
    )

    # 3. Model AIC / BIC
    m_frac = compute_metrics(frac_res)
    cols[2].metric(
        label="AIC (Fractional)",
        value=f"{m_frac.aic:.1f}",
        delta=f"ΔAIC: {int_res.problem.sse - frac_res.sse:.1f}" if int_res else None
    )

    # 4. Runtime / Convergence
    cols[3].metric(
        label="Runtime / Evaluations",
        value=f"{frac_res.runtime:.2f} s",
        delta=f"{frac_res.nfev} evals"
    )


def render_main_tabs(frac_res: FitResult, int_res: FitResult | None = None):
    """Render the tabbed diagnostics and export panels."""
    t1, t2, t3, t4, t5, t6, t7 = st.tabs([
        "📈 Trajectory Fit",
        "🌀 Phase Portrait",
        "📊 Residuals",
        "⚖️ Model Comparison",
        "🎯 Bootstrap CIs",
        "🔍 Alpha Sensitivity",
        "📐 LaTeX & Export"
    ])

    # Tab 1: Trajectory Fit
    with t1:
        st.subheader("Fitted vs. Observed Trajectories")
        fig_tr = plot_trajectory_plotly(frac_res)
        st.plotly_chart(fig_tr, use_container_width=True)

        if int_res:
            with st.expander("Compare with Integer-order Trajectory", expanded=False):
                fig_int = plot_trajectory_plotly(int_res)
                st.plotly_chart(fig_int, use_container_width=True)

    # Tab 2: Phase Portrait
    with t2:
        st.subheader("Phase Space Trajectory")
        fig_phase = plot_phase_portrait_plotly(frac_res)
        st.plotly_chart(fig_phase, use_container_width=True)

    # Tab 3: Residuals
    with t3:
        st.subheader("Residual Analysis")
        fig_res = plot_residuals_matplotlib(frac_res)
        st.pyplot(fig_res)

        st.markdown("**Summary Statistics of Residuals:**")
        resid_df = pd.DataFrame(frac_res.residuals(), columns=list(frac_res.problem.data.names))
        st.dataframe(resid_df.describe().T[["mean", "std", "min", "50%", "max"]])

    # Tab 4: Model Comparison
    with t4:
        st.subheader("Fractional vs Integer-order Model Selection")
        if int_res is not None:
            comp_df = compare_fractional_vs_integer(frac_res, int_res)
            st.dataframe(comp_df.style.highlight_min(subset=["RMSE", "AIC", "BIC"], color="#d4edda"))

            verdict = comp_df.attrs.get("verdict", "")
            daic = comp_df.attrs.get("delta_aic", 0.0)
            if daic > 10:
                st.success(f"🏆 **Verdict**: {verdict} (ΔAIC = {daic:.2f})")
            elif daic > 0:
                st.info(f"ℹ️ **Verdict**: {verdict} (ΔAIC = {daic:.2f})")
            else:
                st.warning(f"⚠️ **Verdict**: {verdict} (ΔAIC = {daic:.2f})")
        else:
            st.info("Integer-order model was not run. Check the option in sidebar to compare.")

        # Ground truth recovery (if synthetic)
        rel_errs = compute_relative_errors(frac_res)
        if rel_errs:
            st.subheader("Ground Truth Parameter Recovery Errors")
            err_df = pd.DataFrame([{"Parameter": k, "Relative Error (%)": f"{v:.3f} %"} for k, v in rel_errs.items()])
            st.table(err_df)

    # Tab 5: Bootstrap CIs
    with t5:
        st.subheader("Bootstrap Uncertainty & Confidence Intervals")
        st.markdown("Quantify parameter estimation uncertainty through residual bootstrap.")
        n_boot = st.slider("Bootstrap Replicates", min_value=10, max_value=60, value=25, step=5)

        if st.button("🚀 Run Bootstrap Analysis"):
            pbar = st.progress(0, text="Resampling residuals...")
            boot_res = run_bootstrap(
                frac_res,
                n_boot=n_boot,
                callback=lambda b, total: pbar.progress(b / total, text=f"Bootstrap replicate {b}/{total}")
            )
            pbar.empty()
            st.success("Bootstrap completed!")
            st.dataframe(boot_res.summary.style.format("{:.4f}"))

    # Tab 6: Alpha Sensitivity
    with t6:
        st.subheader("Loss Profile vs Fractional Order α")
        st.markdown("Examine the loss profile :math:`J(\\alpha)` across orders :math:`\\alpha \\in [0.4, 1.0]`.")

        if st.button("⚡ Compute Sensitivity Curve"):
            with st.spinner("Evaluating loss profile..."):
                sens_df = compute_alpha_sensitivity(frac_res, n_points=25, mode="slice")
            st.line_chart(sens_df.set_index("alpha")["sse"])
            st.caption("Sharp minimum corresponds to the optimal fractional order.")

    # Tab 7: LaTeX & Publication Export
    with t7:
        st.subheader("Mathematical Model Formulation (LaTeX)")
        latex_str = frac_res.problem.model.latex(alpha=frac_res.alpha, params=frac_res.params)
        st.latex(latex_str)

        with st.expander("Show Raw LaTeX Source Code", expanded=False):
            st.code(latex_str, language="latex")

        st.divider()
        st.subheader("Publication-Quality Figure Export (300 DPI)")
        c1, c2 = st.columns(2)

        fig_pub_tr = plot_trajectory_matplotlib(frac_res, dpi=300)
        with c1:
            png_bytes = fig_to_bytes(fig_pub_tr, fmt="png", dpi=300)
            st.download_button(
                label="📥 Download Trajectory Figure (300 DPI PNG)",
                data=png_bytes,
                file_name="fracid_trajectory_fit.png",
                mime="image/png"
            )

        with c2:
            pdf_bytes = fig_to_bytes(fig_pub_tr, fmt="pdf")
            st.download_button(
                label="📥 Download Vector PDF Figure",
                data=pdf_bytes,
                file_name="fracid_trajectory_fit.pdf",
                mime="application/pdf"
            )

        # Download results CSV
        st.divider()
        st.subheader("Download Parameter Estimation Results")
        param_table = pd.DataFrame([{
            "Parameter": k,
            "Estimated Value": v
        } for k, v in frac_res.estimates.items()])
        st.download_button(
            label="📥 Download Estimated Parameters (CSV)",
            data=param_table.to_csv(index=False),
            file_name="fracid_parameters.csv",
            mime="text/csv"
        )
