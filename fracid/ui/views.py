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

    # 3. Model AIC / BIC  (ΔAIC = AIC_int − AIC_frac; positive favours the fractional model)
    m_frac = compute_metrics(frac_res)
    if int_res is not None:
        m_int = compute_metrics(int_res)
        cols[2].metric(
            label="AIC (Fractional)",
            value=f"{m_frac.aic:.1f}",
            delta=f"ΔAIC vs integer: {m_int.aic - m_frac.aic:+.1f}",
            help="ΔAIC = AIC(α=1) − AIC(fractional). Above 10 = decisive support for the fractional model."
        )
    else:
        cols[2].metric(label="AIC (Fractional)", value=f"{m_frac.aic:.1f}")

    # 4. Runtime / Convergence
    cols[3].metric(
        label="Runtime / Evaluations",
        value=f"{frac_res.runtime:.2f} s",
        delta=f"{frac_res.nfev} evals"
    )


def render_main_tabs(frac_res: FitResult, int_res: FitResult | None = None):
    """Render the tabbed diagnostics and export panels."""
    t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs([
        "📈 Trajectory Fit",
        "🌀 Phase Portrait",
        "📊 Residuals",
        "⚖️ Model Comparison",
        "🎯 Bootstrap CIs",
        "🔍 Alpha Sensitivity",
        "📐 LaTeX & Export",
        "📖 Theory"
    ])

    # Tab 1: Trajectory Fit
    with t1:
        st.subheader("Fitted vs. Observed Trajectories")
        fig_tr = plot_trajectory_plotly(frac_res)
        st.plotly_chart(fig_tr, width="stretch")

        if int_res:
            with st.expander("Compare with Integer-order Trajectory", expanded=False):
                fig_int = plot_trajectory_plotly(int_res)
                st.plotly_chart(fig_int, width="stretch")

    # Tab 2: Phase Portrait
    with t2:
        st.subheader("Phase Space Trajectory")
        fig_phase = plot_phase_portrait_plotly(frac_res)
        st.plotly_chart(fig_phase, width="stretch")

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
            meta = frac_res.problem.data.meta
            truth = {"alpha": meta.get("true_alpha"), **meta.get("true_params", {})}
            err_df = pd.DataFrame([{
                "Parameter": k,
                "True": truth.get(k),
                "Estimated": frac_res.estimates.get(k),
                "Relative Error (%)": round(v, 3),
            } for k, v in rel_errs.items()])
            st.dataframe(err_df, hide_index=True, width="stretch")

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
            st.session_state["boot_res"] = boot_res

        boot_res = st.session_state.get("boot_res")
        if boot_res is not None:
            st.success(f"{len(boot_res.samples)} bootstrap replicates · {int(boot_res.ci_level * 100)} % intervals")
            st.dataframe(boot_res.summary.style.format("{:.4f}"))
            st.download_button("📥 Download bootstrap samples (CSV)", boot_res.samples.to_csv(index=False),
                               file_name="fracid_bootstrap_samples.csv", mime="text/csv")

    # Tab 6: Alpha Sensitivity
    with t6:
        st.subheader("Loss Profile vs Fractional Order α")
        st.markdown(
            "**Slice**: other parameters frozen at their estimates (instant). "
            "**Profile**: the other parameters are re-optimised for every α — the curve whose "
            "sharpness actually measures how well α is identified.")
        if "alpha" not in frac_res.estimates:
            st.info("α was fixed in this fit, so there is no α-profile to show.")
        else:
            mode = st.radio("Curve type", ["slice", "profile"], horizontal=True)
            n_pts = st.slider("Grid points", 10, 41, 21, step=1)
            if st.button("⚡ Compute Loss Curve"):
                pbar = st.progress(0.0, text="Evaluating loss curve…")
                st.session_state["sens_df"] = (mode, compute_alpha_sensitivity(
                    frac_res, n_points=n_pts, mode=mode,
                    callback=lambda i, n: pbar.progress(i / n, text=f"α point {i}/{n}")))
                pbar.empty()
            if st.session_state.get("sens_df") is not None:
                used_mode, sens_df = st.session_state["sens_df"]
                import plotly.graph_objects as go
                fig = go.Figure(go.Scatter(x=sens_df["alpha"], y=sens_df["sse"], mode="lines+markers",
                                           name=f"J(α) – {used_mode}"))
                fig.add_vline(x=frac_res.alpha, line_dash="dash",
                              annotation_text=f"α̂ = {frac_res.alpha:.3f}")
                true_a = frac_res.problem.data.meta.get("true_alpha")
                if true_a is not None:
                    fig.add_vline(x=true_a, line_dash="dot", line_color="green",
                                  annotation_text=f"true α = {true_a:.3f}", annotation_position="bottom right")
                fig.update_layout(xaxis_title="α", yaxis_title="normalised SSE  J(α)", yaxis_type="log",
                                  height=420, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, width="stretch")

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

    # Tab 8: Theory
    with t8:
        render_theory()


def render_theory():
    """Mathematical background of the forward and inverse problems."""
    st.subheader("Forward problem")
    st.markdown("Every model is a commensurate-order Caputo system")
    st.latex(r"{}^{C}\!D^{\alpha}\mathbf{x}(t)=\mathbf{f}\bigl(\mathbf{x}(t),\mathbf{x}(t-\tau);\theta\bigr),"
             r"\qquad \mathbf{x}(0)=\mathbf{x}_0,\quad 0<\alpha\le 1,")
    st.latex(r"{}^{C}\!D^{\alpha}x(t)=\frac{1}{\Gamma(1-\alpha)}\int_0^t (t-s)^{-\alpha}x'(s)\,ds .")
    st.markdown(
        "It is integrated with the **Adams–Bashforth–Moulton predictor–corrector** scheme of "
        "Diethelm, Ford & Freed (2002), convergence order $\\min(2,1+\\alpha)$, applied to the equivalent "
        "Volterra equation")
    st.latex(r"\mathbf{x}(t)=\mathbf{x}_0+\frac{1}{\Gamma(\alpha)}\int_0^t (t-s)^{\alpha-1}"
             r"\mathbf{f}\bigl(s,\mathbf{x}(s)\bigr)\,ds .")
    st.subheader("Inverse problem")
    st.markdown("The unknowns $\\vartheta=(\\alpha,\\theta)$ minimise the normalised least-squares misfit")
    st.latex(r"J(\vartheta)=\sum_{i=1}^{M}\sum_{k=1}^{K}\Bigl(\frac{y_k(t_i)-\hat x_{c_k}(t_i;\vartheta)}{s_k}\Bigr)^2,"
             r"\qquad s_k=\operatorname{rms}(y_k).")
    st.markdown(
        "- **Differential evolution** evaluates its whole population in one *batched* ABM sweep "
        "(all candidates integrated simultaneously), then polishes with L-BFGS-B whose finite-difference "
        "gradient also comes from a single batched sweep.\n"
        "- **fPINN**: a neural network $\\hat{\\mathbf x}_{\\phi}(t)$ is trained on data misfit + the residual "
        "of the fractional ODE (L1 discretisation of the Caputo derivative), with $\\alpha,\\theta$ as "
        "trainable scalars.")
    st.subheader("Model selection")
    st.latex(r"\mathrm{AIC}=N\ln\frac{\mathrm{SSE}}{N}+2p,\qquad \mathrm{BIC}=N\ln\frac{\mathrm{SSE}}{N}+p\ln N,"
             r"\qquad \Delta\mathrm{AIC}=\mathrm{AIC}_{\alpha=1}-\mathrm{AIC}_{\hat\alpha}.")
    st.markdown("$\\Delta\\mathrm{AIC}>10$: decisive evidence that memory (fractional order) is needed.")
    st.subheader("References")
    st.markdown(
        "1. K. Diethelm, N. J. Ford, A. D. Freed, *A predictor-corrector approach for the numerical solution "
        "of fractional differential equations*, Nonlinear Dynamics 29 (2002) 3–22.\n"
        "2. I. Podlubny, *Fractional Differential Equations*, Academic Press, 1999.\n"
        "3. G. Pang, L. Lu, G. E. Karniadakis, *fPINNs: Fractional physics-informed neural networks*, "
        "SIAM J. Sci. Comput. 41 (2019) A2603–A2626.\n"
        "4. R. Storn, K. Price, *Differential evolution*, J. Global Optimization 11 (1997) 341–359.")
