"""UI Presentation views and visualization tabs for FracID."""
from __future__ import annotations

import io
import pandas as pd
import streamlit as st

from ..diagnostics import (BootstrapResult, ModelMetrics,
                           compare_fractional_vs_integer,
                           compute_alpha_sensitivity, compute_loss_surface_2d,
                           compute_metrics, compute_profile_ci,
                           compute_relative_errors, fig_to_bytes,
                           plot_phase_portrait_matplotlib,
                           plot_phase_portrait_plotly,
                           plot_residuals_matplotlib,
                           plot_robustness_matplotlib,
                           plot_trajectory_matplotlib, plot_trajectory_plotly,
                           run_bootstrap, run_robustness_study)
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
    if int_res is not None:
        tot_runtime = frac_res.runtime + int_res.runtime
        tot_evals = frac_res.nfev + int_res.nfev
        cols[3].metric(
            label="Total Runtime (Frac + Int)",
            value=f"{tot_runtime:.2f} s",
            delta=f"Frac: {frac_res.runtime:.1f}s | Int: {int_res.runtime:.1f}s",
            help=f"Total: {tot_runtime:.2f} s across {tot_evals} evaluations (Fractional: {frac_res.runtime:.2f} s, Integer: {int_res.runtime:.2f} s)."
        )
    else:
        cols[3].metric(
            label="Runtime / Evaluations",
            value=f"{frac_res.runtime:.2f} s",
            delta=f"{frac_res.nfev} evals"
        )


def render_main_tabs(frac_res: FitResult, int_res: FitResult | None = None):
    """Render the tabbed diagnostics and export panels."""
    t1, t2, t3, t4, t5, t6, t7, t8, t9 = st.tabs([
        "📈 Trajectory Fit",
        "🌀 Phase Portrait",
        "📊 Residuals",
        "⚖️ Model Comparison",
        "🎯 Bootstrap CIs",
        "🔍 Alpha Sensitivity",
        "🧪 Robustness Study",
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

            # Compare with Profile-Likelihood CI if alpha profile has been computed
            sens_state = st.session_state.get("sens_df")
            if sens_state is not None:
                _, sens_df = sens_state
                from ..diagnostics import compute_profile_ci
                ci_lo, ci_hi, _ = sens_df.attrs.get("ci") or compute_profile_ci(frac_res, sens_df)
                if "alpha" in boot_res.summary.index:
                    boot_lo = boot_res.summary.loc["alpha", "CI Lower"]
                    boot_hi = boot_res.summary.loc["alpha", "CI Upper"]
                    ci_comp_df = pd.DataFrame([
                        {"Method": "Residual Bootstrap (95%)", "CI Lower": boot_lo, "CI Upper": boot_hi, "CI Width": boot_hi - boot_lo},
                        {"Method": "Profile Likelihood (95%)", "CI Lower": ci_lo, "CI Upper": ci_hi, "CI Width": ci_hi - ci_lo}
                    ])
                    st.dataframe(ci_comp_df.style.format({"CI Lower": "{:.4f}", "CI Upper": "{:.4f}", "CI Width": "{:.4f}"}), width="stretch")

    # Tab 6: Alpha Sensitivity & 2-D Loss Landscape
    with t6:
        st.subheader("Objective Loss Landscape Analysis")
        view_type = st.radio("Landscape Dimensionality", ["1-D α Curve (Slice / Profile)", "2-D Landscape Heatmap J(α, θ)"], horizontal=True)

        if view_type == "1-D α Curve (Slice / Profile)":
            st.markdown(
                "**Slice**: other parameters frozen at their estimates (instant batched sweep). "
                "**Profile**: the other parameters are re-optimised for every α using continuation — "
                "the curve whose sharpness measures parameter identifiability and profile-likelihood CIs.")
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
                    ci_lo, ci_hi, threshold = sens_df.attrs.get("ci") or compute_profile_ci(frac_res, sens_df)

                    st.info(f"📐 **Profile-Likelihood 95% Confidence Interval for α**: `[{ci_lo:.4f}, {ci_hi:.4f}]` (width: `{ci_hi - ci_lo:.4f}`, cut-off $J={threshold:.4g}$)")

                    fig = go.Figure(go.Scatter(x=sens_df["alpha"], y=sens_df["sse"], mode="lines+markers",
                                               name=f"J(α) – {used_mode}"))
                    fig.add_vline(x=frac_res.alpha, line_dash="dash",
                                  annotation_text=f"α̂ = {frac_res.alpha:.3f}")
                    true_a = frac_res.problem.data.meta.get("true_alpha")
                    if true_a is not None:
                        fig.add_vline(x=true_a, line_dash="dot", line_color="green",
                                      annotation_text=f"true α = {true_a:.3f}", annotation_position="bottom right")

                    # Cut-off line and CI region
                    if np.isfinite(threshold):
                        fig.add_hline(y=threshold, line_dash="dash", line_color="orange",
                                      annotation_text=f"95% CI cut-off ({threshold:.3g})",
                                      annotation_position="top left")
                    if ci_lo < ci_hi:
                        fig.add_vrect(x0=ci_lo, x1=ci_hi, fillcolor="rgba(255, 165, 0, 0.12)",
                                      line_width=0, annotation_text="95% CI", annotation_position="top right")

                    fig.update_layout(xaxis_title="α", yaxis_title="normalised SSE  J(α)", yaxis_type="log",
                                      height=420, margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig, width="stretch")
        else:
            # 2-D Loss Landscape
            st.markdown(
                "Evaluate the 2-D objective loss landscape :math:`\\log_{10} J(\\alpha, \\theta_k)` on a 30×30 grid, "
                "accelerated via population batching (one batched BLAS sweep per row)."
            )
            free_params = list(frac_res.problem.free)
            if not free_params:
                st.info("No free physical parameters to vary alongside α.")
            else:
                sel_param = st.selectbox("Parameter θ to vary", free_params)
                if st.button("⚡ Compute 2-D Landscape (30×30)"):
                    pbar2 = st.progress(0.0, text="Evaluating 2-D loss surface...")
                    a_grid, p_grid, l_grid = compute_loss_surface_2d(
                        frac_res,
                        param_name=sel_param,
                        n_alpha=30,
                        n_param=30,
                        callback=lambda r, tot: pbar2.progress(r / tot, text=f"Evaluating row {r}/{tot}")
                    )
                    pbar2.empty()
                    st.session_state["loss_2d"] = (sel_param, a_grid, p_grid, l_grid)

                if st.session_state.get("loss_2d") is not None:
                    p_name, a_vals, p_vals, l_grid = st.session_state["loss_2d"]
                    import plotly.graph_objects as go
                    log_loss = np.log10(np.maximum(l_grid, 1e-12))
                    fig_2d = go.Figure(data=go.Heatmap(
                        z=log_loss,
                        x=p_vals,
                        y=a_vals,
                        colorscale="Viridis",
                        colorbar=dict(title="log₁₀ SSE")
                    ))
                    # Mark estimate
                    fig_2d.add_trace(go.Scatter(
                        x=[frac_res.params[p_name]],
                        y=[frac_res.alpha],
                        mode="markers+text",
                        marker=dict(symbol="x", size=14, color="red", line=dict(width=2, color="white")),
                        text=["Estimate"],
                        textposition="top right",
                        name="Estimate"
                    ))
                    # Mark ground truth if available
                    meta = frac_res.problem.data.meta
                    if "true_alpha" in meta and p_name in meta.get("true_params", {}):
                        fig_2d.add_trace(go.Scatter(
                            x=[meta["true_params"][p_name]],
                            y=[meta["true_alpha"]],
                            mode="markers+text",
                            marker=dict(symbol="circle", size=12, color="cyan", line=dict(width=2, color="black")),
                            text=["Ground Truth"],
                            textposition="bottom right",
                            name="Ground Truth"
                        ))
                    fig_2d.update_layout(
                        xaxis_title=f"Parameter {p_name}",
                        yaxis_title="Fractional Order α",
                        height=480,
                        margin=dict(l=10, r=10, t=30, b=10)
                    )
                    st.plotly_chart(fig_2d, width="stretch")

    # Tab 7: Robustness Study
    with t7:
        st.subheader("🧪 Identifiability & Noise-Robustness Monte-Carlo Study")
        st.markdown(
            "Evaluate parameter recovery consistency across multiple noise levels "
            "and independent random seeds with Differential Evolution."
        )
        c1, c2 = st.columns([1, 1])
        with c1:
            n_seeds = st.slider("Random seeds per noise level (R)", min_value=3, max_value=10, value=5, step=1)
        with c2:
            noise_opts = [0.0, 1.0, 2.0, 5.0, 10.0, 15.0]
            sel_noise = st.multiselect("Noise levels (%)", noise_opts, default=noise_opts)

        if st.button("🚀 Run Robustness Study", key="btn_robustness"):
            if not sel_noise:
                st.error("Please select at least one noise level.")
            else:
                pbar = st.progress(0.0, text="Starting Monte-Carlo runs...")
                def rob_cb(done, total, cur_noise, cur_seed):
                    pbar.progress(done / total, text=f"Run {done}/{total}: Noise {cur_noise:.1f}% (seed {cur_seed})")

                df_rob = run_robustness_study(
                    model=frac_res.problem.model,
                    noise_levels=sorted(sel_noise),
                    n_seeds=n_seeds,
                    max_iter=60,
                    popsize=12,
                    polish_iter=60,
                    callback=rob_cb,
                )
                pbar.empty()
                st.session_state["robustness_df"] = df_rob

        rob_df = st.session_state.get("robustness_df")
        if rob_df is not None:
            st.success(f"Completed {len(rob_df)} runs across {len(rob_df['noise_pct'].unique())} noise levels.")

            # Summary table
            st.subheader("Summary: Mean Relative Errors by Noise Level")
            mean_cols = [c for c in rob_df.columns if c.endswith("_rel_err_pct")]
            summary_table = rob_df.groupby("noise_pct")[mean_cols].mean()
            st.dataframe(summary_table.style.format("{:.2f}%"), width="stretch")

            # Matplotlib publication figure
            fig_rob = plot_robustness_matplotlib(rob_df, model_name=frac_res.problem.model.name)
            st.pyplot(fig_rob)

            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1:
                png_bytes = fig_to_bytes(fig_rob, fmt="png", dpi=300)
                st.download_button("📸 Download High-Res PNG (300 DPI)", png_bytes,
                                   file_name="fracid_robustness_study.png", mime="image/png", width="stretch")
            with col_d2:
                pdf_bytes = fig_to_bytes(fig_rob, fmt="pdf")
                st.download_button("📄 Download Vector PDF", pdf_bytes,
                                   file_name="fracid_robustness_study.pdf", mime="application/pdf", width="stretch")
            with col_d3:
                csv_bytes = rob_df.to_csv(index=False)
                st.download_button("📥 Download Raw Data (CSV)", csv_bytes,
                                   file_name="fracid_robustness_raw_data.csv", mime="text/csv", width="stretch")

    # Tab 8: LaTeX & Publication Export
    with t8:
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

    # Tab 9: Theory
    with t9:
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
