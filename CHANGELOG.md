# Changelog

All notable changes to FracID are documented in this file.

## [v2.0.0-step1] - Step 1: UX Improvements & Cloud Readiness
- **Cloud Timings**: Renamed accuracy presets to realistic Streamlit Community Cloud execution times: `"Quick (~30 s)"`, `"Standard (~1 min)"`, `"Thorough (several min)"`.
- **Top Run Button**: Added a prominent "🚀 Run System Identification" button at the top of the main view to accommodate narrow screens where the sidebar is collapsed by default.
- **Combined Runtime Metric**: Overview card now reports the combined runtime and evaluation count across both fractional and integer-order fits when integer comparison is enabled.
- **Pre-computed Benchmark**: Added a "✨ Load Example Result" button that instantly loads a pre-computed Fractional Chen identification result from `fracid/assets/example_chen.pkl`.
## [v2.0.0-step2] - Step 2: Smoother α-Profile, Continuation, and Profile CIs
- **Continuation Profile**: `compute_alpha_sensitivity(mode="profile")` now sweeps outward from $\hat\alpha$ in both directions, warm-starting each fit from the neighbouring optimum for robust continuity across multimodal landscapes.
- **Batched Slice Evaluation**: In `mode="slice"`, all grid points are evaluated in a single forward sweep via `problem.sse_batch()`.
- **Profile-Likelihood Confidence Intervals**: Added `compute_profile_ci()`, computing exact likelihood-ratio cut-off confidence intervals $\{\alpha : J(\alpha) - J(\hat\alpha) \le \chi^2_1(0.95) \cdot J(\hat\alpha)/(N - p)\}$.
- **UI Diagnostics**: Highlighted the 95% profile CI region and threshold line on the interactive sensitivity plot, and added an $\alpha$ interval comparison table alongside residual bootstrap in the UI.
## [v2.0.0-step3] - Step 3: Identifiability & Noise-Robustness Study
- **Monte-Carlo Robustness Engine**: Added `fracid.diagnostics.robustness` with `run_robustness_study()`, evaluating parameter recovery across noise levels ($[0, 1, 2, 5, 10, 15]\%$) and independent random seeds.
- **Publication Figures**: Added `plot_robustness_matplotlib()` generating high-resolution 2-panel figures ($\hat\alpha$ error vs noise and parameter recovery bands).
- **Interactive UI Tab**: Introduced dedicated `"🧪 Robustness Study"` tab with live progress tracking, summary error tables, interactive charts, and 300-DPI PNG, vector PDF, and CSV data export buttons.
## [v2.0.0-step4] - Step 4: 2-D Objective Loss Landscape
- **Batched 2-D Loss Surface**: Added `compute_loss_surface_2d()` evaluating $\log_{10} J(\alpha, \theta_k)$ on a 30×30 grid with one batched BLAS sweep per row via `problem.sse_batch()`.
- **Interactive UI Heatmap**: Added 2-D landscape option in the Alpha Sensitivity tab with Plotly heatmap, contour lines, estimate marker ($\times$), and ground-truth marker ($\circ$).
- **Unit Tests**: Added `tests/test_landscape_2d.py` covering grid shape, finiteness, and parameter variations.
