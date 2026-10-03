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
- **Unit Tests**: Added `tests/test_sensitivity_profile.py` verifying continuation, batched slices, and profile CI calculations.
