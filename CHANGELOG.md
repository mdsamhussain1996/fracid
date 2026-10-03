# Changelog

All notable changes to FracID are documented in this file.

## [v2.0.0-step1] - Step 1: UX Improvements & Cloud Readiness
- **Cloud Timings**: Renamed accuracy presets to realistic Streamlit Community Cloud execution times: `"Quick (~30 s)"`, `"Standard (~1 min)"`, `"Thorough (several min)"`.
- **Top Run Button**: Added a prominent "🚀 Run System Identification" button at the top of the main view to accommodate narrow screens where the sidebar is collapsed by default.
- **Combined Runtime Metric**: Overview card now reports the combined runtime and evaluation count across both fractional and integer-order fits when integer comparison is enabled.
- **Pre-computed Benchmark**: Added a "✨ Load Example Result" button that instantly loads a pre-computed Fractional Chen identification result from `fracid/assets/example_chen.pkl`.
- **Unit Tests**: Added `tests/test_ui_ux.py` covering accuracy presets and asset loading.
