# ⚡ FracID: Fractional-Order Dynamical System Identification

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

**FracID** is an end-to-end computational framework and interactive web application for **identifying fractional-order dynamical systems from time-series data**. Given experimental or synthetic measurements, FracID estimates the fractional derivative order $\alpha$ ($0 < \alpha \le 1$) alongside physical parameters $\theta$, quantifies parameter uncertainty with residual bootstrap, and provides statistical criteria (AIC, BIC) to establish whether a fractional-order model is superior to an integer-order model.

---

## 📖 Table of Contents
1. [Key Features](#-key-features)
2. [Mathematical Background](#-mathematical-background)
   - [Caputo Fractional Derivative](#caputo-fractional-derivative)
   - [Adams–Bashforth–Moulton (Diethelm) Solver](#adamsbashforthmoulton-diethelm-solver)
   - [Grünwald–Letnikov Discretization](#grnwaldletnikov-discretization)
   - [Physics-Informed Neural Networks (fPINN)](#physics-informed-neural-networks-fpinn)
   - [Model Selection & Information Criteria](#model-selection--information-criteria)
3. [Architecture & Project Structure](#-architecture--project-structure)
4. [Installation](#-installation)
5. [Quickstart & Usage](#-quickstart--usage)
   - [Running the Streamlit UI](#running-the-streamlit-ui)
   - [Python API Example](#python-api-example)
6. [Benchmark Model Library](#-benchmark-model-library)
7. [Case Study: Fractional Chen System](#-case-study-fractional-chen-system)
8. [Testing & Validation](#-testing--validation)
9. [Deployment](#-deployment)
10. [License](#-license)

---

## ✨ Key Features
- **High-Order Fractional Solvers**:
  - Diethelm Adams–Bashforth–Moulton (ABM) predictor-corrector ($O(h^{\min(2, 1+\alpha)})$ accuracy).
  - Explicit and Newton-implicit Grünwald–Letnikov (GL) schemes.
  - Analytic validation against closed-form scalar and matrix **Mittag-Leffler functions** $E_{\alpha, \beta}(z)$.
- **Extensive Model Library**:
  - 2D Fractional Linear system
  - Fractional Lorenz, Chen, and Chua chaotic attractors
  - Delayed 2-neuron Fractional Hopfield network
  - Fractional SIR epidemic model with memory
- **Dual Estimation Engines**:
  - **Optimization-based**: Bound-constrained Nelder–Mead, L-BFGS-B, and Differential Evolution with unit-cube parameter scaling and graded blow-up penalization.
  - **fPINN (Physics-Informed Neural Network)**: Deep neural network approximating state trajectories, with fractional Caputo residuals evaluated via the **$L1$ discretization scheme** using PyTorch autograd.
- **Statistical Model Selection**:
  - Side-by-side comparison of Fractional vs Integer-order ($\alpha=1$) models reporting RMSE, AIC, AICc, and BIC.
  - Residual diagnostics, Q-Q plots, and residual distribution histograms.
  - Residual bootstrap confidence intervals for $\alpha$ and $\theta$.
  - Sensitivity/profile loss curve $J(\alpha)$ highlighting the global minimum.
- **Publication-Ready Exports**:
  - Equations rendered in aligned LaTeX.
  - Vector PDF and 300 DPI PNG figure downloads.
  - CSV export of estimated parameters and diagnostics.

---

## 📐 Mathematical Background

### Caputo Fractional Derivative
For $\alpha \in (0, 1]$, the Caputo fractional derivative of a function $x(t)$ with base point $t_0=0$ is defined as:
$${}^C D^\alpha x(t) = \frac{1}{\Gamma(1 - \alpha)} \int_0^t (t - s)^{-\alpha} x'(s) \, ds$$
The Caputo derivative is preferred in physical modeling because its Laplace transform involves integer-order initial conditions $x(0) = x_0$, matching classical physical measurements.

### Adams–Bashforth–Moulton (Diethelm) Solver
The Caputo initial value problem ${}^C D^\alpha \mathbf{x}(t) = \mathbf{f}(t, \mathbf{x}(t))$ is equivalent to the Volterra integral equation:
$$\mathbf{x}(t) = \mathbf{x}_0 + \frac{1}{\Gamma(\alpha)} \int_0^t (t - s)^{\alpha - 1} \mathbf{f}(s, \mathbf{x}(s)) \, ds$$
Diethelm's predictor-corrector discretizes this on a uniform grid $t_n = n h$:
- **Predictor** (Product rectangle rule):
  $$\mathbf{x}_{n+1}^P = \mathbf{x}_0 + \frac{h^\alpha}{\Gamma(\alpha+1)} \sum_{j=0}^n b_{j, n+1} \mathbf{f}(t_j, \mathbf{x}_j), \quad b_{j, n+1} = (n + 1 - j)^\alpha - (n - j)^\alpha$$
- **Corrector** (Product trapezoidal rule):
  $$\mathbf{x}_{n+1} = \mathbf{x}_0 + \frac{h^\alpha}{\Gamma(\alpha+2)} \left[ \mathbf{f}(t_{n+1}, \mathbf{x}_{n+1}^P) + \sum_{j=0}^n a_{j, n+1} \mathbf{f}(t_j, \mathbf{x}_j) \right]$$

### Grünwald–Letnikov Discretization
Using binomial coefficients $w_j = (-1)^j \binom{\alpha}{j}$:
$${}^C D^\alpha \mathbf{x}(t_n) \approx h^{-\alpha} \sum_{j=0}^n w_j (\mathbf{x}_{n-j} - \mathbf{x}_0)$$
Yielding the explicit update:
$$\mathbf{x}_{n+1} = \mathbf{x}_0 + h^\alpha \mathbf{f}(t_n, \mathbf{x}_n) - \sum_{j=1}^{n+1} w_j (\mathbf{x}_{n+1-j} - \mathbf{x}_0)$$

### Physics-Informed Neural Networks (fPINN)
A neural network $\mathbf{x}_{\text{NN}}(t; \mathbf{w})$ parameterizes the trajectory. The fractional derivative is computed using the **$L1$ scheme**:
$${}^C D^\alpha \mathbf{x}(t_n) \approx \frac{h^{-\alpha}}{\Gamma(2 - \alpha)} \sum_{k=0}^{n-1} \left( (k+1)^{1-\alpha} - k^{1-\alpha} \right) \left( \mathbf{x}(t_{n-k}) - \mathbf{x}(t_{n-k-1}) \right)$$
The total loss combines data fidelity and fractional ODE residuals:
$$\mathcal{L}(\mathbf{w}, \alpha, \theta) = \frac{1}{M K} \sum_{i, k} \left(\frac{y_k(t_i) - \hat{x}_{c_k}(t_i)}{s_k}\right)^2 + \lambda_{\text{phys}} \frac{1}{N_{\text{colloc}}} \sum_{n=1}^{N_{\text{colloc}}} \left\| {}^C D^\alpha \mathbf{x}(t_n) - \mathbf{f}(\mathbf{x}(t_n); \theta) \right\|^2$$

### Model Selection & Information Criteria
To judge whether the fractional model fits significantly better than the integer-order model ($\alpha=1$):
$$\text{RMSE} = \sqrt{\frac{1}{N}\sum (y - \hat{y})^2}$$
$$\text{AIC} = N \ln\left(\frac{\text{SSE}}{N}\right) + 2p, \qquad \text{BIC} = N \ln\left(\frac{\text{SSE}}{N}\right) + p \ln N$$
$$\Delta\text{AIC} = \text{AIC}_{\text{integer}} - \text{AIC}_{\text{fractional}}$$
- $\Delta\text{AIC} > 10$: **Decisive evidence** favoring the fractional-order model.
- $4 < \Delta\text{AIC} \le 10$: **Substantial evidence** favoring the fractional-order model.
- $|\Delta\text{AIC}| \le 4$: Inconclusive (prefer simpler model).

---

## 📁 Architecture & Project Structure

```
fracid/
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Python dependencies
├── pytest.ini                  # Pytest configuration
├── fracid/
│   ├── solvers/                # Module 1: Numerical forward solvers
│   │   ├── abm.py              # Diethelm predictor-corrector
│   │   ├── gl.py               # Grünwald-Letnikov explicit/implicit
│   │   ├── mittag_leffler.py   # Analytic Mittag-Leffler reference functions
│   │   └── convergence.py      # EOC and refinement convergence testing
│   ├── models/                 # Module 2: Fractional dynamical models
│   │   ├── base.py             # FractionalModel abstract base class & LaTeX renderer
│   │   └── library.py          # Linear, Lorenz, Chen, Chua, Hopfield, SIR
│   ├── data/                   # Module 3: Synthetic data & CSV loading
│   │   ├── dataset.py          # Dataset container and CSV reader/writer
│   │   └── generator.py        # Noise injection (SNR dB/percent), subsampling
│   ├── estimation/             # Module 4: Parameter & order estimation
│   │   ├── problem.py          # FitProblem definition, unit box scaling, blow-up penalty
│   │   ├── least_squares.py    # Nelder-Mead, L-BFGS-B, Differential Evolution
│   │   └── fpinn.py            # Physics-Informed Neural Network with L1 derivative
│   ├── diagnostics/            # Module 5: Comparison and validation
│   │   ├── comparison.py       # Metrics, AIC, BIC, Delta-AIC
│   │   ├── bootstrap.py        # Residual bootstrap confidence intervals
│   │   ├── sensitivity.py      # Loss profile vs order alpha
│   │   └── plots.py            # Matplotlib (300 DPI) & Plotly visualizers
│   └── ui/                     # Module 6: Streamlit interface
│       ├── sidebar.py          # Controls for data, noise, models, bounds
│       └── views.py            # Tab views, LaTeX rendering, export buttons
├── tests/                      # Automated unit test suite (72+ tests)
│   ├── test_solvers.py
│   ├── test_models_data.py
│   ├── test_least_squares.py
│   ├── test_fpinn.py
│   └── test_diagnostics.py
└── notebooks/                  # Case studies and tutorials
    └── case_study_fractional_chen.ipynb
```

---

## 💻 Installation

### Prerequisites
- Python 3.11 (recommended) or 3.10+
- `pip` or `uv`

### Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/mdsamhussain1996/fracid.git
cd fracid

# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quickstart & Usage

### Running the Streamlit UI
Launch the interactive web app locally:
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### Python API Example
```python
from fracid.models import FractionalChen
from fracid.data import generate
from fracid.estimation import FitProblem, fit_least_squares
from fracid.diagnostics import compare_fractional_vs_integer

# 1. Define model and generate synthetic data (alpha = 0.90, 5% noise)
model = FractionalChen(alpha=0.90, params={"a": 35.0, "b": 3.0, "c": 28.0})
dataset = generate(model, T=1.5, h=0.005, noise_percent=5.0, seed=42)

# 2. Formulate identification problem
problem = FitProblem(model, dataset, free=["a", "b", "c"], alpha_bounds=(0.5, 1.0))

# 3. Solve using L-BFGS-B or Differential Evolution
result_frac = fit_least_squares(problem, method="l-bfgs-b")
print(f"Estimated alpha: {result_frac.alpha:.4f} (True: 0.900)")
print(f"Estimated parameters: {result_frac.estimates}")

# 4. Compare with integer-order model (alpha = 1.0)
result_int = fit_least_squares(problem.integer_order(), method="l-bfgs-b")
comparison_table = compare_fractional_vs_integer(result_frac, result_int)
print(comparison_table)
```

---

## 🔬 Benchmark Model Library

| Model Name | Equations | Default $\alpha$ | Default Parameters |
| :--- | :--- | :---: | :--- |
| **Fractional Linear** | ${}^C D^\alpha \mathbf{x} = A\mathbf{x}$ | $0.90$ | $a_{11}=-1, a_{12}=2, a_{21}=-2, a_{22}=-1$ |
| **Fractional Lorenz** | $\sigma(y-x), x(\rho-z)-y, xy-\beta z$ | $0.99$ | $\sigma=10, \rho=28, \beta=8/3$ |
| **Fractional Chen** | $a(y-x), (c-a)x-xz+cy, xy-bz$ | $0.90$ | $a=35, b=3, c=28$ |
| **Fractional Chua** | $a(y-x-g(x)), x-y+z, -by$ | $0.95$ | $a=15.6, b=28, m_0=-1.143, m_1=-0.714$ |
| **Fractional Hopfield** | Delayed 2-neuron network | $0.90$ | $\tau=1.0$ (constant delay) |
| **Fractional SIR** | Epidemic model with memory | $0.85$ | $\beta=0.5, \gamma=0.2, N=1000$ |

---

## 📊 Case Study: Fractional Chen System
The case study notebook in `notebooks/case_study_fractional_chen.ipynb` demonstrates:
- Ground truth: $\alpha=0.90, a=35.0, b=3.0, c=28.0$ with $5\%$ Gaussian noise.
- **Identified Values**:
  - $\hat{\alpha} = 0.8963$ (Relative Error: $0.41\%$)
  - $\hat{a} = 34.40$ (Relative Error: $1.71\%$)
  - $\hat{b} = 2.95$ (Relative Error: $1.50\%$)
  - $\hat{c} = 27.64$ (Relative Error: $1.29\%$)
- **Model Comparison**:
  - Fractional Model AIC: $-123.8$
  - Integer Model ($\alpha=1$) AIC: $+142.1$
  - $\Delta\text{AIC} = 265.9 \gg 10$ (**Decisive evidence for fractional dynamics**).

---

## 🧪 Testing & Validation
Run the full test suite with pytest:
```bash
pytest -v
```
All 72+ unit tests cover:
- Experimental Order of Convergence (EOC) of the ABM and GL numerical schemes.
- Exact agreement with analytic Mittag-Leffler solutions for scalar and matrix differential equations.
- Non-divergence handling and graded objective penalties.
- Robust parameter recovery across all models.
- PyTorch $L1$ Caputo derivative calculation accuracy.

---

## 🌐 Deployment

### Streamlit Community Cloud
1. Push this repository to GitHub.
2. Log in to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub account, select the repository, branch `main`, and main file `app.py`.
4. Click **Deploy**.

---

## 📄 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
