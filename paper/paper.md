---
title: 'FracID: A Python Framework and Web Application for Fractional-Order System Identification'
tags:
  - Python
  - fractional calculus
  - dynamical systems
  - system identification
  - differential evolution
  - physics-informed neural networks
authors:
  - name: Md Samshad Hussain Ansari
    orcid: 0000-0002-7757-3216
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 3 October 2026
bibliography: paper.bib
---

# Summary

Fractional differential equations (FDEs) model memory, hereditary effects, and anomalous transport across rheology, electrochemistry, control engineering, and epidemiology [@Podlubny1999]. In experimental practice, neither the fractional derivative order $\alpha \in (0, 1]$ nor the physical governing parameters $\theta$ are known *a priori*. Standard integer-order identification tools are incapable of estimating fractional orders, while manual trial-and-error fitting is tedious and mathematically unprincipled.

`FracID` is an open-source Python framework and interactive Streamlit web application for identifying commensurate Caputo fractional-order dynamical systems from noisy experimental or synthetic time series. It provides dual estimation engines (vectorized global Differential Evolution with L-BFGS-B polishing, and PyTorch Physics-Informed Neural Networks with $L1$ Caputo residual regularization), statistical model selection criteria (AIC, BIC, $\Delta\text{AIC}$), residual bootstrap uncertainty quantification, and profile-likelihood confidence intervals.

# Statement of Need

A central challenge in fractional modeling is answering whether memory is genuinely present in observational data or whether an integer-order model ($\alpha = 1$) suffices. Existing fractional calculus software packages, such as MATLAB's FOMCON and FOTF [@Tejado2019], primarily target linear transfer functions and frequency-domain control design. They lack native support for nonlinear chaotic systems, population-batched forward integration, rigorous information-theoretic criteria ($\Delta\text{AIC}$), and web-based interactive exploration.

`FracID` bridges this gap by offering:
1. **Population-Batched ABM Integration**: An accelerated Adams–Bashforth–Moulton solver [@Diethelm2002] that integrates an entire candidate population simultaneously, converting convolution loops into batched BLAS matrix multiplications for a 10–20$\times$ speedup over serial evaluation.
2. **Statistically Principled Model Selection**: Direct computation of Akaike and Bayesian Information Criteria (AIC, BIC) to evaluate whether empirical data justifies the inclusion of fractional memory ($\Delta\text{AIC} > 10$).
3. **Dual Uncertainty Quantification**: Residual bootstrap distributions alongside profile-likelihood confidence intervals computed via directional continuation.
4. **Interactive GUI and Paper-Ready Exports**: A Streamlit application featuring live progress feedback, interactive Plotly portraits, and vector PDF/300-DPI PNG figure downloads.

# Key Numerical Methods

## Caputo Fractional Derivative & ABM Predictor-Corrector
For $\alpha \in (0, 1]$, the Caputo fractional initial value problem is formulated as:
$${}^C D^\alpha \mathbf{x}(t) = \mathbf{f}(t, \mathbf{x}(t); \theta), \quad \mathbf{x}(0) = \mathbf{x}_0$$
This is solved numerically using Diethelm's Adams–Bashforth–Moulton (ABM) predictor-corrector scheme on a uniform grid $t_n = n h$ with convergence order $\mathcal{O}(h^{\min(2, 1+\alpha)})$. In `fracid.solvers.batch.abm_solve_batch`, all $S$ candidates in a Differential Evolution population are advanced concurrently in a single Python loop.

## Profile-Likelihood Continuation
To estimate the profile loss $J(\alpha) = \min_\theta J(\alpha, \theta)$, `FracID` sweeps outward in both directions starting from the optimum $\hat\alpha$. Each optimization step is warm-started from its immediate neighbor's solution, ensuring continuous tracking across multimodal valleys. Profile-likelihood confidence intervals are determined by:
$$\text{CI}_{0.95} = \Bigl\{ \alpha : J(\alpha) - J(\hat\alpha) \le \chi^2_1(0.95) \cdot \frac{J(\hat\alpha)}{N - p} \Bigr\}$$

# Verification and Real-Data Demonstration

`FracID` is validated against closed-form scalar and matrix Mittag-Leffler analytic solutions, Experimental Order of Convergence (EOC) tests, and standard chaotic benchmarks (Fractional Chen, Lorenz, Chua, and Hopfield models). 

In real-world verification, `FracID` identifies parameters of the 1978 English Boarding School Influenza epidemic ($N = 763$) [@BMJ1978] using a Fractional SIR model, providing automated model comparison and residual diagnostics reproducing published epidemiological findings.

# References
