r"""Mittag-Leffler functions: the analytic reference for linear Caputo problems.

The two-parameter Mittag-Leffler function is

.. math::
    E_{\alpha,\beta}(z) = \sum_{k=0}^{\infty} \frac{z^k}{\Gamma(\alpha k + \beta)} .

For the scalar Caputo problem :math:`{}^C D^\alpha x = \lambda x,\; x(0)=x_0`
(:math:`0<\alpha\le 1`) the exact solution is

.. math::
    x(t) = x_0\, E_{\alpha,1}(\lambda t^\alpha),

and for the linear system :math:`{}^C D^\alpha \mathbf{x} = A\mathbf{x}`

.. math::
    \mathbf{x}(t) = E_{\alpha,1}(A t^\alpha)\,\mathbf{x}_0 .

Numerical strategy
------------------
* ``z`` small / positive (no cancellation): truncated power series evaluated
  with ``loggamma`` so that the terms never overflow.
* ``z < 0``, :math:`0<\alpha<1`, :math:`\beta=1` (the decaying case, where the
  series suffers catastrophic cancellation): the Stieltjes integral
  representation (Gorenflo, Mainardi)

  .. math::
      E_{\alpha}(-x) = \frac{\sin\alpha\pi}{\alpha\pi}\int_0^\infty
      \frac{e^{-x^{1/\alpha} s^{1/\alpha}}}{s^2 + 2 s\cos\alpha\pi + 1}\,ds ,

  evaluated with adaptive quadrature (the integrand is smooth after the
  substitution :math:`r = s^{1/\alpha}`).
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import integrate
from scipy.special import loggamma, gammaln


def _series(z: complex, alpha: float, beta: float, tol: float = 1e-17, kmax: int = 2000) -> complex:
    """Power series for E_{alpha,beta}(z), beta > 0.

    Terms are formed in log-space (``exp(k log|z| - lnGamma(alpha k + beta))``) so
    that nothing overflows. The loop stops once k is past the peak of the terms
    (located near ``|z|**(1/alpha)``) and the terms are negligible.
    """
    if z == 0:
        return complex(np.exp(-gammaln(beta)))
    total = 0.0 + 0.0j
    logz = np.log(abs(z))
    phase = np.angle(z)
    k_peak = abs(z) ** (1.0 / alpha)
    for k in range(kmax):
        term = np.exp(k * logz - gammaln(alpha * k + beta)) * np.exp(1j * k * phase)
        total += term
        if k > k_peak + 3 and abs(term) < tol * max(1.0, abs(total)):
            break
    return total


def _ml_negative_real(x: float, alpha: float) -> float:
    """E_alpha(-x) for x >= 0 and 0 < alpha < 1 via the Stieltjes integral."""
    if x == 0:
        return 1.0
    ca = np.cos(alpha * np.pi)
    # substitution r = s^(1/alpha) removes the endpoint singularity:
    def integrand(s: float) -> float:
        return np.exp(-(x ** (1.0 / alpha)) * s ** (1.0 / alpha)) / (s * s + 2.0 * s * ca + 1.0)

    # location where the exponential has decayed to e^-45
    s_max = (45.0 / x ** (1.0 / alpha)) ** alpha
    # split the domain: the bulk of the mass is near s ~ s_max/10
    val, _ = integrate.quad(integrand, 0.0, s_max, limit=400, epsabs=1e-14, epsrel=1e-13,
                            points=[s_max / 100.0, s_max / 10.0, s_max / 3.0] if s_max > 1e-8 else None)
    return float(np.sin(alpha * np.pi) / (alpha * np.pi) * val)


def mittag_leffler(z, alpha: float, beta: float = 1.0):
    """Evaluate :math:`E_{\\alpha,\\beta}(z)` for scalar/array ``z``.

    Parameters
    ----------
    z : float, complex or array_like
        Argument(s). Real arrays return real output; complex input returns complex.
    alpha : float
        First parameter, :math:`\\alpha > 0`.
    beta : float
        Second parameter (default 1).

    Notes
    -----
    Accurate to roughly 1e-10 for real ``z`` when :math:`0<\\alpha\\le 1`.
    For complex ``z`` the series is used and a warning is issued when
    :math:`|z|^{1/\\alpha} > 15` (loss of more than ~6 digits).
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    arr = np.asarray(z)
    is_complex = np.iscomplexobj(arr)
    out = np.empty(arr.shape, dtype=complex if is_complex else float)
    flat_in, flat_out = arr.ravel(), out.ravel()
    for i, zi in enumerate(flat_in):
        if alpha == 1.0 and beta == 1.0:
            flat_out[i] = np.exp(zi)
            continue
        if (not is_complex) and zi < 0 and beta == 1.0 and alpha < 1.0:
            if abs(zi) ** (1.0 / alpha) > 8.0:
                flat_out[i] = _ml_negative_real(-float(zi), alpha)
                continue
        if abs(zi) ** (1.0 / alpha) > 15.0 and (is_complex or zi < 0):
            warnings.warn("mittag_leffler: power series used outside its accurate range", RuntimeWarning)
        val = _series(complex(zi), alpha, beta)
        flat_out[i] = val if is_complex else val.real
    return out if arr.ndim else out.item()


def ml_scalar_solution(t, lam: float, alpha: float, x0: float = 1.0):
    r"""Exact solution of :math:`{}^C D^\alpha x = \lambda x,\ x(0)=x_0`: :math:`x_0 E_\alpha(\lambda t^\alpha)`."""
    t = np.asarray(t, dtype=float)
    return x0 * mittag_leffler(lam * t ** alpha, alpha)


def ml_linear_solution(t, A, x0, alpha: float):
    r"""Exact solution of :math:`{}^C D^\alpha \mathbf{x} = A\mathbf{x}` for diagonalisable ``A``.

    With :math:`A = V \Lambda V^{-1}`, :math:`\mathbf{x}(t) = V\,\mathrm{diag}(E_\alpha(\lambda_i t^\alpha))\,V^{-1}\mathbf{x}_0`.
    """
    A = np.asarray(A, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    lam, V = np.linalg.eig(A)
    c = np.linalg.solve(V, x0.astype(complex))
    t = np.atleast_1d(np.asarray(t, dtype=float))
    X = np.empty((t.size, A.shape[0]))
    for i, ti in enumerate(t):
        e = np.array([mittag_leffler(complex(l * ti ** alpha), alpha) if np.iscomplex(l)
                      else mittag_leffler(float(np.real(l)) * ti ** alpha, alpha) for l in lam])
        X[i] = np.real(V @ (e * c))
    return X
