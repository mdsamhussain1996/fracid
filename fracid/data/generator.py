r"""Synthetic data generator.

Pipeline: simulate :math:`\mathbf{x}(t)` with a chosen model → pick observed states
→ subsample → add Gaussian noise at a prescribed signal-to-noise ratio.

Noise model
-----------
Independent Gaussian noise per observed channel :math:`k`,

.. math::
    \sigma_k = \frac{\operatorname{rms}(y_k)}{10^{\mathrm{SNR_{dB}}/20}},\qquad
    \operatorname{rms}(y_k)=\sqrt{\tfrac1M\sum_i y_k(t_i)^2},

so that :math:`\mathrm{SNR_{dB}} = 10\log_{10}(P_{signal}/P_{noise})`.
A "5 % noise" setting corresponds to :math:`\sigma_k = 0.05\,\mathrm{rms}(y_k)`, i.e.
:math:`\mathrm{SNR}=-20\log_{10}0.05\approx 26.0` dB (use ``noise_percent=5``).

.. note:: *Inverse crime.* Generating and fitting with the same solver and step flatters
   the estimator. ``fine_factor>1`` generates the data on a grid ``fine_factor`` times
   finer than the one used for subsequent fitting.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..models.base import FractionalModel
from .dataset import Dataset


def snr_db_from_percent(percent: float) -> float:
    r""":math:`\mathrm{SNR_{dB}} = -20\log_{10}(\text{percent}/100)`."""
    return -20.0 * np.log10(percent / 100.0)


def percent_from_snr_db(snr_db: float) -> float:
    return 100.0 * 10.0 ** (-snr_db / 20.0)


def add_noise(y: np.ndarray, snr_db: float | None = None, noise_percent: float | None = None,
              rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Add per-channel Gaussian noise; returns ``(noisy, sigma)`` (sigma per column)."""
    y = np.asarray(y, float)
    if snr_db is None and noise_percent is None:
        return y.copy(), np.zeros(y.shape[1])
    if snr_db is None:
        snr_db = snr_db_from_percent(noise_percent)
    rng = rng or np.random.default_rng()
    rms = np.sqrt(np.mean(y ** 2, axis=0))
    sigma = rms / 10.0 ** (snr_db / 20.0)
    return y + rng.standard_normal(y.shape) * sigma, sigma


def _resolve_observed(model: FractionalModel, observed) -> tuple[int, ...]:
    if observed is None:
        return tuple(range(model.dim))
    idx = []
    for o in observed:
        idx.append(model.state_names.index(o) if isinstance(o, str) else int(o))
    return tuple(idx)


def generate(model: FractionalModel, T: float | None = None, h: float | None = None, *,
             snr_db: float | None = None, noise_percent: float | None = None,
             observed: Sequence | None = None, subsample: int = 1,
             n_points: int | None = None, fine_factor: int = 1, seed: int | None = 0,
             method: str = "abm", t_min: float = 0.0) -> Dataset:
    """Simulate ``model`` and produce a noisy, possibly subsampled/partial data set.

    Parameters
    ----------
    model : FractionalModel
        Instance carrying the *true* ``alpha``, ``params`` and ``x0``.
    T, h : float
        Horizon and step of the *fitting* grid (defaults from the model).
    snr_db, noise_percent :
        Noise specification (give at most one; none = noise-free).
    observed : sequence of str/int, optional
        States to keep (default: all).
    subsample : int
        Keep every ``subsample``-th grid point.
    n_points : int, optional
        Alternative to ``subsample``: keep ``n_points`` equally spaced samples.
    fine_factor : int
        Generate on a ``fine_factor``-times finer grid (avoid the inverse crime).
    seed : int or None
        RNG seed for reproducibility.
    t_min : float
        Discard samples with :math:`t<t_{min}` after simulating (e.g. transients).
    """
    if snr_db is not None and noise_percent is not None:
        raise ValueError("specify either snr_db or noise_percent, not both")
    T = model.default_T if T is None else T
    h = model.default_h if h is None else h
    ff = max(int(fine_factor), 1)
    sol = model.simulate(T=T, h=h / ff, method=method)
    if not sol.finite:
        raise RuntimeError("simulation diverged; choose a smaller step or other parameters")
    t, x = sol.t[::ff], sol.x[::ff]            # back on the fitting grid
    obs = _resolve_observed(model, observed)
    sel = np.arange(t.size)
    if n_points is not None:
        sel = np.unique(np.round(np.linspace(0, t.size - 1, int(n_points))).astype(int))
    elif subsample > 1:
        sel = sel[::int(subsample)]
    sel = sel[t[sel] >= t_min - 1e-12]
    clean = x[sel][:, list(obs)]
    rng = np.random.default_rng(seed)
    noisy, sigma = add_noise(clean, snr_db, noise_percent, rng)
    if snr_db is None and noise_percent is not None:
        snr_db = snr_db_from_percent(noise_percent)
    meta = dict(model=model.name, true_alpha=model.alpha, true_params=dict(model.params),
                true_x0=model.x0.copy(), snr_db=snr_db, sigma=sigma, seed=seed, h_fit=h,
                T=T, subsample=subsample, observed_names=[model.state_names[i] for i in obs])
    return Dataset(t[sel], noisy, obs, tuple(model.state_names[i] for i in obs), clean, meta)
