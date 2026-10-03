"""Tests of the model library and the synthetic data generator."""
import io

import numpy as np
import pytest

from fracid.data import Dataset, generate, percent_from_snr_db, snr_db_from_percent
from fracid.models import (MODELS, FractionalChen, FractionalHopfield, FractionalLinear,
                           FractionalLorenz, FractionalSIR, get_model)
from fracid.solvers import ml_linear_solution


@pytest.mark.parametrize("name", list(MODELS))
def test_every_model_simulates_and_is_bounded(name):
    m = MODELS[name]()
    sol = m.simulate(T=min(m.default_T, 5.0), h=max(m.default_h, 0.01))
    assert sol.finite and sol.x.shape[1] == m.dim
    assert np.max(np.abs(sol.x)) < 1e4


@pytest.mark.parametrize("name", list(MODELS))
def test_metadata_is_consistent(name):
    m = MODELS[name]()
    assert set(m.params) == set(m.param_names)
    assert set(m.default_free) <= set(m.param_names)
    assert set(m.default_bounds) <= set(m.param_names)
    for k in m.default_free:
        lo, hi = m.default_bounds[k]
        assert lo < m.params[k] < hi        # truth lies inside the default search box


def test_linear_model_matches_mittag_leffler():
    m = FractionalLinear(alpha=0.8)
    sol = m.simulate(T=3.0, h=0.005)
    ex = ml_linear_solution(sol.t, m.matrix(), m.x0, 0.8)
    assert np.max(np.abs(sol.x[sol.t >= 0.5] - ex[sol.t >= 0.5])) < 2e-3


def test_sir_conserves_population():
    m = FractionalSIR()
    sol = m.simulate(T=30.0, h=0.1)
    assert np.allclose(sol.x.sum(axis=1), m.params["N"], atol=1e-8)
    assert sol.x[:, 1].max() > m.x0[1]                 # epidemic grows (R0 = 2.5)


def test_lorenz_rhs_values():
    m = FractionalLorenz()
    f = m.rhs(0.0, [1.0, 2.0, 3.0])
    assert np.allclose(f, [10.0 * 1.0, 1.0 * (28 - 3) - 2.0, 2.0 - 8 / 3 * 3.0])


def test_chen_rhs_values():
    m = FractionalChen()
    f = m.rhs(0.0, [1.0, 2.0, 3.0])
    assert np.allclose(f, [35.0, (28 - 35) * 1 - 3 + 56, 2 - 9])


def test_alpha_one_chen_is_integer_order_and_differs_from_fractional():
    a = FractionalChen(alpha=1.0).simulate(T=1.0, h=0.002).x
    b = FractionalChen(alpha=0.9).simulate(T=1.0, h=0.002).x
    assert np.max(np.abs(a - b)) > 1.0


def test_hopfield_uses_delay():
    m1 = FractionalHopfield(params=dict(tau=1.0))
    m2 = FractionalHopfield(params=dict(tau=1.5))
    a = m1.simulate(T=6.0, h=0.025).x
    b = m2.simulate(T=6.0, h=0.025).x
    assert np.allclose(a[:40], b[:40], atol=1e-9)       # identical until t = tau (first delay)
    assert np.max(np.abs(a - b)) > 1e-3                 # then they differ


def test_unknown_param_rejected_and_aliases():
    with pytest.raises(KeyError):
        FractionalChen(params={"zzz": 1})
    assert isinstance(get_model("chen"), FractionalChen)
    with pytest.raises(KeyError):
        get_model("nope")


def test_latex_contains_estimated_values():
    s = FractionalChen().latex(alpha=0.9123, params=dict(a=34.5, b=3.0, c=28.0))
    assert "0.9123" in s and "34.5" in s and "aligned" in s
    s2 = FractionalHopfield().latex()
    assert "tanh" in s2 and "tau" in s2
    for cls in MODELS.values():
        assert "begin{aligned}" in cls().latex()


# --------------------------------------------------------------------- generator
def test_snr_conversion_roundtrip():
    assert snr_db_from_percent(5.0) == pytest.approx(26.0206, abs=1e-3)
    assert percent_from_snr_db(snr_db_from_percent(7.0)) == pytest.approx(7.0)


def test_generator_noise_level_matches_snr():
    m = FractionalLinear()
    ds = generate(m, T=20.0, h=0.01, snr_db=20.0, seed=1)
    resid = ds.y - ds.clean
    rms = np.sqrt(np.mean(ds.clean ** 2, axis=0))
    measured = 20 * np.log10(rms / resid.std(axis=0))
    assert np.allclose(measured, 20.0, atol=0.6)


def test_generator_percent_and_reproducibility():
    m = FractionalLinear()
    a = generate(m, T=5.0, h=0.02, noise_percent=5.0, seed=7)
    b = generate(m, T=5.0, h=0.02, noise_percent=5.0, seed=7)
    c = generate(m, T=5.0, h=0.02, noise_percent=5.0, seed=8)
    assert np.array_equal(a.y, b.y) and not np.array_equal(a.y, c.y)
    assert a.meta["snr_db"] == pytest.approx(26.02, abs=0.01)


def test_generator_noise_free_equals_simulation():
    m = FractionalChen()
    ds = generate(m, T=1.0, h=0.005)
    sol = m.simulate(T=1.0, h=0.005)
    assert np.allclose(ds.y, sol.x) and np.allclose(ds.y, ds.clean)


def test_generator_subsample_and_partial_observation():
    m = FractionalChen()
    ds = generate(m, T=1.0, h=0.005, observed=["x", "z"], subsample=10, noise_percent=2.0)
    assert ds.y.shape == (21, 2) and ds.observed == (0, 2) and ds.names == ("x", "z")
    ds2 = generate(m, T=1.0, h=0.005, n_points=30)
    assert ds2.n == 30 and ds2.t[0] == 0 and ds2.t[-1] == pytest.approx(1.0)


def test_generator_fine_factor_changes_data_slightly():
    m = FractionalLorenz()
    a = generate(m, T=1.0, h=0.01)
    b = generate(m, T=1.0, h=0.01, fine_factor=4)
    assert a.y.shape == b.y.shape
    assert 0 < np.max(np.abs(a.y - b.y)) < 1.0


def test_dataset_csv_roundtrip_and_validation():
    ds = generate(FractionalLinear(), T=2.0, h=0.05, noise_percent=1.0)
    buf = io.StringIO(ds.to_csv())
    back = Dataset.from_csv(buf)
    assert np.allclose(back.t, ds.t) and np.allclose(back.y, ds.y)
    with pytest.raises(ValueError):
        Dataset(np.array([0, 1, 1.0]), np.zeros((3, 1)))
