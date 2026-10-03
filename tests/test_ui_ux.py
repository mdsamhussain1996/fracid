"""Tests for UX and UI enhancements (Step 1)."""
import os
import pickle
import pytest

from fracid.estimation import FitResult
from fracid.ui.sidebar import ACCURACY_PRESETS


def test_accuracy_presets():
    expected_keys = ["Quick (~30 s)", "Standard (~1 min)", "Thorough (several min)"]
    assert list(ACCURACY_PRESETS.keys()) == expected_keys
    for k in expected_keys:
        assert "max_iter" in ACCURACY_PRESETS[k]
        assert "popsize" in ACCURACY_PRESETS[k]


def test_example_chen_asset_loads():
    pkg_dir = os.path.dirname(os.path.dirname(__file__))
    asset_path = os.path.join(pkg_dir, "fracid", "assets", "example_chen.pkl")
    assert os.path.exists(asset_path), f"Asset {asset_path} does not exist"

    with open(asset_path, "rb") as f:
        data = pickle.load(f)

    assert "frac_res" in data
    assert "int_res" in data
    assert isinstance(data["frac_res"], FitResult)
    assert isinstance(data["int_res"], FitResult)
    assert abs(data["frac_res"].alpha - 0.90) < 0.05
    assert data["int_res"].alpha == 1.0
