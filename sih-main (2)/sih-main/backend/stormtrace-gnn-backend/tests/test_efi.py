import numpy as np
import pytest
from src.efi import compute_efi_field


def test_efi_extreme_value_near_one():
    baseline = np.full((30, 5), 25.0)  # 30 years, 5 points, all normal = 25
    forecast = np.array([25.0, 25.0, 25.0, 25.0, 60.0])  # last point wildly extreme
    efi = compute_efi_field(forecast, baseline)
    assert efi[-1] > 0.9, "an extreme value far above the whole baseline should score near +1"


def test_efi_normal_value_near_zero():
    rng = np.random.default_rng(0)
    baseline = rng.normal(25, 2, (1000, 3))
    forecast = np.array([25.0, 25.0, 25.0])  # right at the baseline mean
    efi = compute_efi_field(forecast, baseline)
    assert np.all(np.abs(efi) < 0.2), "a forecast at the baseline mean should score near 0"


def test_efi_ties_use_midrank():
    baseline = np.full((30, 2), 25.0)
    forecast = np.array([25.0, 10.0])
    efi = compute_efi_field(forecast, baseline)
    assert np.array_equal(efi, np.array([0.0, -1.0]))


def test_efi_requires_nonempty_matching_climatology():
    with pytest.raises(ValueError, match="no samples"):
        compute_efi_field(np.array([1.0]), np.empty((0, 1)))
    with pytest.raises(ValueError, match="n_points"):
        compute_efi_field(np.array([1.0, 2.0]), np.ones((5, 1)))


def test_efi_rejects_nonfinite_inputs_and_is_deterministic():
    forecast = np.array([1.0, 3.0])
    baseline = np.array([[0.0, 2.0], [2.0, 4.0]])
    first = compute_efi_field(forecast, baseline)
    second = compute_efi_field(forecast, baseline)
    assert np.array_equal(first, second)

    with pytest.raises(ValueError, match="must be finite"):
        compute_efi_field(np.array([np.nan]), np.ones((2, 1)))
