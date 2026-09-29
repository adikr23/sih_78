import pytest

from config import settings
from src.diffusion import validate_downscaling_resolutions


def test_true_downscaling_requires_finer_target_resolution():
    validate_downscaling_resolutions(12, 5)
    with pytest.raises(ValueError, match="finer target"):
        validate_downscaling_resolutions(12, 12)
    with pytest.raises(ValueError, match="finer target"):
        validate_downscaling_resolutions(12, 15)


def test_true_downscaling_requires_both_resolutions():
    with pytest.raises(ValueError, match="both input and target"):
        validate_downscaling_resolutions(settings.input_resolution_km, None)
