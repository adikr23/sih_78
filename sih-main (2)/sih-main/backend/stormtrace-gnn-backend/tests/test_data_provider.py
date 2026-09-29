import numpy as np
import pytest

from config import settings
from src.data_provider import NepsGProvider, canonicalize_weather_data, get_data_provider


def test_default_provider_is_public_weather():
    provider = get_data_provider()
    assert provider.provider_name == "public_weather"
    assert provider.label == "synthetic-demo"
    assert provider.metadata["label"] == "synthetic-demo"


def test_unknown_provider_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown data provider"):
        get_data_provider("not_a_real_provider")


def test_neps_g_provider_requires_a_configured_local_file(monkeypatch):
    monkeypatch.setattr(settings, "neps_g_forecast_path", None)
    monkeypatch.setattr(settings, "data_mode", "LOCAL_REAL")
    provider = NepsGProvider()
    with pytest.raises(RuntimeError, match="local path"):
        provider.generate_ensemble(seed=1)


def test_neps_g_provider_reads_local_npz_and_preserves_members(tmp_path, monkeypatch):
    latitude = np.linspace(-90, 90, 4)
    longitude = np.linspace(-180, 90, 4)
    forecast_path = tmp_path / "forecast.npz"
    climatology_path = tmp_path / "climatology.npz"
    ensemble = np.ones((2, 3, 4, 4, 4), dtype=np.float32)
    climatology = np.zeros((5, 4, 4, 4), dtype=np.float32)
    np.savez(
        forecast_path,
        ensemble=ensemble,
        latitude=latitude,
        longitude=longitude,
    )
    np.savez(
        climatology_path,
        climatology=climatology,
        latitude=latitude,
        longitude=longitude,
    )
    monkeypatch.setattr(settings, "neps_g_forecast_path", str(forecast_path))
    monkeypatch.setattr(settings, "era5_climatology_path", str(climatology_path))
    monkeypatch.setattr(settings, "data_mode", "LOCAL_REAL")

    provider = NepsGProvider()
    loaded_ensemble = provider.generate_ensemble()
    loaded_climatology = provider.generate_baseline()
    canonical = canonicalize_weather_data(loaded_ensemble, loaded_climatology, provider)

    assert canonical.ensemble.shape == (2, 3, 4, 4, 4)
    assert canonical.climatology.shape == (5, 4, 4, 4)
    assert canonical.provenance["source_type"] == "neps_g"
    assert canonical.provenance["reference_type"] == "climatology"


def test_canonical_weather_data_rejects_missing_members_and_nonfinite_values():
    provider = get_data_provider("synthetic_demo")
    with pytest.raises(ValueError, match="shape"):
        canonicalize_weather_data(np.ones((2, 4, 3, 3)), np.ones((5, 4, 3, 3)), provider)

    ensemble = np.ones((2, 3, 4, 3, 3))
    ensemble[0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        canonicalize_weather_data(ensemble, np.ones((5, 4, 3, 3)), provider)
