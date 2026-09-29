import json

import numpy as np
import pytest
import xarray as xr
from fastapi.testclient import TestClient

from config import settings
from src.data_provider import (
    Era5Provider,
    canonicalize_weather_data,
    get_data_provider,
    quality_control,
)


def _era5_reference_dataset(include_precipitation=True):
    coordinates = {
        "time": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
        "latitude": [2.0, 1.0],
        "longitude": [20.0, 21.0],
    }
    data = {
        "2m_temperature": (("time", "latitude", "longitude"), np.full((2, 2, 2), 280.0)),
        "10m_u_component_of_wind": (("time", "latitude", "longitude"), np.full((2, 2, 2), 4.0)),
        "10m_v_component_of_wind": (("time", "latitude", "longitude"), np.full((2, 2, 2), -2.0)),
    }
    if include_precipitation:
        data["total_precipitation"] = (
            ("time", "latitude", "longitude"),
            np.full((2, 2, 2), 0.001),
        )
    dataset = xr.Dataset(data, coords=coordinates)
    dataset["2m_temperature"].attrs["units"] = "K"
    dataset["10m_u_component_of_wind"].attrs["units"] = "m s-1"
    dataset["10m_v_component_of_wind"].attrs["units"] = "m s-1"
    if include_precipitation:
        dataset["total_precipitation"].attrs["units"] = "m"
    return dataset


def test_era5_local_file_loading_maps_external_names_and_preserves_metadata(tmp_path, monkeypatch):
    path = tmp_path / "era5_reference.nc"
    _era5_reference_dataset().to_netcdf(path)
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(
            {
                "dataset": "reanalysis-era5-single-levels",
                "variables": ["2m_temperature", "total_precipitation"],
                "date_range": {"start": "2024-01-01", "end": "2024-01-02"},
                "source": "CDS/ERA5",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "era5_climatology_path", str(path))
    monkeypatch.setattr(settings, "data_mode", "LOCAL_REAL")

    provider = Era5Provider()
    values = provider.generate_baseline()
    canonical = canonicalize_weather_data(
        np.repeat(values[0:1, np.newaxis, ...], 2, axis=1),
        values,
        provider,
    )
    assert canonical.climatology.shape == (2, 4, 2, 2)
    assert canonical.variables == ("t2m", "rainfall", "wind_u", "wind_v")
    assert canonical.units == ("K", "m", "m s-1", "m s-1")
    assert canonical.provenance["source_variables"] == (
        "2m_temperature",
        "total_precipitation",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    )
    assert canonical.provenance["quality_control"]["status"] == "passed"
    assert canonical.provenance["data_mode"] == "LOCAL_REAL"
    assert canonical.provenance["download_provenance"]["source"] == "CDS/ERA5"


def test_era5_local_zarr_loading(tmp_path, monkeypatch):
    path = tmp_path / "era5_reference.zarr"
    _era5_reference_dataset().to_zarr(path)
    monkeypatch.setattr(settings, "era5_climatology_path", str(path))
    monkeypatch.setattr(settings, "data_mode", "LOCAL_REAL")

    values = Era5Provider().generate_baseline()
    assert values.shape == (2, 4, 2, 2)


def test_invalid_era5_file_fails_clearly(tmp_path, monkeypatch):
    path = tmp_path / "not_a_dataset.nc"
    path.write_text("not a NetCDF file", encoding="ascii")
    monkeypatch.setattr(settings, "era5_climatology_path", str(path))
    monkeypatch.setattr(settings, "data_mode", "LOCAL_REAL")
    with pytest.raises(Exception):
        Era5Provider().generate_baseline()


def test_missing_required_era5_variable_is_data_unavailable(tmp_path, monkeypatch):
    path = tmp_path / "missing_variable.nc"
    _era5_reference_dataset(include_precipitation=False).to_netcdf(path)
    monkeypatch.setattr(settings, "era5_climatology_path", str(path))
    monkeypatch.setattr(settings, "data_mode", "LOCAL_REAL")
    with pytest.raises(ValueError, match="DATA_UNAVAILABLE.*rainfall"):
        Era5Provider().generate_baseline()


def test_qc_rejects_invalid_physics_and_duplicate_coordinates():
    forecast = np.ones((1, 1, 4, 2, 2), dtype=np.float32)
    climatology = np.ones((2, 4, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="physically implausible"):
        quality_control(
            forecast,
            climatology,
            ("t2m", "rainfall", "wind_u", "wind_v"),
            ("K", "m", "m s-1", "m s-1"),
            np.array([1.0, 2.0]),
            np.array([10.0, 11.0]),
        )
    with pytest.raises(ValueError, match="strictly ordered and duplicate-free"):
        quality_control(
            np.full_like(forecast, 280.0),
            np.full_like(climatology, 280.0),
            ("t2m", "rainfall", "wind_u", "wind_v"),
            ("K", "m", "m s-1", "m s-1"),
            np.array([1.0, 2.0]),
            np.array([10.0, 10.0]),
        )


def test_real_mode_requires_real_nepsg_forecast_and_era5(monkeypatch):
    monkeypatch.setattr(settings, "data_mode", "REAL")
    monkeypatch.setattr(settings, "neps_g_forecast_path", None)
    monkeypatch.setattr(settings, "era5_climatology_path", None)
    monkeypatch.setattr(settings, "cds_uid", "configured")
    monkeypatch.setattr(settings, "cds_api_key", "configured")
    with pytest.raises(RuntimeError, match="NEPS-G: CONFIGURED BUT DATA UNAVAILABLE"):
        get_data_provider()


def test_local_real_mode_uses_only_configured_local_real_files(tmp_path, monkeypatch):
    neps_path = tmp_path / "neps.grib2"
    era5_path = tmp_path / "era5.nc"
    neps_path.write_bytes(b"fixture")
    era5_path.write_bytes(b"fixture")
    monkeypatch.setattr(settings, "data_mode", "LOCAL_REAL")
    monkeypatch.setattr(settings, "neps_g_forecast_path", str(neps_path))
    monkeypatch.setattr(settings, "era5_climatology_path", str(era5_path))
    monkeypatch.setattr(settings, "cds_uid", None)
    monkeypatch.setattr(settings, "cds_api_key", None)
    provider = get_data_provider()
    assert provider.provider_name == "neps_g"


def test_synthetic_demo_remains_explicit(monkeypatch):
    monkeypatch.setattr(settings, "data_mode", "SYNTHETIC_DEMO")
    provider = get_data_provider("synthetic_demo")
    assert provider.label == "synthetic-demo"
    assert provider.generate_ensemble(n_steps=1, n_members=1, grid_size=4).shape == (1, 1, 4, 4, 4)
    with pytest.raises(ValueError, match="cannot be selected in SYNTHETIC_DEMO"):
        get_data_provider("neps_g")


def test_system_status_never_exposes_cds_or_redis_secrets(monkeypatch):
    from src.main import app

    monkeypatch.setattr(settings, "data_mode", "SYNTHETIC_DEMO")
    monkeypatch.setattr(settings, "cds_uid", "private-user-id")
    monkeypatch.setattr(settings, "cds_api_key", "private-cds-key")
    monkeypatch.setattr(settings, "redis_url", "redis://private-user:private-password@localhost:6379/0")
    response = TestClient(app).get("/api/system/status")
    assert response.status_code == 200
    body = response.text
    assert "private-user-id" not in body
    assert "private-cds-key" not in body
    assert "private-password" not in body


def test_real_mode_system_status_reports_missing_nepsg_explicitly(monkeypatch):
    from src.era5_cds import make_data_source_status

    monkeypatch.setattr(settings, "data_mode", "REAL")
    monkeypatch.setattr(settings, "neps_g_forecast_path", None)
    status = make_data_source_status()
    assert status["neps_g"]["status"] == "NEPS-G: CONFIGURED BUT DATA UNAVAILABLE"
    assert status["neps_g"]["available"] is False
