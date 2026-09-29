from __future__ import annotations

from pathlib import Path

import cfgrib
import numpy as np
import pytest
import xarray as xr

from config import settings
from src.data_provider import NCMRWFTIGGEProvider, canonicalize_weather_data, get_data_provider


def _group(
    variables: dict[str, tuple[np.ndarray, str, str]],
    *,
    initialization: str = "2020-05-16T00:00:00",
    members: tuple[int, ...] = (1, 2),
    steps: tuple[int, ...] = (0, 6, 12),
):
    latitude = np.asarray([30.0, 29.88], dtype=np.float64)
    longitude = np.asarray([70.02, 70.20, 70.38], dtype=np.float64)
    step_values = np.asarray(steps, dtype="timedelta64[h]")
    valid_times = np.datetime64(initialization, "ns") + step_values.astype("timedelta64[ns]")
    data_vars = {}
    for name, (values, units, step_type) in variables.items():
        data_vars[name] = (
            ("number", "step", "latitude", "longitude"),
            values,
            {"units": units, "GRIB_stepType": step_type},
        )
    return xr.Dataset(
        data_vars,
        coords={
            "number": np.asarray(members),
            "step": step_values,
            "time": np.datetime64(initialization, "ns"),
            "valid_time": ("step", valid_times),
            "latitude": latitude,
            "longitude": longitude,
        },
        attrs={
            "GRIB_centre": "dems",
            "GRIB_centreDescription": "New Delhi",
            "GRIB_dataType": "pf",
            "GRIB_edition": 2,
        },
    )


def _mock_groups():
    shape = (2, 3, 2, 3)
    return [
        _group(
            {
                "u10": (np.full(shape, 4.0, dtype=np.float32), "m s**-1", "instant"),
                "v10": (np.full(shape, -2.0, dtype=np.float32), "m s**-1", "instant"),
            }
        ),
        _group(
            {"t2m": (np.full(shape, 290.0, dtype=np.float32), "K", "instant")}
        ),
        _group(
            {"tp": (np.full(shape, 12.0, dtype=np.float32), "kg m**-2", "accum")}
        ),
    ]


def test_ncmrwf_tigge_groups_form_canonical_ensemble_and_provenance(
    tmp_path, monkeypatch
):
    source = tmp_path / "forecast.grib"
    source.write_bytes(b"mock GRIB fixture")
    groups = _mock_groups()
    reference = np.zeros((2, 4, 2, 3), dtype=np.float32)
    reference[:, 0] = 290.0
    monkeypatch.setattr(cfgrib, "open_datasets", lambda *args, **kwargs: groups)
    monkeypatch.setattr(settings, "neps_g_forecast_path", str(source))
    monkeypatch.setattr(settings, "data_mode", "REAL")
    monkeypatch.setattr(settings, "ensemble_member", None)
    monkeypatch.setattr(settings, "data_domain", None)

    provider = NCMRWFTIGGEProvider()
    ensemble = provider.generate_ensemble()
    canonical = canonicalize_weather_data(
        ensemble,
        reference,
        provider,
    )

    assert canonical.ensemble.shape == (3, 2, 4, 2, 3)
    assert canonical.variables == ("t2m", "rainfall", "wind_u", "wind_v")
    assert canonical.units == ("K", "m", "m s**-1", "m s**-1")
    assert canonical.ensemble[0, 0, 1, 0, 0] == pytest.approx(0.012)
    metadata = canonical.provenance
    assert metadata["centre"] == "dems"
    assert metadata["centre_description"] == "New Delhi"
    assert metadata["data_type"] == "pf"
    assert metadata["grib_edition"] == 2
    assert metadata["initialization_time"] == "2020-05-16T00:00:00"
    assert metadata["ensemble_members"] == ("1", "2")
    assert metadata["lead_time"] == (0.0, 6.0, 12.0)
    assert metadata["valid_time"] == (
        "2020-05-16T00:00:00",
        "2020-05-16T06:00:00",
        "2020-05-16T12:00:00",
    )
    assert "not a rate" in metadata["variable_semantics"]["rainfall"]
    assert metadata["quality_control"]["status"] == "passed"


def test_ncmrwf_tigge_rejects_missing_variables_and_wrong_tp_step_type(
    tmp_path, monkeypatch
):
    source = tmp_path / "forecast.grib"
    source.write_bytes(b"mock GRIB fixture")
    monkeypatch.setattr(settings, "neps_g_forecast_path", str(source))
    monkeypatch.setattr(settings, "data_mode", "LOCAL_REAL")
    monkeypatch.setattr(settings, "ensemble_member", None)

    missing_groups = _mock_groups()[:2]
    monkeypatch.setattr(
        cfgrib,
        "open_datasets",
        lambda *args, **kwargs: missing_groups,
    )
    with pytest.raises(ValueError, match="missing required variables.*tp"):
        NCMRWFTIGGEProvider().generate_ensemble()

    wrong_groups = _mock_groups()
    wrong_groups[-1]["tp"].attrs["GRIB_stepType"] = "instant"
    monkeypatch.setattr(
        cfgrib,
        "open_datasets",
        lambda *args, **kwargs: wrong_groups,
    )
    with pytest.raises(ValueError, match="tp stepType must be 'accum'"):
        NCMRWFTIGGEProvider().generate_ensemble()


def test_real_mode_uses_configured_tigge_file_and_fails_if_missing(monkeypatch):
    monkeypatch.setattr(settings, "data_mode", "REAL")
    monkeypatch.setattr(settings, "neps_g_forecast_path", None)
    with pytest.raises(RuntimeError, match="NEPS-G: CONFIGURED BUT DATA UNAVAILABLE"):
        get_data_provider()


@pytest.mark.skipif(
    not settings.neps_g_forecast_path
    or not Path(settings.neps_g_forecast_path).is_file(),
    reason="local NCMRWF TIGGE GRIB is not available",
)
def test_real_ncmrwf_tigge_grib_smoke(monkeypatch):
    monkeypatch.setattr(settings, "data_mode", "REAL")
    monkeypatch.setattr(settings, "ensemble_member", None)
    monkeypatch.setattr(settings, "data_domain", None)
    monkeypatch.setattr(settings, "forecast_lead_hours", None)
    monkeypatch.setattr(settings, "forecast_start_time", None)
    monkeypatch.setattr(settings, "forecast_end_time", None)
    monkeypatch.setattr(settings, "forecast_initialization_time", None)

    provider = NCMRWFTIGGEProvider()
    ensemble = provider.generate_ensemble()

    assert ensemble.shape == (41, 11, 4, 209, 195)
    assert provider.metadata["uri"].endswith("tigge_ncmrwf_amphan_20200516_00.grib")
    assert provider.metadata["source"] == "NCMRWF TIGGE"
    assert provider.metadata["centre"] == "dems"
    assert provider.metadata["data_type"] == "pf"
    assert provider.metadata["grib_edition"] == 2
    assert provider.metadata["initialization_time"] == "2020-05-16T00:00:00"
    assert provider.metadata["region"] == {
        "latitude_min": pytest.approx(5.04),
        "latitude_max": pytest.approx(30.0),
        "longitude_min": pytest.approx(70.02),
        "longitude_max": pytest.approx(104.94),
    }
    assert provider.metadata["ensemble_count"] == 11
    assert provider.metadata["lead_time_range_hours"] == [0.0, 240.0]
    assert provider.metadata["lead_time"] == tuple(float(step) for step in range(0, 241, 6))
    assert provider.metadata["valid_time"][0] == "2020-05-16T00:00:00"
    assert provider.metadata["valid_time"][-1] == "2020-05-26T00:00:00"
    assert provider.metadata["ensemble_members"] == tuple(str(member) for member in range(1, 12))
    assert provider.metadata["grid_resolution_degrees"]["latitude_degrees"] == pytest.approx(0.12)
    assert provider.metadata["grid_resolution_degrees"]["longitude_degrees"] == pytest.approx(0.18)
    assert provider.metadata["variables"] == ("t2m", "rainfall", "wind_u", "wind_v")
    assert provider.metadata["source_variables"] == ("t2m", "tp", "u10", "v10")
    assert provider.metadata["processing_version"] == "ncmrwf-tigge-grib-v1"
    assert provider.metadata["variable_semantics"]["rainfall"].endswith("not a rate")
    assert provider.metadata["variable_units"] == ("K", "m", "m s**-1", "m s**-1")
