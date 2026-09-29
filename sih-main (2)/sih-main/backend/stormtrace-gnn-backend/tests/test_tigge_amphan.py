from __future__ import annotations

import json

import numpy as np

from config import settings
import src.tigge_amphan as tigge
from src.severity import classify_demo_severity


class _FixtureProvider:
    def __init__(self):
        self.latitude = np.asarray([5.0, 10.0, 15.0, 20.0])
        self.longitude = np.asarray([75.0, 80.0, 85.0, 90.0, 95.0])
        self.metadata = {
            "variables": ("t2m", "rainfall", "wind_u", "wind_v"),
            "valid_time": ("2020-05-16T00:00:00", "2020-05-16T06:00:00"),
            "lead_time": (0.0, 6.0),
            "initialization_time": "2020-05-16T00:00:00",
            "uri": "fixture.grib",
            "source": "NCMRWF TIGGE",
            "variable_semantics": {
                "rainfall": "accumulated water-equivalent depth; not a rate"
            },
        }

    def generate_ensemble(self):
        field_shape = (2, 2, 4, 4, 5)
        values = np.zeros(field_shape, dtype=np.float32)
        temperature_pattern = np.linspace(0.0, 3.0, 20, dtype=np.float32).reshape(4, 5)
        wind_pattern = np.linspace(0.0, 20.0, 20, dtype=np.float32).reshape(4, 5)
        values[:, :, 0] = (
            300.0
            + np.arange(2, dtype=np.float32)[:, None, None, None] * 0.1
            + temperature_pattern
        )
        values[:, :, 1] = 0.01
        values[:, :, 2] = 5.0 + wind_pattern
        values[:, :, 3] = 2.0
        self.metadata["ensemble_count"] = 2
        return values

    def grid_coordinates(self, shape):
        assert shape == (4, 5)
        return self.latitude.copy(), self.longitude.copy()


def _write_reference(cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    times = np.asarray(
        [
            f"{year}-05-{day:02d}T{hour:02d}:00:00"
            for year in range(2011, 2020)
            for day in range(1, 32)
            for hour in (0, 6, 12, 18)
        ],
        dtype="datetime64[s]",
    )
    latitude = np.asarray([5.0, 10.0, 15.0, 20.0])
    longitude = np.asarray([75.0, 80.0, 85.0, 90.0, 95.0])
    base = np.arange(times.size, dtype=np.float32)[:, None, None] * 0.1 + 280.0
    field = np.broadcast_to(base, (times.size, 4, 5)).copy()
    np.savez_compressed(
        cache_dir / tigge._REFERENCE_CACHE,
        times=times,
        latitude=latitude,
        longitude=longitude,
        t2m=field,
    )
    (cache_dir / (tigge._REFERENCE_CACHE + ".json")).write_text(
        json.dumps(
            {
                "years": list(range(2011, 2020)),
                "contains_2020": False,
                "variables_used_for_reference": ["t2m"],
            }
        ),
        encoding="utf-8",
    )


def test_tigge_cache_precomputes_reference_proxy_gnn_and_geographic_tracks(
    tmp_path, monkeypatch
):
    _write_reference(tmp_path)
    monkeypatch.setattr(tigge, "NCMRWFTIGGEProvider", _FixtureProvider)
    monkeypatch.setattr(settings, "data_domain", None)
    monkeypatch.setattr(settings, "data_mode", "SYNTHETIC_DEMO")

    result = tigge.build_tigge_amphan_cache(tmp_path, training_epochs=2)
    replay = tigge.load_tigge_replay(tmp_path)
    field = tigge.load_tigge_field(tmp_path, "anomaly_score", 0, visualization=True)
    rainfall = tigge.load_tigge_field(tmp_path, "rainfall", 0)

    assert result["status"] == "READY"
    assert replay["mode"] == "REAL_FORECAST_REPLAY"
    assert replay["data_status"]["reference"]["contains_2020"] is False
    assert replay["timeline"][1]["lead_time_hours"] == 6.0
    assert field["latitude"] == [5.0, 10.0, 15.0, 20.0]
    assert field["longitude"] == [75.0, 80.0, 85.0, 90.0, 95.0]
    assert "wind spatial percentile proxy" in field["classification"]
    assert rainfall["status"] == "ACCUMULATED_AMOUNT"
    assert "not a rate" in rainfall["classification"]
    assert replay["verification"]["verification_status"] == "NOT_AVAILABLE / EXPERIMENTAL"


def test_tigge_reference_rejects_2020_contamination(tmp_path):
    _write_reference(tmp_path)
    reference_meta_path = tmp_path / (tigge._REFERENCE_CACHE + ".json")
    payload = json.loads(reference_meta_path.read_text(encoding="utf-8"))
    payload["years"].append(2020)
    reference_meta_path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        tigge.build_tigge_amphan_cache(tmp_path)
    except ValueError as exc:
        assert "2011–2019 reference" in str(exc)
    else:
        raise AssertionError("2020 reference contamination must be rejected")


def test_project_severity_is_a_deterministic_anomaly_coverage_tier():
    assert classify_demo_severity(0.02) == "NORMAL"
    assert classify_demo_severity(0.12) == "WATCH"
    assert classify_demo_severity(0.25) == "HIGH"
    assert classify_demo_severity(0.40) == "EXTREME"


def test_spatial_percentile_preserves_ties_without_fabricating_a_gradient():
    percentile = tigge._spatial_percentile(np.ones((1, 2, 3), dtype=np.float32))
    assert np.all(percentile == 0.5)


def test_tigge_replay_api_serves_cached_forecast_fields_and_limitations(
    tmp_path, monkeypatch
):
    from fastapi.testclient import TestClient

    from src.main import app

    times = np.asarray(["2020-05-16T00:00:00"], dtype="datetime64[s]")
    np.savez_compressed(
        tmp_path / tigge.TIGGE_ANALYSIS_CACHE,
        times=times,
        latitude=np.asarray([5.0, 10.0]),
        longitude=np.asarray([75.0, 80.0]),
        lead_time_hours=np.asarray([0.0]),
        t2m=np.full((1, 2, 2), 300.0, dtype=np.float32),
        wind_speed=np.ones((1, 2, 2), dtype=np.float32),
        rainfall_accumulation=np.ones((1, 2, 2), dtype=np.float32) * 0.01,
        efi=np.ones((1, 2, 2), dtype=np.float32) * 0.5,
        wind_anomaly_proxy=np.ones((1, 2, 2), dtype=np.float32) * 0.4,
        anomaly_score=np.ones((1, 2, 2), dtype=np.float32) * 0.5,
        gnn_probability=np.ones((1, 2, 2), dtype=np.float32) * 0.7,
    )
    replay = {
        "event": "Cyclone Amphan",
        "event_period": "2020-05-16T00:00:00/2020-05-16T00:00:00",
        "status": "READY",
        "data_source": "NCMRWF TIGGE / NEPS-G",
        "mode": "REAL_FORECAST_REPLAY",
        "forecast_initialization_time": "2020-05-16T00:00:00",
        "progress": [],
        "timeline": [
            {
                "time": "2020-05-16T00:00:00",
                "valid_time": "2020-05-16T00:00:00",
                "lead_time_hours": 0.0,
                "efi_max": 0.5,
                "anomaly_max": 0.5,
                "gnn_probability_max": 0.7,
                "active_nodes": 0,
                "track_count": 0,
            }
        ],
        "tracks": [],
        "dominant_track": None,
        "alert": {
            "event": "Cyclone Amphan",
            "event_period": "2020-05-16T00:00:00/2020-05-16T00:00:00",
            "centroid": {"latitude": 10.0, "longitude": 80.0},
            "severity": "WATCH",
        },
        "alerts": [
            {
                "event": "Cyclone Amphan",
                "time": "2020-05-16T00:00:00",
                "location": {"latitude": 10.0, "longitude": 80.0},
                "severity": "WATCH",
                "confidence": 0.7,
                "anomaly_score": 0.5,
                "temperature_efi_like": 0.5,
                "lead_time_hours": 0.0,
                "source": "NCMRWF TIGGE (dems)",
                "model_version": "fixture",
                "notice": "Not operational",
            }
        ],
        "spatial_bounds": {
            "latitude_min": 5.0,
            "latitude_max": 10.0,
            "longitude_min": 75.0,
            "longitude_max": 80.0,
        },
        "grid_dimensions": {"latitude": 2, "longitude": 2},
        "physics": {"physics_status": "PHYSICS_PROXY", "physics_method": "none", "reason": "fixture"},
        "verification": {
            "verification_status": "NOT_AVAILABLE / EXPERIMENTAL",
            "reason": "fixture",
            "diagnostics": {},
        },
        "model_status": {
            "classification": "EXPERIMENTAL",
            "model_version": "fixture",
            "training_epochs": 1,
            "calibrated": False,
            "accuracy": "NOT AVAILABLE",
        },
        "data_status": {
            "reference": {"years": list(range(2011, 2020)), "contains_2020": False},
            "rainfall": {"status": "ACCUMULATED_PRECIPITATION_NOT_RATE"},
        },
        "provenance": {"forecast": {}},
        "anomaly_classification": "Experimental proxy",
        "data_limitations": ["Experimental; not verified"],
    }
    (tmp_path / tigge.TIGGE_RESULT_CACHE).write_text(
        json.dumps(replay), encoding="utf-8"
    )
    monkeypatch.setattr(settings, "imdaa_processed_dir", str(tmp_path))
    client = TestClient(app)

    started = client.post("/api/tigge/amphan/replay")
    overview = client.get("/api/tigge/amphan/overview")
    field = client.get(
        "/api/tigge/amphan/field",
        params={"variable": "gnn_probability", "time_index": 0, "visualization": True},
    )
    alert = client.get("/api/tigge/amphan/alert", params={"time_index": 0})
    verification = client.get("/api/tigge/amphan/verification")

    assert started.status_code == 200
    assert started.json()["mode"] == "REAL_FORECAST_REPLAY"
    assert overview.status_code == 200
    assert overview.json()["data_source"] == "NCMRWF TIGGE / NEPS-G"
    assert field.status_code == 200
    assert field.json()["latitude"] == [5.0, 10.0]
    assert field.headers["cache-control"].startswith("public, max-age=60")
    cached_field = client.get(
        "/api/tigge/amphan/field",
        params={"variable": "gnn_probability", "time_index": 0, "visualization": True},
        headers={"If-None-Match": field.headers["etag"]},
    )
    assert cached_field.status_code == 304
    assert alert.status_code == 200
    assert alert.json()["lead_time_hours"] == 0.0
    assert verification.json()["verification_status"] == "NOT_AVAILABLE / EXPERIMENTAL"
