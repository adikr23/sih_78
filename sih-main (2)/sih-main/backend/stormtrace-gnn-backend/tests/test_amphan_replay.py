from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from config import settings
from src.amphan_replay import (
    ANALYSIS_CACHE,
    EVENT_CACHE,
    REFERENCE_CACHE,
    _cache_inputs,
    _run_normalized_unet,
    load_cached_alert,
    load_cached_analysis_field,
    load_cached_field,
    load_cached_tracks,
)


def _write(path: Path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_reference_cache_rejects_2020_leak(tmp_path):
    reference = tmp_path / REFERENCE_CACHE
    event = tmp_path / EVENT_CACHE
    reference.touch()
    event.touch()
    _write(reference.with_suffix(".npz.json"), {"years": [*range(2011, 2020), 2020]})
    _write(event.with_suffix(".npz.json"), {"event_period": "2020-05-16/2020-05-21"})

    with pytest.raises(ValueError, match="only years 2011–2019"):
        _cache_inputs(tmp_path)


def test_cached_t2m_field_keeps_real_geographic_coordinates(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "imdaa_rainfall_mode", "disabled")
    np.savez_compressed(
        tmp_path / ANALYSIS_CACHE,
        times=np.asarray(["2020-05-16T00:00:00"], dtype="datetime64[s]"),
        latitude=np.asarray([5.0, 10.0]),
        longitude=np.asarray([75.0, 80.0]),
        t2m=np.asarray([[[300.0, 301.0], [302.0, 303.0]]]),
        rainfall=np.ones((1, 2, 2), dtype=np.float32),
        wind_u=np.zeros((1, 2, 2), dtype=np.float32),
        wind_v=np.zeros((1, 2, 2), dtype=np.float32),
        efi=np.zeros((1, 2, 2), dtype=np.float32),
        gnn_probability=np.zeros((1, 2, 2), dtype=np.float32),
        crop_times=np.asarray("2020-05-16T00:00:00", dtype="datetime64[s]"),
        crop_latitude=np.asarray([5.0, 10.0]),
        crop_longitude=np.asarray([75.0, 80.0]),
        downscaled_latitude=np.asarray([5.0, 7.5, 10.0]),
        downscaled_longitude=np.asarray([75.0, 77.5, 80.0]),
        crop_t2m=np.asarray([[300.0, 301.0], [302.0, 303.0]], dtype=np.float32),
        crop_efi=np.zeros((2, 2), dtype=np.float32),
        unet=np.full((3, 3), 301.0, dtype=np.float32),
        diffusion=np.full((3, 3), 302.0, dtype=np.float32),
    )
    _write(
        tmp_path / "amphan_replay.json",
        {
            "timeline": [{"time": "2020-05-16T00:00:00"}],
            "event": "Cyclone Amphan",
            "event_period": "2020-05-16/2020-05-21",
            "data_source": "IMDAA",
            "alert": {"centroid": {"latitude": 5.0, "longitude": 75.0}},
            "physics": {"physics_status": "PHYSICS_PROXY"},
            "verification": {"verification_status": "NOT_AVAILABLE"},
            "data_status": {
                "rainfall": {
                    "status": "UNRESOLVED_SEMANTICS",
                    "source_variable": "param8.1.0",
                }
            },
            "provenance": {"event": {"records": {}}},
        },
    )

    result = load_cached_field(tmp_path, "t2m", 0)
    rainfall = load_cached_field(tmp_path, "rainfall", 0)
    sampled = load_cached_field(tmp_path, "t2m", 0, visualization=True, max_points=2)
    downscaling = load_cached_analysis_field(tmp_path, "unet", visualization=True)

    assert result["latitude"] == [5.0, 10.0]
    assert result["longitude"] == [75.0, 80.0]
    assert result["valid_time"] == "2020-05-16T00:00:00"
    assert rainfall["status"] == "UNAVAILABLE"
    assert rainfall["available"] is False
    assert rainfall["values"] == []
    assert sampled["visualization_only"] is True
    assert sampled["source_shape"] == [2, 2]
    assert sampled["returned_shape"] == [2, 2]
    assert downscaling["valid_time"] == "2020-05-16T00:00:00"
    assert "not verified" in downscaling["classification"]


def test_normalized_unet_stays_in_a_temperature_scale():
    field = np.linspace(299.0, 306.0, 16 * 16, dtype=np.float32).reshape(16, 16)

    result = _run_normalized_unet(field, epochs=1)

    assert result.shape == (32, 32)
    assert np.isfinite(result).all()
    assert result.min() > 200.0
    assert result.max() < 400.0


def test_amphan_overview_api_returns_cached_historical_schema(tmp_path, monkeypatch):
    from src.main import app

    monkeypatch.setattr(settings, "imdaa_processed_dir", str(tmp_path))
    expected = {
        "event": "Cyclone Amphan",
        "event_period": "2020-05-16/2020-05-21",
        "status": "READY",
        "data_source": "IMDAA",
        "mode": "HISTORICAL_REPLAY",
        "timeline": [],
        "physics": {"physics_status": "PHYSICS_PROXY"},
        "verification": {"verification_status": "NOT_AVAILABLE"},
        "data_status": {"rainfall": {"status": "UNRESOLVED_SEMANTICS"}},
        "alert": {"centroid": {"latitude": 20.0, "longitude": 88.0}},
    }
    _write(tmp_path / "amphan_replay.json", expected)
    response = TestClient(app).get("/api/amphan/overview")

    assert response.status_code == 200
    assert response.json()["event"] == "Cyclone Amphan"
    assert response.json()["data_source"] == "IMDAA"
    assert response.json()["alert"]["centroid"] == {"latitude": 20.0, "longitude": 88.0}
    assert response.json()["demo_summary"]["reference_period"] == "May 2011–2019"
    assert "NEPS-G forecast source is not connected." in response.json()["demo_summary"]["data_limitations"]


def test_time_indexed_alert_and_track_use_cached_coordinates(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "imdaa_rainfall_mode", "disabled")
    np.savez_compressed(
        tmp_path / ANALYSIS_CACHE,
        times=np.asarray(["2020-05-16T00:00:00"], dtype="datetime64[s]"),
        latitude=np.asarray([5.0, 10.0]),
        longitude=np.asarray([75.0, 80.0]),
        efi=np.asarray([[[0.1, 0.2], [0.3, 0.9]]], dtype=np.float32),
        crop_t2m=np.asarray([[300.0, 301.0]], dtype=np.float32),
        unet=np.asarray([[300.5, 301.2]], dtype=np.float32),
        diffusion=np.asarray([[300.6, 301.1]], dtype=np.float32),
    )
    _write(
        tmp_path / "amphan_replay.json",
        {
            "event": "Cyclone Amphan",
            "event_period": "2020-05-16/2020-05-21",
            "data_source": "IMDAA",
            "mode": "HISTORICAL_REPLAY",
            "timeline": [{"time": "2020-05-16T00:00:00"}],
            "alert": {},
            "physics": {"physics_status": "PHYSICS_PROXY", "physics_score": 0.0},
            "verification": {"verification_status": "NOT_AVAILABLE"},
            "data_status": {"rainfall": {"status": "UNRESOLVED_SEMANTICS"}},
            "tracks": [
                {
                    "track_id": "track-1",
                    "track_points": [
                        {
                            "valid_time": "2020-05-16T00:00:00",
                            "latitude": 8.0,
                            "longitude": 77.0,
                            "efi": 0.5,
                            "confidence": 0.6,
                            "geographic_bbox": {
                                "latitude_min": 7.0,
                                "latitude_max": 9.0,
                                "longitude_west": 76.0,
                                "longitude_east": 78.0,
                            },
                        }
                    ],
                }
            ],
        },
    )

    alert = load_cached_alert(tmp_path, 0)
    tracks = load_cached_tracks(tmp_path, 0)

    assert alert["location"] == {"latitude": 10.0, "longitude": 80.0}
    assert alert["time"] == "2020-05-16T00:00:00"
    assert alert["notice"] == "Not a live operational warning."
    assert alert["track"]["confidence"] == 0.6
    assert tracks["dominant_track"]["track_id"] == "track-1"
