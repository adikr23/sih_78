"""Cached NCMRWF TIGGE Amphan forecast replay and geographic anomaly diagnostics."""

from __future__ import annotations

import json
import copy
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import sparse
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import rankdata

from config import settings
from src.amphan_replay import _field_statistics, _save_npz, _visualization_subset
from src.data_provider import NCMRWFTIGGEProvider
from src.gnn import AnomalyGNN, train_gnn
from src.mesh import scipy_csr_to_torch_sparse
from src.severity import classify_demo_severity
from src.tracking import track_anomaly_4d


TIGGE_ANALYSIS_CACHE = "tigge_amphan_analysis.npz"
TIGGE_RESULT_CACHE = "tigge_amphan_replay.json"
_REFERENCE_CACHE = "reference_may_2011_2019.npz"
_REFERENCE_YEARS = list(range(2011, 2020))
_CACHE_VERSION = "tigge-amphan-replay-v2"
_TRACK_MAX_LINK_CHORD_DISTANCE = 0.05


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Required TIGGE replay metadata is missing or invalid: {path}") from exc


@lru_cache(maxsize=8)
def _cached_replay(path_text: str, modified_ns: int, file_size: int) -> dict[str, Any]:
    del modified_ns, file_size
    return _read_json(Path(path_text))


@lru_cache(maxsize=24)
def _cached_analysis_array(path_text: str, modified_ns: int, array_name: str) -> np.ndarray:
    del modified_ns
    with np.load(path_text, allow_pickle=False) as archive:
        array = np.asarray(archive[array_name])
    array.setflags(write=False)
    return array


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _grid_adjacency(rows: int, columns: int) -> sparse.csr_matrix:
    """Four-neighbour graph over the actual regional forecast grid."""
    node_ids = np.arange(rows * columns, dtype=np.int64).reshape(rows, columns)
    source = np.concatenate(
        (node_ids[:, :-1].ravel(), node_ids[:-1, :].ravel())
    )
    target = np.concatenate(
        (node_ids[:, 1:].ravel(), node_ids[1:, :].ravel())
    )
    edges = np.concatenate(
        (
            np.stack((source, target)),
            np.stack((target, source)),
        ),
        axis=1,
    )
    adjacency = sparse.coo_matrix(
        (np.ones(edges.shape[1], dtype=np.float32), (edges[0], edges[1])),
        shape=(rows * columns, rows * columns),
    ).tocsr()
    adjacency.data[:] = 1.0
    adjacency.setdiag(1.0)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse_sqrt_degree = np.power(degree, -0.5, where=degree > 0)
    return (
        sparse.diags(inverse_sqrt_degree)
        @ adjacency
        @ sparse.diags(inverse_sqrt_degree)
    ).tocsr()


def _geographic_features(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    latitude_radians, longitude_radians = np.radians(
        np.meshgrid(latitude, longitude, indexing="ij")
    )
    return np.stack(
        (
            np.sin(latitude_radians),
            np.cos(latitude_radians),
            np.sin(longitude_radians),
            np.cos(longitude_radians),
        ),
        axis=-1,
    ).reshape(-1, 4).astype(np.float32)


def _interpolate_reference(
    reference: np.ndarray,
    reference_latitude: np.ndarray,
    reference_longitude: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    times: np.ndarray,
) -> dict[int, np.ndarray]:
    if reference.ndim != 3 or reference.shape[0] != times.size:
        raise ValueError("IMDAA reference temperature and timestamps do not align")
    if reference.shape[1:] != (reference_latitude.size, reference_longitude.size):
        raise ValueError("IMDAA reference temperature grid does not match its coordinates")
    if not (
        np.all(np.diff(reference_latitude) > 0)
        and np.all(np.diff(reference_longitude) > 0)
    ):
        raise ValueError("IMDAA reference coordinates must be ascending")
    if (
        latitude.min() < reference_latitude.min()
        or latitude.max() > reference_latitude.max()
        or longitude.min() < reference_longitude.min()
        or longitude.max() > reference_longitude.max()
    ):
        raise ValueError("TIGGE replay region extends outside the IMDAA reference grid")

    query_latitude, query_longitude = np.meshgrid(latitude, longitude, indexing="ij")
    query_points = np.column_stack((query_latitude.ravel(), query_longitude.ravel()))
    result: dict[int, np.ndarray] = {}
    time_hours = np.asarray(
        [int(np.datetime_as_string(value, unit="h")[11:13]) for value in times]
    )
    for hour in (0, 6, 12, 18):
        hour_indices = np.flatnonzero(time_hours == hour)
        if not hour_indices.size:
            continue
        interpolated = np.empty(
            (hour_indices.size, latitude.size, longitude.size), dtype=np.float32
        )
        for output_index, reference_index in enumerate(hour_indices):
            interpolator = RegularGridInterpolator(
                (reference_latitude, reference_longitude),
                reference[reference_index],
                method="linear",
                bounds_error=True,
            )
            interpolated[output_index] = interpolator(query_points).reshape(
                latitude.size, longitude.size
            )
        result[hour] = interpolated.reshape(hour_indices.size, -1)
    if not result:
        raise ValueError("IMDAA reference has no synoptic-hour samples for the TIGGE forecast")
    return result


def _temperature_efi_like(
    forecast_members: np.ndarray,
    baseline_by_hour: dict[int, np.ndarray],
    valid_times: np.ndarray,
) -> np.ndarray:
    steps, members, rows, columns = forecast_members.shape
    result = np.empty((steps, rows, columns), dtype=np.float32)
    hours = np.asarray(
        [int(np.datetime_as_string(value, unit="h")[11:13]) for value in valid_times]
    )
    for step, hour in enumerate(hours):
        if int(hour) not in baseline_by_hour:
            raise ValueError(f"IMDAA reference has no samples for valid-time hour {hour:02d} UTC")
        baseline = baseline_by_hour[int(hour)]
        member_scores = np.empty((members, rows * columns), dtype=np.float32)
        for member in range(members):
            values = forecast_members[step, member].reshape(-1)
            below = (baseline < values[None, :]).mean(axis=0)
            tied = (baseline == values[None, :]).mean(axis=0)
            member_scores[member] = 2.0 * (below + 0.5 * tied) - 1.0
        result[step] = member_scores.mean(axis=0).reshape(rows, columns)
    return result


def _spatial_percentile(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(values.shape[0], -1)
    scores = np.empty_like(flat, dtype=np.float32)
    for step, field in enumerate(flat):
        scores[step] = (rankdata(field, method="average") - 1.0) / max(
            field.size - 1, 1
        )
    return scores.reshape(values.shape)


def _unit_sphere_grid(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    lat_grid, lon_grid = np.meshgrid(np.radians(latitude), np.radians(longitude), indexing="ij")
    return np.stack(
        (
            np.cos(lat_grid) * np.cos(lon_grid),
            np.cos(lat_grid) * np.sin(lon_grid),
            np.sin(lat_grid),
        ),
        axis=-1,
    ).reshape(-1, 3)


def _summary(replay: dict[str, Any]) -> dict[str, Any]:
    return {
        "project": "STORMTRACE-GNN",
        "event": replay["event"],
        "event_period": replay["event_period"],
        "dataset": replay["data_source"],
        "reference_period": "May 2011–2019 IMDAA temperature; UTC-hour matched",
        "spatial_bounds": replay["spatial_bounds"],
        "grid_dimensions": replay["grid_dimensions"],
        "number_of_timestamps": len(replay["timeline"]),
        "processing_stages": replay["progress"],
        "alert": replay["alert"],
        "physics_status": replay["physics"]["physics_status"],
        "verification_status": replay["verification"]["verification_status"],
        "rainfall_status": replay["data_status"]["rainfall"]["status"],
        "model_status": replay["model_status"],
        "data_limitations": replay["data_limitations"],
    }


def build_tigge_amphan_cache(
    cache_dir: str | Path,
    forecast_path: str | Path | None = None,
    *,
    training_epochs: int = 20,
) -> dict[str, Any]:
    """Read the local TIGGE file once and cache geographic fields/tracks for offline API use."""
    cache_root = Path(cache_dir)
    reference_path = cache_root / _REFERENCE_CACHE
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"2011–2019 IMDAA reference cache is required: {reference_path}; "
            "build it with scripts/build_imdaa_cache.py"
        )
    reference_metadata = _read_json(reference_path.with_suffix(".npz.json"))
    if reference_metadata.get("years") != _REFERENCE_YEARS or reference_metadata.get("contains_2020"):
        raise ValueError("TIGGE baseline must be the separate IMDAA 2011–2019 reference")
    if reference_metadata.get("variables_used_for_reference") != ["t2m"]:
        raise ValueError("TIGGE MVP requires the audited IMDAA t2m-only reference cache")
    with np.load(reference_path, allow_pickle=False) as archive:
        reference_times = np.asarray(archive["times"]).astype("datetime64[s]")
        reference = np.asarray(archive["t2m"], dtype=np.float32)
        reference_latitude = np.asarray(archive["latitude"], dtype=np.float64)
        reference_longitude = np.asarray(archive["longitude"], dtype=np.float64)
    reference_time_strings = np.datetime_as_string(reference_times, unit="s")
    reference_years = sorted({int(value[:4]) for value in reference_time_strings})
    expected_reference_times = np.asarray(
        [
            f"{year}-05-{day:02d}T{hour:02d}:00:00"
            for year in _REFERENCE_YEARS
            for day in range(1, 32)
            for hour in (0, 6, 12, 18)
        ],
        dtype="datetime64[s]",
    )
    if reference_years != _REFERENCE_YEARS or not np.array_equal(
        np.sort(reference_times), expected_reference_times
    ):
        raise ValueError(
            "IMDAA reference samples must be exactly the 1,116 May timestamps "
            "from 2011–2019, with no 2020 samples"
        )

    original_path = settings.neps_g_forecast_path
    original_mode = settings.data_mode
    original_domain = settings.data_domain
    if forecast_path is not None:
        settings.neps_g_forecast_path = str(forecast_path)
    settings.data_mode = "REAL"
    # The reference does not cover the full TIGGE domain; only use their shared region.
    settings.data_domain = "5.04,30.0,75.0,99.96"
    try:
        provider = NCMRWFTIGGEProvider()
        ensemble = provider.generate_ensemble()
        latitude, longitude = provider.grid_coordinates(ensemble.shape[-2:])
        metadata = provider.metadata.copy()
    finally:
        settings.neps_g_forecast_path = original_path
        settings.data_mode = original_mode
        settings.data_domain = original_domain

    if ensemble.shape[2] != 4 or tuple(metadata["variables"]) != (
        "t2m",
        "rainfall",
        "wind_u",
        "wind_v",
    ):
        raise ValueError("TIGGE cache requires the canonical four-variable forecast tensor")
    valid_times = np.asarray(metadata["valid_time"], dtype="datetime64[s]")
    if valid_times.shape != (ensemble.shape[0],):
        raise ValueError("TIGGE forecast valid times do not match forecast steps")

    baseline_by_hour = _interpolate_reference(
        reference,
        reference_latitude,
        reference_longitude,
        latitude,
        longitude,
        reference_times,
    )
    t2m_members = ensemble[:, :, 0]
    temperature = t2m_members.mean(axis=1)
    wind_speed_members = np.hypot(ensemble[:, :, 2], ensemble[:, :, 3])
    wind_speed = wind_speed_members.mean(axis=1)
    rainfall_accumulation = ensemble[:, :, 1].mean(axis=1)
    temperature_efi = _temperature_efi_like(t2m_members, baseline_by_hour, valid_times)
    temperature_spatial_percentile = _spatial_percentile(temperature)
    wind_spatial_percentile = _spatial_percentile(wind_speed)
    wind_anomaly_proxy = np.clip(2.0 * wind_spatial_percentile - 1.0, 0.0, 1.0)
    anomaly_score = np.maximum(
        0.5 * np.clip(temperature_efi, 0.0, 1.0),
        wind_anomaly_proxy,
    )

    rows, columns = temperature.shape[-2:]
    latitude_grid, longitude_grid = np.meshgrid(latitude, longitude, indexing="ij")
    geographic = _geographic_features(latitude, longitude)
    wind_standard = (wind_speed - np.median(wind_speed)) / max(
        float(np.std(wind_speed)), 1e-6
    )
    feature_fields = np.stack(
        (
            np.clip(temperature_efi, 0.0, 1.0),
            wind_anomaly_proxy,
            temperature_spatial_percentile,
            np.broadcast_to(wind_standard, wind_speed.shape),
        ),
        axis=-1,
    )
    features = np.concatenate(
        (
            feature_fields.reshape(
                temperature.shape[0], rows * columns, feature_fields.shape[-1]
            ),
            np.broadcast_to(
                geographic[None, :, :],
                (temperature.shape[0], geographic.shape[0], geographic.shape[1]),
            ),
        ),
        axis=-1,
    ).astype(np.float32)
    graph = _grid_adjacency(rows, columns)
    graph_torch = scipy_csr_to_torch_sparse(graph)
    training_step = int(anomaly_score.max(axis=(1, 2)).argmax())
    training_features = torch.as_tensor(features[training_step], dtype=torch.float32)
    training_labels = torch.as_tensor(
        (
            (wind_spatial_percentile[training_step] >= 0.9)
            | (temperature_spatial_percentile[training_step] >= 0.95)
        ).reshape(-1).astype(np.float32)
    )
    if not torch.any(training_labels > 0.5):
        raise ValueError("TIGGE anomaly proxy produced no training labels at the configured threshold")
    torch.manual_seed(20200516)
    model, loss_history = train_gnn(
        training_features,
        training_labels,
        graph_torch,
        epochs=training_epochs,
    )
    probabilities = np.empty_like(anomaly_score, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for step in range(temperature.shape[0]):
            probabilities[step] = model(
                torch.as_tensor(features[step], dtype=torch.float32), graph_torch
            ).cpu().numpy().reshape(rows, columns)

    anomaly_coverage = np.mean(anomaly_score >= 0.9, axis=(1, 2))
    severity = [
        classify_demo_severity(float(anomaly_coverage[step]))
        for step in range(temperature.shape[0])
    ]
    lead_hours = [float(value) for value in metadata["lead_time"]]
    valid_time_strings = [str(value) for value in valid_times]
    vertex_coordinates = _unit_sphere_grid(latitude, longitude)
    bbox, tracks = track_anomaly_4d(
        probabilities.reshape(probabilities.shape[0], -1),
        graph,
        vertex_coordinates,
        threshold=0.65,
        max_gap_steps=1,
        forecast_time=metadata["initialization_time"],
        valid_times=valid_time_strings,
        lead_times_hours=lead_hours,
        efi_per_step=temperature_efi.reshape(temperature_efi.shape[0], -1),
        anomaly_per_step=anomaly_score.reshape(anomaly_score.shape[0], -1),
        severity_per_step=severity,
        max_link_distance=_TRACK_MAX_LINK_CHORD_DISTANCE,
    )
    timeline = []
    for step, valid_time in enumerate(valid_time_strings):
        active = sum(
            1
            for track in tracks
            for point in track["track_points"]
            if point["valid_time"] == valid_time
        )
        timeline.append(
            {
                "time": valid_time,
                "valid_time": valid_time,
                "lead_time_hours": lead_hours[step],
                "efi_max": float(np.max(temperature_efi[step])),
                "anomaly_max": float(np.max(anomaly_score[step])),
                "anomaly_coverage": float(anomaly_coverage[step]),
                "gnn_probability_max": float(np.max(probabilities[step])),
                "active_nodes": int(np.sum(probabilities[step] >= 0.65)),
                "track_count": active,
            }
        )
    alert_steps = []
    for step in range(len(timeline)):
        peak_row, peak_column = np.unravel_index(
            np.argmax(anomaly_score[step]), anomaly_score[step].shape
        )
        active_track_points = [
            point
            for track in tracks
            for point in track["track_points"]
            if point["valid_time"] == valid_time_strings[step]
        ]
        selected_track_point = max(
            active_track_points,
            key=lambda point: point.get("confidence") or 0.0,
            default=None,
        )
        if selected_track_point is None:
            alert_latitude = float(latitude[peak_row])
            alert_longitude = float(longitude[peak_column])
            alert_confidence = float(probabilities[step, peak_row, peak_column])
            alert_efi = float(temperature_efi[step, peak_row, peak_column])
            alert_anomaly = float(anomaly_score[step, peak_row, peak_column])
        else:
            alert_latitude = float(selected_track_point["latitude"])
            alert_longitude = float(selected_track_point["longitude"])
            alert_confidence = float(selected_track_point["confidence"])
            alert_efi = float(selected_track_point["efi"])
            alert_anomaly = float(selected_track_point["anomaly_score"])
        alert_steps.append(
            {
                "event": "Cyclone Amphan",
                "time": valid_time_strings[step],
                "location": {
                    "latitude": alert_latitude,
                    "longitude": alert_longitude,
                },
                "severity": severity[step],
                "confidence": alert_confidence,
                "anomaly_score": alert_anomaly,
                "severity_basis": {
                    "metric": "regional cell fraction with combined proxy score >= 0.9",
                    "anomaly_coverage": float(anomaly_coverage[step]),
                    "calibration": "project-only uncalibrated thresholds, not official warnings",
                },
                "temperature_efi_like": alert_efi,
                "lead_time_hours": lead_hours[step],
                "source": "NCMRWF TIGGE (dems)",
                "model_version": _CACHE_VERSION,
                "status": "FORECAST_REPLAY",
                "operational_warning": False,
                "notice": "Research prototype output; not a live or official meteorological warning.",
                "bounding_box": (
                    None
                    if selected_track_point is None
                    else selected_track_point["geographic_bbox"]
                ),
            }
        )

    forecast_init = str(metadata["initialization_time"])
    replay = {
        "event": "Cyclone Amphan",
        "event_period": f"{valid_time_strings[0]}/{valid_time_strings[-1]}",
        "status": "READY",
        "data_source": "NCMRWF TIGGE / NEPS-G",
        "mode": "REAL_FORECAST_REPLAY",
        "forecast_initialization_time": forecast_init,
        "progress": [
            {"stage": stage, "status": "READY"}
            for stage in (
                "NCMRWF TIGGE ingestion",
                "IMDAA 2011–2019 temperature reference",
                "Temperature EFI-like anomaly",
                "Wind spatial anomaly proxy",
                "Geographic GNN probability",
                "Temporal storm tracking",
                "Severity and alert",
            )
        ],
        "timeline": timeline,
        "tracks": tracks,
        "dominant_track": bbox,
        "alerts": alert_steps,
        "alert": {
            **alert_steps[0],
            "centroid": alert_steps[0]["location"],
            "track_id": None if bbox is None else bbox.get("track_id"),
        },
        "spatial_bounds": {
            "latitude_min": float(latitude.min()),
            "latitude_max": float(latitude.max()),
            "longitude_min": float(longitude.min()),
            "longitude_max": float(longitude.max()),
        },
        "grid_dimensions": {"latitude": rows, "longitude": columns},
        "physics": {
            "physics_status": "PHYSICS_PROXY",
            "physics_method": "Not calculated for this replay",
            "reason": "Specific humidity is unavailable; no MFC or validated physics score is reported.",
        },
        "verification": {
            "verification_status": "NOT_AVAILABLE / EXPERIMENTAL",
            "reason": "No independent high-resolution truth or verified cyclone track is configured.",
            "diagnostics": {
                "track_count": len(tracks),
                "detected_points": int(sum(item["active_nodes"] for item in timeline)),
                "maximum_anomaly_score": float(np.max(anomaly_score)),
                "maximum_anomaly_coverage": float(np.max(anomaly_coverage)),
                "maximum_lead_time_hours": max(lead_hours),
                "forecast_skill": "NOT AVAILABLE",
            },
        },
        "model_status": {
            "classification": "EXPERIMENTAL weakly supervised demonstration model",
            "model_version": _CACHE_VERSION,
            "training_epochs": int(training_epochs),
            "track_probability_threshold": 0.65,
            "track_max_centroid_chord_distance": _TRACK_MAX_LINK_CHORD_DISTANCE,
            "track_link_gate": "Experimental geographic continuity heuristic; not validated storm motion.",
            "training_labels": "derived from this forecast's temperature/wind anomaly proxies",
            "loss_history": loss_history,
            "calibrated": False,
            "accuracy": "NOT AVAILABLE",
        },
        "data_status": {
            "reference": {
                "source": "IMDAA reanalysis",
                "years": _REFERENCE_YEARS,
                "variable": "t2m only",
                "samples": int(reference_times.size),
                "contains_2020": False,
                "spatial_regridding": "bilinear to the common TIGGE/reference domain",
            },
            "rainfall": {
                "status": "ACCUMULATED_PRECIPITATION_NOT_RATE",
                "source_variable": "tp",
                "source_units": "kg m**-2",
                "canonical_units": "m",
                "analysis": "excluded from anomaly/GNN analysis",
                "semantics": metadata["variable_semantics"]["rainfall"],
            },
            "forecast": {
                "status": "READY",
                "initialization_time": forecast_init,
                "ensemble_count": int(ensemble.shape[1]),
                "lead_time_hours": lead_hours,
                "source_file": metadata["uri"],
            },
        },
        "provenance": {
            "forecast": metadata,
            "reference": reference_metadata,
            "processing_version": _CACHE_VERSION,
            "gnn_training": {
                "independent_ground_truth": False,
                "coordinates_in_features": ["sin_latitude", "cos_latitude", "sin_longitude", "cos_longitude"],
            },
        },
        "anomaly_classification": (
            "Combined demonstration score: positive temperature percentile EFI-like "
            "score and same-forecast within-domain wind spatial percentile proxy."
        ),
        "data_limitations": [
            "Forecast source is a real local NCMRWF TIGGE perturbed ensemble, not a live feed.",
            "Temperature reference is IMDAA May 2011–2019 only; it is not the 2020 event.",
            "The reference cache contains temperature only; wind anomaly is a same-forecast spatial percentile proxy, not climatological EFI.",
            "Combined anomaly and GNN scores are experimental and uncalibrated; no accuracy or forecast skill is claimed.",
            "Track detections use a 0.65 uncalibrated GNN-score threshold; project severity tiers use proxy anomalous-area coverage and are not verified or official warnings.",
            "Accumulated tp is preserved as a water-equivalent depth and excluded from anomaly analysis; it is not a rainfall rate.",
            "Specific humidity is unavailable; MFC is not calculated and physics remains PHYSICS_PROXY.",
            "Independent verification is NOT_AVAILABLE / EXPERIMENTAL.",
            "No validated 12 km to 5 km downscaling is claimed.",
        ],
    }

    cache_root.mkdir(parents=True, exist_ok=True)
    analysis_path = cache_root / TIGGE_ANALYSIS_CACHE
    _save_npz(
        analysis_path,
        times=valid_times,
        lead_time_hours=np.asarray(lead_hours, dtype=np.float32),
        latitude=latitude,
        longitude=longitude,
        t2m=temperature.astype(np.float32),
        wind_speed=wind_speed.astype(np.float32),
        rainfall_accumulation=rainfall_accumulation.astype(np.float32),
        efi=temperature_efi.astype(np.float32),
        wind_anomaly_proxy=wind_anomaly_proxy.astype(np.float32),
        anomaly_score=anomaly_score.astype(np.float32),
        gnn_probability=probabilities.astype(np.float32),
    )
    _write_json(cache_root / TIGGE_RESULT_CACHE, replay)
    return {
        "status": "READY",
        "analysis_cache": str(analysis_path),
        "result_cache": str(cache_root / TIGGE_RESULT_CACHE),
        "steps": len(timeline),
        "ensemble_members": int(ensemble.shape[1]),
        "tracks": len(tracks),
        "event_period": replay["event_period"],
        "source_file": metadata["uri"],
    }


def load_tigge_replay(cache_dir: str | Path) -> dict[str, Any]:
    path = Path(cache_dir) / TIGGE_RESULT_CACHE
    if not path.is_file():
        raise FileNotFoundError(
            "Prepared NCMRWF TIGGE replay cache is missing; run "
            "python scripts/build_tigge_amphan_cache.py"
        )
    stat = path.stat()
    replay = copy.deepcopy(_cached_replay(str(path.resolve()), stat.st_mtime_ns, stat.st_size))
    replay["demo_summary"] = _summary(replay)
    return replay


def load_tigge_field(
    cache_dir: str | Path,
    variable: str,
    time_index: int,
    visualization: bool = False,
    max_points: int = 64,
) -> dict[str, Any]:
    names = {
        "t2m": ("t2m", "K"),
        "wind_speed": ("wind_speed", "m s**-1"),
        "rainfall": ("rainfall_accumulation", "m"),
        "efi": ("efi", "dimensionless"),
        "wind_anomaly_proxy": ("wind_anomaly_proxy", "dimensionless"),
        "anomaly_score": ("anomaly_score", "dimensionless"),
        "gnn_probability": ("gnn_probability", "probability"),
    }
    if variable not in names:
        raise ValueError(f"unsupported TIGGE field {variable!r}; choose from {sorted(names)}")
    replay = load_tigge_replay(cache_dir)
    if time_index < 0 or time_index >= len(replay["timeline"]):
        raise IndexError(f"time_index must be between 0 and {len(replay['timeline']) - 1}")
    key, units = names[variable]
    analysis_path = Path(cache_dir) / TIGGE_ANALYSIS_CACHE
    analysis_stat = analysis_path.stat()
    modified_ns = analysis_stat.st_mtime_ns
    values = np.asarray(_cached_analysis_array(str(analysis_path.resolve()), modified_ns, key)[time_index], dtype=np.float32)
    latitude = np.asarray(_cached_analysis_array(str(analysis_path.resolve()), modified_ns, "latitude"), dtype=np.float64)
    longitude = np.asarray(_cached_analysis_array(str(analysis_path.resolve()), modified_ns, "longitude"), dtype=np.float64)
    valid_time = str(np.asarray(_cached_analysis_array(str(analysis_path.resolve()), modified_ns, "times")[time_index]).astype("datetime64[s]"))
    values, latitude, longitude, visualization_metadata = _visualization_subset(
        values, latitude, longitude, max_points, visualization
    )
    rainfall = variable == "rainfall"
    return {
        "variable": variable,
        "status": "ACCUMULATED_AMOUNT" if rainfall else "READY",
        "valid_time": valid_time,
        "lead_time_hours": replay["timeline"][time_index]["lead_time_hours"],
        "units": units,
        "latitude": latitude.tolist(),
        "longitude": longitude.tolist(),
        "values": values.tolist(),
        "statistics": _field_statistics(values),
        "available": True,
        **visualization_metadata,
        "classification": (
            "GRIB accumulated precipitation depth, not a rate; excluded from anomaly analysis"
            if rainfall
            else replay["anomaly_classification"]
            if variable in {"efi", "wind_anomaly_proxy", "anomaly_score"}
            else "NCMRWF TIGGE forecast field"
            if variable in {"t2m", "wind_speed"}
            else replay["model_status"]["classification"]
        ),
        "provenance": replay["provenance"]["forecast"],
    }


def load_tigge_tracks(cache_dir: str | Path, time_index: int) -> dict[str, Any]:
    replay = load_tigge_replay(cache_dir)
    if time_index < 0 or time_index >= len(replay["timeline"]):
        raise IndexError(f"time_index must be between 0 and {len(replay['timeline']) - 1}")
    valid_time = replay["timeline"][time_index]["time"]
    active_tracks = []
    for track in replay["tracks"]:
        point = next(
            (item for item in track["track_points"] if item["valid_time"] == valid_time),
            None,
        )
        if point is not None:
            active_tracks.append({**track, "selected_track_point": point})
    dominant = max(
        active_tracks,
        key=lambda track: track["selected_track_point"].get("confidence") or 0.0,
        default=None,
    )
    return {
        "event": replay["event"],
        "time": valid_time,
        "active_tracks": active_tracks,
        "dominant_track": dominant,
    }


def load_tigge_alert(cache_dir: str | Path, time_index: int) -> dict[str, Any]:
    replay = load_tigge_replay(cache_dir)
    if time_index < 0 or time_index >= len(replay["alerts"]):
        raise IndexError(f"time_index must be between 0 and {len(replay['alerts']) - 1}")
    alert = dict(replay["alerts"][time_index])
    selected_tracks = load_tigge_tracks(cache_dir, time_index)
    track = selected_tracks["dominant_track"]
    point = track["selected_track_point"] if track else None
    alert["track"] = point
    alert["bounding_box"] = point.get("geographic_bbox") if point else None
    alert["confidence_status"] = "uncalibrated demonstration-model output"
    return alert
