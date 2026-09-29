"""Deterministic, cache-backed Amphan historical replay preparation and reads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config import settings
from src.diffusion import train_and_generate
from src.efi import compute_efi_field
from src.gnn import train_gnn
from src.mesh import (
    build_icosahedral_mesh,
    build_sparse_adjacency,
    grid_to_mesh,
    mesh_to_geographic,
    mesh_to_grid,
    scipy_csr_to_torch_sparse,
)
from src.physics import moisture_convergence_score_numpy
from src.severity import compute_alert
from src.tracking import track_anomaly_4d
from src.unet import train_and_run_unet


REFERENCE_CACHE = "reference_may_2011_2019.npz"
EVENT_CACHE = "amphan_may_2020.npz"
ANALYSIS_CACHE = "amphan_analysis.npz"
RESULT_CACHE = "amphan_replay.json"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"IMDAA cache provenance is unavailable or invalid: {path}") from exc


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _great_circle_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radians = np.radians([lat1, lon1, lat2, lon2])
    a_lat, a_lon, b_lat, b_lon = radians
    delta_lat = b_lat - a_lat
    delta_lon = b_lon - a_lon
    haversine = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(a_lat) * np.cos(b_lat) * np.sin(delta_lon / 2) ** 2
    )
    return float(2 * 6371.0 * np.arcsin(np.sqrt(np.clip(haversine, 0, 1))))


def _cache_inputs(cache_dir: Path):
    reference_path = cache_dir / REFERENCE_CACHE
    event_path = cache_dir / EVENT_CACHE
    if not reference_path.is_file() or not event_path.is_file():
        raise FileNotFoundError(
            "IMDAA replay caches are missing; run "
            "python scripts/build_imdaa_cache.py first"
        )
    reference_provenance = _read_json(reference_path.with_suffix(".npz.json"))
    event_provenance = _read_json(event_path.with_suffix(".npz.json"))
    if reference_provenance.get("years") != list(range(2011, 2020)):
        raise ValueError("IMDAA reference cache must contain only years 2011–2019")
    if 2020 in reference_provenance.get("years", []):
        raise ValueError("2020 must not be included in the IMDAA reference cache")
    if event_provenance.get("event_period") != "2020-05-16/2020-05-21":
        raise ValueError("IMDAA Amphan cache does not cover the expected event period")
    return reference_path, event_path, reference_provenance, event_provenance


def _rainfall_mode() -> str:
    mode = settings.imdaa_rainfall_mode.strip().lower()
    if mode not in {"disabled", "raw_unverified"}:
        raise ValueError("IMDAA_RAINFALL_MODE must be 'disabled' or 'raw_unverified'")
    return mode


def _run_normalized_unet(crop_t2m: np.ndarray, epochs: int) -> np.ndarray:
    center = float(np.mean(crop_t2m))
    scale = max(float(np.std(crop_t2m)), 1.0)
    normalized = (crop_t2m - center) / scale
    coarse_tensor = torch.as_tensor(normalized[None, None], dtype=torch.float32)
    output = train_and_run_unet(coarse_tensor, epochs=epochs).detach().cpu().numpy().squeeze()
    output = output * scale + center
    if not np.isfinite(output).all():
        raise ValueError("Amphan U-Net prototype produced non-finite values")
    return output.astype(np.float32)


def _field_statistics(values: np.ndarray) -> dict[str, float]:
    finite_values = np.asarray(values, dtype=np.float64).ravel()
    if finite_values.size == 0 or not np.isfinite(finite_values).all():
        raise ValueError("Amphan comparison fields must contain finite values")
    p05, p50, p95 = np.percentile(finite_values, [5, 50, 95])
    return {
        "minimum": float(finite_values.min()),
        "maximum": float(finite_values.max()),
        "mean": float(finite_values.mean()),
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
    }


def _demo_summary(payload: dict[str, Any], cache_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "project": "STORMTRACE-GNN",
        "event": payload["event"],
        "event_period": payload["event_period"],
        "dataset": payload["data_source"],
        "reference_period": "May 2011–2019",
        "processing_stages": [
            "IMDAA",
            "Regional preprocessing",
            "May 2011–2019 reference",
            "EFI",
            "Icosahedral Mesh",
            "GNN anomaly tracking",
            "Temporal Tracking",
            "Regional Crop",
            "U-Net / Diffusion experimental downscaling",
            "Severity / Alert",
        ],
        "alert": payload["alert"],
        "physics_status": payload["physics"]["physics_status"],
        "verification_status": payload["verification"]["verification_status"],
        "rainfall_status": payload["data_status"]["rainfall"]["status"],
        "data_limitations": [
            "Historical IMDAA replay only; not a live forecast or operational warning.",
            "NEPS-G forecast source is not connected.",
            "Raw APCP rainfall is preserved but its units and accumulation semantics are unresolved; it is excluded from analysis.",
            "Specific humidity q is unavailable; moisture-flux convergence was not calculated.",
            "Physics diagnostics are PHYSICS_PROXY.",
            "Independent high-resolution verification is NOT_AVAILABLE.",
            "U-Net and diffusion outputs are experimental reconstructions, not validated 5 km observations.",
        ],
    }
    analysis_path = cache_dir / ANALYSIS_CACHE
    if not analysis_path.is_file():
        summary.update(
            spatial_bounds=None,
            grid_dimensions=None,
            number_of_timestamps=len(payload["timeline"]),
            model_comparison={},
        )
        return summary

    with np.load(analysis_path, allow_pickle=False) as archive:
        latitudes = np.asarray(archive["latitude"], dtype=np.float64)
        longitudes = np.asarray(archive["longitude"], dtype=np.float64)
        timestamps = np.asarray(archive["times"])
        summary.update(
            spatial_bounds={
                "latitude_min": float(latitudes.min()),
                "latitude_max": float(latitudes.max()),
                "longitude_min": float(longitudes.min()),
                "longitude_max": float(longitudes.max()),
            },
            grid_dimensions={
                "latitude": int(latitudes.size),
                "longitude": int(longitudes.size),
            },
            number_of_timestamps=int(timestamps.size),
            model_comparison={
                "classification": "Experimental amplitude-preserving reconstruction",
                "units": "K",
                "independent_verification": "NOT_AVAILABLE",
                "fields": {
                    "input_coarse": _field_statistics(archive["crop_t2m"]),
                    "unet_experimental": _field_statistics(archive["unet"]),
                    "diffusion_experimental": _field_statistics(archive["diffusion"]),
                },
            },
        )
    return summary


def precompute_amphan_replay(cache_dir: str | Path) -> dict[str, Any]:
    """Build scientific outputs once; API replay requests only read the caches."""
    rainfall_mode = _rainfall_mode()
    cache_root = Path(cache_dir)
    reference_path, event_path, reference_meta, event_meta = _cache_inputs(cache_root)
    with np.load(reference_path, allow_pickle=False) as reference_cache:
        reference_t2m = np.asarray(reference_cache["t2m"], dtype=np.float32)
        reference_times = np.asarray(reference_cache["times"]).astype("datetime64[s]")
        latitudes = np.asarray(reference_cache["latitude"], dtype=np.float64)
        longitudes = np.asarray(reference_cache["longitude"], dtype=np.float64)
        reference_percentiles = np.asarray(
            reference_cache["percentiles_by_hour"], dtype=np.float32
        )
        percentile_levels = np.asarray(reference_cache["percentile_levels"], dtype=np.int32)
    with np.load(event_path, allow_pickle=False) as event_cache:
        event = {
            key: np.asarray(event_cache[key])
            for key in ("times", "t2m", "rainfall", "wind_u", "wind_v")
        }
        event_times = event["times"].astype("datetime64[s]")
        fields = {
            key: np.asarray(event[key], dtype=np.float32)
            for key in ("t2m", "rainfall", "wind_u", "wind_v")
        }
        event_latitudes = np.asarray(event_cache["latitude"], dtype=np.float64)
        event_longitudes = np.asarray(event_cache["longitude"], dtype=np.float64)

    if not (
        np.array_equal(latitudes, event_latitudes)
        and np.array_equal(longitudes, event_longitudes)
    ):
        raise ValueError("IMDAA reference and event caches must use the same geographic grid")
    if reference_t2m.shape[0] != reference_times.size or reference_t2m.shape[1:] != (
        len(latitudes),
        len(longitudes),
    ):
        raise ValueError("IMDAA reference t2m data do not align with its time and coordinate arrays")
    if event_times.size != 24 or any(values.shape[0] != 24 for values in fields.values()):
        raise ValueError("Amphan event cache must contain 24 records for each required variable")
    if 2020 in {int(str(value)[:4]) for value in reference_times}:
        raise ValueError("2020 event samples leaked into the IMDAA reference cache")

    threshold = float(settings.amphan_efi_threshold)
    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("AMPHAN_EFI_THRESHOLD must be finite and within (0, 1]")
    efi_fields = np.empty_like(fields["t2m"], dtype=np.float32)
    for timestep, valid_time in enumerate(event_times):
        hour = int(np.datetime_as_string(valid_time, unit="h")[11:13])
        reference_indices = np.flatnonzero(
            np.array(
                [int(np.datetime_as_string(value, unit="h")[11:13]) for value in reference_times]
            )
            == hour
        )
        if reference_indices.size == 0:
            raise ValueError(f"IMDAA reference has no t2m samples for UTC hour {hour:02d}")
        baseline = reference_t2m[reference_indices].reshape(reference_indices.size, -1)
        forecast = fields["t2m"][timestep].reshape(-1)
        efi_fields[timestep] = compute_efi_field(forecast, baseline).reshape(
            len(latitudes), len(longitudes)
        )

    # Restrict the existing spherical mesh to the documented IMDAA region.
    vertices_all, faces = build_icosahedral_mesh()
    mesh_latitudes, mesh_longitudes = mesh_to_geographic(vertices_all)
    region_mask = (
        (mesh_latitudes >= latitudes[0] - 0.5)
        & (mesh_latitudes <= latitudes[-1] + 0.5)
        & (mesh_longitudes >= longitudes[0] - 0.5)
        & (mesh_longitudes <= longitudes[-1] + 0.5)
    )
    selected_indices = np.flatnonzero(region_mask)
    if selected_indices.size < 4:
        raise ValueError(
            "configured icosahedral mesh has too few vertices in the Amphan analysis region"
        )
    vertices = vertices_all[selected_indices]
    adjacency_all = build_sparse_adjacency(faces, len(vertices_all))
    adjacency = adjacency_all[selected_indices][:, selected_indices].tocsr()
    adjacency_torch = scipy_csr_to_torch_sparse(adjacency)
    mesh_efi = np.stack(
        [grid_to_mesh(field, vertices, latitudes, longitudes) for field in efi_fields]
    ).astype(np.float32)

    mesh_features = []
    for index, valid_time in enumerate(event_times):
        hour = int(np.datetime_as_string(valid_time, unit="h")[11:13])
        reference_indices = np.flatnonzero(
            np.array(
                [int(np.datetime_as_string(value, unit="h")[11:13]) for value in reference_times]
            )
            == hour
        )
        baseline_grid = reference_t2m[reference_indices]
        baseline_mean = baseline_grid.mean(axis=0)
        baseline_std = np.maximum(baseline_grid.std(axis=0), 1e-4)
        normalized = (fields["t2m"][index] - baseline_mean) / baseline_std
        mesh_normalized = grid_to_mesh(normalized, vertices, latitudes, longitudes)
        mesh_features.append(np.column_stack((mesh_normalized, mesh_efi[index])))
    mesh_features = np.asarray(mesh_features, dtype=np.float32)

    torch.manual_seed(20200516)
    np.random.seed(20200516)
    training_index = int(np.argmax(mesh_efi.max(axis=1)))
    gnn_input = torch.as_tensor(mesh_features[training_index], dtype=torch.float32)
    gnn_labels = torch.as_tensor(
        (mesh_efi[training_index] >= threshold).astype(np.float32)
    )
    model, loss_history = train_gnn(
        gnn_input,
        gnn_labels,
        adjacency_torch,
        epochs=min(30, max(1, settings.gnn_epochs)),
    )
    model.eval()
    predictions = np.empty((len(event_times), len(vertices)), dtype=np.float32)
    with torch.no_grad():
        for index in range(len(event_times)):
            predictions[index] = model(
                torch.as_tensor(mesh_features[index], dtype=torch.float32),
                adjacency_torch,
            ).cpu().numpy()

    track_threshold = float(settings.amphan_gnn_threshold)
    if not np.isfinite(track_threshold) or not 0 < track_threshold < 1:
        raise ValueError("AMPHAN_GNN_THRESHOLD must be finite and within (0, 1)")
    track_bbox, tracks = track_anomaly_4d(
        predictions,
        adjacency,
        vertices,
        threshold=track_threshold,
        max_gap_steps=1,
        valid_times=tuple(str(value) for value in event_times),
        efi_per_step=mesh_efi,
    )
    probability_fields = np.stack(
        [
            mesh_to_grid(
                predictions[index],
                vertices,
                (len(latitudes), len(longitudes)),
                latitudes,
                longitudes,
            )
            for index in range(len(event_times))
        ]
    ).astype(np.float32)

    peak_step, peak_row, peak_col = np.unravel_index(
        int(np.argmax(efi_fields)), efi_fields.shape
    )
    if track_bbox:
        geo_bbox = track_bbox["geographic_bbox"]
        center_lat = float(track_bbox["latitude"])
        center_lon = float(track_bbox["longitude"])
        crop_bounds = (
            float(geo_bbox["latitude_min"]) - 0.8,
            float(geo_bbox["latitude_max"]) + 0.8,
            float(geo_bbox["longitude_west"]) - 0.8,
            float(geo_bbox["longitude_east"]) + 0.8,
        )
        track_id = track_bbox.get("track_id")
    else:
        center_lat = float(latitudes[peak_row])
        center_lon = float(longitudes[peak_col])
        crop_bounds = (center_lat - 1.5, center_lat + 1.5, center_lon - 1.5, center_lon + 1.5)
        track_id = None
    crop_rows = np.flatnonzero((latitudes >= crop_bounds[0]) & (latitudes <= crop_bounds[1]))
    crop_cols = np.flatnonzero(
        (longitudes >= crop_bounds[2]) & (longitudes <= crop_bounds[3])
    )
    if crop_rows.size < 8 or crop_cols.size < 8:
        row_start = max(0, peak_row - 7)
        row_end = min(len(latitudes), peak_row + 9)
        col_start = max(0, peak_col - 7)
        col_end = min(len(longitudes), peak_col + 9)
        crop_rows = np.arange(row_start, row_end)
        crop_cols = np.arange(col_start, col_end)
    crop_rows = _center_limit(crop_rows, len(latitudes), 48)
    crop_cols = _center_limit(crop_cols, len(longitudes), 48)
    crop_t2m = fields["t2m"][peak_step][np.ix_(crop_rows, crop_cols)]
    crop_efi = efi_fields[peak_step][np.ix_(crop_rows, crop_cols)]
    crop_latitudes = latitudes[crop_rows]
    crop_longitudes = longitudes[crop_cols]
    downscaled_latitudes = np.linspace(
        crop_latitudes[0] - np.diff(crop_latitudes)[0] / 4,
        crop_latitudes[-1] + np.diff(crop_latitudes)[0] / 4,
        len(crop_latitudes) * 2,
    )
    downscaled_longitudes = np.linspace(
        crop_longitudes[0] - np.diff(crop_longitudes)[0] / 4,
        crop_longitudes[-1] + np.diff(crop_longitudes)[0] / 4,
        len(crop_longitudes) * 2,
    )

    torch.manual_seed(20200516)
    unet_output = _run_normalized_unet(crop_t2m, epochs=1)
    diffusion_output, coarse_upscaled, _ = train_and_generate(
        crop_t2m[None, None, :, :].astype(np.float32),
        variable_index=0,
        training_epochs=2,
    )
    diffusion_field = diffusion_output.detach().cpu().numpy().squeeze()
    coarse_field = coarse_upscaled.detach().cpu().numpy().squeeze()
    physics_score = moisture_convergence_score_numpy(diffusion_field, coarse_field)
    physics = {
        "physics_score": physics_score,
        "physics_status": "PHYSICS_PROXY",
        "physics_method": "laplacian_smoothness_proxy",
        "mfc_available": False,
        "reason": "IMDAA inputs do not include specific humidity q; MFC was not calculated.",
    }

    lat_grid, lon_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
    alert = compute_alert(
        np.maximum(efi_fields[peak_step], 0.0),
        float(efi_fields[peak_step].max()),
        physics_score,
        latitude_grid=lat_grid,
        longitude_grid=lon_grid,
    )
    centroid_lat = alert["latitude"]
    centroid_lon = alert["longitude"]
    radius_km = None
    if track_bbox:
        box = track_bbox["geographic_bbox"]
        radius_km = max(
            _great_circle_distance_km(centroid_lat, centroid_lon, box["latitude_min"], box["longitude_west"]),
            _great_circle_distance_km(centroid_lat, centroid_lon, box["latitude_max"], box["longitude_east"]),
        )
    alert_payload = {
        "event": "Cyclone Amphan",
        "event_period": "2020-05-16/2020-05-21",
        "centroid": {"latitude": centroid_lat, "longitude": centroid_lon},
        "severity": alert["severity"].upper(),
        "severity_label": "prototype severity tiers",
        "radius_km": radius_km,
        "efi": float(alert["model_output"]["efi"]),
        "track_id": track_id,
        "track_available": track_bbox is not None,
        "data_source": "IMDAA",
        "mode": "HISTORICAL_REPLAY",
        "provenance": {
            "reference_source": "IMDAA May 2011–2019",
            "event_source": "IMDAA May 2020 reanalysis",
            "reference_excludes_event_year": True,
            "efi_variable": "t2m",
            "rainfall_status": event_meta["rainfall"],
        },
    }
    timeline = []
    for index, valid_time in enumerate(event_times):
        timeline.append(
            {
                "time": str(valid_time),
                "efi_max": float(efi_fields[index].max()),
                "efi_min": float(efi_fields[index].min()),
                "anomaly_coverage": float(np.mean(efi_fields[index] >= threshold)),
                "gnn_probability_max": float(predictions[index].max()),
                "active_nodes": int(np.count_nonzero(predictions[index] >= track_threshold)),
                "track_count": sum(
                    1
                    for track in tracks
                    if track["t_start"] <= index <= track["t_end"]
                ),
            }
        )
    anomaly_coverage = float(np.mean(efi_fields >= threshold))
    track_continuity = (
        float(len(track_bbox["track_points"]) / len(event_times)) if track_bbox else 0.0
    )
    model_diagnostic = {
        "label": "model-to-model diagnostic",
        "mae": float(np.mean(np.abs(unet_output - diffusion_field))),
        "rmse": float(np.sqrt(np.mean(np.square(unet_output - diffusion_field)))),
        "unet_peak": float(np.max(unet_output)),
        "diffusion_peak": float(np.max(diffusion_field)),
        "reference_target_available": False,
    }
    result = {
        "event": "Cyclone Amphan",
        "event_period": "2020-05-16/2020-05-21",
        "status": "READY",
        "data_source": "IMDAA",
        "mode": "HISTORICAL_REPLAY",
        "progress": [
            {"stage": "Data", "status": "complete"},
            {"stage": "EFI", "status": "complete"},
            {"stage": "GNN", "status": "complete", "supervision": "weak EFI-threshold labels"},
            {"stage": "Tracking", "status": "complete" if track_bbox else "no_track_above_threshold"},
            {"stage": "Downscaling", "status": "complete", "classification": "PROTOTYPE / PROXY TARGET"},
            {"stage": "Physics", "status": "complete", "classification": "PHYSICS_PROXY"},
            {"stage": "Alert", "status": "complete"},
        ],
        "alert": alert_payload,
        "timeline": timeline,
        "tracks": tracks,
        "dominant_track": track_bbox,
        "efi_threshold": threshold,
        "gnn_threshold": track_threshold,
        "mesh": {
            "type": "icosahedral",
            "vertices_in_region": int(len(vertices)),
            "subdivisions": int(settings.mesh_subdivisions),
        },
        "downscaling": {
            "status": "PROTOTYPE / PROXY TARGET",
            "label": "2x experimental downscaled field",
            "reference_target_available": False,
            "model_to_model_diagnostic": model_diagnostic,
            "crop": {
                "latitude_min": float(latitudes[crop_rows].min()),
                "latitude_max": float(latitudes[crop_rows].max()),
                "longitude_min": float(longitudes[crop_cols].min()),
                "longitude_max": float(longitudes[crop_cols].max()),
            },
        },
        "physics": physics,
        "verification": {
            "verification_status": "NOT_AVAILABLE",
            "reason": "No independent observational labels or genuine high-resolution reference target are available.",
            "stage1_descriptive_metrics": {
                "anomaly_coverage": anomaly_coverage,
                "track_continuity": track_continuity,
                "centroid_displacement": {
                    "status": "NOT_AVAILABLE",
                    "reason": "No independent cyclone-center reference track was configured.",
                },
            },
            "downscaling_metrics": {
                "status": "NOT_AVAILABLE",
                "reason": "No genuine approximately 5 km target dataset is available.",
            },
            "model_to_model_diagnostic": model_diagnostic,
        },
        "data_status": {
            "reference": {
                "status": "READY",
                "years": list(range(2011, 2020)),
                "samples": int(len(reference_times)),
                "contains_2020": False,
                "percentiles": percentile_levels.tolist(),
                "reference_source": "IMDAA May reanalysis",
            },
            "rainfall": event_meta["rainfall"],
            "rainfall_analysis_mode": rainfall_mode,
            "event": {
                "status": "READY",
                "sample_count": int(len(event_times)),
                "required_variable_counts": {
                    name: int(values.shape[0]) for name, values in fields.items()
                },
            },
        },
        "provenance": {
            "reference": reference_meta,
            "event": event_meta,
            "gnn_training": {
                "method": "self-supervised labels from deterministic t2m EFI threshold",
                "independent_ground_truth": False,
                "loss_history": loss_history,
            },
        },
    }
    analysis_path = cache_root / ANALYSIS_CACHE
    _save_npz(
        analysis_path,
        times=event_times,
        latitude=latitudes,
        longitude=longitudes,
        t2m=fields["t2m"],
        rainfall=fields["rainfall"],
        wind_u=fields["wind_u"],
        wind_v=fields["wind_v"],
        efi=efi_fields,
        gnn_probability=probability_fields,
        crop_times=event_times[peak_step],
        crop_latitude=crop_latitudes,
        crop_longitude=crop_longitudes,
        downscaled_latitude=downscaled_latitudes,
        downscaled_longitude=downscaled_longitudes,
        crop_t2m=crop_t2m,
        crop_efi=crop_efi,
        unet=unet_output.astype(np.float32),
        diffusion=diffusion_field.astype(np.float32),
    )
    _write_json(analysis_path.with_suffix(".npz.json"), {
        "data_source": "IMDAA",
        "mode": "HISTORICAL_REPLAY",
        "source_reference_years": list(range(2011, 2020)),
        "event_period": "2020-05-16/2020-05-21",
        "rainfall_semantics": "UNRESOLVED; excluded from EFI and downscaling",
        "generated_fields": ["t2m", "rainfall raw", "wind_u", "wind_v", "EFI", "GNN probability"],
    })
    _write_json(cache_root / RESULT_CACHE, result)
    return result


def _center_limit(indices: np.ndarray, dimension_size: int, maximum: int) -> np.ndarray:
    if indices.size <= maximum:
        return indices
    center = int(np.median(indices))
    start = max(0, min(dimension_size - maximum, center - maximum // 2))
    return np.arange(start, start + maximum)


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_cached_replay(
    cache_dir: str | Path,
    include_demo_summary: bool = True,
) -> dict[str, Any]:
    cache_root = Path(cache_dir)
    path = cache_root / RESULT_CACHE
    if not path.is_file():
        raise FileNotFoundError(
            "Prepared Amphan replay results are missing; run python scripts/build_imdaa_cache.py"
        )
    replay = _read_json(path)
    if include_demo_summary:
        replay["demo_summary"] = _demo_summary(replay, cache_root)
    return replay


def _visualization_subset(
    values: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    maximum_points: int,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    source_shape = [int(values.shape[-2]), int(values.shape[-1])]
    if not enabled:
        return values, latitude, longitude, {
            "visualization_downsampled": False,
            "source_shape": source_shape,
            "returned_shape": source_shape,
        }
    row_indices = np.unique(
        np.linspace(0, values.shape[-2] - 1, min(maximum_points, values.shape[-2]), dtype=int)
    )
    column_indices = np.unique(
        np.linspace(0, values.shape[-1] - 1, min(maximum_points, values.shape[-1]), dtype=int)
    )
    subset = values[np.ix_(row_indices, column_indices)]
    return subset, latitude[row_indices], longitude[column_indices], {
        "visualization_downsampled": subset.shape != values.shape,
        "visualization_only": True,
        "source_shape": source_shape,
        "returned_shape": [int(subset.shape[0]), int(subset.shape[1])],
        "max_points_per_axis": maximum_points,
    }


def load_cached_field(
    cache_dir: str | Path,
    variable: str,
    time_index: int,
    visualization: bool = False,
    max_points: int = 64,
) -> dict[str, Any]:
    allowed = {"t2m", "rainfall", "wind_u", "wind_v", "efi", "gnn_probability"}
    if variable not in allowed:
        raise ValueError(f"unsupported Amphan field {variable!r}; choose from {sorted(allowed)}")
    analysis_path = Path(cache_dir) / ANALYSIS_CACHE
    if not analysis_path.is_file():
        raise FileNotFoundError("Prepared Amphan field cache is missing; build the IMDAA cache first")
    replay = load_cached_replay(cache_dir, include_demo_summary=False)
    rainfall_mode = _rainfall_mode()
    if time_index < 0 or time_index >= len(replay["timeline"]):
        raise IndexError(
            f"time_index must be between 0 and {len(replay['timeline']) - 1}"
        )
    if variable == "rainfall" and rainfall_mode != "raw_unverified":
        return {
            "variable": variable,
            "status": "UNAVAILABLE",
            "reason": (
                "Rainfall analysis is disabled because APCP units and accumulation "
                "semantics are unresolved. Set "
                "IMDAA_RAINFALL_MODE=raw_unverified to inspect unmodified cached values."
            ),
            "units": None,
            "time_index": time_index,
            "valid_time": replay["timeline"][time_index]["time"],
            "latitude": [],
            "longitude": [],
            "values": [],
            "available": False,
            "provenance": replay["data_status"]["rainfall"],
        }
    with np.load(analysis_path, allow_pickle=False) as archive:
        times = np.asarray(archive["times"]).astype("datetime64[s]")
        if time_index < 0 or time_index >= len(times):
            raise IndexError(f"time_index must be between 0 and {len(times) - 1}")
        values = np.asarray(archive[variable][time_index], dtype=np.float32)
        latitude = np.asarray(archive["latitude"], dtype=np.float64)
        longitude = np.asarray(archive["longitude"], dtype=np.float64)
    values, latitude, longitude, visualization_metadata = _visualization_subset(
        values, latitude, longitude, max_points, visualization
    )
    status = "READY"
    if variable == "rainfall":
        status = "RAW_UNVERIFIED"
    return {
        "variable": variable,
        "status": status,
        "valid_time": str(times[time_index]),
        "units": {
            "t2m": "K",
            "rainfall": None,
            "wind_u": "m s**-1",
            "wind_v": "m s**-1",
            "efi": "dimensionless",
            "gnn_probability": "probability",
        }[variable],
        "latitude": latitude.tolist(),
        "longitude": longitude.tolist(),
        "values": values.tolist(),
        **visualization_metadata,
        "provenance": (
            replay["data_status"]["rainfall"]
            if variable == "rainfall"
            else replay["provenance"]["event"]["records"].get(str(times[time_index]), {})
        ),
    }


def load_cached_analysis_field(
    cache_dir: str | Path,
    variable: str,
    visualization: bool = False,
    max_points: int = 48,
) -> dict[str, Any]:
    if variable not in {"crop_t2m", "crop_efi", "unet", "diffusion"}:
        raise ValueError(f"unsupported Amphan analysis field {variable!r}")
    replay = load_cached_replay(cache_dir, include_demo_summary=False)
    with np.load(Path(cache_dir) / ANALYSIS_CACHE, allow_pickle=False) as archive:
        is_crop = variable in {"crop_t2m", "crop_efi"}
        raw_values = np.asarray(archive[variable], dtype=np.float32)
        crop_times = np.asarray(archive["crop_times"]).astype("datetime64[s]").reshape(-1)
        if not crop_times.size:
            raise ValueError("Amphan analysis cache contains no downscaling snapshot time")
        peak_step = max(
            range(min(len(replay["timeline"]), crop_times.size)),
            key=lambda index: replay["timeline"][index].get("efi_max", 0.0),
            default=0,
        )
        latitude = np.asarray(
            archive["crop_latitude"] if is_crop else archive["downscaled_latitude"],
            dtype=np.float64,
        )
        longitude = np.asarray(
            archive["crop_longitude"] if is_crop else archive["downscaled_longitude"],
            dtype=np.float64,
        )
        values, latitude, longitude, visualization_metadata = _visualization_subset(
            raw_values, latitude, longitude, max_points, visualization
        )
        return {
            "variable": variable,
            "status": "READY",
            "valid_time": str(crop_times[peak_step]),
            "units": "K" if variable in {"crop_t2m", "unet", "diffusion"} else "dimensionless",
            "latitude": latitude.tolist(),
            "longitude": longitude.tolist(),
            "values": values.tolist(),
            "statistics": _field_statistics(raw_values),
            **visualization_metadata,
            "classification": (
                "EXPERIMENTAL amplitude-preserving reconstruction; not verified against independent high-resolution data"
                if variable in {"unet", "diffusion"}
                else "IMDAA historical reanalysis crop"
            ),
        }


def load_cached_alert(cache_dir: str | Path, time_index: int) -> dict[str, Any]:
    replay = load_cached_replay(cache_dir, include_demo_summary=False)
    if time_index < 0 or time_index >= len(replay["timeline"]):
        raise IndexError(
            f"time_index must be between 0 and {len(replay['timeline']) - 1}"
        )
    with np.load(Path(cache_dir) / ANALYSIS_CACHE, allow_pickle=False) as archive:
        efi = np.asarray(archive["efi"][time_index], dtype=np.float32)
        latitudes = np.asarray(archive["latitude"], dtype=np.float64)
        longitudes = np.asarray(archive["longitude"], dtype=np.float64)
    latitude_grid, longitude_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
    computed = compute_alert(
        np.maximum(efi, 0.0),
        float(efi.max()),
        float(replay["physics"]["physics_score"]),
        latitude_grid=latitude_grid,
        longitude_grid=longitude_grid,
    )
    valid_time = replay["timeline"][time_index]["time"]
    active_points = [
        point
        for track in replay["tracks"]
        for point in track["track_points"]
        if point["valid_time"] == valid_time
    ]
    track_point = max(
        active_points,
        key=lambda point: point.get("confidence") or 0.0,
        default=None,
    )
    return {
        "event": replay["event"],
        "time": valid_time,
        "location": {
            "latitude": computed["latitude"],
            "longitude": computed["longitude"],
        },
        "severity": computed["severity"].upper(),
        "severity_label": "prototype severity tiers; uncalibrated",
        "efi": computed["model_output"]["efi"],
        "confidence": computed["confidence"],
        "confidence_status": "uncalibrated prototype score",
        "status": "HISTORICAL_REPLAY",
        "operational_warning": False,
        "track": track_point,
        "bounding_box": track_point.get("geographic_bbox") if track_point else None,
        "notice": "Not a live operational warning.",
    }


def load_cached_tracks(cache_dir: str | Path, time_index: int) -> dict[str, Any]:
    replay = load_cached_replay(cache_dir, include_demo_summary=False)
    if time_index < 0 or time_index >= len(replay["timeline"]):
        raise IndexError(
            f"time_index must be between 0 and {len(replay['timeline']) - 1}"
        )
    valid_time = replay["timeline"][time_index]["time"]
    active_tracks = []
    for track in replay["tracks"]:
        point = next(
            (item for item in track["track_points"] if item["valid_time"] == valid_time),
            None,
        )
        if point is not None:
            active_tracks.append({**track, "selected_track_point": point})
    dominant_track = max(
        active_tracks,
        key=lambda track: track["selected_track_point"].get("confidence") or 0.0,
        default=None,
    )
    return {
        "event": replay["event"],
        "time": valid_time,
        "active_tracks": active_tracks,
        "dominant_track": dominant_track,
    }
