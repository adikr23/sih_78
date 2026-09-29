"""
Pinpoint centroid + severity tier, per blueprint section 3 final stage.

Severity is a derived, uncalibrated prototype score. Peak amplitude is used only
when an explicit variable-specific reference range is supplied; within-field
min/max normalization is not evidence of meteorological severity.
"""
import numpy as np

from config import settings


def classify_demo_severity(score: float) -> str:
    """Project-only tier for uncalibrated anomalous-area coverage."""
    if not np.isfinite(score) or score < 0.0 or score > 1.0:
        raise ValueError("demo anomaly coverage must be finite and within [0, 1]")
    if score < 0.1:
        return "NORMAL"
    if score < 0.2:
        return "WATCH"
    if score < 0.35:
        return "HIGH"
    return "EXTREME"


def compute_alert(
    generated_field: np.ndarray,
    efi_field_at_mesh_peak: float,
    physics_score: float,
    amplitude_reference: tuple = None,
    latitude_grid: np.ndarray = None,
    longitude_grid: np.ndarray = None,
) -> dict:
    field = np.asarray(generated_field).squeeze()
    if field.ndim != 2 or field.size == 0 or not np.isfinite(field).all():
        raise ValueError("generated_field must be a finite two-dimensional field")
    if not np.isfinite(efi_field_at_mesh_peak) or not np.isfinite(physics_score):
        raise ValueError("EFI and physics scores must be finite")
    peak_idx = np.unravel_index(field.argmax(), field.shape)
    peak_val = float(field.max())
    normalized_amp = None
    if amplitude_reference is not None:
        if len(amplitude_reference) != 2:
            raise ValueError("amplitude_reference must contain a lower and upper bound")
        lower, upper = map(float, amplitude_reference)
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError("amplitude reference bounds must be finite and strictly increasing")
        normalized_amp = float(np.clip((peak_val - lower) / (upper - lower), 0, 1))

    efi_norm = float(np.clip((efi_field_at_mesh_peak + 1) / 2, 0, 1))  # EFI in [-1,1] -> [0,1]
    physics_norm = float(np.clip(physics_score, 0, 1))

    raw_score = 0.35 * efi_norm + 0.10 * physics_norm
    if normalized_amp is not None:
        raw_score += 0.55 * normalized_amp
    confidence = float(np.clip(0.5 * raw_score + 0.5 * physics_norm, 0, 1))

    if raw_score < settings.severity_low_thresh:
        severity = "low"
    elif raw_score < settings.severity_high_thresh:
        severity = "moderate"
    else:
        severity = "severe"

    # a severe call with low physics confidence is down-graded one tier, so the
    # alert API never emits "severe" purely off an unsupported generative peak
    if severity == "severe" and physics_norm < 0.35:
        severity = "moderate"

    centroid_latitude = None
    centroid_longitude = None
    if latitude_grid is not None and longitude_grid is not None:
        latitude_grid = np.asarray(latitude_grid)
        longitude_grid = np.asarray(longitude_grid)
        if latitude_grid.shape == field.shape and longitude_grid.shape == field.shape:
            centroid_latitude = float(latitude_grid[peak_idx])
            centroid_longitude = float(longitude_grid[peak_idx])

    return {
        "centroid_row": int(peak_idx[0]),
        "centroid_col": int(peak_idx[1]),
        "latitude": centroid_latitude,
        "longitude": centroid_longitude,
        "peak_value": peak_val,
        "severity": severity,
        "derived_severity": severity,
        "model_output": {
            "peak_value": peak_val,
            "efi": float(efi_field_at_mesh_peak),
            "physics_score": physics_norm,
        },
        "confidence": round(confidence, 3),
        "radius_km": None,
        "raw_score": round(float(raw_score), 3),
        "normalized_amplitude": normalized_amp,
        "amplitude_available": normalized_amp is not None,
        "alert_threshold": {
            "low": settings.severity_low_thresh,
            "high": settings.severity_high_thresh,
        },
        "calibration_status": "uncalibrated",
    }
