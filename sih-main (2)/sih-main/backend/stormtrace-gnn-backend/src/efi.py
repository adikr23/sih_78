"""
Extreme Forecast Index (EFI): how unusual is the forecast at each mesh point
compared to that SAME point's 30-year climatological distribution.

Fix vs a naive version: EFI must be computed per-point against that point's
own baseline distribution, not against a single grid-wide scalar (averaging
away the baseline's spatial distribution would silently break the metric,
since a hotspot could easily look "normal" against a flattened baseline).

Vectorized: no python loop over mesh points. For P points and Y baseline
years this is an (P, Y) broadcasted comparison done in one numpy call.
"""
import numpy as np

from src.mesh import grid_to_mesh_multivar


def compute_efi_field(forecast_mesh_values: np.ndarray, baseline_mesh_values: np.ndarray) -> np.ndarray:
    """
    forecast_mesh_values: (n_points,) forecast at each mesh point, single variable
    baseline_mesh_values: (n_reference_samples, n_points) climatological reference values
        at each mesh point for the same variable and comparable valid-time context
    returns: (n_points,) empirical percentile anomaly score in [-1, 1]

    This percentile mapping is an EFI-like prototype, not the operational ECMWF EFI.
    The supplied reference samples must represent the variable, location, season, and
    forecast/valid-time context being evaluated.
    """
    forecast_mesh_values = np.asarray(forecast_mesh_values)
    baseline_mesh_values = np.asarray(baseline_mesh_values)
    if forecast_mesh_values.ndim != 1:
        raise ValueError("forecast_mesh_values must have shape (n_points,)")
    if baseline_mesh_values.ndim != 2 or baseline_mesh_values.shape[1] != forecast_mesh_values.shape[0]:
        raise ValueError("baseline_mesh_values must have shape (n_reference_samples, n_points)")
    if baseline_mesh_values.shape[0] == 0:
        raise ValueError("climatological reference distribution has no samples")
    if not np.isfinite(forecast_mesh_values).all() or not np.isfinite(baseline_mesh_values).all():
        raise ValueError("forecast and climatological reference values must be finite")

    below = (baseline_mesh_values < forecast_mesh_values[None, :]).mean(axis=0)
    tied = (baseline_mesh_values == forecast_mesh_values[None, :]).mean(axis=0)
    percentile = below + 0.5 * tied
    return 2 * percentile - 1


def compute_efi_multivar(
    forecast_mesh: np.ndarray,
    baseline_years_grid: np.ndarray,
    vertices: np.ndarray,
    latitudes: np.ndarray = None,
    longitudes: np.ndarray = None,
) -> dict:
    """
    forecast_mesh: (n_points, n_vars) forecast field already projected to the mesh
    baseline_years_grid: (n_reference_samples, n_vars, grid, grid) reference distribution
    vertices: spherical mesh vertices used to project reference samples onto the mesh
    latitudes/longitudes: optional one-dimensional grid coordinates in degrees

    Returns dict variable_name -> efi array (n_points,), plus a combined "max_efi"
    field taking the element-wise max across variables (an anomaly can dominate
    via temperature OR rainfall OR wind). The percentile score remains approximate;
    callers should provide a temporally matched climatology, such as ERA5, when available.
    """
    from config import settings

    forecast_mesh = np.asarray(forecast_mesh)
    baseline_years_grid = np.asarray(baseline_years_grid)
    if forecast_mesh.ndim != 2:
        raise ValueError("forecast_mesh must have shape (n_points, n_vars)")
    if baseline_years_grid.ndim != 4:
        raise ValueError("baseline_years_grid must have shape (n_reference_samples, n_vars, grid, grid)")
    if baseline_years_grid.shape[0] == 0:
        raise ValueError("climatological reference distribution has no samples")
    if baseline_years_grid.shape[1] != forecast_mesh.shape[1]:
        raise ValueError("forecast and reference variable dimensions must match")
    if forecast_mesh.shape[0] != np.asarray(vertices).shape[0]:
        raise ValueError("forecast_mesh must contain one row per mesh vertex")
    n_years = baseline_years_grid.shape[0]
    n_vars = baseline_years_grid.shape[1]
    n_points = vertices.shape[0]
    if n_vars > len(settings.variable_names):
        raise ValueError("reference has more variables than configured variable_names")

    baseline_mesh = np.zeros((n_years, n_vars, n_points), dtype=np.float32)
    for y in range(n_years):
        baseline_mesh[y] = grid_to_mesh_multivar(
            baseline_years_grid[y], vertices, latitudes, longitudes
        ).T

    efi_per_var = {}
    stacked = np.zeros((n_vars, n_points), dtype=np.float32)
    for v in range(n_vars):
        efi_v = compute_efi_field(forecast_mesh[:, v], baseline_mesh[:, v, :])
        name = settings.variable_names[v]
        efi_per_var[name] = efi_v
        stacked[v] = efi_v

    efi_per_var["combined_max"] = stacked.max(axis=0)
    return efi_per_var
