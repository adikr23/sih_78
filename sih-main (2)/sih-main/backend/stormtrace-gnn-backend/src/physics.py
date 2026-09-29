"""
Physics-informed constraints (blueprint section 3.3 / 8): the model is
penalized, not hard-constrained, if it generates a physically implausible
field -- e.g. a sharp rainfall/temperature spike with no matching wind
convergence feeding it.

Two checks are implemented:
  1. moisture_convergence_penalty  -- a differentiable torch penalty used
     INSIDE the diffusion training loss (diffusion.py imports this).
  2. physics_consistency_score     -- a post-hoc numpy score (0..1, higher
     is better) used for the physics-check panel / API report, comparing the
     generated field's peak location against the ensemble's own agreement
     (low spread = ensemble agreed something unusual was there = trustworthy).

A field with a sharp local peak that has no support in the negative Laplacian
(i.e. no local convergence pattern) is penalized -- this is a lightweight
stand-in for real moisture-convergence-from-wind-divergence physics, as the
blueprint itself notes should start as ONE conservation check before adding
the full fluid-dynamics set.
"""
import numpy as np
import torch
import torch.nn.functional as F


EARTH_RADIUS_M = 6_371_000.0

_LAPLACIAN_KERNEL = torch.tensor(
    [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]
).view(1, 1, 3, 3)


def moisture_flux_convergence(
    specific_humidity: torch.Tensor,
    wind_u: torch.Tensor,
    wind_v: torch.Tensor,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    specific_humidity_units: str = "kg kg-1",
    wind_units: str = "m s-1",
) -> torch.Tensor:
    """Calculate spherical MFC, -div(qV), in s^-1 on a regular lat/lon grid."""
    if not (
        specific_humidity.shape == wind_u.shape == wind_v.shape
        and specific_humidity.ndim == 4
        and specific_humidity.shape[1] == 1
    ):
        raise ValueError("q, u, and v must share shape (batch, 1, latitude, longitude)")
    if specific_humidity.shape[-2] < 2 or specific_humidity.shape[-1] < 2:
        raise ValueError("MFC requires at least two latitude and longitude coordinates")
    q_units = specific_humidity_units.strip().lower().replace(" ", "")
    if q_units not in ("kg/kg", "kgkg-1", "kgkg^-1"):
        raise ValueError("specific humidity q must use kg kg-1 units")
    v_units = wind_units.strip().lower().replace(" ", "")
    if v_units not in ("m/s", "ms-1", "ms^-1", "ms**-1"):
        raise ValueError("horizontal wind u and v must use m s-1 units")
    if not (
        torch.isfinite(specific_humidity).all()
        and torch.isfinite(wind_u).all()
        and torch.isfinite(wind_v).all()
    ):
        raise ValueError("q, u, and v must contain only finite values")

    latitude = np.asarray(latitudes, dtype=np.float64)
    longitude = np.asarray(longitudes, dtype=np.float64)
    if latitude.shape != (specific_humidity.shape[-2],) or longitude.shape != (
        specific_humidity.shape[-1],
    ):
        raise ValueError("latitude and longitude coordinates must match the q/u/v grid")
    if (
        not np.isfinite(latitude).all()
        or not np.isfinite(longitude).all()
        or np.any((latitude < -90) | (latitude > 90))
    ):
        raise ValueError("MFC coordinates must be finite and latitude must be within [-90, 90]")
    if not (
        np.all(np.diff(latitude) > 0) or np.all(np.diff(latitude) < 0)
    ) or not (
        np.all(np.diff(longitude) > 0) or np.all(np.diff(longitude) < 0)
    ):
        raise ValueError("MFC latitude and longitude coordinates must be strictly monotonic")

    device = specific_humidity.device
    dtype = specific_humidity.dtype
    latitude_radians = torch.as_tensor(np.radians(latitude), dtype=dtype, device=device)
    longitude_radians = torch.as_tensor(np.radians(longitude), dtype=dtype, device=device)
    cos_latitude = torch.cos(latitude_radians).clamp(min=1e-6)
    q_u_cos = specific_humidity * wind_u * cos_latitude[None, None, :, None]
    q_v = specific_humidity * wind_v
    d_q_u_cos_dlon = torch.gradient(
        q_u_cos, spacing=(latitude_radians, longitude_radians), dim=(-2, -1), edge_order=1
    )[1]
    d_q_v_dlat = torch.gradient(
        q_v, spacing=(latitude_radians, longitude_radians), dim=(-2, -1), edge_order=1
    )[0]
    divergence = (
        d_q_u_cos_dlon / (EARTH_RADIUS_M * cos_latitude[None, None, :, None])
        + d_q_v_dlat / EARTH_RADIUS_M
    )
    return -divergence


def physics_constraint_diagnostic(variable_names=None, variable_units=None, has_geo_coordinates=False):
    names = set(variable_names or ())
    units = dict(zip(variable_names or (), variable_units or ()))
    q_name = next(iter(names.intersection({"q", "specific_humidity", "hus"})), None)
    u_name = next(iter(names.intersection({"u", "wind_u", "u10"})), None)
    v_name = next(iter(names.intersection({"v", "wind_v", "v10"})), None)
    q_present = q_name is not None
    u_present = u_name is not None
    v_present = v_name is not None
    missing = []
    if not q_present:
        missing.append("specific_humidity (q)")
    if not u_present:
        missing.append("zonal_wind (u)")
    if not v_present:
        missing.append("meridional_wind (v)")
    if not has_geo_coordinates:
        missing.append("latitude/longitude grid spacing")
    if q_present and not units.get(q_name):
        missing.append("specific humidity units")
    if (u_present or v_present) and not all(units.get(name) for name in (u_name, v_name) if name):
        missing.append("horizontal wind units")

    if (
        q_present
        and u_present
        and v_present
        and has_geo_coordinates
        and not missing
    ):
        return {
            "method": "moisture_flux_convergence",
            "available": True,
            "missing_variables": [],
            "units": "s-1",
            "is_hard_constraint": False,
        }
    return {
        "method": "laplacian_proxy",
        "available": False,
        "missing_variables": missing,
        "units": None,
        "is_hard_constraint": False,
        "reason": "MFC requires q, u, v and geographic grid coordinates; the loss uses the labeled proxy.",
    }


def moisture_convergence_penalty(
    generated_field: torch.Tensor,
    coarse_condition: torch.Tensor,
    specific_humidity: torch.Tensor = None,
    wind_u: torch.Tensor = None,
    wind_v: torch.Tensor = None,
    latitudes: np.ndarray = None,
    longitudes: np.ndarray = None,
    specific_humidity_units: str = "kg kg-1",
    wind_units: str = "m s-1",
) -> torch.Tensor:
    """
    generated_field, coarse_condition: (B,1,H,W)
    When q/u/v and coordinates are supplied, uses spherical moisture-flux
    convergence. Otherwise it retains the explicitly approximate Laplacian proxy.
    """
    if generated_field.shape != coarse_condition.shape or generated_field.ndim != 4:
        raise ValueError("generated_field and coarse_condition must share shape (batch, 1, H, W)")
    physical_inputs = (specific_humidity, wind_u, wind_v, latitudes, longitudes)
    if any(value is not None for value in physical_inputs):
        if any(value is None for value in physical_inputs):
            raise ValueError("MFC requires q, u, v, latitude, and longitude together")
        convergence = torch.relu(
            moisture_flux_convergence(
                specific_humidity,
                wind_u,
                wind_v,
                latitudes,
                longitudes,
                specific_humidity_units,
                wind_units,
            )
        )
    else:
        kernel = _LAPLACIAN_KERNEL.to(generated_field.dtype)
        lap_condition = F.conv2d(coarse_condition, kernel, padding=1)
        convergence = torch.relu(-lap_condition)

    convergence_norm = convergence / (convergence.amax(dim=(-1, -2), keepdim=True) + 1e-6)
    peak_norm = generated_field / (generated_field.amax(dim=(-1, -2), keepdim=True).abs() + 1e-6)
    unsupported_peak = torch.relu(peak_norm - convergence_norm)
    return unsupported_peak.mean()


def physics_consistency_score(generated_field: np.ndarray, ensemble_var_field: np.ndarray) -> float:
    """
    generated_field: (1,1,H,W) or (H,W) diffusion output
    ensemble_var_field: (n_members, grid, grid) raw ensemble for the same variable
    Returns 0..1, higher = more physically consistent with ensemble agreement at the peak.
    """
    from scipy.ndimage import zoom

    generated = np.asarray(generated_field).squeeze()
    ensemble_spread = ensemble_var_field.std(axis=0)

    scale = generated.shape[0] / ensemble_spread.shape[0]
    spread_resized = zoom(ensemble_spread, scale)

    peak_idx = np.unravel_index(generated.argmax(), generated.shape)
    agreement_at_peak = 1 - min(spread_resized[peak_idx] / (spread_resized.max() + 1e-6), 1.0)
    return float(np.clip(agreement_at_peak, 0.0, 1.0))


def moisture_convergence_score_numpy(generated_field: np.ndarray, coarse_condition: np.ndarray) -> float:
    """Numpy mirror of the torch training penalty, for the post-hoc physics-check panel."""
    with torch.no_grad():
        g = torch.tensor(np.asarray(generated_field), dtype=torch.float32).reshape(1, 1, *generated_field.shape[-2:])
        c = torch.tensor(np.asarray(coarse_condition), dtype=torch.float32).reshape(1, 1, *coarse_condition.shape[-2:])
        penalty = moisture_convergence_penalty(g, c).item()
    return float(np.clip(1.0 - penalty, 0.0, 1.0))  # convert penalty -> score, higher is better
