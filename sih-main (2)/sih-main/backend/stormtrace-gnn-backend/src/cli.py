from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

from config import settings
from src.db import init_db


REQUIRED_PACKAGES = ("fastapi", "numpy", "xarray", "zarr", "torch", "sqlalchemy", "pydantic_settings")
CANONICAL_SOURCE_ALIASES = {
    "t2m": {"t2m", "2m_temperature", "temperature_2m"},
    "rainfall": {"rainfall", "tp", "total_precipitation", "precipitation"},
    "wind_u": {"wind_u", "u10", "10m_u_component_of_wind", "u"},
    "wind_v": {"wind_v", "v10", "10m_v_component_of_wind", "v"},
}


def _result(status: str, message: str) -> Dict[str, str]:
    return {"status": status, "message": message}


def _check_python() -> Dict[str, str]:
    if sys.version_info < (3, 10):
        return _result("INVALID", f"Python {sys.version_info.major}.{sys.version_info.minor} is unsupported; Python 3.10+ is required")
    return _result("READY", f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")


def _check_dependencies() -> Dict[str, str]:
    missing: List[str] = []
    for package in REQUIRED_PACKAGES:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    if missing:
        return _result("INVALID", "Missing required packages: " + ", ".join(missing))
    optional_missing = []
    for package in ("cdsapi", "cfgrib", "redis"):
        try:
            __import__(package)
        except ImportError:
            optional_missing.append(package)
    note = (
        "Optional real-data packages unavailable: " + ", ".join(optional_missing)
        if optional_missing
        else "Required and optional ingestion packages are importable"
    )
    return _result("READY", note)


def _check_mode() -> Dict[str, str]:
    mode = settings.data_mode.strip().upper()
    if mode not in {"REAL", "LOCAL_REAL", "SYNTHETIC_DEMO"}:
        return _result("INVALID", "DATA_MODE must be REAL, LOCAL_REAL, or SYNTHETIC_DEMO")
    if mode == "SYNTHETIC_DEMO":
        return _result("DEMO ONLY", "Synthetic data is enabled and must not be used as real-data evidence")
    return _result("READY", f"DATA_MODE={mode}")


def _is_local_dataset_readable(path_value: str, kind: str) -> tuple[bool, str]:
    path = Path(path_value).expanduser()
    if not path.exists():
        return False, f"Configured {kind} path does not exist"
    forecast = kind in {"NEPS-G", "ERA5 forecast"}
    if path.suffix.lower() == ".npz":
        try:
            import numpy as np

            with np.load(path, allow_pickle=False) as archive:
                array_keys = ("ensemble", "forecast") if forecast else ("climatology", "baseline")
                array_key = next((key for key in array_keys if key in archive), None)
                if array_key is None:
                    return False, f"{kind} NPZ is missing the expected {array_keys} array"
                if "latitude" not in archive or "longitude" not in archive:
                    return False, f"{kind} NPZ is missing latitude/longitude coordinates"
                if forecast and (archive[array_key].ndim != 5 or "ensemble_members" not in archive):
                    return False, "NEPS-G forecast NPZ must include a 5D ensemble and ensemble_members"
                expected_dimensions = 5 if forecast else 4
                if archive[array_key].ndim != expected_dimensions:
                    return False, f"{kind} array must have {expected_dimensions} dimensions"
            return True, "local dataset container can be read"
        except Exception as exc:
            return False, f"{kind} dataset cannot be read: {exc}"
    try:
        if kind == "NEPS-G" and path.suffix.lower() in {".grib", ".grb", ".grib2", ".grb2"}:
            import cfgrib

            groups = cfgrib.open_datasets(
                str(path),
                backend_kwargs={"indexpath": ""},
            )
            if not groups:
                return False, "NEPS-G GRIB contains no readable cfgrib groups"
            try:
                required_variables = {"t2m", "tp", "u10", "v10"}
                available_variables = set().union(*(set(group.data_vars) for group in groups))
                missing = required_variables - available_variables
                if missing:
                    return False, (
                        "NEPS-G TIGGE GRIB is missing required variables: "
                        + ", ".join(sorted(missing))
                    )
                for group in groups:
                    variables = required_variables & set(group.data_vars)
                    if not variables:
                        continue
                    if not ({"latitude", "lat"} & set(group.coords)) or not (
                        {"longitude", "lon"} & set(group.coords)
                    ):
                        return False, "NEPS-G TIGGE GRIB group is missing geographic coordinates"
                    if "step" not in group.dims or "number" not in group.dims:
                        return False, "NEPS-G TIGGE GRIB requires step and number ensemble dimensions"
                return True, "NCMRWF TIGGE GRIB groups expose all required fields and ensemble dimensions"
            finally:
                for group in groups:
                    group.close()

        import xarray as xr

        if path.suffix.lower() == ".zarr":
            dataset = xr.open_zarr(path)
        else:
            open_options = {"engine": "cfgrib"} if path.suffix.lower() in {".grib", ".grb", ".grib2", ".grb2"} else {}
            dataset = xr.open_dataset(path, **open_options)
        try:
            coords = set(dataset.coords)
            if not ({"latitude", "lat"} & coords) or not ({"longitude", "lon"} & coords):
                return False, f"{kind} dataset must expose one-dimensional latitude and longitude coordinates"
            if not dataset.data_vars:
                return False, f"{kind} dataset contains no data variables"
            available_variables = set(dataset.data_vars)
            configured_sources = (
                [value.strip() for value in settings.data_variables.split(",") if value.strip()]
                if settings.data_variables
                else None
            )
            required_variables = (
                set(configured_sources)
                if configured_sources
                else set.union(*CANONICAL_SOURCE_ALIASES.values())
            )
            if configured_sources:
                missing = required_variables - available_variables
            else:
                missing = {
                    canonical
                    for canonical, aliases in CANONICAL_SOURCE_ALIASES.items()
                    if not (aliases & available_variables)
                }
            if missing:
                return False, f"{kind} dataset is missing required variables: {', '.join(sorted(missing))}"
            dimensions = set(dataset.dims)
            sample_dimensions = {"step", "forecast_step", "lead_time", "time", "valid_time"}
            if not dimensions & sample_dimensions:
                return False, f"{kind} dataset is missing a forecast/time dimension"
            if kind == "NEPS-G" and not dimensions & {"number", "ensemble_member", "member", "realization"}:
                return False, "NEPS-G dataset is missing an ensemble-member dimension"
        finally:
            dataset.close()
        return True, "local dataset opens and exposes geographic coordinates"
    except Exception as exc:
        return False, f"{kind} dataset cannot be opened: {exc}"


def _check_cds() -> Dict[str, str]:
    from src.era5_cds import cds_configured

    configured = cds_configured()
    if not configured:
        return _result("NOT CONFIGURED", "CDS_URL, CDS_UID, and CDS_API_KEY are required for live downloads")
    try:
        from src.era5_cds import validate_cds_url, validate_area, validate_variables

        validate_cds_url(settings.cds_url)
        validate_area([float(value.strip()) for value in settings.era5_download_area.split(",")])
        validate_variables(settings.era5_variables.split(","), settings.era5_dataset)
    except (TypeError, ValueError) as exc:
        return _result("INVALID", f"CDS/ERA5 configuration is invalid: {exc}")
    try:
        import cdsapi  # noqa: F401
    except ImportError:
        return _result("CONFIGURED BUT DATA UNAVAILABLE", "CDS credentials configured but cdsapi is not installed")
    return _result("CONFIGURED BUT UNVERIFIED", "CDS configuration passed local checks; authentication has not been tested")


def _check_era5() -> Dict[str, str]:
    configured_paths = [settings.era5_climatology_path, settings.era5_forecast_path]
    for index, path in enumerate(configured_paths):
        if path:
            readable, detail = _is_local_dataset_readable(
                path,
                "ERA5 forecast" if index == 1 else "ERA5",
            )
            if not readable:
                return _result("INVALID", detail)
    if settings.era5_climatology_path:
        return _result("READY", "Local ERA5 climatology opens successfully")
    from src.era5_cds import cds_configured
    if settings.data_mode.strip().upper() == "REAL" and cds_configured():
        if not (settings.era5_start_date and settings.era5_end_date):
            return _result(
                "CONFIGURED BUT DATA UNAVAILABLE",
                "ERA5_START_DATE and ERA5_END_DATE are required before requesting a reference period",
            )
        return _result("CONFIGURED BUT UNVERIFIED", "CDS credentials are present; no ERA5 data has been downloaded or validated")
    if settings.data_mode.strip().upper() == "LOCAL_REAL":
        return _result("CONFIGURED BUT DATA UNAVAILABLE", "LOCAL_REAL requires ERA5_CLIMATOLOGY_PATH")
    return _result("NOT CONFIGURED", "Configure ERA5_CLIMATOLOGY_PATH or CDS credentials and a reference period")


def _check_neps_g() -> Dict[str, str]:
    if not settings.neps_g_forecast_path:
        if settings.neps_g_active or settings.data_mode.strip().upper() in {"REAL", "LOCAL_REAL"}:
            return _result("CONFIGURED BUT DATA UNAVAILABLE", "NEPS-G: CONFIGURED BUT DATA UNAVAILABLE; set NEPS_G_FORECAST_PATH to an official local file")
        return _result("NOT CONFIGURED", "No local official NEPS-G forecast file is configured")
    readable, detail = _is_local_dataset_readable(settings.neps_g_forecast_path, "NEPS-G")
    if not readable:
        return _result("INVALID", detail)
    return _result("READY", "Configured NEPS-G file opens and exposes geographic coordinates")


def _check_cache() -> Dict[str, str]:
    try:
        path = Path(settings.era5_cache_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".system-check-write-test"
        probe.write_text("ok", encoding="ascii")
        probe.unlink()
        return _result("READY", f"ERA5 cache is writable: {path}")
    except OSError:
        return _result("INVALID", "ERA5 cache directory is not writable")


def _check_high_resolution_target() -> Dict[str, str]:
    if not settings.high_resolution_target_path:
        return _result("NOT CONFIGURED", "No high-resolution reference dataset is configured")
    path = Path(settings.high_resolution_target_path).expanduser()
    if not path.exists():
        return _result("INVALID", "Configured high-resolution reference path does not exist")
    return _result(
        "CONFIGURED BUT UNVERIFIED",
        "A target path exists, but its genuine approximately 5 km resolution and validation suitability are not verified",
    )


def _check_database() -> Dict[str, str]:
    try:
        init_db()
        return _result("READY", "Database initialization succeeded")
    except Exception as exc:
        return _result("INVALID", f"Database initialization failed: {exc}")


def _check_job_store() -> Dict[str, str]:
    if settings.job_store_backend.strip().lower() != "redis":
        return _result("DEMO ONLY", "Using process-local job state; not suitable for multi-worker operation")
    try:
        import redis

        redis.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
        ).ping()
        return _result("READY", "Configured Redis job store is reachable")
    except Exception:
        return _result("CONFIGURED BUT DATA UNAVAILABLE", "Redis job store is not reachable; endpoint and credentials were not emitted")


def _check_frontend_config() -> Dict[str, str]:
    value = os.getenv("VITE_API_BASE_URL")
    return (
        _result("READY", "VITE_API_BASE_URL is set")
        if value
        else _result("NOT CONFIGURED", "VITE_API_BASE_URL is not set")
    )


def run_system_check() -> Dict[str, Dict[str, str]]:
    return {
        "DATA MODE": _check_mode(),
        "Python": _check_python(),
        "required Python dependencies": _check_dependencies(),
        "CDS credentials/configuration": _check_cds(),
        "ERA5 local data": _check_era5(),
        "ERA5 cache": _check_cache(),
        "NEPS-G configuration/availability": _check_neps_g(),
        "high-resolution target": _check_high_resolution_target(),
        "database": _check_database(),
        "Redis/job store": _check_job_store(),
        "frontend API configuration": _check_frontend_config(),
    }


def _cli():
    parser = argparse.ArgumentParser(description="STORMTRACE-GNN system check")
    parser.add_argument("command", choices=["system-check"], nargs="?", default="system-check")
    args = parser.parse_args()
    if args.command == "system-check":
        report = run_system_check()
        for name, result in report.items():
            print(f"{name}: {result['status']} - {result['message']}")
        if any(result["status"] == "INVALID" for result in report.values()):
            raise SystemExit(1)


if __name__ == "__main__":
    _cli()
