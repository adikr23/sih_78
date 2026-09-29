from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config import settings

try:
    import cdsapi
except ImportError:  # pragma: no cover - optional in demo/local-only installs
    cdsapi = None


_ERA5_SINGLE_LEVEL_VARIABLES = {
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
    "surface_pressure",
    "mean_sea_level_pressure",
}
_CDS_TARGET_DIR = Path("C:/stormtrace_cds_cache")


def cds_configured() -> bool:
    return _resolve_cds_credentials() is not None


def _read_cdsapi_config() -> Dict[str, str]:
    config_path = Path.home() / ".cdsapirc"
    if not config_path.is_file():
        return {}
    values: Dict[str, str] = {}
    try:
        for line in config_path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() in {"url", "key"}:
                values[key.strip()] = value.strip().strip("\"'")
    except OSError:
        return {}
    return values


def _resolve_cds_credentials() -> Optional[Tuple[str, str]]:
    if settings.cds_uid and settings.cds_api_key:
        return settings.cds_url, f"{settings.cds_uid}:{settings.cds_api_key}"
    local_config = _read_cdsapi_config()
    api_key = local_config.get("key")
    if not api_key:
        return None
    configured_url = os.getenv("CDS_URL") or local_config.get("url") or settings.cds_url
    return configured_url, api_key


def validate_cds_url(url: str) -> str:
    if not isinstance(url, str) or not re.fullmatch(r"https://[^/\s]+(?:/[^/\s]*)*", url.strip()):
        raise ValueError("CDS_URL must be an HTTPS URL")
    return url.strip().rstrip("/")


def validate_area(area: Sequence[float]) -> Tuple[float, float, float, float]:
    try:
        values = tuple(float(value) for value in area)
    except (TypeError, ValueError) as exc:
        raise ValueError("ERA5 area must contain numeric north,west,south,east bounds") from exc
    if len(values) != 4:
        raise ValueError("ERA5 area must contain north,west,south,east bounds")
    north, west, south, east = values
    if not (-90 <= south < north <= 90):
        raise ValueError("ERA5 latitude bounds must satisfy -90 <= south < north <= 90")
    if not (-180 <= west < east <= 360):
        raise ValueError("ERA5 longitude bounds must satisfy -180 <= west < east <= 360")
    return values


def validate_variables(variables: Sequence[str], dataset: str) -> List[str]:
    if dataset != "reanalysis-era5-single-levels":
        raise ValueError(
            f"Invalid or unsupported ERA5_DATASET {dataset!r}; "
            "this provider currently supports reanalysis-era5-single-levels"
        )
    names = [str(value).strip() for value in variables if str(value).strip()]
    if not names:
        raise ValueError("ERA5_VARIABLES must contain at least one variable")
    if len(set(names)) != len(names):
        raise ValueError("ERA5_VARIABLES must not contain duplicate variables")
    if dataset == "reanalysis-era5-single-levels":
        invalid = sorted(set(names) - _ERA5_SINGLE_LEVEL_VARIABLES)
        if invalid:
            raise ValueError(
                "invalid variables for reanalysis-era5-single-levels: " + ", ".join(invalid)
            )
    return names


def validate_date_range(start_date: str, end_date: str) -> Tuple[date, date]:
    if not all(
        isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)
        for value in (start_date, end_date)
    ):
        raise ValueError("ERA5 dates must use YYYY-MM-DD format")
    try:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
    except (TypeError, ValueError) as exc:
        raise ValueError("ERA5 dates must use YYYY-MM-DD format") from exc
    if start > end:
        raise ValueError("ERA5 start date must be on or before end date")
    if end > datetime.now(timezone.utc).date():
        raise ValueError("ERA5 end date cannot be in the future")
    return start, end


def validate_times(times: Sequence[str]) -> List[str]:
    normalized = []
    for value in times:
        text = str(value).strip()
        if not re.fullmatch(r"\d{2}:\d{2}", text):
            raise ValueError(f"invalid ERA5 time {text!r}; expected HH:MM")
        try:
            parsed = datetime.strptime(text, "%H:%M")
        except ValueError as exc:
            raise ValueError(f"invalid ERA5 time {text!r}; expected HH:MM") from exc
        if parsed.minute not in (0, 30):
            raise ValueError("ERA5 time minute must be 00 or 30")
        normalized.append(parsed.strftime("%H:%M"))
    if not normalized:
        raise ValueError("at least one ERA5 time is required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("ERA5 times must not contain duplicates")
    return normalized


def _cache_path_for_request(request_key: str, suffix: str = ".nc") -> Path:
    cache_dir = Path(settings.era5_cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(request_key.encode("utf-8")).hexdigest()
    return cache_dir / f"era5_{digest[:24]}{suffix}"


class Era5CDSDownloader:
    def __init__(self, client=None):
        credentials = _resolve_cds_credentials()
        if credentials is None:
            raise RuntimeError("CDS credentials are missing; set CDS_UID/CDS_API_KEY or configure the backend user's .cdsapirc")
        if cdsapi is None and client is None:
            raise RuntimeError("cdsapi is not installed; install the backend requirements to enable ERA5 downloads")
        url, key = credentials
        url = validate_cds_url(url)
        self.client = client or cdsapi.Client(
            url=url,
            key=key,
            timeout=settings.cds_timeout_seconds,
            quiet=True,
        )

    def _build_request(
        self,
        start_date: str,
        end_date: str,
        variables: Optional[Sequence[str]] = None,
        times: Optional[Sequence[str]] = None,
        area: Optional[Sequence[float]] = None,
        resolution: Optional[str] = None,
    ) -> Dict[str, Any]:
        start, end = validate_date_range(start_date, end_date)
        selected_variables = validate_variables(
            variables or [v.strip() for v in settings.era5_variables.split(",")],
            settings.era5_dataset,
        )
        selected_times = validate_times(times or [f"{hour:02d}:00" for hour in range(24)])
        try:
            configured_area = (
                area
                if area is not None
                else [float(v.strip()) for v in settings.era5_download_area.split(",")]
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                "ERA5_DOWNLOAD_AREA must contain north,west,south,east numeric bounds"
            ) from exc
        selected_area = validate_area(configured_area)
        selected_resolution = resolution or settings.era5_resolution
        try:
            lat_step, lon_step = (float(value.strip()) for value in selected_resolution.split("/"))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("ERA5_RESOLUTION must be formatted as latitude_step/longitude_step") from exc
        if not (0 < lat_step <= 1 and 0 < lon_step <= 1):
            raise ValueError("ERA5_RESOLUTION grid steps must be greater than 0 and no greater than 1 degree")
        request: Dict[str, Any] = {
            "product_type": ["reanalysis"],
            "variable": selected_variables,
            "year": [f"{start.year:04d}"],
            "month": [f"{start.month:02d}"],
            "day": [f"{day.day:02d}" for day in _dates(start, end)],
            "time": selected_times,
            "area": list(selected_area),
            "data_format": "netcdf",
            "download_format": "unarchived",
            "grid": [lat_step, lon_step],
        }
        return request

    def _build_monthly_requests(
        self,
        start_date: str,
        end_date: str,
        variables: Optional[Sequence[str]],
        times: Optional[Sequence[str]],
        area: Optional[Sequence[float]],
        resolution: Optional[str],
    ) -> List[Dict[str, Any]]:
        start, end = validate_date_range(start_date, end_date)
        month_start = date(start.year, start.month, 1)
        requests = []
        while month_start <= end:
            next_month = date(
                month_start.year + (month_start.month == 12),
                1 if month_start.month == 12 else month_start.month + 1,
                1,
            )
            period_start = max(start, month_start)
            period_end = min(end, next_month.fromordinal(next_month.toordinal() - 1))
            requests.append(
                self._build_request(
                    period_start.isoformat(),
                    period_end.isoformat(),
                    variables,
                    times,
                    area,
                    resolution,
                )
            )
            month_start = next_month
        return requests

    def download(
        self,
        start_date: str,
        end_date: str,
        variables: Optional[Sequence[str]] = None,
        times: Optional[Sequence[str]] = None,
        area: Optional[Sequence[float]] = None,
        resolution: Optional[str] = None,
    ) -> Path:
        requests = self._build_monthly_requests(start_date, end_date, variables, times, area, resolution)
        key = json.dumps({"dataset": settings.era5_dataset, "requests": requests}, sort_keys=True)
        cache_path = _cache_path_for_request(key)
        metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
        if cache_path.is_file() and cache_path.stat().st_size > 0:
            try:
                cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cached_metadata = None
            if (
                isinstance(cached_metadata, dict)
                and cached_metadata.get("dataset") == settings.era5_dataset
                and cached_metadata.get("date_range") == {"start": start_date, "end": end_date}
                and cached_metadata.get("variables") == requests[0]["variable"]
            ):
                return cache_path

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        _CDS_TARGET_DIR.mkdir(parents=True, exist_ok=True)
        temporary_files = []
        try:
            for index, request in enumerate(requests):
                temporary_path = _CDS_TARGET_DIR / f"{cache_path.stem}.part{index:03d}{cache_path.suffix}"
                temporary_files.append(temporary_path)
                self.client.retrieve(
                    settings.era5_dataset,
                    request,
                    temporary_path.resolve().as_posix(),
                )
                if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
                    raise RuntimeError("CDS request completed without producing a non-empty data file")
            if len(temporary_files) == 1:
                shutil.move(str(temporary_files[0]), str(cache_path))
            else:
                try:
                    import xarray as xr
                except ImportError as exc:
                    raise RuntimeError("Combining monthly ERA5 downloads requires xarray") from exc
                opened = [xr.open_dataset(path) for path in temporary_files]
                try:
                    combined = xr.combine_by_coords(opened)
                    combined.to_netcdf(cache_path)
                    combined.close()
                finally:
                    for dataset in opened:
                        dataset.close()
        except Exception as exc:
            for temporary_path in temporary_files:
                if temporary_path.exists():
                    temporary_path.unlink()
            message = str(exc)
            lowered = message.lower()
            if "licence" in lowered or "license" in lowered:
                raise RuntimeError(
                    "CDS access is blocked because the required ERA5 dataset licence has not been accepted; "
                    "accept the dataset licence in the CDS account and retry"
                ) from exc
            if any(token in lowered for token in ("401", "403", "unauthorized", "forbidden", "authentication", "invalid key")):
                raise RuntimeError("CDS authentication failed; verify CDS_UID, CDS_API_KEY, and account access") from exc
            if any(token in lowered for token in ("dataset", "not found", "unknown dataset")):
                raise RuntimeError(f"Invalid or unavailable CDS dataset {settings.era5_dataset!r}: {message}") from exc
            if isinstance(exc, RuntimeError) and "without producing" in message:
                raise
            raise RuntimeError(f"ERA5/CDS download failed: {message}") from exc

        self._write_metadata(
            metadata_path,
            cache_path,
            start_date,
            end_date,
            requests[0],
            requests[0]["variable"],
        )
        return cache_path

    def _write_metadata(
        self,
        metadata_path: Path,
        cache_path: Path,
        start_date: str,
        end_date: str,
        request: Dict[str, Any],
        variables: Sequence[str],
    ) -> None:
        metadata = {
            "dataset": settings.era5_dataset,
            "variables": list(variables),
            "date_range": {"start": start_date, "end": end_date},
            "times": request["time"],
            "area": request["area"],
            "resolution": request["grid"],
            "download_timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "CDS/ERA5",
            "local_file_path": str(cache_path),
        }
        temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".part")
        temporary_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        temporary_metadata.replace(metadata_path)

    def download_climatology(
        self,
        variables: Optional[Sequence[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Path:
        start_date = start_date or settings.era5_start_date
        end_date = end_date or settings.era5_end_date
        if not start_date or not end_date:
            raise ValueError(
                "ERA5 climatology download requires ERA5_START_DATE and ERA5_END_DATE; "
                "configure a reference period before requesting data"
            )
        return self.download(
            start_date=start_date[:10],
            end_date=end_date[:10],
            variables=variables,
        )

    def check_availability(self) -> Dict[str, Any]:
        return {
            "configured": cds_configured(),
            "dataset": settings.era5_dataset,
            "source": "CDS/ERA5",
            "status": "configured" if cds_configured() else "not_configured",
            "resolution": settings.era5_resolution,
            "area": settings.era5_download_area,
            "cache_dir": str(Path(settings.era5_cache_dir).expanduser()),
        }


def _dates(start: date, end: date):
    from datetime import timedelta

    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def ensure_era5_climatology() -> str:
    downloader = Era5CDSDownloader()
    path = downloader.download_climatology()
    return str(path)


def _configured_local_file(path: Optional[str]) -> Tuple[bool, bool]:
    if not path:
        return False, False
    candidate = Path(path).expanduser()
    return True, candidate.exists() and (candidate.is_file() or candidate.is_dir())


def make_data_source_status() -> Dict[str, Any]:
    era5_configured, era5_local_available = _configured_local_file(settings.era5_climatology_path)
    neps_configured, neps_available = _configured_local_file(settings.neps_g_forecast_path)
    target_configured, target_available = _configured_local_file(settings.high_resolution_target_path)
    cache_dir = Path(settings.era5_cache_dir).expanduser()
    cached_era5_available = cache_dir.is_dir() and any(
        path.is_file() and path.stat().st_size > 0
        for path in cache_dir.glob("era5_*.nc")
    )
    if era5_local_available:
        era5_status = "configured_but_unverified"
    elif cached_era5_available:
        era5_status = "cached_data_unverified"
    elif cds_configured() and cdsapi is not None:
        era5_status = "configured_not_downloaded"
    elif era5_configured:
        era5_status = "configured_but_data_unavailable"
    else:
        era5_status = "not_configured"
    if neps_available:
        neps_status = "configured_but_unverified"
    elif (
        settings.neps_g_active
        or settings.data_mode.strip().upper() in {"REAL", "LOCAL_REAL"}
        or neps_configured
    ):
        neps_status = "NEPS-G: CONFIGURED BUT DATA UNAVAILABLE"
    else:
        neps_status = "not_configured"
    return {
        "era5": {
            "configured": era5_configured or cds_configured(),
            "available": era5_local_available or cached_era5_available,
            "status": era5_status,
            "source": "local ERA5 file" if era5_local_available else ("ERA5 cache" if cached_era5_available else "CDS/ERA5"),
            "resolution": settings.data_spatial_resolution_km or settings.era5_resolution,
            "time_coverage": settings.forecast_start_time or "not inspected",
        },
        "neps_g": {
            "configured": neps_configured or settings.neps_g_active,
            "available": neps_available,
            "status": neps_status,
            "source": "NEPS-G local files",
            "resolution": settings.data_spatial_resolution_km or "not inspected",
            "time_coverage": settings.forecast_start_time or "not inspected",
        },
        "high_resolution_reference": {
            "configured": target_configured,
            "available": False,
            "status": "configured_but_unverified" if target_available else "not_configured",
            "source": "local high-resolution reference",
            "resolution": settings.target_resolution_km or "not configured",
            "time_coverage": settings.forecast_start_time or "not inspected",
        },
    }
