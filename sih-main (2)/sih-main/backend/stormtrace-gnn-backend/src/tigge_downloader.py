"""Secure, atomic NCMRWF TIGGE retrieval through the configured CDS client.

The client is injected in tests and credentials are resolved by the existing
CDS mechanism. No credential value is included in requests returned to APIs
or in provenance.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from config import settings
from src.data_provider import _read_ncmrwf_tigge_grib
from src.era5_cds import _resolve_cds_credentials, cdsapi, validate_cds_url


class TiggeDownloadError(RuntimeError):
    pass


def _cycle_candidates(now: datetime, lookback: int) -> Iterable[datetime]:
    cycle = now.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    for offset in range(max(1, lookback)):
        day = cycle - timedelta(days=offset)
        for hour in (12, 0):
            yield day.replace(hour=hour)


class TIGGEDownloader:
    def __init__(self, client=None, now: Callable[[], datetime] | None = None):
        self._now = now or (lambda: datetime.now(timezone.utc))
        credentials = _resolve_cds_credentials()
        if credentials is None:
            raise TiggeDownloadError(
                "LIVE INGESTION UNAVAILABLE: CDS/ECDS credentials are not configured"
            )
        if client is None and cdsapi is None:
            raise TiggeDownloadError("LIVE INGESTION UNAVAILABLE: cdsapi is not installed")
        url, key = credentials
        self.client = client or cdsapi.Client(
            url=validate_cds_url(url),
            key=key,
            timeout=settings.cds_timeout_seconds,
            quiet=True,
        )

    @staticmethod
    def _region() -> list[float]:
        values = [float(value.strip()) for value in settings.tigge_region.split(",")]
        if len(values) != 4:
            raise ValueError("TIGGE_REGION must be north,west,south,east")
        north, west, south, east = values
        if not (-90 <= south < north <= 90 and -180 <= west < east <= 360):
            raise ValueError("TIGGE_REGION has invalid geographic bounds")
        return values

    def _request(self, cycle: datetime) -> dict[str, Any]:
        return {
            "origin": settings.tigge_origin,
            "product_type": "ensemble_perturbed_forecasts",
            "variable": [
                "2m_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "total_precipitation",
            ],
            "date": cycle.strftime("%Y-%m-%d"),
            "time": f"{cycle.hour:02d}:00",
            "leadtime_hour": [str(value) for value in range(0, settings.tigge_max_lead_hours + 1, 6)],
            "format": "grib",
            "area": self._region(),
        }

    def _validate_and_activate(self, temporary: Path, cycle: datetime, request: dict[str, Any]) -> dict[str, Any]:
        values, metadata = _read_ncmrwf_tigge_grib(temporary)
        cache_dir = Path(settings.tigge_cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / f"tigge_ncmrwf_{cycle.strftime('%Y%m%d%H')}.grib"
        temporary.replace(target)
        provenance = {
            "source": "NCMRWF TIGGE",
            "dataset": settings.tigge_dataset,
            "mode": "TIGGE_AUTO",
            "initialization": metadata.get("initialization_time"),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "path": str(target),
            "variables": metadata.get("source_variables"),
            "members": metadata.get("ensemble_members"),
            "steps": metadata.get("valid_time"),
            "grid": metadata.get("grid_dimensions"),
            "region": metadata.get("region"),
            "units": metadata.get("variable_units"),
            "centre": metadata.get("centre"),
            "data_type": metadata.get("data_type"),
            "grib_edition": metadata.get("grib_edition"),
            "request": {key: value for key, value in request.items() if key not in {"key", "token"}},
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "processing_version": metadata.get("processing_version"),
        }
        (target.with_suffix(".json")).write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
        return {"success": True, **provenance}

    def download_cycle(self, cycle: datetime) -> dict[str, Any]:
        request = self._request(cycle)
        cache_dir = Path(settings.tigge_cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / f"tigge_ncmrwf_{cycle.strftime('%Y%m%d%H')}.grib"
        if target.is_file():
            try:
                _read_ncmrwf_tigge_grib(target)
                return {"success": True, "skipped": True, "path": str(target), "mode": "TIGGE_AUTO"}
            except Exception:
                target.unlink(missing_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(prefix="tigge_", suffix=".grib", dir=cache_dir, delete=False) as handle:
                temporary_path = Path(handle.name)
            self.client.retrieve(settings.tigge_dataset, request, str(temporary_path))
            return self._validate_and_activate(temporary_path, cycle, request)
        except Exception as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise TiggeDownloadError(f"TIGGE cycle {cycle.isoformat()} failed: {exc}") from exc

    def discover_and_download(self) -> dict[str, Any]:
        errors = []
        for cycle in _cycle_candidates(self._now(), settings.tigge_lookback_cycles):
            try:
                return self.download_cycle(cycle)
            except TiggeDownloadError as exc:
                errors.append(str(exc))
        raise TiggeDownloadError(
            "No valid NCMRWF TIGGE cycle found in configured lookback: " + " | ".join(errors)
        )
