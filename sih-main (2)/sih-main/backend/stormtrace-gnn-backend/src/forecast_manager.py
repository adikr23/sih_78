"""Persistent state for the current forecast cycle.

Derived TIGGE products continue to use the existing cache builder. This
manager makes their current-cycle state explicit and provides one place for
future ingestion/processing orchestration.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import settings
from src.live_ingestion import ingest_local_forecast
from src.tigge_amphan import TIGGE_ANALYSIS_CACHE, TIGGE_RESULT_CACHE, load_tigge_replay


class ForecastManager:
    def __init__(self) -> None:
        self.state_path = Path(settings.forecast_state_path)

    def _cache_state(self) -> dict[str, Any] | None:
        cache_dir = Path(settings.imdaa_processed_dir)
        if not all((cache_dir / name).is_file() for name in (TIGGE_ANALYSIS_CACHE, TIGGE_RESULT_CACHE)):
            return None
        try:
            replay = load_tigge_replay(settings.imdaa_processed_dir)
        except (FileNotFoundError, OSError, RuntimeError, ValueError):
            return None
        forecast = replay.get("provenance", {}).get("forecast", {})
        valid_times = forecast.get("valid_time") or [
            item.get("time") for item in replay.get("timeline", [])
        ]
        initialization = forecast.get("initialization_time")
        latest = valid_times[-1] if valid_times else None
        age_minutes = None
        if initialization:
            try:
                init_time = datetime.fromisoformat(str(initialization).replace("Z", "+00:00"))
                if init_time.tzinfo is None:
                    init_time = init_time.replace(tzinfo=timezone.utc)
                age_minutes = max(0.0, (datetime.now(timezone.utc) - init_time).total_seconds() / 60.0)
            except ValueError:
                age_minutes = None
        source_file = forecast.get("uri") or forecast.get("source_file")
        ingestion_time = datetime.fromtimestamp(
            max((cache_dir / name).stat().st_mtime for name in (TIGGE_ANALYSIS_CACHE, TIGGE_RESULT_CACHE)),
            tz=timezone.utc,
        ).isoformat()
        return {
            "forecast_id": f"{forecast.get('provider', 'tigge')}:{initialization or 'unknown'}",
            "source": "NCMRWF TIGGE",
            "mode": settings.data_mode.upper(),
            "initialization_time": initialization,
            "latest_valid_time": latest,
            "available_steps": len(valid_times),
            "ensemble_members": forecast.get("ensemble_count", 0),
            "variables": ["t2m", "rainfall", "wind_u", "wind_v"],
            "processing_status": "READY",
            "ingestion_time": ingestion_time,
            "cache_location": str(cache_dir),
            "provenance": forecast,
            "error_state": None,
            "age_minutes": age_minutes,
            "stale": age_minutes is None or age_minutes > settings.forecast_stale_after_minutes,
            "status": "STALE" if age_minutes is None or age_minutes > settings.forecast_stale_after_minutes else "READY",
            "note": "Cached local TIGGE cycle; automatic remote retrieval is not configured.",
            "remote_ingestion": "unavailable",
            "source_file": source_file,
        }

    def status(self) -> dict[str, Any]:
        if settings.tigge_auto_enabled and self.state_path.is_file():
            try:
                return json.loads(self.state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        state = self._cache_state()
        if state is not None:
            return state
        return {
            "forecast_id": None,
            "source": "NCMRWF TIGGE",
            "mode": settings.data_mode.upper(),
            "initialization_time": None,
            "latest_valid_time": None,
            "age_minutes": None,
            "available_steps": 0,
            "ensemble_members": 0,
            "variables": [],
            "processing_status": "ERROR",
            "ingestion_time": None,
            "cache_location": str(Path(settings.imdaa_processed_dir)),
            "provenance": {},
            "error_state": "LIVE INGESTION UNAVAILABLE: forecast cache is missing",
            "stale": True,
            "status": "ERROR",
        }

    def ingest(self) -> dict[str, Any]:
        """Validate a local cycle and persist its state; processing remains cached."""
        forecast = ingest_local_forecast()
        state = self.status()
        state.update({
            "initialization_time": forecast.metadata.get("initialization_time"),
            "ensemble_members": forecast.members,
            "available_steps": forecast.steps,
            "variables": list(forecast.metadata.get("variables", [])),
            "processing_status": "VALIDATED",
            "ingestion_time": datetime.now(timezone.utc).isoformat(),
            "source_file": forecast.metadata.get("source_file"),
            "provenance": {key: value for key, value in forecast.metadata.items() if key not in {"latitude", "longitude"}},
            "error_state": None,
            "status": "READY",
            "stale": False,
        })
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
        return state

    def ingest_auto(self) -> dict[str, Any]:
        from src.tigge_downloader import TIGGEDownloader

        attempt = datetime.now(timezone.utc).isoformat()
        try:
            result = TIGGEDownloader().discover_and_download()
            state = self.status()
            state.update({
                "forecast_id": f"ncmrwf-tigge:{result.get('initialization')}",
                "source": "NCMRWF TIGGE",
                "mode": "TIGGE_AUTO",
                "initialization_time": result.get("initialization"),
                "available_steps": len(result.get("steps") or []),
                "ensemble_members": len(result.get("members") or []),
                "variables": result.get("variables") or [],
                "processing_status": "VALIDATED",
                "ingestion_time": result.get("retrieved_at"),
                "cache_location": result.get("path"),
                "provenance": result,
                "error_state": None,
                "last_attempt": attempt,
                "last_success": result.get("retrieved_at"),
                "stale": False,
                "status": "READY",
            })
        except Exception as exc:
            state = self.status()
            state.update({
                "mode": "TIGGE_AUTO",
                "last_attempt": attempt,
                "error_state": str(exc),
                "status": "STALE" if state.get("forecast_id") else "ERROR",
                "stale": True,
            })
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
        return state


forecast_manager = ForecastManager()
