"""
Alerting API (blueprint deliverable #4) + pipeline trigger + historical replay.

The pipeline (mesh build + GNN training + diffusion training/sampling) takes
real wall-clock time, so /pipeline/run does NOT block the request: it enqueues
a background job and returns a job_id immediately, matching the blueprint's
"Diffusion inference latency" guardrail ("keep the live path optional and
clearly labelled as slower"). Historical Replay Mode uses a FIXED seed per
named event so the same "Cyclone Amphan" replay is reproducible and offline-
safe for judging, exactly as blueprint section 11 specifies.
"""
import traceback
import uuid
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from config import settings
from src.data_provider import get_data_provider
from src.db import get_alert, get_latest_alerts, get_metrics, init_db
from src.era5_cds import make_data_source_status
from src.amphan_replay import (
    ANALYSIS_CACHE,
    EVENT_CACHE,
    REFERENCE_CACHE,
    load_cached_alert,
    load_cached_analysis_field,
    load_cached_field,
    load_cached_replay,
    load_cached_tracks,
)
from src.tigge_amphan import (
    TIGGE_ANALYSIS_CACHE,
    TIGGE_RESULT_CACHE,
    load_tigge_alert,
    load_tigge_field,
    load_tigge_replay,
    load_tigge_tracks,
)
from src.job_store import get_job_store
from src.forecast_manager import forecast_manager
from src.forecast_scheduler import start_forecast_scheduler
from src.pipeline import run_full_pipeline
from src.schemas import JobStatus, PipelineRunRequest

app = FastAPI(
    title="STORMTRACE-GNN Backend",
    description="Spherical-mesh GNN anomaly tracking + physics-constrained diffusion downscaling API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)

JOB_STORE = get_job_store()
HISTORICAL_REPLAY_SEEDS = {"cyclone_amphan": 2020, "north_india_heatwave": 2022}


@app.on_event("startup")
def startup():
    init_db()
    start_forecast_scheduler()


@app.get("/")
def api_root():
    return {
        "service": app.title,
        "status": "running",
        "health": "/health",
        "api_docs": "/docs",
        "openapi": "/openapi.json",
        "frontend": "http://127.0.0.1:5173/",
    }


def _provider_summary():
    try:
        provider = get_data_provider()
        metadata = provider.metadata
        return {
            "provider": metadata.get("provider"),
            "label": metadata.get("label"),
            "description": metadata.get("description"),
            "data_mode": settings.data_mode,
        }
    except Exception as exc:
        return {
            "provider": "unavailable",
            "label": "unavailable",
            "description": str(exc),
            "data_mode": settings.data_mode,
        }


def _tigge_cache_ready() -> bool:
    cache_dir = Path(settings.imdaa_processed_dir)
    return all((cache_dir / name).is_file() for name in (TIGGE_ANALYSIS_CACHE, TIGGE_RESULT_CACHE))


def _effective_status() -> dict:
    """Expose the active forecast capability without conflating legacy demo mode."""
    ready = _tigge_cache_ready()
    if ready:
        return {
            "backend": "online",
            "forecast_source": "NCMRWF TIGGE",
            "forecast_status": "ready",
            "forecast_data": "real",
            "data_mode": "LOCAL_REAL",
            "ensemble_members": 11,
            "reference_dataset": "IMDAA May 2011–2019",
            "gnn_status": "experimental",
            "tracking_status": "derived",
            "downscaling_status": "experimental",
            "verification_status": "not_available",
            "mfc_status": "not_available",
        }
    return {
        "backend": "online",
        "forecast_source": "NCMRWF TIGGE",
        "forecast_status": "unavailable",
        "forecast_data": "not_available",
        "data_mode": settings.data_mode,
        "ensemble_members": 0,
        "reference_dataset": "IMDAA May 2011–2019",
        "gnn_status": "experimental",
        "tracking_status": "derived",
        "downscaling_status": "experimental",
        "verification_status": "not_available",
        "mfc_status": "not_available",
    }


@app.get("/health")
@app.get("/api/health")
def health():
    provider = _provider_summary()
    status = _effective_status()
    return {
        "status": "backend is working",
        "version": app.version,
        "data_provider": "neps_g" if status["forecast_data"] == "real" else provider["provider"],
        "data_source": status["forecast_source"] if status["forecast_data"] == "real" else provider["label"],
        "source_label": status["forecast_source"] if status["forecast_data"] == "real" else provider["label"],
        "source_description": "Local cached NCMRWF TIGGE forecast; no remote download performed." if status["forecast_data"] == "real" else provider["description"],
        **status,
    }


@app.get("/api/system/status")
def system_status():
    provider = _provider_summary()
    data_sources = make_data_source_status()
    imdaa_cache = Path(settings.imdaa_processed_dir)
    imdaa_available = all(
        (imdaa_cache / name).is_file()
        for name in (REFERENCE_CACHE, EVENT_CACHE, ANALYSIS_CACHE, "amphan_replay.json")
    )
    data_sources["imdaa_historical_replay"] = {
        "configured": Path(settings.imdaa_archive_dir).is_dir(),
        "available": imdaa_available,
        "status": "READY" if imdaa_available else "NOT PREPROCESSED",
        "source": "local IMDAA reanalysis",
        "resolution": "0.12 degree",
        "time_coverage": "May 2011–2019 reference; 2020-05-16/2020-05-21 event",
        "reference_contains_2020": False,
        "rainfall_status": (
            "UNRESOLVED_SEMANTICS_RAW_UNVERIFIED"
            if settings.imdaa_rainfall_mode.strip().lower() == "raw_unverified"
            else "UNRESOLVED_SEMANTICS_DISABLED"
        ),
    }
    tigge_available = all(
        (Path(settings.imdaa_processed_dir) / name).is_file()
        for name in (TIGGE_ANALYSIS_CACHE, TIGGE_RESULT_CACHE)
    )
    data_sources["ncmrwf_tigge_amphan_replay"] = {
        "configured": bool(settings.neps_g_forecast_path),
        "available": tigge_available,
        "status": "READY" if tigge_available else "NOT PRECOMPUTED",
        "source": "NCMRWF TIGGE (dems)",
        "forecast_path": settings.neps_g_forecast_path,
        "reference": "IMDAA May 2011–2019 t2m only",
        "reference_contains_2020": False,
    }
    return {
        "status": "ok",
        **_effective_status(),
        "job_store_backend": settings.job_store_backend,
        "job_store_mode": (
            "shared_redis" if settings.job_store_backend.lower() == "redis" else "local_fallback"
        ),
        "provider": provider,
        "data_sources": data_sources,
        "backend": "FastAPI",
    }


@app.get("/api/system/data-sources")
def system_data_sources():
    status = make_data_source_status()
    imdaa_cache = Path(settings.imdaa_processed_dir)
    imdaa_available = all(
        (imdaa_cache / name).is_file()
        for name in (REFERENCE_CACHE, EVENT_CACHE, ANALYSIS_CACHE, "amphan_replay.json")
    )
    status["imdaa_historical_replay"] = {
        "configured": Path(settings.imdaa_archive_dir).is_dir(),
        "available": imdaa_available,
        "status": "READY" if imdaa_available else "NOT PREPROCESSED",
        "source": "local IMDAA reanalysis",
        "resolution": "0.12 degree",
        "time_coverage": "May 2011–2019 reference; 2020-05-16/2020-05-21 event",
        "reference_contains_2020": False,
        "rainfall_status": (
            "UNRESOLVED_SEMANTICS_RAW_UNVERIFIED"
            if settings.imdaa_rainfall_mode.strip().lower() == "raw_unverified"
            else "UNRESOLVED_SEMANTICS_DISABLED"
        ),
    }
    tigge_available = all(
        (imdaa_cache / name).is_file()
        for name in (TIGGE_ANALYSIS_CACHE, TIGGE_RESULT_CACHE)
    )
    status["ncmrwf_tigge_amphan_replay"] = {
        "configured": bool(settings.neps_g_forecast_path),
        "available": tigge_available,
        "status": "READY" if tigge_available else "NOT PRECOMPUTED",
        "source": "NCMRWF TIGGE (dems)",
        "forecast_path": settings.neps_g_forecast_path,
        "reference": "IMDAA May 2011–2019 t2m only",
        "reference_contains_2020": False,
    }
    return status


def _amphan_cached(callable_, *args, **kwargs):
    try:
        return callable_(settings.imdaa_processed_dir, *args, **kwargs)
    except (FileNotFoundError, OSError, RuntimeError, ValueError, IndexError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/amphan/replay")
def run_amphan_replay():
    """Load the precomputed historical replay; this endpoint never trains models."""
    return _amphan_cached(load_cached_replay)


@app.get("/api/amphan/overview")
def amphan_overview():
    return _amphan_cached(load_cached_replay)


@app.get("/api/amphan/timeline")
def amphan_timeline():
    replay = _amphan_cached(load_cached_replay)
    return {
        "event": replay["event"],
        "mode": replay["mode"],
        "stages": replay["progress"],
        "timeline": replay["timeline"],
        "data_status": replay["data_status"],
    }


@app.get("/api/amphan/field")
def amphan_field(
    variable: str = Query(default="t2m"),
    time_index: int = Query(default=0, ge=0, le=23),
    visualization: bool = Query(default=False),
    max_points: int = Query(default=64, ge=2, le=256),
):
    if variable in {"unet", "diffusion", "crop_t2m", "crop_efi"}:
        return _amphan_cached(
            load_cached_analysis_field,
            variable,
            visualization=visualization,
            max_points=min(max_points, 48),
        )
    return _amphan_cached(
        load_cached_field,
        variable,
        time_index,
        visualization=visualization,
        max_points=max_points,
    )


@app.get("/api/amphan/tracks")
def amphan_tracks(time_index: int = Query(default=0, ge=0, le=23)):
    return _amphan_cached(load_cached_tracks, time_index)


@app.get("/api/amphan/alert")
def amphan_alert(time_index: int = Query(default=0, ge=0, le=23)):
    return _amphan_cached(load_cached_alert, time_index)


@app.get("/api/amphan/verification")
def amphan_verification():
    replay = _amphan_cached(load_cached_replay)
    return replay["verification"]


def _tigge_cached(callable_, *args, **kwargs):
    try:
        return callable_(settings.imdaa_processed_dir, *args, **kwargs)
    except (FileNotFoundError, OSError, RuntimeError, ValueError, IndexError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _cached_field_headers(variable: str, time_index: int, visualization: bool, max_points: int) -> dict[str, str]:
    analysis_path = Path(settings.imdaa_processed_dir) / TIGGE_ANALYSIS_CACHE
    try:
        modified = analysis_path.stat().st_mtime_ns
    except OSError:
        modified = 0
    etag = f'"tigge-{modified}-{variable}-{time_index}-{int(visualization)}-{max_points}"'
    return {"ETag": etag, "Cache-Control": "public, max-age=60, stale-while-revalidate=300"}


@app.post("/api/tigge/amphan/replay")
def run_tigge_amphan_replay():
    """Load precomputed real TIGGE inference; never parses GRIB or trains per request."""
    return _tigge_cached(load_tigge_replay)


@app.get("/api/tigge/amphan/overview")
def tigge_amphan_overview():
    return _tigge_cached(load_tigge_replay)


@app.get("/api/tigge/amphan/field")
def tigge_amphan_field(
    request: Request,
    response: Response,
    variable: str = Query(default="t2m"),
    time_index: int = Query(default=0, ge=0, le=40),
    visualization: bool = Query(default=False),
    max_points: int = Query(default=64, ge=2, le=256),
):
    headers = _cached_field_headers(variable, time_index, visualization, max_points)
    response.headers.update(headers)
    if request.headers.get("if-none-match") == headers["ETag"]:
        return Response(status_code=304, headers=headers)
    return _tigge_cached(
        load_tigge_field,
        variable,
        time_index,
        visualization=visualization,
        max_points=max_points,
    )


@app.get("/api/tigge/amphan/tracks")
def tigge_amphan_tracks(time_index: int = Query(default=0, ge=0, le=40)):
    return _tigge_cached(load_tigge_tracks, time_index)


@app.get("/api/tigge/amphan/alert")
def tigge_amphan_alert(time_index: int = Query(default=0, ge=0, le=40)):
    return _tigge_cached(load_tigge_alert, time_index)


@app.get("/api/tigge/amphan/verification")
def tigge_amphan_verification():
    replay = _tigge_cached(load_tigge_replay)
    return replay["verification"]


# Generalized cache-backed API aliases. Legacy Amphan routes remain unchanged.
@app.get("/api/v1/system/status")
def v1_system_status():
    return system_status()


@app.get("/api/v1/system/data-sources")
def v1_system_data_sources():
    return system_data_sources()


@app.get("/api/v1/forecast/status")
def v1_forecast_status():
    return forecast_manager.status()


@app.get("/api/v1/forecast/ingestion-status")
def v1_forecast_ingestion_status():
    state = forecast_manager.status()
    return {
        "mode": state.get("mode", settings.data_mode.upper()),
        "source": state.get("source", "NCMRWF TIGGE"),
        "enabled": bool(settings.tigge_auto_enabled or settings.data_mode.upper() == "TIGGE_AUTO"),
        "status": state.get("status"),
        "last_attempt": state.get("last_attempt"),
        "last_success": state.get("last_success"),
        "active_initialization": state.get("initialization_time"),
        "latest_valid_time": state.get("latest_valid_time"),
        "forecast_age_minutes": state.get("age_minutes"),
        "stale": state.get("stale", True),
        "error": state.get("error_state"),
    }

@app.get("/api/v1/forecast/latest")
def v1_forecast_latest():
    replay = _tigge_cached(load_tigge_replay)
    replay["forecast_state"] = forecast_manager.status()
    return replay


@app.get("/api/v1/forecast/timeline")
def v1_forecast_timeline():
    replay = _tigge_cached(load_tigge_replay)
    return {
        "status": replay["status"],
        "event": replay["event"],
        "initialization_time": replay["forecast_initialization_time"],
        "timeline": replay["timeline"],
        "provenance": replay["provenance"]["forecast"],
        "forecast_state": forecast_manager.status(),
    }


@app.get("/api/v1/forecast/field")
def v1_forecast_field(
    request: Request,
    response: Response,
    variable: str = Query(default="t2m"),
    time_index: int = Query(default=0, ge=0),
    visualization: bool = Query(default=True),
    max_points: int = Query(default=64, ge=2, le=256),
):
    headers = _cached_field_headers(variable, time_index, visualization, max_points)
    response.headers.update(headers)
    if request.headers.get("if-none-match") == headers["ETag"]:
        return Response(status_code=304, headers=headers)
    return _tigge_cached(
        load_tigge_field,
        variable,
        time_index,
        visualization=visualization,
        max_points=max_points,
    )


@app.get("/api/v1/forecast/ensemble")
def v1_forecast_ensemble():
    replay = _tigge_cached(load_tigge_replay)
    forecast = replay["data_status"]["forecast"]
    return {
        "status": forecast["status"],
        "ensemble_count": forecast["ensemble_count"],
        "member_ids": replay["provenance"]["forecast"].get("ensemble_members", []),
        "note": "API field products are cached ensemble means; raw member arrays are not returned.",
        "forecast_state": forecast_manager.status(),
    }


@app.get("/api/v1/anomaly/efi")
def v1_anomaly_efi(
    time_index: int = Query(default=0, ge=0),
    visualization: bool = Query(default=True),
    max_points: int = Query(default=64, ge=2, le=256),
):
    return tigge_amphan_field("anomaly_score", time_index, visualization, max_points)


@app.get("/api/v1/anomaly/gnn")
def v1_anomaly_gnn(
    time_index: int = Query(default=0, ge=0),
    visualization: bool = Query(default=True),
    max_points: int = Query(default=64, ge=2, le=256),
):
    return tigge_amphan_field("gnn_probability", time_index, visualization, max_points)


@app.get("/api/v1/tracking/current")
def v1_tracking_current(time_index: int = Query(default=0, ge=0)):
    return tigge_amphan_tracks(time_index)


@app.get("/api/v1/tracking/history")
def v1_tracking_history():
    replay = _tigge_cached(load_tigge_replay)
    return {
        "tracks": replay["tracks"],
        "event": replay.get("event"),
        "status": replay["status"],
        "forecast_state": forecast_manager.status(),
    }


@app.get("/api/v1/alerts/current")
def v1_alerts_current(time_index: int = Query(default=0, ge=0)):
    return tigge_amphan_alert(time_index)


@app.get("/api/v1/downscaling/status")
def v1_downscaling_status():
    return {
        "status": "experimental",
        "output": "2x model-derived spatial refinement",
        "matched_tigge_output": False,
        "independent_5km_verification": "NOT_AVAILABLE",
        "note": "Cached U-Net and diffusion snapshots are not matched to a TIGGE timestep.",
    }


@app.get("/api/data/available")
def data_available():
    try:
        provider = get_data_provider()
        return {
            "data_mode": settings.data_mode,
            "available": True,
            "provider": provider.metadata.get("provider"),
            "label": provider.metadata.get("label"),
            "description": provider.metadata.get("description"),
        }
    except Exception as exc:
        return {
            "data_mode": settings.data_mode,
            "available": False,
            "reason": str(exc),
            "provider": "unavailable",
        }


@app.get("/api/data/qc")
def api_data_qc():
    return {
        "status": "not_available",
        "reason": "QC results are recorded with each completed analysis; no persisted QC report is available yet",
        "data_mode": settings.data_mode,
    }


@app.get("/api/forecast/runs")
def forecast_runs():
    return {"runs": [], "data_mode": settings.data_mode}


@app.get("/api/forecast/{run_id}")
def forecast_run(run_id: str):
    return {"run_id": run_id, "status": "not_available", "reason": "no run has been persisted yet", "data_mode": settings.data_mode}


@app.get("/api/forecast/{run_id}/fields")
def forecast_fields(run_id: str):
    return {"run_id": run_id, "fields": [], "data_mode": settings.data_mode}


@app.get("/api/efi")
def efi():
    return {"status": "not_available", "reason": "EFI requires a valid real forecast and climatology reference", "data_mode": settings.data_mode}


@app.get("/api/anomalies")
def anomalies():
    return {"anomalies": [], "data_mode": settings.data_mode}


@app.get("/api/tracks")
def tracks():
    return {"tracks": [], "data_mode": settings.data_mode}


@app.get("/api/tracks/{track_id}")
def track(track_id: str):
    return {"track_id": track_id, "status": "not_available", "reason": "no track stored for this identifier", "data_mode": settings.data_mode}


@app.get("/api/alerts")
def api_alerts():
    return get_latest_alerts(limit=25)


@app.get("/api/physics/validation")
def physics_validation():
    return {"status": "not_available", "reason": "physics validation requires real variables and reference fields", "data_mode": settings.data_mode}


@app.get("/api/verification")
def verification():
    return {"status": "not_available", "reason": "verification metrics require a valid reference dataset", "data_mode": settings.data_mode}


@app.get("/api/historical/events")
def historical_events():
    return {"events": list(HISTORICAL_REPLAY_SEEDS.keys()), "data_mode": settings.data_mode}


@app.post("/api/analysis/run")
def analysis_run(req: PipelineRunRequest, background_tasks: BackgroundTasks):
    return trigger_pipeline(req, background_tasks)


@app.get("/api/analysis/{job_id}")
def analysis_status(job_id: str):
    return pipeline_status(job_id)


@app.get("/api/models/status")
def models_status():
    return {"status": "ok", "models": {"GNN": "ready", "U-Net": "prototype", "Diffusion": "prototype/proxy"}, "data_mode": settings.data_mode}


def _execute_job(job_id: str, event_name: str, seed: Optional[int]):
    JOB_STORE.mark_running(job_id)

    def progress(msg: str):
        JOB_STORE.set_progress(job_id, msg)

    try:
        result = run_full_pipeline(event_name=event_name, seed=seed, progress_cb=progress)
        JOB_STORE.mark_done(job_id, result)
    except Exception as e:
        JOB_STORE.mark_failed(job_id, f"{e}\n{traceback.format_exc()}")


@app.post("/pipeline/run", response_model=JobStatus)
def trigger_pipeline(req: PipelineRunRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    JOB_STORE.create_job(job_id)
    background_tasks.add_task(_execute_job, job_id, req.event_name, req.seed)
    return JobStatus(job_id=job_id, status="queued")


@app.get("/pipeline/status/{job_id}", response_model=JobStatus)
def pipeline_status(job_id: str):
    job = JOB_STORE.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return JobStatus(job_id=job_id, **job)


@app.post("/pipeline/run_sync")
def trigger_pipeline_sync(req: PipelineRunRequest):
    """Blocking variant for quick local testing / curl. Prefer /pipeline/run for real clients."""
    try:
        return run_full_pipeline(event_name=req.event_name, seed=req.seed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/replay/{event_key}")
def historical_replay(event_key: str, background_tasks: BackgroundTasks):
    """Offline-safe judging path: known historical event -> fixed seed -> reproducible run."""
    if event_key not in HISTORICAL_REPLAY_SEEDS:
        raise HTTPException(
            status_code=404,
            detail=f"unknown replay event. Available: {list(HISTORICAL_REPLAY_SEEDS.keys())}",
        )
    job_id = str(uuid.uuid4())
    JOB_STORE.create_job(job_id)
    background_tasks.add_task(_execute_job, job_id, event_key, HISTORICAL_REPLAY_SEEDS[event_key])
    return JobStatus(job_id=job_id, status="queued")


@app.get("/alerts/latest")
def latest_alerts(
    limit: int = Query(10, ge=1, le=200),
    severity: Optional[str] = Query(None, pattern="^(low|moderate|severe)$"),
    event_name: Optional[str] = None,
):
    return get_latest_alerts(limit=limit, severity=severity, event_name=event_name)


@app.get("/alerts/{alert_id}")
def alert_by_id(alert_id: int):
    alert = get_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="alert not found")
    return alert


@app.get("/metrics")
def metrics():
    return get_metrics()


@app.get("/replay/events")
def list_replay_events():
    return {"available_events": list(HISTORICAL_REPLAY_SEEDS.keys())}
