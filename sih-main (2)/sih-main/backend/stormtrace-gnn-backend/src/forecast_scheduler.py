"""Non-blocking forecast-cycle scheduler.

Remote retrieval is intentionally not invented here. When enabled, the
scheduler validates the configured local cycle and records failures clearly.
"""

from __future__ import annotations

import logging
import threading
from time import sleep

from config import settings
from src.forecast_manager import forecast_manager

logger = logging.getLogger(__name__)


def _run() -> None:
    while True:
        try:
            if settings.tigge_auto_enabled or settings.data_mode.upper() == "TIGGE_AUTO":
                forecast_manager.ingest_auto()
            else:
                forecast_manager.ingest()
        except Exception:
            logger.exception("Forecast auto-ingestion failed")
        sleep(max(1, settings.forecast_interval_minutes * 60))


def start_forecast_scheduler() -> threading.Thread | None:
    if not (settings.forecast_auto_ingest or settings.tigge_auto_enabled):
        return None
    worker = threading.Thread(target=_run, name="forecast-ingestion", daemon=True)
    worker.start()
    return worker
