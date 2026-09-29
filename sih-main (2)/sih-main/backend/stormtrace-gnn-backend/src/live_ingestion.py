"""Validated local forecast ingestion shared by real-data services.

This module deliberately delegates GRIB decoding to the existing NCMRWF
provider. It adds a small normalized object for services that need forecast
state without exposing provider internals.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config import settings
from src.data_provider import NCMRWFTIGGEProvider


@dataclass(frozen=True)
class NormalizedForecast:
    values: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    metadata: dict[str, Any]

    @property
    def steps(self) -> int:
        return int(self.values.shape[0])

    @property
    def members(self) -> int:
        return int(self.values.shape[1])


def locate_forecast(path: str | None = None) -> Path:
    candidate = Path(path or settings.neps_g_forecast_path or "").expanduser()
    if not candidate.is_file():
        raise FileNotFoundError(
            "LIVE INGESTION UNAVAILABLE: configured NCMRWF TIGGE forecast file is missing"
        )
    return candidate


def ingest_local_forecast(path: str | None = None) -> NormalizedForecast:
    """Read and validate one local TIGGE cycle using the canonical provider."""
    source = locate_forecast(path)
    provider = NCMRWFTIGGEProvider()
    values = np.asarray(provider.generate_ensemble(), dtype=np.float32)
    if values.ndim != 5:
        raise ValueError("NCMRWF TIGGE forecast must have shape (time, member, variable, latitude, longitude)")
    latitude, longitude = provider.grid_coordinates(values.shape[-2:])
    if not np.isfinite(values).all():
        raise ValueError("NCMRWF TIGGE forecast contains NaN or infinite values")
    metadata = dict(provider.metadata)
    metadata.update({
        "source_file": str(source),
        "latitude": np.asarray(latitude, dtype=np.float64),
        "longitude": np.asarray(longitude, dtype=np.float64),
        "canonical_dimensions": ["time", "member", "variable", "latitude", "longitude"],
    })
    return NormalizedForecast(values, np.asarray(latitude), np.asarray(longitude), metadata)
