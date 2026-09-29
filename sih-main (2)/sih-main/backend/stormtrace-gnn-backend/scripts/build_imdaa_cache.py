"""Build compact IMDAA reference/event caches and precompute replay outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import settings
from src.amphan_replay import (
    EVENT_CACHE,
    REFERENCE_CACHE,
    _write_json,
    _save_npz,
    precompute_amphan_replay,
)
from src.imdaa_adapter import load_imdaa_period


def _load_year(archive_dir: Path, year: int):
    return load_imdaa_period(
        archive_dir,
        f"{year}-05-01",
        f"{year}-05-31",
    )


def build_cache(archive_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_fields: list[np.ndarray] = []
    reference_times: list[np.ndarray] = []
    reference_year_counts: dict[str, int] = {}
    reference_latitude = reference_longitude = None
    for year in range(2011, 2020):
        print(f"Reading IMDAA reference May {year} (regional subset only)...", flush=True)
        period = _load_year(archive_dir, year)
        if reference_latitude is None:
            reference_latitude = period.latitude
            reference_longitude = period.longitude
        elif not (
            np.array_equal(reference_latitude, period.latitude)
            and np.array_equal(reference_longitude, period.longitude)
        ):
            raise ValueError(f"IMDAA grid for {year} does not match prior reference years")
        reference_fields.append(period.fields["t2m"])
        reference_times.append(period.times)
        reference_year_counts[str(year)] = int(period.times.size)

    reference_t2m = np.concatenate(reference_fields, axis=0)
    reference_time_values = np.concatenate(reference_times)
    years_in_reference = sorted(
        {int(str(value)[:4]) for value in reference_time_values}
    )
    if years_in_reference != list(range(2011, 2020)) or 2020 in years_in_reference:
        raise ValueError("Reference cache must contain May 2011–2019 only")
    percentile_levels = np.asarray([50, 75, 90, 95, 99], dtype=np.int32)
    percentiles_by_hour = np.empty(
        (4, len(percentile_levels), len(reference_latitude), len(reference_longitude)),
        dtype=np.float32,
    )
    for hour_index, hour in enumerate((0, 6, 12, 18)):
        indices = np.flatnonzero(
            np.array(
                [int(np.datetime_as_string(value, unit="h")[11:13]) for value in reference_time_values]
            )
            == hour
        )
        if indices.size != 9 * 31:
            raise ValueError(
                f"expected 279 May reference samples for {hour:02d} UTC, found {indices.size}"
            )
        percentiles_by_hour[hour_index] = np.percentile(
            reference_t2m[indices], percentile_levels, axis=0
        ).astype(np.float32)
    reference_path = output_dir / REFERENCE_CACHE
    _save_npz(
        reference_path,
        times=reference_time_values,
        latitude=reference_latitude,
        longitude=reference_longitude,
        t2m=reference_t2m,
        percentile_levels=percentile_levels,
        percentiles_by_hour=percentiles_by_hour,
        percentile_hours=np.asarray([0, 6, 12, 18], dtype=np.int8),
    )
    reference_meta = {
        "dataset": "IMDAA",
        "source": "local IMDAA reanalysis ZIP archives",
        "years": years_in_reference,
        "year_sample_counts": reference_year_counts,
        "sample_count": int(reference_time_values.size),
        "period": "May 2011–2019",
        "contains_2020": False,
        "variables_used_for_reference": ["t2m"],
        "rainfall": {
            "status": "UNRESOLVED_SEMANTICS",
            "included_in_reference": False,
            "units": None,
            "reason": "APCP units and accumulation interval are not verified.",
        },
        "region": {
            "latitude_min": float(reference_latitude.min()),
            "latitude_max": float(reference_latitude.max()),
            "longitude_min": float(reference_longitude.min()),
            "longitude_max": float(reference_longitude.max()),
            "latitude_count": int(reference_latitude.size),
            "longitude_count": int(reference_longitude.size),
        },
        "percentiles": percentile_levels.tolist(),
        "percentile_context": "location-specific May samples grouped by UTC synoptic hour",
        "quality_control": {"status": "PASS", "missing_or_nonfinite_values": 0},
    }
    _write_json(reference_path.with_suffix(".npz.json"), reference_meta)

    print("Reading IMDAA Amphan event 2020-05-16 through 2020-05-21...", flush=True)
    event = load_imdaa_period(
        archive_dir,
        "2020-05-16",
        "2020-05-21",
    )
    if event.times.size != 24:
        raise ValueError(f"expected 24 Amphan timestamps, found {event.times.size}")
    if not (
        np.array_equal(reference_latitude, event.latitude)
        and np.array_equal(reference_longitude, event.longitude)
    ):
        raise ValueError("IMDAA event grid does not match the 2011–2019 reference grid")
    event_path = output_dir / EVENT_CACHE
    _save_npz(
        event_path,
        times=event.times,
        latitude=event.latitude,
        longitude=event.longitude,
        **event.fields,
    )
    event_meta = {
        **event.provenance,
        "event_period": "2020-05-16/2020-05-21",
        "sample_count": int(event.times.size),
        "reference_years": years_in_reference,
        "rainfall": {
            **event.provenance["rainfall"],
            "mode": settings.imdaa_rainfall_mode,
        },
        "variable_source_counts": {
            variable: int(event.times.size)
            for variable in ("t2m", "rainfall", "wind_u", "wind_v")
        },
    }
    _write_json(event_path.with_suffix(".npz.json"), event_meta)
    print("Precomputing EFI, mesh, weakly-supervised GNN, tracking, downscaling and physics proxy...", flush=True)
    result = precompute_amphan_replay(output_dir)
    return {
        "reference_cache": str(reference_path),
        "event_cache": str(event_path),
        "analysis_cache": str(output_dir / "amphan_analysis.npz"),
        "result_cache": str(output_dir / "amphan_replay.json"),
        "status": result["status"],
        "reference_samples": result["data_status"]["reference"]["samples"],
        "event_samples": result["data_status"]["event"]["sample_count"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", default=settings.imdaa_archive_dir)
    parser.add_argument("--output-dir", default=settings.imdaa_processed_dir)
    args = parser.parse_args()
    outcome = build_cache(Path(args.archive_dir), Path(args.output_dir))
    print(json.dumps(outcome, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
