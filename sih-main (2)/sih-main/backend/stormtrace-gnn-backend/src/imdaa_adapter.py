"""Read-only adapter for the local IMDAA monthly ZIP archives."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import re
from pathlib import Path
from typing import Any, Iterable, Iterator
import zipfile

import netCDF4
import numpy as np


_FILES = {
    "TMP-2m": ("t2m", "2t", 2.0),
    "APCP-sfc": ("rainfall", "param8.1.0", None),
    "UGRD-10m": ("wind_u", "10u", 10.0),
    "VGRD-10m": ("wind_v", "10v", 10.0),
}
_TIMESTAMP = re.compile(r"_(\d{10})_")
_HOURS = (0, 6, 12, 18)
_DEFAULT_BOUNDS = (5.0, 30.0, 75.0, 100.0)


@dataclass(frozen=True)
class IMDAAPeriod:
    times: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    fields: dict[str, np.ndarray]
    provenance: dict[str, Any]


def _expected_timestamps(start_date: str, end_date: str) -> list[datetime]:
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("IMDAA dates must use YYYY-MM-DD format") from exc
    if end < start:
        raise ValueError("IMDAA end date must be on or after start date")
    current = start
    timestamps = []
    while current <= end:
        timestamps.extend(current.replace(hour=hour) for hour in _HOURS)
        current += timedelta(days=1)
    return timestamps


def _index_archive_members(archive: zipfile.ZipFile, year: int) -> dict[tuple[str, str], str]:
    indexed: dict[tuple[str, str], str] = {}
    for name in archive.namelist():
        if not name.lower().endswith(".nc"):
            continue
        basename = Path(name).name
        match = _TIMESTAMP.search(basename)
        if not match:
            continue
        stamp = match.group(1)
        prefix = basename.split("_", 1)[0]
        if prefix not in _FILES or stamp[:4] != str(year):
            continue
        key = (prefix, stamp)
        if key in indexed:
            raise ValueError(f"duplicate IMDAA member for {prefix} timestamp {stamp}")
        indexed[key] = name
    return indexed


def _coordinate(dataset: netCDF4.Dataset, name: str, units: str) -> np.ndarray:
    if name not in dataset.variables:
        raise ValueError(f"IMDAA file is missing its {name} coordinate")
    coordinate = dataset.variables[name]
    if coordinate.ndim != 1 or coordinate.dimensions != (name,):
        raise ValueError(f"IMDAA {name} coordinate must be one-dimensional")
    actual_units = getattr(coordinate, "units", None)
    if actual_units != units:
        raise ValueError(f"IMDAA {name} coordinate units must be {units!r}, got {actual_units!r}")
    values = np.asarray(coordinate[:], dtype=np.float64)
    if not np.isfinite(values).all() or not np.all(np.diff(values) > 0):
        raise ValueError(f"IMDAA {name} coordinates must be finite, ascending, and unique")
    return values


def _crop_indices(
    latitude: np.ndarray,
    longitude: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    south, north, west, east = bounds
    if not (-90 <= south < north <= 90 and -180 <= west < east <= 360):
        raise ValueError("IMDAA bounds must be south,north,west,east within geographic limits")
    lat_indices = np.flatnonzero((latitude >= south) & (latitude <= north))
    lon_indices = np.flatnonzero((longitude >= west) & (longitude <= east))
    if lat_indices.size < 2 or lon_indices.size < 2:
        raise ValueError("IMDAA region must contain at least two coordinates per axis")
    return lat_indices, lon_indices


def _read_member(
    archive: zipfile.ZipFile,
    member_name: str,
    prefix: str,
    filename_time: datetime,
    bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    canonical, expected_variable, expected_height = _FILES[prefix]
    payload = archive.read(member_name)
    try:
        dataset = netCDF4.Dataset("imdaa-member.nc", mode="r", memory=payload)
    except Exception as exc:
        raise ValueError(f"invalid IMDAA NetCDF member {member_name}: {exc}") from exc

    with dataset:
        latitude = _coordinate(dataset, "lat", "degrees_north")
        longitude = _coordinate(dataset, "lon", "degrees_east")
        lat_indices, lon_indices = _crop_indices(latitude, longitude, bounds)

        if "time" not in dataset.variables:
            raise ValueError(f"IMDAA member {member_name} is missing the time coordinate")
        time_coordinate = dataset.variables["time"]
        raw_times = np.asarray(time_coordinate[:])
        if raw_times.ndim != 1 or raw_times.size == 0:
            raise ValueError(f"IMDAA time coordinate in {member_name} must be non-empty and one-dimensional")
        decoded_times = netCDF4.num2date(
            raw_times,
            time_coordinate.units,
            calendar=getattr(time_coordinate, "calendar", "standard"),
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )
        decoded_strings = [value.strftime("%Y-%m-%dT%H:%M:%S") for value in decoded_times]
        expected_decoded = filename_time + (timedelta(hours=1) if canonical == "rainfall" else timedelta())
        expected_string = expected_decoded.strftime("%Y-%m-%dT%H:%M:%S")
        if any(value != expected_string for value in decoded_strings):
            raise ValueError(
                f"IMDAA time mismatch in {member_name}: filename implies "
                f"{filename_time.isoformat()}, decoded values are {decoded_strings}"
            )

        if expected_variable not in dataset.variables:
            raise ValueError(
                f"IMDAA member {member_name} is missing expected variable {expected_variable!r}"
            )
        variable = dataset.variables[expected_variable]
        required_dims = ("time", "lat", "lon") if expected_height is None else (
            "time",
            "height",
            "lat",
            "lon",
        )
        if set(variable.dimensions) != set(required_dims) or len(variable.dimensions) != len(required_dims):
            raise ValueError(
                f"IMDAA variable {expected_variable} dimensions {variable.dimensions} "
                f"do not match {required_dims}"
            )
        values = np.ma.asarray(variable[:])
        permutation = [variable.dimensions.index(dim) for dim in required_dims]
        values = np.transpose(values, permutation)

        selected_height = None
        if expected_height is not None:
            if "height" not in dataset.variables:
                raise ValueError(f"IMDAA variable {expected_variable} has no height coordinate")
            height_coordinate = dataset.variables["height"]
            if getattr(height_coordinate, "units", None) != "m":
                raise ValueError(f"IMDAA height units must be 'm' for {expected_variable}")
            heights = np.asarray(height_coordinate[:], dtype=np.float64)
            matching = np.flatnonzero(np.isclose(heights, expected_height, rtol=0, atol=1e-6))
            if matching.size != 1 or heights.size != 1:
                raise ValueError(
                    f"IMDAA {expected_variable} must contain exactly one {expected_height:g} m level; "
                    f"found {heights.tolist()}"
                )
            selected_height = float(heights[matching[0]])
            values = values[:, int(matching[0]), :, :]
        if values.shape[0] != len(decoded_times):
            raise ValueError(f"IMDAA time dimension does not align with coordinate in {member_name}")

        record_values = values[0]
        if values.shape[0] > 1:
            if canonical not in {"wind_u", "wind_v"}:
                raise ValueError(f"unexpected duplicate records in {member_name}")
            first = np.ma.asarray(values[0])
            if any(not np.ma.allequal(first, np.ma.asarray(values[index])) for index in range(1, values.shape[0])):
                raise ValueError(f"non-identical duplicate wind records in {member_name}")
        record_values = np.ma.asarray(record_values)[lat_indices][:, lon_indices]
        masked = np.ma.getmaskarray(record_values)
        values_filled = np.asarray(record_values.filled(np.nan), dtype=np.float32)
        invalid = masked | ~np.isfinite(values_filled)
        fill_values = [
            getattr(variable, attribute)
            for attribute in ("_FillValue", "missing_value")
            if hasattr(variable, attribute)
        ]
        for fill_value in fill_values:
            invalid |= values_filled == np.float32(fill_value)
        if np.any(invalid):
            raise ValueError(
                f"IMDAA {canonical} contains {int(invalid.sum())} missing or non-finite values "
                f"in the selected region ({member_name})"
            )

        if canonical == "t2m" and np.any((values_filled < 180) | (values_filled > 350)):
            raise ValueError(f"IMDAA t2m contains values outside the broad physical QC range in {member_name}")
        if canonical in {"wind_u", "wind_v"} and np.any(np.abs(values_filled) > 150):
            raise ValueError(f"IMDAA {canonical} contains values outside the broad physical QC range")
        if canonical == "rainfall" and np.any(values_filled < 0):
            raise ValueError("IMDAA APCP has negative values; refusing to interpret the field")

        metadata = {
            "source_archive_member": member_name,
            "source_variable": expected_variable,
            "units": getattr(variable, "units", None),
            "parameter": getattr(variable, "param", None),
            "height_m": selected_height,
            "filename_time": filename_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "decoded_netcdf_times": decoded_strings,
            "decoded_minus_filename_hours": 1 if canonical == "rainfall" else 0,
            "duplicate_records": int(values.shape[0]),
            "identical_duplicate_records_removed_logically": max(0, int(values.shape[0]) - 1),
            "global_attributes": {
                name: str(getattr(dataset, name))
                for name in ("Conventions", "CDI", "CDO", "history")
                if name in dataset.ncattrs()
            },
        }
        return (
            values_filled,
            latitude[lat_indices],
            longitude[lon_indices],
            metadata,
            latitude,
            longitude,
        )


def load_imdaa_period(
    archive_dir: str | Path,
    start_date: str,
    end_date: str,
    *,
    bounds: tuple[float, float, float, float] = _DEFAULT_BOUNDS,
) -> IMDAAPeriod:
    """Load four canonical variables from ZIP members without extracting files."""
    root = Path(archive_dir)
    requested = _expected_timestamps(start_date, end_date)
    if not root.is_dir():
        raise FileNotFoundError(f"IMDAA archive directory does not exist: {root}")
    south, north, west, east = bounds
    expected_years = sorted({value.year for value in requested})
    all_rows: dict[str, dict[str, Any]] = {}
    grid_lat = grid_lon = None
    reference_full_latitude = reference_full_longitude = None

    for year in expected_years:
        archive_path = root / f"{year}.zip"
        if not archive_path.is_file():
            raise FileNotFoundError(f"IMDAA archive is missing for {year}: {archive_path}")
        try:
            archive = zipfile.ZipFile(archive_path)
        except (OSError, zipfile.BadZipFile) as exc:
            raise ValueError(f"invalid IMDAA ZIP archive {archive_path}: {exc}") from exc
        with archive:
            indexed = _index_archive_members(archive, year)
            for timestamp in (value for value in requested if value.year == year):
                stamp = timestamp.strftime("%Y%m%d%H")
                record: dict[str, Any] = {"fields": {}, "provenance": {}}
                for prefix, (canonical, _, _) in _FILES.items():
                    member_name = indexed.get((prefix, stamp))
                    if member_name is None:
                        raise FileNotFoundError(
                            f"IMDAA {prefix} file is missing for timestamp {stamp} in {archive_path.name}"
                        )
                    (
                        array,
                        latitude,
                        longitude,
                        source_meta,
                        full_latitude,
                        full_longitude,
                    ) = _read_member(
                        archive,
                        member_name,
                        prefix,
                        timestamp,
                        bounds,
                    )
                    if reference_full_latitude is None:
                        reference_full_latitude = full_latitude
                        reference_full_longitude = full_longitude
                        grid_lat, grid_lon = latitude, longitude
                    elif not (
                        np.array_equal(full_latitude, reference_full_latitude)
                        and np.array_equal(full_longitude, reference_full_longitude)
                    ):
                        raise ValueError(f"IMDAA grid mismatch in {archive_path.name}:{member_name}")
                    elif not (
                        np.array_equal(latitude, grid_lat) and np.array_equal(longitude, grid_lon)
                    ):
                        raise ValueError(f"IMDAA cropped grid mismatch in {archive_path.name}:{member_name}")
                    record["fields"][canonical] = array
                    record["provenance"][canonical] = source_meta
                all_rows[timestamp.strftime("%Y-%m-%dT%H:%M:%S")] = record

    ordered_times = sorted(all_rows)
    if len(ordered_times) != len(requested):
        raise ValueError("IMDAA period contains duplicate or missing valid timestamps")
    fields = {
        canonical: np.stack(
            [all_rows[time_value]["fields"][canonical] for time_value in ordered_times]
        ).astype(np.float32, copy=False)
        for canonical in ("t2m", "rainfall", "wind_u", "wind_v")
    }
    provenance = {
        "dataset": "IMDAA",
        "source": "local IMDAA reanalysis ZIP archives",
        "data_mode": "HISTORICAL_REPLAY",
        "requested_period": {"start": ordered_times[0], "end": ordered_times[-1]},
        "reference_years": expected_years,
        "region": {"south": south, "north": north, "west": west, "east": east},
        "canonical_variables": list(fields),
        "records": {time_value: all_rows[time_value]["provenance"] for time_value in ordered_times},
        "rainfall": {
            "status": "UNRESOLVED_SEMANTICS",
            "enabled_for_efi": False,
            "units": None,
            "source_variable": "param8.1.0",
            "parameter": "8.1.0",
            "decoded_time_offset_hours": 1,
            "interpretation": (
                "Raw APCP values are preserved. Units and accumulation interval are not verified; "
                "filename time is the canonical valid time and decoded NetCDF time is provenance only."
            ),
        },
        "quality_control": {
            "status": "PASS",
            "missing_or_nonfinite_values": 0,
            "coordinate_order": "ascending",
            "duplicate_coordinates": 0,
            "wind_duplicate_policy": "identical records verified and reduced to one logical record",
        },
    }
    return IMDAAPeriod(
        times=np.asarray(ordered_times, dtype="datetime64[s]"),
        latitude=np.asarray(grid_lat, dtype=np.float64),
        longitude=np.asarray(grid_lon, dtype=np.float64),
        fields=fields,
        provenance=provenance,
    )


def iter_imdaa_years(
    archive_dir: str | Path,
    years: Iterable[int],
    *,
    bounds: tuple[float, float, float, float] = _DEFAULT_BOUNDS,
) -> Iterator[IMDAAPeriod]:
    """Yield one May at a time, keeping the reference build's memory bounded."""
    for year in years:
        if year >= 2020:
            raise ValueError("IMDAA reference years must exclude 2020 and later")
        yield load_imdaa_period(
            archive_dir,
            f"{year}-05-01",
            f"{year}-05-31",
            bounds=bounds,
        )
