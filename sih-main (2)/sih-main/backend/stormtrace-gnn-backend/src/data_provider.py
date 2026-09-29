from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from config import settings


_VARIABLE_ALIASES = {
    "t2m": ("t2m", "2m_temperature", "temperature_2m"),
    "rainfall": ("rainfall", "tp", "total_precipitation", "precipitation"),
    "wind_u": ("wind_u", "u10", "10m_u_component_of_wind", "u"),
    "wind_v": ("wind_v", "v10", "10m_v_component_of_wind", "v"),
    "q": ("q", "specific_humidity"),
}


@dataclass(frozen=True)
class CanonicalWeatherFields:
    ensemble: np.ndarray
    climatology: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray
    variables: Tuple[str, ...]
    units: Tuple[Optional[str], ...]
    provenance: Dict[str, Any]


def _coordinate(dataset, aliases, label):
    for name in aliases:
        if name in dataset.coords:
            coordinate = dataset.coords[name]
            values = np.asarray(coordinate.values)
            if values.ndim != 1:
                raise ValueError(f"{label} coordinate must be one-dimensional")
            return name, values
    raise ValueError(f"dataset is missing a {label} coordinate ({', '.join(aliases)})")


def _dimension(dataset, aliases, label):
    for name in aliases:
        if name in dataset.dims:
            return name
    raise ValueError(f"dataset is missing the {label} dimension ({', '.join(aliases)})")


def _crop_domain(values, latitude, longitude):
    if not settings.data_domain:
        return values, latitude, longitude
    try:
        south, north, west, east = (
            float(value.strip()) for value in settings.data_domain.split(",")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("DATA_DOMAIN must be south,north,west,east in degrees") from exc
    if south >= north or south < -90 or north > 90:
        raise ValueError("DATA_DOMAIN latitude bounds must satisfy -90 <= south < north <= 90")
    if west >= east or west < -180 or east > 360:
        raise ValueError("DATA_DOMAIN longitude bounds must be ordered and within [-180, 360]")

    normalized_longitudes = np.mod(longitude, 360.0)
    west_360, east_360 = np.mod([west, east], 360.0)
    if west_360 >= east_360:
        raise ValueError("DATA_DOMAIN ranges crossing the antimeridian are not supported")
    latitude_indices = np.flatnonzero((latitude >= south) & (latitude <= north))
    longitude_indices = np.flatnonzero(
        (normalized_longitudes >= west_360) & (normalized_longitudes <= east_360)
    )
    if latitude_indices.size < 2 or longitude_indices.size < 2:
        raise ValueError("DATA_DOMAIN selects fewer than two grid coordinates on an axis")
    values = np.take(np.take(values, latitude_indices, axis=-2), longitude_indices, axis=-1)
    return values, latitude[latitude_indices], longitude[longitude_indices]


def _read_local_dataset(path: str, field_kind: str):
    if not path:
        raise RuntimeError(f"A local path must be configured for the {field_kind} dataset")
    if "://" in path:
        raise ValueError("Only local dataset paths are supported; automatic downloads are disabled")
    file_path = Path(path).expanduser()
    if not file_path.exists() or not (file_path.is_file() or file_path.is_dir()):
        raise FileNotFoundError(f"Configured {field_kind} dataset does not exist: {file_path}")

    if file_path.suffix.lower() == ".npz":
        with np.load(file_path, allow_pickle=False) as archive:
            key = next(
                (candidate for candidate in (("ensemble", "forecast") if field_kind == "forecast" else ("climatology", "baseline"))
                 if candidate in archive),
                None,
            )
            if key is None:
                raise ValueError(f"{file_path} must contain an '{field_kind}' array")
            values = np.asarray(archive[key])
            if "latitude" not in archive or "longitude" not in archive:
                raise ValueError(f"{file_path} must contain latitude and longitude coordinates")
            members = None
            if field_kind == "forecast":
                if values.ndim != 5:
                    raise ValueError(
                        "forecast NPZ array must have shape (steps, members, variables, latitude, longitude)"
                    )
                members = np.asarray(
                    archive["ensemble_members"] if "ensemble_members" in archive else np.arange(values.shape[1])
                )
                if members.shape != (values.shape[1],):
                    raise ValueError("ensemble_members must contain one identifier per ensemble member")
                if settings.ensemble_member is not None:
                    matches = np.flatnonzero(members.astype(str) == str(settings.ensemble_member))
                    if matches.size != 1:
                        raise ValueError(
                            f"configured ensemble member {settings.ensemble_member!r} is not present exactly once"
                        )
                    values = values[:, matches, ...]
                    members = members[matches]
            latitude = np.asarray(archive["latitude"], dtype=np.float64)
            longitude = np.asarray(archive["longitude"], dtype=np.float64)
            values, latitude, longitude = _crop_domain(values, latitude, longitude)
            metadata = {
                "latitude": latitude,
                "longitude": longitude,
                "variable_units": tuple(None for _ in settings.variable_names),
                "time": None,
                "lead_time": None,
                "valid_time": None,
                "initialization_time": settings.forecast_initialization_time,
                "ensemble_members": None if members is None else tuple(str(value) for value in members),
                "uri": str(file_path),
            }
        return values, metadata

    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError("Reading NetCDF/GRIB files requires the xarray dependency") from exc

    engine = "cfgrib" if file_path.suffix.lower() in (".grib", ".grb", ".grb2", ".grib2") else None
    if file_path.is_dir() and file_path.suffix.lower() != ".zarr":
        raise ValueError(f"Unsupported dataset directory: {file_path}; Zarr paths must end with .zarr")
    if file_path.suffix.lower() == ".zarr":
        try:
            dataset_context = xr.open_zarr(file_path)
        except ImportError as exc:
            raise RuntimeError("Reading Zarr datasets requires the zarr dependency") from exc
    else:
        open_options = {"engine": engine} if engine else {}
        dataset_context = xr.open_dataset(file_path, **open_options)
    with dataset_context as dataset:
        initialization_time = settings.forecast_initialization_time
        if field_kind == "forecast":
            sample_candidates = ("step", "forecast_step", "lead_time", "valid_time", "time")
            selected_sample = next((name for name in sample_candidates if name in dataset.dims), None)
            if selected_sample in ("step", "forecast_step", "lead_time"):
                for init_dim in ("forecast_reference_time", "time"):
                    if init_dim not in dataset.dims or init_dim == selected_sample:
                        continue
                    if initialization_time is not None:
                        try:
                            dataset = dataset.sel({init_dim: initialization_time})
                        except (KeyError, ValueError) as exc:
                            raise ValueError(
                                f"configured initialization time {initialization_time!r} is not present"
                            ) from exc
                    elif dataset.sizes[init_dim] == 1:
                        initialization_time = str(np.asarray(dataset[init_dim].values[0]))
                        dataset = dataset.isel({init_dim: 0})
                    else:
                        raise ValueError(
                            f"forecast has multiple {init_dim} values; configure FORECAST_INITIALIZATION_TIME"
                        )

        lat_name, latitudes = _coordinate(dataset, ("latitude", "lat"), "latitude")
        lon_name, longitudes = _coordinate(dataset, ("longitude", "lon"), "longitude")
        lat_dim, lon_dim = dataset[lat_name].dims[0], dataset[lon_name].dims[0]
        configured_source_names = tuple(
            name.strip() for name in (settings.data_variables or "").split(",") if name.strip()
        )
        if configured_source_names and len(configured_source_names) != len(settings.variable_names):
            raise ValueError("DATA_VARIABLES must contain one source variable per configured model variable")

        if field_kind == "forecast":
            sample_dim = _dimension(dataset, ("step", "forecast_step", "lead_time", "time"), "forecast-step")
            member_dim = next(
                (name for name in ("number", "ensemble_member", "member", "realization") if name in dataset.dims),
                None,
            )
        else:
            sample_dim = _dimension(dataset, ("year", "time", "valid_time", "sample"), "climatology-sample")
            member_dim = None

        arrays = []
        units = []
        source_variable_names = []
        expected_dims = {sample_dim, lat_dim, lon_dim}
        if member_dim:
            expected_dims.add(member_dim)
        explicit_source_names = (
            tuple(name.strip() for name in settings.data_variables.split(",") if name.strip())
            if settings.data_variables
            else None
        )
        if explicit_source_names and len(explicit_source_names) != len(settings.variable_names):
            raise ValueError("DATA_VARIABLES must contain one source variable per configured model variable")
        for variable_index, canonical_name in enumerate(settings.variable_names):
            if explicit_source_names:
                candidates = (explicit_source_names[variable_index],)
            else:
                candidates = _VARIABLE_ALIASES.get(canonical_name, (canonical_name,))
            source_name = next((name for name in candidates if name in dataset.data_vars), None)
            if source_name is None:
                expected = ", ".join(candidates)
                raise ValueError(
                    f"DATA_UNAVAILABLE: required canonical variable '{canonical_name}' is missing "
                    f"(accepted source names: {expected})"
                )
            if source_name not in dataset.data_vars:
                raise ValueError(f"dataset is missing configured variable '{source_name}'")
            field = dataset[source_name]
            if set(field.dims) != expected_dims:
                raise ValueError(
                    f"variable '{source_name}' has dimensions {field.dims}; expected {tuple(expected_dims)}"
                )
            order = (sample_dim, member_dim, lat_dim, lon_dim) if member_dim else (sample_dim, lat_dim, lon_dim)
            values = np.asarray(field.transpose(*order).values)
            if not np.isfinite(values).all():
                raise ValueError(f"variable '{source_name}' contains non-finite values")
            arrays.append(values)
            units.append(field.attrs.get("units"))
            source_variable_names.append(source_name)

        sample_values = np.asarray(dataset[sample_dim].values) if sample_dim in dataset.coords else None
        sample_indices = (
            np.arange(sample_values.size, dtype=np.int64) if sample_values is not None else None
        )
        if field_kind == "forecast" and settings.forecast_lead_hours is not None:
            if sample_dim not in ("step", "forecast_step", "lead_time") or sample_values is None:
                raise ValueError("FORECAST_LEAD_HOURS requires a forecast step/lead_time coordinate")
            if np.issubdtype(sample_values.dtype, np.timedelta64):
                lead_values = sample_values / np.timedelta64(1, "h")
            else:
                try:
                    lead_values = sample_values.astype(np.float64)
                except (TypeError, ValueError) as exc:
                    raise ValueError("forecast lead coordinates must be numeric hours or timedeltas") from exc
            lead_indices = np.flatnonzero(np.isclose(lead_values, settings.forecast_lead_hours))
            if lead_indices.size != 1:
                raise ValueError(
                    f"configured lead time {settings.forecast_lead_hours} hours is not present exactly once"
                )
            arrays = [array[lead_indices] for array in arrays]
            sample_values = sample_values[lead_indices]
            sample_indices = lead_indices

        if field_kind == "forecast" and (settings.forecast_start_time or settings.forecast_end_time):
            if sample_values is None:
                raise ValueError("forecast time range filtering requires a time coordinate")
            range_values = sample_values
            if sample_dim in ("step", "forecast_step", "lead_time") and "valid_time" in dataset.coords:
                candidate_times = np.squeeze(np.asarray(dataset.coords["valid_time"].values))
                if (
                    candidate_times.ndim != 1
                    or sample_indices is None
                    or candidate_times.size <= int(sample_indices.max())
                ):
                    raise ValueError("valid_time coordinate must align with forecast steps for time filtering")
                range_values = candidate_times[sample_indices]
            try:
                time_values = range_values.astype("datetime64[ns]")
                start = (
                    np.datetime64(settings.forecast_start_time, "ns")
                    if settings.forecast_start_time
                    else None
                )
                end = (
                    np.datetime64(settings.forecast_end_time, "ns")
                    if settings.forecast_end_time
                    else None
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "FORECAST_START_TIME/FORECAST_END_TIME require datetime-valued valid-time coordinates"
                ) from exc
            time_mask = np.ones(sample_values.shape, dtype=bool)
            if start is not None:
                time_mask &= time_values >= start
            if end is not None:
                time_mask &= time_values <= end
            selected_indices = np.flatnonzero(time_mask)
            if selected_indices.size == 0:
                raise ValueError("configured forecast time range selects no dataset samples")
            arrays = [array[selected_indices] for array in arrays]
            sample_values = sample_values[selected_indices]
            sample_indices = sample_indices[selected_indices]

        if member_dim:
            stacked = np.stack(arrays, axis=2)
            members = np.asarray(dataset[member_dim].values)
            if settings.ensemble_member is not None:
                matches = np.flatnonzero(members.astype(str) == str(settings.ensemble_member))
                if matches.size != 1:
                    raise ValueError(
                        f"configured ensemble member {settings.ensemble_member!r} is not present exactly once"
                    )
                stacked = stacked[:, matches, ...]
                members = members[matches]
        else:
            stacked = np.stack(arrays, axis=1)
            members = None

        if sample_values is not None and np.unique(sample_values).size != sample_values.size:
            raise ValueError(f"{field_kind} dataset contains duplicate {sample_dim} timestamps")
        stacked, latitudes, longitudes = _crop_domain(
            stacked, np.asarray(latitudes, dtype=np.float64), np.asarray(longitudes, dtype=np.float64)
        )
        valid_time = None
        if field_kind == "forecast" and "valid_time" in dataset.coords:
            valid_time_values = np.squeeze(np.asarray(dataset.coords["valid_time"].values))
            if (
                valid_time_values.ndim == 1
                and sample_indices is not None
                and sample_indices.size
                and valid_time_values.size > int(sample_indices.max())
            ):
                valid_time = tuple(str(value) for value in valid_time_values[sample_indices])
            else:
                valid_time = None
        elif field_kind == "forecast" and sample_dim == "time" and sample_values is not None:
            valid_time = tuple(str(value) for value in sample_values)
        metadata = {
            "latitude": np.asarray(latitudes, dtype=np.float64),
            "longitude": np.asarray(longitudes, dtype=np.float64),
            "variable_units": tuple(units),
            "time": sample_values,
            "lead_time": (
                tuple(float(value) for value in lead_values[lead_indices])
                if field_kind == "forecast" and settings.forecast_lead_hours is not None
                else (
                    tuple(float(value) for value in sample_values)
                    if field_kind == "forecast" and sample_dim in ("step", "forecast_step", "lead_time")
                    and sample_values is not None
                    else None
                )
            ),
            "valid_time": valid_time,
            "ensemble_members": None if members is None else tuple(str(value) for value in members),
            "variables": tuple(settings.variable_names),
            "source_variables": tuple(source_variable_names),
            "initialization_time": dataset.attrs.get("forecast_reference_time", initialization_time),
            "uri": str(file_path),
        }
    sidecar = file_path.with_suffix(file_path.suffix + ".json")
    if sidecar.is_file():
        try:
            metadata["download_provenance"] = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"ERA5 download provenance sidecar is invalid: {sidecar}") from exc
    return stacked, metadata


class WeatherDataProvider(ABC):
    provider_name: str = "unknown"
    label: str = "unknown-source"
    description: str = "Provider metadata is not configured."
    requires_credentials: bool = False
    source_type: str = "unknown"

    def __init__(self):
        self._coordinates = None
        self._field_metadata: Dict[str, Any] = {}

    @property
    def metadata(self) -> Dict[str, Any]:
        metadata = {
            "provider": self.provider_name,
            "label": self.label,
            "description": self.description,
            "requires_credentials": self.requires_credentials,
            "source_type": self.source_type,
            "dataset_name": settings.source_dataset_name or self.provider_name,
            "dataset_version": None,
            "uri": None,
            "initialization_time": settings.forecast_initialization_time,
            "valid_time": None,
            "lead_time": None,
            "spatial_resolution_km": settings.data_spatial_resolution_km,
            "variables": tuple(settings.variable_names),
            "ensemble_members": None,
            "processing_version": "1",
        }
        metadata.update(self._field_metadata)
        return metadata

    def grid_coordinates(self, grid_size: int):
        grid_shape = grid_size if isinstance(grid_size, tuple) else (grid_size, grid_size)
        if self._coordinates is not None:
            latitude, longitude = self._coordinates
            if latitude.shape != (grid_shape[0],) or longitude.shape != (grid_shape[1],):
                raise ValueError("dataset coordinates do not match the configured grid dimensions")
            return latitude.copy(), longitude.copy()
        if self.source_type != "synthetic":
            raise ValueError("the selected data source did not provide geographic coordinates")
        return (
            np.linspace(-90.0, 90.0, grid_shape[0]),
            np.linspace(-180.0, 180.0, grid_shape[1], endpoint=False),
        )

    @abstractmethod
    def generate_ensemble(
        self,
        n_steps: Optional[int] = None,
        n_members: Optional[int] = None,
        grid_size: Optional[int] = None,
        seed: int = 0,
    ):
        raise NotImplementedError

    @abstractmethod
    def generate_baseline(
        self,
        grid_size: Optional[int] = None,
        n_years: Optional[int] = None,
        seed: int = 1,
    ):
        raise NotImplementedError


def _require_real_mode(provider_name: str) -> None:
    if settings.data_mode.strip().upper() == "SYNTHETIC_DEMO":
        raise RuntimeError(
            f"{provider_name} is a real-data provider and cannot run in SYNTHETIC_DEMO mode"
        )


class PublicWeatherProvider(WeatherDataProvider):
    provider_name = "public_weather"
    label = "synthetic-demo"
    description = "Synthetic demonstration data; no real public forecast feed is connected."
    requires_credentials = False
    source_type = "synthetic"

    def generate_ensemble(
        self,
        n_steps: Optional[int] = None,
        n_members: Optional[int] = None,
        grid_size: Optional[int] = None,
        seed: int = 0,
    ):
        from src.data_gen import generate_synthetic_ensemble

        return generate_synthetic_ensemble(
            n_steps=n_steps,
            n_members=n_members,
            grid_size=grid_size,
            seed=seed,
        )

    def generate_baseline(
        self,
        grid_size: Optional[int] = None,
        n_years: Optional[int] = None,
        seed: int = 1,
    ):
        from src.data_gen import generate_baseline_climatology

        return generate_baseline_climatology(
            grid_size=grid_size,
            n_years=n_years,
            seed=seed,
        )


class LocalFixtureProvider(PublicWeatherProvider):
    provider_name = "local_fixture"
    label = "local-fixture"
    description = "Cached local fixture for offline development and smoke testing."


class SyntheticDemoProvider(PublicWeatherProvider):
    provider_name = "synthetic_demo"
    label = "synthetic-demo"
    description = "Synthetic demo mode used for rapid local development and demos."


_TIGGE_VARIABLES = {
    "t2m": ("t2m", "K", "instant"),
    "rainfall": ("tp", "m", "accum"),
    "wind_u": ("u10", "m s**-1", "instant"),
    "wind_v": ("v10", "m s**-1", "instant"),
}
_TIGGE_PROCESSING_VERSION = "ncmrwf-tigge-grib-v1"


def _normalized_grib_units(units: Any) -> str:
    return "".join(str(units or "").lower().replace("metres", "m").replace("meters", "m").split())


def _validate_tigge_variable_units(name: str, units: Any) -> None:
    normalized = _normalized_grib_units(units)
    accepted = {
        "t2m": {"k", "kelvin"},
        "tp": {"kgm**-2", "kgm-2", "kgm^-2"},
        "u10": {"ms**-1", "ms-1", "ms^-1"},
        "v10": {"ms**-1", "ms-1", "ms^-1"},
    }[name]
    if normalized not in accepted:
        raise ValueError(
            f"NCMRWF TIGGE {name} has unsupported or missing units {units!r}; "
            f"expected {sorted(accepted)}"
        )


def _tigge_step_hours(step_values: np.ndarray) -> np.ndarray:
    values = np.asarray(step_values)
    if np.issubdtype(values.dtype, np.timedelta64):
        hours = values / np.timedelta64(1, "h")
    else:
        hours = values.astype(np.float64)
    hours = np.asarray(hours, dtype=np.float64)
    if hours.ndim != 1 or not np.isfinite(hours).all() or np.any(hours < 0):
        raise ValueError("NCMRWF TIGGE step coordinate must contain finite non-negative lead times")
    if np.unique(hours).size != hours.size or np.any(np.diff(hours) <= 0):
        raise ValueError("NCMRWF TIGGE lead times must be strictly increasing and unique")
    return hours


def _tigge_datetime_values(values: Any, label: str) -> np.ndarray:
    times = np.asarray(values).astype("datetime64[s]")
    if times.ndim != 1 or times.size == 0 or np.isnat(times).any():
        raise ValueError(f"NCMRWF TIGGE {label} must be a one-dimensional datetime coordinate")
    return times


def _read_ncmrwf_tigge_grib(file_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import cfgrib
    except ImportError as exc:
        raise RuntimeError("Reading the NCMRWF TIGGE GRIB requires the cfgrib dependency") from exc

    try:
        datasets = cfgrib.open_datasets(
            str(file_path),
            backend_kwargs={"indexpath": ""},
        )
    except Exception as exc:
        raise ValueError(f"Unable to open NCMRWF TIGGE GRIB groups from {file_path}: {exc}") from exc
    if not datasets:
        raise ValueError(f"NCMRWF TIGGE GRIB contains no cfgrib groups: {file_path}")

    try:
        variable_groups: dict[str, Any] = {}
        for dataset in datasets:
            for variable_name in _TIGGE_VARIABLES.values():
                name = variable_name[0]
                if name in dataset.data_vars:
                    if name in variable_groups:
                        raise ValueError(
                            f"NCMRWF TIGGE GRIB contains ambiguous duplicate variable {name!r}"
                        )
                    variable_groups[name] = dataset
        missing = sorted(
            spec[0] for spec in _TIGGE_VARIABLES.values() if spec[0] not in variable_groups
        )
        if missing:
            raise ValueError(
                "NCMRWF TIGGE GRIB is missing required variables: " + ", ".join(missing)
            )

        first_dataset = variable_groups["t2m"]
        required_centre = str(first_dataset.attrs.get("GRIB_centre", "")).lower()
        data_types = {
            str(
                variable_groups[source_name][source_name].attrs.get(
                    "GRIB_dataType",
                    variable_groups[source_name].attrs.get("GRIB_dataType", ""),
                )
            ).lower()
            for source_name, _, _ in _TIGGE_VARIABLES.values()
        }
        if len(data_types) != 1:
            raise ValueError(
                "NCMRWF TIGGE variables must share one GRIB data type; "
                f"got {sorted(data_types)}"
            )
        data_type = data_types.pop()
        edition = int(first_dataset.attrs.get("GRIB_edition", 0))
        if required_centre != "dems":
            raise ValueError(
                f"NCMRWF TIGGE source must have centre 'dems', got {required_centre!r}"
            )
        if data_type != "pf":
            raise ValueError(f"NCMRWF TIGGE source must be perturbed forecast 'pf', got {data_type!r}")
        if edition != 2:
            raise ValueError(f"NCMRWF TIGGE source must be GRIB edition 2, got {edition}")

        latitude = np.asarray(first_dataset["latitude"].values, dtype=np.float64)
        longitude = np.asarray(first_dataset["longitude"].values, dtype=np.float64)
        if (
            latitude.ndim != 1
            or longitude.ndim != 1
            or latitude.size < 2
            or longitude.size < 2
            or not np.isfinite(latitude).all()
            or not np.isfinite(longitude).all()
        ):
            raise ValueError("NCMRWF TIGGE latitude/longitude must be finite one-dimensional coordinates")
        for axis_name, values in (("latitude", latitude), ("longitude", longitude)):
            differences = np.diff(values)
            if not (np.all(differences > 0) or np.all(differences < 0)):
                raise ValueError(f"NCMRWF TIGGE {axis_name} must be strictly monotonic and unique")

        reference_time = first_dataset.coords.get("time")
        if reference_time is None:
            raise ValueError("NCMRWF TIGGE groups are missing the forecast initialization time")
        initialization_values = np.asarray(reference_time.values).reshape(-1)
        if initialization_values.size != 1:
            raise ValueError("NCMRWF TIGGE file must contain exactly one forecast initialization")
        initialization_time = initialization_values[0].astype("datetime64[s]")
        if np.isnat(initialization_time):
            raise ValueError("NCMRWF TIGGE initialization time is invalid")
        if settings.forecast_initialization_time is not None:
            requested_initialization = np.datetime64(settings.forecast_initialization_time, "s")
            if requested_initialization != initialization_time:
                raise ValueError(
                    f"configured initialization {settings.forecast_initialization_time!r} "
                    f"does not match the GRIB initialization {str(initialization_time)}"
                )

        step_values = np.asarray(first_dataset.coords["step"].values)
        lead_hours = _tigge_step_hours(step_values)
        valid_time_coordinate = first_dataset.coords.get("valid_time")
        if valid_time_coordinate is None:
            raise ValueError("NCMRWF TIGGE groups are missing actual valid_time coordinates")
        valid_time_values = np.asarray(valid_time_coordinate.values)
        if valid_time_values.ndim > 1:
            flattened = valid_time_values.reshape(-1, valid_time_values.shape[-1])
            if not np.all(flattened == flattened[0]):
                raise ValueError("NCMRWF TIGGE valid_time differs across ensemble members")
            valid_time_values = flattened[0]
        valid_times = _tigge_datetime_values(valid_time_values, "valid_time")
        if valid_times.size != lead_hours.size:
            raise ValueError("NCMRWF TIGGE valid_time coordinate does not align with lead times")

        member_values = np.asarray(first_dataset.coords["number"].values)
        if member_values.ndim != 1 or member_values.size == 0 or np.unique(member_values).size != member_values.size:
            raise ValueError("NCMRWF TIGGE number coordinate must contain unique ensemble members")
        if settings.ensemble_member is not None:
            member_indices = np.flatnonzero(member_values.astype(str) == str(settings.ensemble_member))
            if member_indices.size != 1:
                raise ValueError(
                    f"configured ensemble member {settings.ensemble_member!r} is not present exactly once"
                )
        else:
            member_indices = np.arange(member_values.size)

        for variable_name, dataset in variable_groups.items():
            if not np.array_equal(latitude, np.asarray(dataset["latitude"].values)) or not np.array_equal(
                longitude, np.asarray(dataset["longitude"].values)
            ):
                raise ValueError("NCMRWF TIGGE variable groups do not share identical coordinates")
            if not np.array_equal(step_values, np.asarray(dataset.coords["step"].values)):
                raise ValueError("NCMRWF TIGGE variable groups do not share identical lead times")
            group_members = np.asarray(dataset.coords["number"].values)
            if not np.array_equal(member_values, group_members):
                raise ValueError("NCMRWF TIGGE variable groups do not share identical ensemble members")
            group_valid = np.asarray(dataset.coords["valid_time"].values)
            if group_valid.ndim > 1:
                group_valid = group_valid.reshape(-1, group_valid.shape[-1])[0]
            if not np.array_equal(valid_times, _tigge_datetime_values(group_valid, "valid_time")):
                raise ValueError("NCMRWF TIGGE variable groups do not share identical valid times")

        configured_variables = tuple(settings.variable_names)
        expected_variables = ("t2m", "rainfall", "wind_u", "wind_v")
        if len(configured_variables) != len(expected_variables) or set(configured_variables) != set(expected_variables):
            raise ValueError(
                "NCMRWF TIGGE adapter requires canonical variables t2m,rainfall,wind_u,wind_v"
            )
        output = np.empty(
            (
                lead_hours.size,
                member_indices.size,
                len(configured_variables),
                latitude.size,
                longitude.size,
            ),
            dtype=np.float32,
        )
        units: list[Optional[str]] = [None] * len(configured_variables)
        source_variables: list[Optional[str]] = [None] * len(configured_variables)
        variable_semantics: dict[str, str] = {}
        for canonical_name, (source_name, canonical_unit, expected_step_type) in _TIGGE_VARIABLES.items():
            dataset = variable_groups[source_name]
            field = dataset[source_name]
            if set(field.dims) != {"number", "step", "latitude", "longitude"} or len(field.dims) != 4:
                raise ValueError(
                    f"NCMRWF TIGGE {source_name} dimensions {field.dims} do not match "
                    "(number, step, latitude, longitude)"
                )
            actual_step_type = str(field.attrs.get("GRIB_stepType", "")).lower()
            if actual_step_type != expected_step_type:
                raise ValueError(
                    f"NCMRWF TIGGE {source_name} stepType must be {expected_step_type!r}, "
                    f"got {actual_step_type!r}"
                )
            raw_units = field.attrs.get("units")
            _validate_tigge_variable_units(source_name, raw_units)
            variable_index = configured_variables.index(canonical_name)
            ordered = field.transpose("step", "number", "latitude", "longitude")
            selected = ordered.isel(number=member_indices)
            target = output[:, :, variable_index, :, :]
            raw_values = np.asarray(selected.values)
            if not np.isfinite(raw_values).all():
                raise ValueError(f"NCMRWF TIGGE {source_name} contains missing or non-finite values")
            if source_name == "tp":
                # 1 kg m-2 of liquid-water equivalent is 0.001 m accumulated depth.
                np.multiply(raw_values, np.float32(0.001), out=target, casting="unsafe")
                variable_semantics[canonical_name] = (
                    "accumulated precipitation amount per GRIB forecast accumulation interval; "
                    "converted from kg m-2 to m water-equivalent depth; not a rate"
                )
            else:
                np.copyto(target, raw_values, casting="unsafe")
                variable_semantics[canonical_name] = "instantaneous field"
            units[variable_index] = canonical_unit
            source_variables[variable_index] = source_name

        step_indices = np.arange(lead_hours.size)
        if settings.forecast_lead_hours is not None:
            matches = np.flatnonzero(np.isclose(lead_hours, settings.forecast_lead_hours))
            if matches.size != 1:
                raise ValueError(
                    f"configured lead time {settings.forecast_lead_hours} hours is not present exactly once"
                )
            step_indices = matches
        if settings.forecast_start_time or settings.forecast_end_time:
            start = (
                np.datetime64(settings.forecast_start_time, "s")
                if settings.forecast_start_time
                else np.datetime64("1678-01-01", "s")
            )
            end = (
                np.datetime64(settings.forecast_end_time, "s")
                if settings.forecast_end_time
                else np.datetime64("2261-12-31", "s")
            )
            step_indices = step_indices[
                (valid_times[step_indices] >= start) & (valid_times[step_indices] <= end)
            ]
            if step_indices.size == 0:
                raise ValueError("configured forecast valid-time range selects no TIGGE leads")
        output = output[step_indices]
        valid_times = valid_times[step_indices]
        lead_hours = lead_hours[step_indices]

        output, latitude, longitude = _crop_domain(output, latitude, longitude)
        grid_resolution = {
            "latitude_degrees": float(np.median(np.abs(np.diff(latitude)))),
            "longitude_degrees": float(np.median(np.abs(np.diff(longitude)))),
        }
        metadata = {
            "uri": str(file_path.resolve()),
            "source": "NCMRWF TIGGE",
            "centre": "dems",
            "centre_description": str(first_dataset.attrs.get("GRIB_centreDescription", "New Delhi")),
            "data_type": data_type,
            "grib_edition": edition,
            "initialization_time": str(initialization_time),
            "lead_time": tuple(float(value) for value in lead_hours),
            "valid_time": tuple(str(value) for value in valid_times),
            "ensemble_members": tuple(str(value) for value in member_values[member_indices]),
            "variables": configured_variables,
            "source_variables": tuple(source_variables),
            "variable_units": tuple(units),
            "variable_semantics": variable_semantics,
            "precipitation_conversion": {
                "source_units": "kg m**-2",
                "canonical_units": "m",
                "factor": 0.001,
                "meaning": "accumulated water-equivalent depth conversion; no rate conversion",
            },
            "region": {
                "latitude_min": float(latitude.min()),
                "latitude_max": float(latitude.max()),
                "longitude_min": float(longitude.min()),
                "longitude_max": float(longitude.max()),
            },
            "grid_resolution_degrees": grid_resolution,
            "grid_dimensions": [int(latitude.size), int(longitude.size)],
            "ensemble_count": int(member_indices.size),
            "lead_time_range_hours": [float(lead_hours.min()), float(lead_hours.max())],
            "processing_version": _TIGGE_PROCESSING_VERSION,
            "data_mode": settings.data_mode.upper(),
        }
        return output, {
            **metadata,
            "latitude": latitude,
            "longitude": longitude,
        }
    finally:
        for dataset in datasets:
            dataset.close()


class NepsGProvider(WeatherDataProvider):
    provider_name = "neps_g"
    label = "neps-g"
    description = "Reads locally configured NEPS-G ensemble files; no data is downloaded."
    source_type = "neps_g"

    def generate_ensemble(
        self,
        n_steps: Optional[int] = None,
        n_members: Optional[int] = None,
        grid_size: Optional[int] = None,
        seed: int = 0,
    ):
        _require_real_mode("NEPS-G")
        if not settings.neps_g_forecast_path:
            raise RuntimeError("NEPS-G: CONFIGURED BUT DATA UNAVAILABLE (local path required)")
        file_path = Path(settings.neps_g_forecast_path).expanduser()
        if file_path.suffix.lower() in {".grib", ".grb", ".grib2", ".grb2"}:
            values, metadata = _read_ncmrwf_tigge_grib(file_path)
        else:
            values, metadata = _read_local_dataset(settings.neps_g_forecast_path, "forecast")
        if values.ndim != 5:
            raise ValueError("NEPS-G forecast must have shape (steps, members, variables, latitude, longitude)")
        self._coordinates = (metadata["latitude"], metadata["longitude"])
        self._field_metadata = {
            **{key: value for key, value in metadata.items() if key not in {"latitude", "longitude"}},
            "variables": tuple(settings.variable_names),
            "source_variables": metadata.get("source_variables"),
            "variable_units": metadata["variable_units"],
            "ensemble_members": metadata["ensemble_members"],
            "valid_time": metadata["valid_time"],
            "lead_time": metadata["lead_time"],
            "initialization_time": metadata["initialization_time"],
            "data_mode": settings.data_mode,
            "source": metadata.get("source", "NEPS-G local file"),
        }
        return values

    def generate_baseline(
        self,
        grid_size: Optional[int] = None,
        n_years: Optional[int] = None,
        seed: int = 1,
    ):
        _require_real_mode("NEPS-G")
        from src.era5_cds import cds_configured
        if not settings.era5_climatology_path and not cds_configured():
            raise RuntimeError("ERA5: CONFIGURED BUT DATA UNAVAILABLE")
        reference_provider = Era5Provider()
        climatology = reference_provider.generate_baseline(grid_size, n_years, seed)
        reference_latitude, reference_longitude = reference_provider.grid_coordinates(climatology.shape[-2:])
        if self._coordinates is None:
            self._coordinates = (reference_latitude, reference_longitude)
        elif not (
            np.allclose(self._coordinates[0], reference_latitude, atol=1e-6)
            and np.allclose(self._coordinates[1], reference_longitude, atol=1e-6)
        ):
            raise ValueError("NEPS-G forecast and ERA5 climatology grids must match before EFI calculation")

        forecast_units = self._field_metadata.get("variable_units")
        reference_units = reference_provider.metadata.get("variable_units")
        if forecast_units and reference_units and all(forecast_units) and all(reference_units):
            if tuple(forecast_units) != tuple(reference_units):
                raise ValueError("NEPS-G forecast and ERA5 climatology units must match before EFI calculation")
        self._field_metadata.update(
            {
                "reference_source": "era5",
                "reference_uri": reference_provider.metadata.get("uri"),
                "reference_variable_units": reference_units,
                "reference_time": reference_provider.metadata.get("climatology_time"),
                "reference_download_provenance": reference_provider.metadata.get("download_provenance"),
            }
        )
        return climatology


class NCMRWFTIGGEProvider(NepsGProvider):
    """NCMRWF TIGGE adapter using grouped cfgrib fields through the NEPS-G contract."""


class Era5Provider(WeatherDataProvider):
    provider_name = "era5"
    label = "era5-climatology"
    description = "Reads a locally configured ERA5 climatology/reference dataset; no data is downloaded unless configured to do so."
    source_type = "era5"

    def generate_ensemble(
        self,
        n_steps: Optional[int] = None,
        n_members: Optional[int] = None,
        grid_size: Optional[int] = None,
        seed: int = 0,
    ):
        _require_real_mode("ERA5")
        if not settings.era5_forecast_path:
            raise RuntimeError("DATA_UNAVAILABLE: configure ERA5_FORECAST_PATH to load local ERA5 time-series data")
        values, metadata = _read_local_dataset(settings.era5_forecast_path, "forecast")
        if values.ndim == 4:
            values = values[:, np.newaxis, ...]
            metadata["ensemble_members"] = ("deterministic",)
        if values.ndim != 5:
            raise ValueError("ERA5 forecast must have shape (time, members, variables, latitude, longitude)")
        self._coordinates = (metadata["latitude"], metadata["longitude"])
        self._field_metadata = {
            **metadata,
            "variables": tuple(settings.variable_names),
            "variable_units": metadata["variable_units"],
            "data_mode": settings.data_mode,
            "source": "ERA5 local file",
        }
        return values

    def generate_baseline(
        self,
        grid_size: Optional[int] = None,
        n_years: Optional[int] = None,
        seed: int = 1,
    ):
        _require_real_mode("ERA5")
        downloaded_from_cds = False
        if settings.era5_climatology_path:
            values, metadata = _read_local_dataset(settings.era5_climatology_path, "climatology")
        elif settings.data_mode.upper() == "REAL":
            from src.era5_cds import cds_configured
            if not cds_configured():
                raise RuntimeError("DATA_UNAVAILABLE: ERA5 climatology requires CDS credentials or ERA5_CLIMATOLOGY_PATH")
            from src.era5_cds import ensure_era5_climatology
            settings.era5_climatology_path = ensure_era5_climatology()
            downloaded_from_cds = True
            values, metadata = _read_local_dataset(settings.era5_climatology_path, "climatology")
        else:
            raise RuntimeError("DATA_UNAVAILABLE: ERA5 climatology is required; configure ERA5_CLIMATOLOGY_PATH")
        if values.ndim != 4:
            raise ValueError("ERA5 climatology must have shape (samples, variables, latitude, longitude)")
        self._coordinates = (metadata["latitude"], metadata["longitude"])
        self._field_metadata = {
            "uri": metadata["uri"],
            "variables": tuple(settings.variable_names),
            "variable_units": metadata["variable_units"],
            "climatology_time": None if metadata["time"] is None else tuple(str(value) for value in metadata["time"]),
            "reference_period": (
                (str(metadata["time"][0]), str(metadata["time"][-1]))
                if metadata["time"] is not None and len(metadata["time"])
                else None
            ),
            "data_mode": settings.data_mode,
            "source": "CDS/ERA5" if downloaded_from_cds else "ERA5 local file",
            "source_variables": metadata.get("source_variables"),
            "download_provenance": metadata.get("download_provenance"),
            "reference_source": "ERA5 climatology",
        }
        return values


class Era5CDSProvider(Era5Provider):
    provider_name = "era5_cds"
    label = "era5-cds"
    description = "Live ERA5/CDS provider using configured environment credentials and local cache."
    source_type = "era5_cds"

    def generate_baseline(
        self,
        grid_size: Optional[int] = None,
        n_years: Optional[int] = None,
        seed: int = 1,
    ):
        _require_real_mode("ERA5/CDS")
        from src.era5_cds import cds_configured
        if not cds_configured():
            raise RuntimeError("ERA5/CDS: CONFIGURED BUT DATA UNAVAILABLE")
        from src.era5_cds import ensure_era5_climatology
        if not settings.era5_climatology_path:
            settings.era5_climatology_path = ensure_era5_climatology()
        return super().generate_baseline(grid_size, n_years, seed)


def canonicalize_weather_data(
    ensemble: np.ndarray,
    climatology: np.ndarray,
    provider: WeatherDataProvider,
) -> CanonicalWeatherFields:
    ensemble = np.asarray(ensemble)
    climatology = np.asarray(climatology)
    if ensemble.ndim != 5:
        raise ValueError("ensemble must have shape (steps, members, variables, latitude, longitude)")
    if climatology.ndim != 4:
        raise ValueError("climatology must have shape (samples, variables, latitude, longitude)")
    if ensemble.shape[0] == 0 or ensemble.shape[1] == 0 or climatology.shape[0] == 0:
        raise ValueError("forecast steps, ensemble members, and climatology samples must not be empty")
    if ensemble.shape[2] != climatology.shape[1]:
        raise ValueError("forecast and climatology variable dimensions must match")
    if ensemble.shape[2] != len(settings.variable_names):
        raise ValueError("weather variable dimension does not match configured variable_names")
    if ensemble.shape[-2:] != climatology.shape[-2:]:
        raise ValueError("forecast and climatology must use the same latitude/longitude grid")
    if not np.isfinite(ensemble).all() or not np.isfinite(climatology).all():
        raise ValueError("forecast and climatology fields must not contain NaN or infinite values")

    latitude, longitude = provider.grid_coordinates(ensemble.shape[-2:])
    for name, coordinate, lower, upper in (
        ("latitude", latitude, -90.0, 90.0),
        ("longitude", longitude, -180.0, 360.0),
    ):
        if not np.isfinite(coordinate).all() or np.any((coordinate < lower) | (coordinate > upper)):
            raise ValueError(f"{name} coordinates must be finite and within [{lower}, {upper}]")
        differences = np.diff(coordinate)
        if len(differences) and not (np.all(differences > 0) or np.all(differences < 0)):
            raise ValueError(f"{name} coordinates must be strictly monotonic without duplicates")
    if np.unique(np.round(np.mod(longitude, 360.0), decimals=8)).size != len(longitude):
        raise ValueError("longitude coordinates must not contain duplicate meridians")

    units = tuple(provider.metadata.get("variable_units", (None,) * ensemble.shape[2]))
    if len(units) != ensemble.shape[2]:
        raise ValueError("variable unit metadata must have one entry per weather variable")
    provenance = provider.metadata.copy()
    qc = quality_control(
        ensemble,
        climatology,
        tuple(settings.variable_names),
        units,
        latitude,
        longitude,
        strict_physics=provider.source_type != "synthetic",
    )
    provenance["quality_control"] = qc
    provenance["data_mode"] = settings.data_mode.upper()
    provenance["reference_type"] = (
        "synthetic_proxy" if provider.source_type == "synthetic" else "climatology"
    )
    provenance.setdefault("reference_source", "synthetic demo climatology" if provider.source_type == "synthetic" else "real climatology")
    return CanonicalWeatherFields(
        ensemble=ensemble,
        climatology=climatology,
        latitudes=latitude,
        longitudes=longitude,
        variables=tuple(settings.variable_names),
        units=units,
        provenance=provenance,
    )


def quality_control(
    ensemble: np.ndarray,
    climatology: np.ndarray,
    variables: Tuple[str, ...],
    units: Tuple[Optional[str], ...],
    latitude: np.ndarray,
    longitude: np.ndarray,
    strict_physics: bool = True,
) -> Dict[str, Any]:
    if ensemble.ndim != 5 or climatology.ndim != 4:
        raise ValueError("QC failed: expected forecast (steps,members,variables,lat,lon) and climatology (samples,variables,lat,lon)")
    if ensemble.shape[2] != len(variables) or climatology.shape[1] != len(variables):
        raise ValueError("QC failed: variable dimensions do not match canonical variable names")
    if ensemble.shape[-2:] != climatology.shape[-2:] or ensemble.shape[-2:] != (len(latitude), len(longitude)):
        raise ValueError("QC failed: forecast, climatology, and geographic grid dimensions must match")
    for label, values in (("forecast", ensemble), ("climatology", climatology)):
        if not np.isfinite(values).all():
            raise ValueError(f"QC failed: {label} contains missing, NaN, or infinite values; no values were discarded")
    for name, coordinates, minimum, maximum in (
        ("latitude", np.asarray(latitude), -90.0, 90.0),
        ("longitude", np.asarray(longitude), -180.0, 360.0),
    ):
        if coordinates.ndim != 1 or not np.isfinite(coordinates).all():
            raise ValueError(f"QC failed: {name} coordinates must be finite one-dimensional arrays")
        if np.any((coordinates < minimum) | (coordinates > maximum)):
            raise ValueError(f"QC failed: {name} coordinate outside expected geographic bounds")
        delta = np.diff(coordinates)
        if delta.size and not (np.all(delta > 0) or np.all(delta < 0)):
            raise ValueError(f"QC failed: {name} coordinates must be strictly ordered and duplicate-free")
    if np.unique(np.round(np.mod(longitude, 360.0), decimals=8)).size != len(longitude):
        raise ValueError("QC failed: longitude has duplicate meridians")

    checks = []
    if strict_physics:
        for index, name in enumerate(variables):
            values = np.concatenate(
                (
                    ensemble[:, :, index].reshape(-1),
                    climatology[:, index].reshape(-1),
                )
            )
            unit = (units[index] or "").lower()
            if name in {"t2m", "temperature"}:
                if not unit:
                    checks.append({"variable": name, "status": "range_check_unavailable_missing_units"})
                    continue
                bounds = (-100.0, 70.0) if "c" in unit and "k" not in unit else (150.0, 350.0)
            elif name in {"rainfall", "tp", "precipitation"}:
                bounds = (0.0, float("inf"))
            elif name == "q":
                bounds = (0.0, 0.1)
            elif name in {"wind_u", "wind_v"}:
                bounds = (-150.0, 150.0)
            else:
                checks.append({"variable": name, "status": "range_check_not_defined", "units": units[index]})
                continue
            if np.any(values < bounds[0]) or np.any(values > bounds[1]):
                raise ValueError(
                    f"QC failed: physically implausible values for '{name}' with units {units[index]!r}; "
                    "check source variable mapping and units"
                )
            checks.append({"variable": name, "status": "passed", "units": units[index]})
    return {
        "status": "passed",
        "dimensions": {
            "forecast": list(ensemble.shape),
            "climatology": list(climatology.shape),
        },
        "coordinate_order": {
            "latitude": "ascending" if len(latitude) < 2 or latitude[0] < latitude[-1] else "descending",
            "longitude": "ascending" if len(longitude) < 2 or longitude[0] < longitude[-1] else "descending",
        },
        "missing_or_nonfinite_values": 0,
        "physical_checks": checks if strict_physics else "skipped_for_synthetic_demo",
        "corrections_applied": [],
    }


_PROVIDER_REGISTRY = {
    "public_weather": PublicWeatherProvider,
    "local_fixture": LocalFixtureProvider,
    "synthetic_demo": SyntheticDemoProvider,
    "neps_g": NepsGProvider,
    "ncmrwf_tigge": NCMRWFTIGGEProvider,
    "era5": Era5Provider,
    "era5_cds": Era5CDSProvider,
}


def _normalize_provider_name(name: Optional[str]) -> str:
    provider_name = (name or settings.data_provider or "public_weather").strip().lower().replace("-", "_")
    aliases = {
        "demo": "synthetic_demo",
        "public": "public_weather",
        "public_forecast": "public_weather",
        "local": "local_fixture",
        "fixture": "local_fixture",
        "neps": "neps_g",
        "nepsg": "neps_g",
        "ncmrwf": "ncmrwf_tigge",
        "tigge": "ncmrwf_tigge",
        "ncmrwf_tigge": "ncmrwf_tigge",
        "real": "real",
        "local_real": "local_real",
        "synthetic": "synthetic_demo",
    }
    return aliases.get(provider_name, provider_name)


def _local_file_available(path: Optional[str]) -> bool:
    if not path:
        return False
    candidate = Path(path).expanduser()
    return candidate.exists() and (candidate.is_file() or candidate.is_dir())


def get_data_provider(provider_name: Optional[str] = None) -> WeatherDataProvider:
    normalized = _normalize_provider_name(provider_name)
    mode = settings.data_mode.strip().upper()
    if mode not in {"REAL", "LOCAL_REAL", "TIGGE_AUTO", "SYNTHETIC_DEMO"}:
        raise ValueError("DATA_MODE must be REAL, LOCAL_REAL, TIGGE_AUTO, or SYNTHETIC_DEMO")
    if mode == "TIGGE_AUTO":
        if not _local_file_available(settings.neps_g_forecast_path):
            raise RuntimeError("LIVE INGESTION UNAVAILABLE: TIGGE_AUTO has no active validated local forecast")
        return NCMRWFTIGGEProvider()
    if mode == "REAL":
        if provider_name is not None and normalized not in {"real", "neps_g", "ncmrwf_tigge"}:
            raise ValueError("REAL mode requires the configured real NEPS-G forecast provider")
        if not _local_file_available(settings.neps_g_forecast_path):
            raise RuntimeError("NEPS-G: CONFIGURED BUT DATA UNAVAILABLE; set NEPS_G_FORECAST_PATH to an official local forecast file")
        from src.era5_cds import cds_configured
        if not _local_file_available(settings.era5_climatology_path) and not cds_configured():
            raise RuntimeError("ERA5: CONFIGURED BUT DATA UNAVAILABLE; set ERA5_CLIMATOLOGY_PATH or CDS credentials")
        return NCMRWFTIGGEProvider()
    if mode == "LOCAL_REAL":
        if provider_name is not None and normalized not in {"local_real", "neps_g", "ncmrwf_tigge"}:
            raise ValueError("LOCAL_REAL mode requires the configured local NEPS-G forecast provider")
        if not _local_file_available(settings.neps_g_forecast_path):
            raise RuntimeError("NEPS-G: CONFIGURED BUT DATA UNAVAILABLE; LOCAL_REAL requires NEPS_G_FORECAST_PATH")
        if not _local_file_available(settings.era5_climatology_path):
            raise RuntimeError("DATA_UNAVAILABLE: LOCAL_REAL requires ERA5_CLIMATOLOGY_PATH")
        return NCMRWFTIGGEProvider()
    if normalized in {"real", "local_real"}:
        raise ValueError("Provider selector does not override DATA_MODE; use DATA_MODE=REAL or LOCAL_REAL explicitly")
    if normalized in {"neps_g", "ncmrwf_tigge", "era5", "era5_cds"}:
        raise ValueError(
            "Real-data providers cannot be selected in SYNTHETIC_DEMO mode; set DATA_MODE=REAL or LOCAL_REAL"
        )
    provider_class = _PROVIDER_REGISTRY.get(normalized)
    if provider_class is None:
        available = ", ".join(sorted(_PROVIDER_REGISTRY))
        raise ValueError(f"Unknown data provider '{normalized}'. Available providers: {available}")
    return provider_class()
