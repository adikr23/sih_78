from __future__ import annotations

from datetime import datetime
from pathlib import Path
import zipfile

import netCDF4
import numpy as np
import pytest

from src.imdaa_adapter import iter_imdaa_years, load_imdaa_period


def _write_member(path: Path, prefix: str, stamp: str, *, height=None, unequal=False):
    timestamp = datetime.strptime(stamp, "%Y%m%d%H")
    path.parent.mkdir(parents=True, exist_ok=True)
    variable_names = {
        "TMP-2m": ("2t", 300.0, 2.0),
        "APCP-sfc": ("param8.1.0", 1.0, None),
        "UGRD-10m": ("10u", 5.0, 10.0),
        "VGRD-10m": ("10v", -2.0, 10.0),
    }
    variable, value, expected_height = variable_names[prefix]
    selected_height = height if height is not None else expected_height
    time_count = 2 if prefix in {"UGRD-10m", "VGRD-10m"} else 1
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("time", time_count)
        dataset.createDimension("lat", 4)
        dataset.createDimension("lon", 4)
        if selected_height is not None:
            dataset.createDimension("height", 1)
            height_coordinate = dataset.createVariable("height", "f8", ("height",))
            height_coordinate[:] = [selected_height]
            height_coordinate.units = "m"
        latitude = dataset.createVariable("lat", "f8", ("lat",))
        latitude[:] = [0.0, 10.0, 20.0, 30.0]
        latitude.units = "degrees_north"
        longitude = dataset.createVariable("lon", "f8", ("lon",))
        longitude[:] = [70.0, 80.0, 90.0, 100.0]
        longitude.units = "degrees_east"
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = f"hours since {timestamp.year}-{timestamp.month}-{timestamp.day} {timestamp.hour:02d}:00:00"
        time[:] = [1.0 if prefix == "APCP-sfc" else 0.0] * time_count
        dimensions = ("time", "lat", "lon") if selected_height is None else (
            "time",
            "height",
            "lat",
            "lon",
        )
        values = dataset.createVariable(variable, "f4", dimensions)
        if prefix == "APCP-sfc":
            values.param = "8.1.0"
        data_shape = (time_count, 4, 4) if selected_height is None else (time_count, 1, 4, 4)
        data = np.full(data_shape, value, dtype=np.float32)
        if unequal and time_count > 1:
            data[1, ..., 0, 0] += 1
        values[:] = data


def _make_archive(tmp_path: Path, *, unequal_winds=False, t2m_height=None):
    archive_path = tmp_path / "2020.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        members = []
        for hour in (0, 6, 12, 18):
            stamp = f"20200516{hour:02d}"
            for prefix in ("TMP-2m", "APCP-sfc", "UGRD-10m", "VGRD-10m"):
                member_path = tmp_path / f"{prefix}_{stamp}_fixture.nc"
                _write_member(
                    member_path,
                    prefix,
                    stamp,
                    height=t2m_height if prefix == "TMP-2m" else None,
                    unequal=unequal_winds and prefix == "UGRD-10m",
                )
                archive.write(member_path, member_path.name)
                members.append(member_path)
        for member in members:
            member.unlink()
    return tmp_path


def test_imdaa_zip_maps_variables_and_normalizes_times(tmp_path):
    archive_dir = _make_archive(tmp_path)
    loaded = load_imdaa_period(archive_dir, "2020-05-16", "2020-05-16")

    assert set(loaded.fields) == {"t2m", "rainfall", "wind_u", "wind_v"}
    assert loaded.fields["t2m"].shape == (4, 3, 3)
    assert loaded.fields["rainfall"].shape == (4, 3, 3)
    assert str(loaded.times[0]) == "2020-05-16T00:00:00"
    first = loaded.provenance["records"]["2020-05-16T00:00:00"]
    assert first["rainfall"]["decoded_netcdf_times"] == ["2020-05-16T01:00:00"]
    assert first["rainfall"]["decoded_minus_filename_hours"] == 1
    assert first["wind_u"]["duplicate_records"] == 2
    assert first["wind_u"]["identical_duplicate_records_removed_logically"] == 1
    assert loaded.provenance["rainfall"]["status"] == "UNRESOLVED_SEMANTICS"


def test_imdaa_rejects_nonidentical_duplicate_wind_records(tmp_path):
    archive_dir = _make_archive(tmp_path, unequal_winds=True)
    with pytest.raises(ValueError, match="non-identical duplicate wind records"):
        load_imdaa_period(archive_dir, "2020-05-16", "2020-05-16")


def test_imdaa_requires_exact_single_height(tmp_path):
    archive_dir = _make_archive(tmp_path, t2m_height=10.0)
    with pytest.raises(ValueError, match="must contain exactly one 2 m level"):
        load_imdaa_period(archive_dir, "2020-05-16", "2020-05-16")


def test_imdaa_rejects_missing_requested_member(tmp_path):
    archive_dir = _make_archive(tmp_path)
    archive_path = archive_dir / "2020.zip"
    with zipfile.ZipFile(archive_path, "a") as archive:
        removed_member = next(
            name for name in archive.namelist() if name.startswith("TMP-2m_2020051600")
        )
        contents = [(name, archive.read(name)) for name in archive.namelist() if name != removed_member]
    archive_path.unlink()
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, data in contents:
            archive.writestr(name, data)
    with pytest.raises(FileNotFoundError, match="TMP-2m file is missing"):
        load_imdaa_period(archive_dir, "2020-05-16", "2020-05-16")


def test_reference_year_iterator_rejects_2020(tmp_path):
    iterator = iter_imdaa_years(tmp_path, [2020])
    with pytest.raises(ValueError, match="exclude 2020"):
        next(iterator)
