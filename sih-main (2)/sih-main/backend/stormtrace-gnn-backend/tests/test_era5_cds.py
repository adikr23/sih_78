import json

import pytest

from config import settings
from src.era5_cds import Era5CDSDownloader


class FakeCDSClient:
    def __init__(self):
        self.calls = []

    def retrieve(self, dataset, request, target=None):
        self.calls.append((dataset, dict(request)))
        with open(target, "wb") as stream:
            stream.write(b"fake-netcdf-fixture")


class FailingCDSClient:
    def __init__(self, error):
        self.error = error

    def retrieve(self, dataset, request, target=None):
        raise self.error


def _configure_cds(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "cds_uid", "test-user")
    monkeypatch.setattr(settings, "cds_api_key", "test-key-not-for-output")
    monkeypatch.setattr(settings, "cds_url", "https://cds.example.test/api")
    monkeypatch.setattr(settings, "era5_dataset", "reanalysis-era5-single-levels")
    monkeypatch.setattr(settings, "era5_variables", "2m_temperature")
    monkeypatch.setattr(settings, "era5_download_area", "10,20,0,30")
    monkeypatch.setattr(settings, "era5_resolution", "0.25/0.25")
    monkeypatch.setattr(settings, "era5_cache_dir", str(tmp_path / "cache"))


def test_missing_cds_credentials_fail_before_client_creation(monkeypatch):
    monkeypatch.setattr(settings, "cds_uid", None)
    monkeypatch.setattr(settings, "cds_api_key", None)
    monkeypatch.setattr("src.era5_cds._read_cdsapi_config", lambda: {})
    with pytest.raises(RuntimeError, match="credentials are missing"):
        Era5CDSDownloader(client=FakeCDSClient())


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"start_date": "2024-02-02", "end_date": "2024-02-01"}, "on or before"),
        ({"start_date": "not-a-date", "end_date": "2024-02-01"}, "YYYY-MM-DD"),
        ({"start_date": "2024-02-01", "end_date": "2024-02-01", "area": [95, 0, 0, 10]}, "latitude bounds"),
        ({"start_date": "2024-02-01", "end_date": "2024-02-01", "variables": ["not_a_variable"]}, "invalid variables"),
        ({"start_date": "2024-02-01", "end_date": "2024-02-01", "times": ["25:00"]}, "invalid ERA5 time"),
    ],
)
def test_invalid_cds_request_configuration_is_rejected(monkeypatch, tmp_path, kwargs, message):
    _configure_cds(monkeypatch, tmp_path)
    downloader = Era5CDSDownloader(client=FakeCDSClient())
    with pytest.raises(ValueError, match=message):
        downloader.download(**kwargs)


def test_invalid_dataset_is_rejected_without_cds_request(monkeypatch, tmp_path):
    _configure_cds(monkeypatch, tmp_path)
    monkeypatch.setattr(settings, "era5_dataset", "invented-dataset")
    client = FakeCDSClient()
    downloader = Era5CDSDownloader(client=client)
    with pytest.raises(ValueError, match="Invalid or unsupported ERA5_DATASET"):
        downloader.download("2024-02-01", "2024-02-01", variables=["2m_temperature"])
    assert client.calls == []


def test_cds_licence_failure_is_reported_as_access_blocked(monkeypatch, tmp_path):
    _configure_cds(monkeypatch, tmp_path)
    downloader = Era5CDSDownloader(client=FailingCDSClient(Exception("403 required licences not accepted")))
    with pytest.raises(RuntimeError, match="dataset licence has not been accepted"):
        downloader.download("2024-02-01", "2024-02-01", variables=["2m_temperature"])


def test_cds_authentication_and_download_failures_are_distinguished(monkeypatch, tmp_path):
    _configure_cds(monkeypatch, tmp_path)
    unauthorized = Era5CDSDownloader(client=FailingCDSClient(Exception("401 unauthorized")))
    with pytest.raises(RuntimeError, match="CDS authentication failed"):
        unauthorized.download("2024-02-01", "2024-02-01", variables=["2m_temperature"])

    _configure_cds(monkeypatch, tmp_path / "generic_failure")
    failed_download = Era5CDSDownloader(client=FailingCDSClient(Exception("service unavailable")))
    with pytest.raises(RuntimeError, match="ERA5/CDS download failed"):
        failed_download.download("2024-02-01", "2024-02-01", variables=["2m_temperature"])


def test_cache_miss_downloads_and_writes_metadata_then_cache_hit_skips_request(monkeypatch, tmp_path):
    _configure_cds(monkeypatch, tmp_path)
    client = FakeCDSClient()
    downloader = Era5CDSDownloader(client=client)

    path = downloader.download(
        "2024-02-01",
        "2024-02-01",
        variables=["2m_temperature"],
        times=["00:00", "01:00"],
        area=[10, 20, 0, 30],
        resolution="0.25/0.25",
    )
    assert path.is_file()
    assert len(client.calls) == 1
    assert client.calls[0][0] == "reanalysis-era5-single-levels"
    assert client.calls[0][1]["time"] == ["00:00", "01:00"]
    assert "target" not in client.calls[0][1]
    metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
    assert metadata["dataset"] == "reanalysis-era5-single-levels"
    assert metadata["variables"] == ["2m_temperature"]
    assert metadata["date_range"] == {"start": "2024-02-01", "end": "2024-02-01"}
    assert metadata["source"] == "CDS/ERA5"
    assert metadata["local_file_path"] == str(path)

    second_path = downloader.download(
        "2024-02-01",
        "2024-02-01",
        variables=["2m_temperature"],
        times=["00:00", "01:00"],
        area=[10, 20, 0, 30],
        resolution="0.25/0.25",
    )
    assert second_path == path
    assert len(client.calls) == 1
