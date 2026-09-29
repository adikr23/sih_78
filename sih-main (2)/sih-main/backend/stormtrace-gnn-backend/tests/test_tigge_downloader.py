from datetime import datetime, timezone

import pytest

from config import settings
from src.tigge_downloader import TIGGEDownloader, TiggeDownloadError, _cycle_candidates


def test_cycle_discovery_checks_recent_00_and_12_cycles():
    cycles = list(_cycle_candidates(datetime(2026, 9, 28, 10, tzinfo=timezone.utc), 2))
    assert [cycle.hour for cycle in cycles] == [12, 0, 12, 0]
    assert cycles[0].date().isoformat() == "2026-09-28"


def test_downloader_requires_cds_credentials(monkeypatch):
    monkeypatch.setattr("src.tigge_downloader._resolve_cds_credentials", lambda: None)
    with pytest.raises(TiggeDownloadError, match="credentials"):
        TIGGEDownloader()


def test_downloader_request_preserves_scientific_parameters(monkeypatch):
    monkeypatch.setattr("src.tigge_downloader._resolve_cds_credentials", lambda: ("https://example.test", "redacted"))
    downloader = TIGGEDownloader(client=object())
    request = downloader._request(datetime(2026, 9, 28, 0, tzinfo=timezone.utc))
    assert request["origin"] == settings.tigge_origin
    assert request["product_type"] == "ensemble_perturbed_forecasts"
    assert request["format"] == "grib"
    assert request["leadtime_hour"][-1] == str(settings.tigge_max_lead_hours)
    assert "total_precipitation" in request["variable"]
    assert "redacted" not in str(request)
