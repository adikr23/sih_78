from fastapi.testclient import TestClient

from src.main import app


def test_forecast_status_exposes_current_cached_cycle():
    response = TestClient(app).get("/api/v1/forecast/status")
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "NCMRWF TIGGE"
    assert body["available_steps"] == 41
    assert body["ensemble_members"] == 11
    assert body["stale"] is True


def test_forecast_field_rejects_unknown_dynamic_variable():
    response = TestClient(app).get("/api/v1/forecast/field?variable=humidity")
    assert response.status_code == 503
    assert "unsupported TIGGE field" in response.json()["detail"]
