from fastapi.testclient import TestClient

from src.main import app


def test_root_endpoint_points_to_health_docs_and_frontend():
    response = TestClient(app).get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "running"
    assert body["health"] == "/health"
    assert body["api_docs"] == "/docs"
