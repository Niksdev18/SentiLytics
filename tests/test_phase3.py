"""Comprehensive Flask test-client tests for Phase 3."""

import time
import pytest
from src.api.app import create_app
from src.config import AppConfig


@pytest.fixture
def test_client(tmp_path):
    """Create Flask test client with an isolated temporary workspace."""
    cfg = AppConfig(demo_mode=True)
    cfg.root_dir = tmp_path

    # Stubbed fast engine
    class FastEngine:
        status = "ready"
        model_description = "Test Stub FinBERT"
        is_fine_tuned = False

        def load(self):
            self.status = "ready"
            return True

        def predict_batch(self, texts):
            return [
                {
                    "label": "Positive" if "profit" in t.lower() else "Neutral",
                    "confidence": 0.88,
                    "probabilities": {"Positive": 0.88, "Neutral": 0.08, "Negative": 0.04},
                }
                for t in texts
            ]

    # Stubbed pipeline
    def stub_pipeline(config, engine):
        return {
            "source": "live",
            "generated_at": "2026-10-08T00:00:00Z",
            "model_name": "Stub Model",
            "metrics": {"accuracy_pct": 80.0, "correct": 8, "total": 10},
            "companies": {"Apple": {"accuracy_pct": 100.0, "total": 2}},
            "rows": [
                {
                    "title": "Stub article",
                    "url": "#",
                    "company": "Apple",
                    "sentiment": "Positive",
                    "confidence": 0.90,
                    "suggestion": "Consider Buy",
                    "change_pct": 2.5,
                }
            ],
        }

    app = create_app(config=cfg, engine=FastEngine(), pipeline_fn=stub_pipeline)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client, app


def test_home_page_renders(test_client):
    """GET / renders successfully with context."""
    client, app = test_client
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"<!DOCTYPE html>" in resp.data or b"<html" in resp.data


def test_dashboard_page_renders(test_client):
    """GET /dashboard renders successfully."""
    client, app = test_client
    resp = client.get("/dashboard")
    assert resp.status_code == 200


def test_about_page_renders(test_client):
    """GET /about renders successfully."""
    client, app = test_client
    resp = client.get("/about")
    assert resp.status_code == 200


def test_api_results(test_client):
    """GET /api/results returns valid JSON schema."""
    client, app = test_client
    resp = client.get("/api/results")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "rows" in data
    assert "metrics" in data
    assert "source" in data


def test_api_summary(test_client):
    """GET /api/summary returns company dictionary."""
    client, app = test_client
    resp = client.get("/api/summary")
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, dict)


def test_api_model_metrics(test_client):
    """GET /api/model_metrics returns metrics payload."""
    client, app = test_client
    resp = client.get("/api/model_metrics")
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, dict)


def test_api_predict_valid_single(test_client):
    """POST /api/predict accepts single text."""
    client, app = test_client
    resp = client.post("/api/predict", json={"text": "Apple beats profit estimates"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["count"] == 1
    assert data["predictions"][0]["label"] == "Positive"


def test_api_predict_valid_batch(test_client):
    """POST /api/predict accepts list of texts."""
    client, app = test_client
    resp = client.post("/api/predict", json={"texts": ["Apple reports record profit", "Tesla stock unchanged"]})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["count"] == 2


def test_api_predict_validation_errors(test_client):
    """POST /api/predict enforces input limits and schema."""
    client, app = test_client

    # Empty payload
    assert client.post("/api/predict", json={}).status_code == 400

    # Non-json / invalid types
    assert client.post("/api/predict", json={"wrong_key": "val"}).status_code == 400

    # Empty texts list
    assert client.post("/api/predict", json={"texts": []}).status_code == 400

    # More than 50 items
    over_limit = ["sample text"] * 51
    assert client.post("/api/predict", json={"texts": over_limit}).status_code == 400

    # Single text exceeding 2000 chars
    giant_text = "word " * 600  # > 2000 chars
    assert client.post("/api/predict", json={"text": giant_text}).status_code == 400


def test_api_predict_model_unavailable(tmp_path):
    """POST /api/predict returns 503 when model is unavailable."""
    cfg = AppConfig(allow_hub_fallback=False)
    cfg.root_dir = tmp_path

    class UnavailableEngine:
        status = "unavailable"
        model_description = "None"
        is_fine_tuned = False

        def load(self):
            return False

    app = create_app(config=cfg, engine=UnavailableEngine())
    with app.test_client() as client:
        resp = client.post("/api/predict", json={"text": "Hello market"})
        assert resp.status_code == 503
        data = resp.get_json()
        assert "unavailable" in data["error"].lower()


def test_refresh_202_and_409_concurrency(test_client):
    """POST /refresh returns 202 on start, 409 when already in progress."""
    client, app = test_client

    # Simulate long running pipeline
    def slow_pipeline(config, engine):
        time.sleep(0.4)
        return {"source": "live", "rows": [], "metrics": {}}

    app.state.pipeline_fn = slow_pipeline

    # First call -> 202 Accepted
    resp1 = client.post("/refresh")
    assert resp1.status_code == 202
    assert resp1.get_json()["status"] == "accepted"

    # Immediate second call while first is running -> 409 Conflict
    resp2 = client.post("/refresh")
    assert resp2.status_code == 409
    assert resp2.get_json()["status"] == "conflict"

    # Wait for completion
    time.sleep(0.6)
    assert app.state.is_refreshing is False


def test_health_endpoint(test_client):
    """GET /health returns health metrics."""
    client, app = test_client
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert "model_status" in data
    assert "is_refreshing" in data


def test_network_failure_fallback_keeps_last_results(test_client):
    """When pipeline refresh fails, previous results are preserved and error is surfaced."""
    client, app = test_client
    initial_rows = list(app.state.results.get("rows", []))

    def failing_pipeline(config, engine):
        raise ConnectionError("Network unreachable")

    app.state.pipeline_fn = failing_pipeline
    app.state.run_refresh_task()

    # Results preserved
    assert app.state.results.get("rows") == initial_rows
    # Error surfaced in state and health
    assert "Network unreachable" in app.state.last_error
    health_data = client.get("/health").get_json()
    assert "Network unreachable" in health_data["last_error"]
