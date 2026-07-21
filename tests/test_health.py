from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.vision import VisionProviderError


client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "ttb-label-verification",
        "environment": "local",
        "batch_max_labels": 10,
        "batch_upload_max_labels": 300,
    }


def test_health_returns_configured_batch_limit(monkeypatch) -> None:
    monkeypatch.setenv("BATCH_MAX_LABELS", "6")

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["batch_max_labels"] == 6


def test_health_returns_configured_batch_upload_limit(monkeypatch) -> None:
    monkeypatch.setenv("BATCH_UPLOAD_MAX_LABELS", "240")

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["batch_upload_max_labels"] == 240


def test_root_serves_frontend() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "TTB Label Check" in response.text
    assert "Choose label image" in response.text
    assert "Government Warning" in response.text
    assert "/static/app.js" in response.text


def test_frontend_script_calls_verify_endpoint() -> None:
    response = client.get("/static/app.js")
    results = client.get("/static/results.js")

    assert response.status_code == 200
    assert 'fetch("/verify"' in response.text
    assert 'formData.append("application_data"' in response.text
    assert "NEEDS REVIEW" in results.text


def test_deep_health_returns_model_status(monkeypatch) -> None:
    def check_model(self):
        return {"id": self.model, "object": "model"}

    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-nano")
    monkeypatch.setattr("backend.app.main.OpenAIVisionService.check_model", check_model)

    response = client.get("/health/deep")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["provider"] == "openai"
    assert response.json()["model"] == "gpt-5.4-nano"
    assert isinstance(response.json()["latency_ms"], int)


def test_deep_health_returns_503_for_unavailable_model(monkeypatch) -> None:
    def check_model(self):
        raise VisionProviderError("unknown model")

    monkeypatch.setenv("OPENAI_MODEL", "stale-model")
    monkeypatch.setattr("backend.app.main.OpenAIVisionService.check_model", check_model)

    response = client.get("/health/deep")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "error"
    assert body["provider"] == "openai"
    assert body["model"] == "stale-model"
    assert body["error"]["code"] == "model_unavailable"
