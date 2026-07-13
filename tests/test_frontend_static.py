from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_frontend_verify_request_has_client_timeout() -> None:
    response = client.get("/static/app.js")
    results = client.get("/static/results.js")

    assert response.status_code == 200
    assert "AbortController" in response.text
    assert "requestTimeoutMs = 5000" in response.text
    assert "The label took too long to read." in response.text
    assert "Line breaks do not matter." in results.text


def test_frontend_requires_manual_government_warning() -> None:
    response = client.get("/static/app.js")

    assert response.status_code == 200
    assert "standardGovernmentWarning" not in response.text
    assert "According to the Surgeon General" not in response.text
    assert "prefillStandardWarning" not in response.text


def test_frontend_batch_view_is_available() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "Batch Labels" in response.text
    assert "Add label images" in response.text
    assert "back-image-input" not in response.text
    assert "Choose back label image" not in response.text
    assert "batch-image-input" in response.text
    assert "multiple" in response.text
    assert "Check All Labels" in response.text
    assert 'accept="image/*"' in response.text
    assert 'id="batch-limit"' in response.text
    assert "Copy First Label to All" not in response.text


def test_frontend_batch_request_has_summary_drilldown_and_progress() -> None:
    response = client.get("/static/app.js")
    results = client.get("/static/results.js")

    assert response.status_code == 200
    assert 'fetch("/verify/batch"' in response.text
    assert 'formData.append("image_ids"' in response.text
    assert 'formData.append("images"' in response.text
    assert "data-back-image-row" not in response.text
    assert "backFile" not in response.text
    assert "batchRows = batchRows.concat" in response.text
    assert "batchImageInput.value = \"\"" in response.text
    assert "batchCopyFirstButton" not in response.text
    assert "Approved" in results.text
    assert "Needs Review" in results.text
    assert "Total" in results.text
    assert "<details" in results.text
    assert "Expected" in results.text
    assert "Found" in results.text
    assert "Checking ${count} label" in response.text
    assert "progress-bar" in results.text
    assert 'role="progressbar"' in results.text
    assert 'aria-label="Checking labels"' in results.text


def test_frontend_uses_spec_response_field_names() -> None:
    response = client.get("/static/results.js")

    assert response.status_code == 200
    assert "data.overall_verdict" in response.text
    assert "data.results" in response.text
    assert "data.summary" in response.text
    assert "result.found" in response.text
    assert "data.verdict" not in response.text
    assert "data.fields" not in response.text
    assert "result.actual" not in response.text


def test_frontend_uses_spec_application_field_names() -> None:
    html = client.get("/").text
    js = client.get("/static/app.js").text

    assert 'name="class_type"' in html
    assert 'name="producer"' in html
    assert 'name="product_class"' not in html
    assert 'name="producer_name"' not in html
    assert '"class_type"' in js
    assert '"producer"' in js
    assert '"product_class"' not in js
    assert '"producer_name"' not in js
    assert 'fetch("/health")' in js
    assert 'inputmode="decimal"' in html
    assert "free-tier service may be starting up" in js


def test_frontend_accessibility_hooks_are_present() -> None:
    html = client.get("/").text
    css = client.get("/static/styles.css").text
    js = client.get("/static/app.js").text

    assert 'aria-live="polite"' in html
    assert 'role="tablist"' in html
    assert 'aria-controls="batch-panel"' in html
    assert 'aria-busy' in js
    assert ".mode-button:focus" in css
    assert ".secondary-button:focus" in css
    assert ".batch-item summary:focus-visible" in css
    assert ".image-preview[hidden]" in css
