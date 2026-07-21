import json
import logging

import pytest
from fastapi.testclient import TestClient

from backend.app.api import MAX_IMAGE_BYTES, get_vision_service
from backend.app.main import app
from backend.app.models import ExtractedLabel, VerificationVerdict
from backend.app.vision import (
    FakeVisionService,
    VisionConfigurationError,
    VisionInvalidImageError,
    VisionParseError,
    VisionProviderError,
    VisionRateLimitError,
    VisionTimeoutError,
)


WARNING = "GOVERNMENT WARNING: EXACTLY AS PRINTED"


def application_payload(**overrides) -> dict:
    payload = {
        "brand_name": "Sunset Ridge",
        "class_type": "Cabernet Sauvignon",
        "producer": "North Valley Estate Winery LLC",
        "country_of_origin": "USA",
        "abv": "45%",
        "net_contents": "750 mL",
        "government_warning": WARNING,
    }
    payload.update(overrides)
    return payload


def extracted_label(**overrides) -> ExtractedLabel:
    payload = {
        "brand_name": "SUNSET RIDGE",
        "class_type": "Sauvignon Cabernet",
        "producer": "North Valley Estate Winery, LLC",
        "country_of_origin": "USA",
        "abv": "45% Alc./Vol. (90 Proof)",
        "net_contents": "750ml",
        "government_warning": WARNING,
        "government_warning_heading_bold": True,
        "raw_text": "SUNSET RIDGE\nGOVERNMENT WARNING: EXACTLY AS PRINTED",
        "extraction_confidence": 0.97,
    }
    payload.update(overrides)
    return ExtractedLabel(**payload)


def post_verify(
    client: TestClient,
    *,
    image_bytes: bytes = b"label-image",
    content_type: str = "image/jpeg",
    application_data: dict | str | None = None,
):
    data = {}
    if application_data is None:
        data["application_data"] = json.dumps(application_payload())
    elif isinstance(application_data, str):
        data["application_data"] = application_data
    else:
        data["application_data"] = json.dumps(application_data)

    return client.post(
        "/verify",
        data=data,
        files={"image": ("label.jpg", image_bytes, content_type)},
    )


@pytest.fixture
def fake_service() -> FakeVisionService:
    return FakeVisionService(label=extracted_label())


@pytest.fixture
def client(fake_service: FakeVisionService):
    app.dependency_overrides[get_vision_service] = lambda: fake_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_verify_returns_pass_result_with_latency(
    client: TestClient,
    fake_service: FakeVisionService,
) -> None:
    response = post_verify(client)

    assert response.status_code == 200
    body = response.json()
    assert body["overall_verdict"] == VerificationVerdict.APPROVED
    assert isinstance(body["latency_ms"], int)
    assert body["latency_ms"] >= 0
    assert "verdict" not in body
    assert "fields" not in body
    assert body["extracted_label"]["government_warning"] == WARNING
    assert body["extracted_label"]["raw_text"].startswith("SUNSET RIDGE")
    assert body["extracted_label"]["extraction_confidence"] == 0.97
    assert body["extracted_label"]["government_warning_heading_bold"] is True
    assert len(body["results"]) == 7
    assert all(field["status"] == "PASS" for field in body["results"])
    assert all("found" in field for field in body["results"])
    assert all("match_type" in field for field in body["results"])
    assert all("actual" not in field for field in body["results"])
    assert all("strategy" not in field for field in body["results"])
    assert all("score" not in field for field in body["results"])
    assert len(fake_service.calls) == 1


def test_verify_success_logs_verdict_and_latency(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="backend.app.verify")

    response = post_verify(client)

    assert response.status_code == 200
    assert "verify completed verdict=VerificationVerdict.APPROVED latency_ms=" in caplog.text


def test_verify_returns_needs_review_and_surfaces_warning_text(
    client: TestClient,
    fake_service: FakeVisionService,
) -> None:
    misread_warning = "Government Warning: EXACTLY AS PRINTED"
    fake_service.label = extracted_label(government_warning=misread_warning)

    response = post_verify(client)

    assert response.status_code == 200
    body = response.json()
    assert body["overall_verdict"] == VerificationVerdict.NEEDS_REVIEW
    assert body["extracted_label"]["government_warning"] == misread_warning

    warning_result = next(
        field for field in body["results"] if field["field"] == "government_warning"
    )
    assert warning_result["status"] == "FAIL"
    assert warning_result["expected"] == WARNING
    assert warning_result["found"] == misread_warning
    assert warning_result["match_type"] == "exact_case_sensitive_whitespace_normalized+bold_heading"


def test_verify_rejects_warning_when_heading_is_not_bold(
    client: TestClient,
    fake_service: FakeVisionService,
) -> None:
    fake_service.label = extracted_label(government_warning_heading_bold=False)

    response = post_verify(client)

    assert response.status_code == 200
    body = response.json()
    assert body["overall_verdict"] == VerificationVerdict.NEEDS_REVIEW
    warning_result = next(
        field for field in body["results"] if field["field"] == "government_warning"
    )
    assert warning_result["status"] == "FAIL"
    assert warning_result["found"] == WARNING


def test_verify_accepts_warning_with_label_line_breaks(

    client: TestClient,
    fake_service: FakeVisionService,
) -> None:
    wrapped_warning = "GOVERNMENT\n WARNING:\n  EXACTLY AS\n PRINTED"
    fake_service.label = extracted_label(government_warning=wrapped_warning)

    response = post_verify(client)

    assert response.status_code == 200
    body = response.json()
    assert body["overall_verdict"] == VerificationVerdict.APPROVED
    warning_result = next(
        field for field in body["results"] if field["field"] == "government_warning"
    )
    assert warning_result["status"] == "PASS"
    assert warning_result["found"] == wrapped_warning


def test_verify_passes_image_bytes_and_content_type_to_vision_service(
    client: TestClient,
    fake_service: FakeVisionService,
) -> None:
    upload = b"uploaded-label"

    response = post_verify(client, image_bytes=upload, content_type="image/png")

    assert response.status_code == 200
    assert len(fake_service.calls) == 1
    assert fake_service.calls[0].image_bytes == upload
    assert fake_service.calls[0].content_type == "image/png"


def test_verify_rejects_missing_image(client: TestClient) -> None:
    response = client.post(
        "/verify",
        data={"application_data": json.dumps(application_payload())},
    )

    assert_error(response, 400, "missing_image", "Upload a label image.")


def test_verify_rejects_missing_application_data(client: TestClient) -> None:
    response = client.post(
        "/verify",
        files={"image": ("label.jpg", b"label-image", "image/jpeg")},
    )

    assert_error(
        response,
        400,
        "missing_application_data",
        "Include application data for this label.",
    )


def test_verify_rejects_malformed_application_json(client: TestClient) -> None:
    response = post_verify(client, application_data="{bad json")

    assert_error(
        response,
        400,
        "invalid_application_data",
        "Application data must be valid JSON.",
    )


def test_verify_rejects_missing_required_application_field(client: TestClient) -> None:
    payload = application_payload()
    payload.pop("government_warning")

    response = post_verify(client, application_data=payload)

    assert_error(
        response,
        400,
        "invalid_application_data",
        "Application data must include exactly the required label fields.",
    )


def test_verify_rejects_extra_application_field(client: TestClient) -> None:
    response = post_verify(
        client,
        application_data=application_payload(unexpected="not allowed"),
    )

    assert_error(
        response,
        400,
        "invalid_application_data",
        "Application data must include exactly the required label fields.",
    )


def test_verify_rejects_unsupported_file_type(client: TestClient) -> None:
    response = post_verify(client, content_type="text/plain")

    assert_error(
        response,
        400,
        "invalid_image_type",
        "Upload a JPEG, PNG, WebP, HEIC, or HEIF label image.",
    )


def test_verify_failure_logs_error_code_and_latency(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="backend.app.verify")

    response = post_verify(client, content_type="text/plain")

    assert response.status_code == 400
    assert "verify failed code=invalid_image_type status_code=400 latency_ms=" in caplog.text


def test_verify_rejects_empty_file(client: TestClient) -> None:
    response = post_verify(client, image_bytes=b"")

    assert_error(response, 400, "empty_image", "Upload a non-empty label image.")


def test_verify_rejects_large_file(client: TestClient) -> None:
    response = post_verify(client, image_bytes=b"x" * (MAX_IMAGE_BYTES + 1))

    assert_error(response, 413, "image_too_large", "Upload a label image smaller than 8 MB.")


def test_verify_maps_vision_timeout_to_504(
    client: TestClient,
    fake_service: FakeVisionService,
) -> None:
    fake_service.error = VisionTimeoutError("timeout")

    response = post_verify(client)

    assert_error(
        response,
        504,
        "vision_timeout",
        "The label image took too long to process. Try a clearer or smaller image.",
    )


def test_verify_maps_missing_vision_configuration_to_503(
    client: TestClient,
    fake_service: FakeVisionService,
) -> None:
    fake_service.error = VisionConfigurationError("missing key")

    response = post_verify(client)

    assert_error(
        response,
        503,
        "vision_not_configured",
        "Vision service is not configured.",
    )


def test_verify_maps_vision_invalid_image_to_400(
    client: TestClient,
    fake_service: FakeVisionService,
) -> None:
    fake_service.error = VisionInvalidImageError("invalid")

    response = post_verify(client)

    assert_error(response, 400, "invalid_image", "Upload a readable label image.")


def test_verify_maps_vision_provider_error_to_502(
    client: TestClient,
    fake_service: FakeVisionService,
) -> None:
    fake_service.error = VisionProviderError("provider details")

    response = post_verify(client)

    assert_error(
        response,
        502,
        "vision_provider_error",
        "The vision service could not process the label image.",
    )


def test_verify_maps_vision_rate_limit_to_503(
    client: TestClient,
    fake_service: FakeVisionService,
) -> None:
    fake_service.error = VisionRateLimitError("quota reached")

    response = post_verify(client)

    assert_error(
        response,
        503,
        "vision_rate_limited",
        "The vision service is temporarily busy. Try again in a minute.",
    )


def test_verify_maps_vision_parse_error_to_502(
    client: TestClient,
    fake_service: FakeVisionService,
) -> None:
    fake_service.error = VisionParseError("bad parse")

    response = post_verify(client)

    assert_error(
        response,
        502,
        "vision_parse_error",
        "The vision service returned an unreadable extraction result.",
    )


def test_verify_error_responses_do_not_include_tracebacks(
    client: TestClient,
    fake_service: FakeVisionService,
) -> None:
    fake_service.error = VisionProviderError("internal provider stack detail")

    response = post_verify(client)

    assert "Traceback" not in response.text
    assert "VisionProviderError" not in response.text
    assert "internal provider stack detail" not in response.text


def assert_error(response, status_code: int, code: str, message: str) -> None:
    assert response.status_code == status_code
    body = response.json()
    assert body["error"] == {
        "code": code,
        "message": message,
    }
    assert isinstance(body["latency_ms"], int)
    assert "Traceback" not in response.text
