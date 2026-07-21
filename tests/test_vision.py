from io import BytesIO
import json
from urllib.error import HTTPError

import pytest
from PIL import Image

from backend.app.models import ExtractedLabel
from backend.app.vision import (
    EXTRACTED_LABEL_FIELDS,
    FakeVisionService,
    OpenAIVisionService,
    VisionConfigurationError,
    VisionInvalidImageError,
    VisionParseError,
    VisionRateLimitError,
    VisionTimeoutError,
    extracted_label_json_schema,
    preprocess_label_image,
)


WARNING = "GOVERNMENT WARNING: EXACTLY AS PRINTED"


def image_bytes(size: tuple[int, int] = (800, 600)) -> bytes:
    image = Image.new("RGB", size, "white")
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def extracted_payload(**overrides) -> dict[str, str | float | bool | None]:
    payload = {
        "brand_name": "Sunset Ridge",
        "class_type": "Cabernet Sauvignon",
        "producer": "North Valley Estate Winery LLC",
        "country_of_origin": "USA",
        "abv": "45% Alc./Vol. (90 Proof)",
        "net_contents": "750 mL",
        "government_warning": WARNING,
        "government_warning_heading_bold": True,
        "raw_text": "Sunset Ridge\nGOVERNMENT WARNING: EXACTLY AS PRINTED",
        "extraction_confidence": 0.94,
    }
    payload.update(overrides)
    return payload


class StubTransport:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls = []

    def __call__(self, request_body, timeout_seconds, model):
        self.calls.append(
            {
                "request_body": request_body,
                "timeout_seconds": timeout_seconds,
                "model": model,
            }
        )
        if self.error is not None:
            raise self.error
        return self.response


class Timeout(Exception):
    pass


def test_fake_vision_service_returns_label_and_records_call() -> None:
    label = ExtractedLabel(brand_name="Sunset Ridge")
    service = FakeVisionService(label=label)
    upload = b"not inspected by fake"

    result = service.extract_label(upload, content_type="image/jpeg")

    assert result == label
    assert len(service.calls) == 1
    assert service.calls[0].image_bytes == upload
    assert service.calls[0].content_type == "image/jpeg"


def test_openai_service_defaults_to_locked_model(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    service = OpenAIVisionService(transport=StubTransport())

    assert service.model == "gpt-5.4-nano"


def test_preprocess_downscales_and_reencodes_to_jpeg_data_url() -> None:
    processed = preprocess_label_image(image_bytes((2400, 1200)))

    assert processed.content_type == "image/jpeg"
    assert processed.data_url.startswith("data:image/jpeg;base64,")
    assert processed.original_width == 2400
    assert processed.original_height == 1200
    assert max(processed.width, processed.height) == 768
    assert processed.data.startswith(b"\xff\xd8")
    assert len(processed.data) < 500_000


def test_preprocess_size_and_quality_are_configurable(monkeypatch) -> None:
    monkeypatch.setenv("IMAGE_MAX_LONG_SIDE", "900")
    monkeypatch.setenv("IMAGE_JPEG_QUALITY", "72")

    processed = preprocess_label_image(image_bytes((2400, 1200)))

    assert max(processed.width, processed.height) == 900


def test_preprocess_ignores_invalid_size_and_quality_env(monkeypatch) -> None:
    monkeypatch.setenv("IMAGE_MAX_LONG_SIDE", "200")
    monkeypatch.setenv("IMAGE_JPEG_QUALITY", "120")

    processed = preprocess_label_image(image_bytes((2400, 1200)))

    assert max(processed.width, processed.height) == 768


def test_invalid_image_fails_before_provider_call() -> None:
    transport = StubTransport(response=openai_response(extracted_payload()))
    service = OpenAIVisionService(transport=transport)

    with pytest.raises(VisionInvalidImageError):
        service.extract_label(b"not an image")

    assert transport.calls == []


def test_openai_service_uses_strict_schema_and_low_detail_image() -> None:
    transport = StubTransport(response=openai_response(extracted_payload()))
    service = OpenAIVisionService(
        transport=transport,
        model="gpt-test-model",
        timeout_seconds=3.5,
    )

    result = service.extract_label(image_bytes(), content_type="image/png")

    assert result.brand_name == "Sunset Ridge"
    assert result.government_warning == WARNING
    assert result.raw_text.startswith("Sunset Ridge")
    assert result.extraction_confidence == 0.94

    [call] = transport.calls
    assert call["model"] == "gpt-test-model"
    assert call["timeout_seconds"] == 3.5

    request_body = call["request_body"]
    assert request_body["model"] == "gpt-test-model"
    assert request_body["temperature"] == 0
    assert request_body["max_output_tokens"] == 900
    assert request_body["store"] is False

    response_format = request_body["text"]["format"]
    assert response_format["type"] == "json_schema"
    assert response_format["name"] == "extracted_label"
    assert response_format["strict"] is True
    assert response_format["schema"] == extracted_label_json_schema()

    schema = response_format["schema"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == list(EXTRACTED_LABEL_FIELDS)
    assert schema["properties"]["raw_text"] == {"type": ["string", "null"]}
    assert schema["properties"]["government_warning_heading_bold"] == {
        "type": ["boolean", "null"]
    }
    assert schema["properties"]["extraction_confidence"] == {
        "type": ["number", "null"],
        "minimum": 0,
        "maximum": 1,
    }

    prompt = request_body["input"][0]["content"][0]["text"]
    assert "class_type" in prompt
    assert "producer" in prompt
    assert "raw_text" in prompt
    assert "extraction_confidence" in prompt
    assert "character-for-character" in prompt
    assert "complete visible producer/bottler name and address" in prompt
    assert "government_warning_heading_bold" in prompt
    assert "Do not complete it from memory." in prompt

    image_part = request_body["input"][0]["content"][1]
    assert image_part["type"] == "input_image"
    assert image_part["detail"] == "low"
    assert image_part["image_url"].startswith("data:image/jpeg;base64,")
    assert len(image_part["image_url"]) > 100


def test_openai_service_reads_timeout_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "3.75")
    transport = StubTransport(response=openai_response(extracted_payload()))
    service = OpenAIVisionService(transport=transport)

    service.extract_label(image_bytes())

    [call] = transport.calls
    assert call["timeout_seconds"] == 3.75


def test_openai_service_uses_default_timeout_for_invalid_environment(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "not-a-number")
    transport = StubTransport(response=openai_response(extracted_payload()))
    service = OpenAIVisionService(transport=transport)

    service.extract_label(image_bytes())

    [call] = transport.calls
    assert call["timeout_seconds"] == 4.5


def test_non_label_or_unreadable_image_returns_all_nulls_from_model() -> None:
    null_payload = {field: None for field in EXTRACTED_LABEL_FIELDS}
    transport = StubTransport(response=openai_response(null_payload))
    service = OpenAIVisionService(transport=transport)

    result = service.extract_label(image_bytes())

    assert result == ExtractedLabel()


def test_misread_warning_is_preserved_from_structured_output() -> None:
    misread_warning = "GOVERNMENT WARNlNG: EXACTLY AS PRlNTED"
    transport = StubTransport(
        response=openai_response(
            extracted_payload(government_warning=misread_warning)
        )
    )
    service = OpenAIVisionService(transport=transport)

    result = service.extract_label(image_bytes())

    assert result.government_warning == misread_warning


def test_malformed_json_raises_parse_error() -> None:
    transport = StubTransport(response=openai_text_response("{not-json"))
    service = OpenAIVisionService(transport=transport)

    with pytest.raises(VisionParseError):
        service.extract_label(image_bytes())


def test_missing_or_extra_structured_fields_raise_parse_error() -> None:
    missing_payload = extracted_payload()
    missing_payload.pop("government_warning")
    missing_transport = StubTransport(response=openai_response(missing_payload))

    with pytest.raises(VisionParseError, match="missing fields"):
        OpenAIVisionService(transport=missing_transport).extract_label(image_bytes())

    extra_payload = extracted_payload(extra_field="not allowed")
    extra_transport = StubTransport(response=openai_response(extra_payload))

    with pytest.raises(VisionParseError, match="extra fields"):
        OpenAIVisionService(transport=extra_transport).extract_label(image_bytes())


def test_provider_timeout_maps_to_typed_timeout_error() -> None:
    transport = StubTransport(error=Timeout("timed out"))
    service = OpenAIVisionService(transport=transport)

    with pytest.raises(VisionTimeoutError):
        service.extract_label(image_bytes())


def test_openai_429_maps_to_rate_limit_error(monkeypatch) -> None:
    def raise_rate_limit(request, timeout):
        raise HTTPError(
            request.full_url,
            429,
            "Too Many Requests",
            hdrs=None,
            fp=BytesIO(b'{"error":{"status":"RESOURCE_EXHAUSTED"}}'),
        )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("backend.app.openai_client.urlopen", raise_rate_limit)
    service = OpenAIVisionService()

    with pytest.raises(VisionRateLimitError):
        service.extract_label(image_bytes())


def test_missing_openai_api_key_raises_configuration_error(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    service = OpenAIVisionService()

    with pytest.raises(VisionConfigurationError, match="OPENAI_API_KEY"):
        service.extract_label(image_bytes())


def test_model_smoke_check_uses_configured_model() -> None:
    def model_transport(timeout_seconds, model):
        return {"id": model, "object": "model", "timeout_seconds": timeout_seconds}

    service = OpenAIVisionService(
        model="gpt-5.4-nano",
        timeout_seconds=1.5,
        model_transport=model_transport,
    )

    assert service.check_model() == {
        "id": "gpt-5.4-nano",
        "object": "model",
        "timeout_seconds": 1.5,
    }


def openai_response(payload: dict[str, str | float | bool | None]) -> dict:
    return openai_text_response(json.dumps(payload))


def openai_text_response(text: str) -> dict:
    return {
        "text": {
            "format": {"type": "json_schema"},
            "verbosity": "medium",
        },
        "output": [
            {
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                    }
                ],
            }
        ]
    }
