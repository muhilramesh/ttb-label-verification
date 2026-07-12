from __future__ import annotations

import base64
from dataclasses import dataclass, field
from io import BytesIO
import json
import logging
import os
import socket
import time
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import ValidationError

from backend.app.models import ExtractedLabel


logger = logging.getLogger("backend.app.vision")

DEFAULT_VISION_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_TIMEOUT_SECONDS = 4.5
OPENAI_MODEL_ENV = "OPENAI_MODEL"
OPENAI_TIMEOUT_ENV = "OPENAI_TIMEOUT_SECONDS"
IMAGE_MAX_LONG_SIDE_ENV = "IMAGE_MAX_LONG_SIDE"
IMAGE_JPEG_QUALITY_ENV = "IMAGE_JPEG_QUALITY"
OPENAI_RESPONSES_ENDPOINT = "https://api.openai.com/v1/responses"
OPENAI_MODELS_ENDPOINT = "https://api.openai.com/v1/models/{model}"
MAX_IMAGE_LONG_SIDE = 768
JPEG_QUALITY = 70

EXTRACTED_LABEL_FIELDS = (
    "brand_name",
    "class_type",
    "producer",
    "country_of_origin",
    "abv",
    "net_contents",
    "government_warning",
    "raw_text",
    "extraction_confidence",
)

VISION_EXTRACTION_PROMPT = """
Extract visible alcohol-label text into JSON fields:
brand_name, class_type, producer, country_of_origin, abv, net_contents, government_warning,
raw_text, extraction_confidence.
Use null for unknown, unreadable, or non-label fields. Do not infer, correct, normalize, or guess.
For raw_text, copy all visible label text you can read from the image.
For extraction_confidence, return a number from 0 to 1 for overall extraction confidence.
For government_warning, copy visible warning text character-for-character, preserving capitalization,
punctuation, spacing, line breaks, and OCR-like mistakes. Do not complete it from memory.
""".strip()


class VisionServiceError(Exception):
    """Base error for image extraction failures."""


class VisionConfigurationError(VisionServiceError):
    """Raised when the vision service is missing required configuration."""


class VisionInvalidImageError(VisionServiceError):
    """Raised when the uploaded bytes are not a valid image."""


class VisionTimeoutError(VisionServiceError):
    """Raised when the model call times out."""


class VisionRateLimitError(VisionServiceError):
    """Raised when the model provider rejects the request for quota/rate limits."""


class VisionProviderError(VisionServiceError):
    """Raised when the model provider fails before returning usable output."""


class VisionParseError(VisionServiceError):
    """Raised when provider output cannot be validated as an ExtractedLabel."""


class VisionService(Protocol):
    def extract_label(
        self,
        image_bytes: bytes,
        content_type: str | None = None,
    ) -> ExtractedLabel:
        ...


@dataclass(frozen=True)
class ProcessedImage:
    data: bytes
    content_type: str
    data_url: str
    original_width: int
    original_height: int
    width: int
    height: int


@dataclass(frozen=True)
class VisionServiceCall:
    image_bytes: bytes
    content_type: str | None


@dataclass
class FakeVisionService:
    label: ExtractedLabel = field(default_factory=ExtractedLabel)
    error: Exception | None = None
    calls: list[VisionServiceCall] = field(default_factory=list)

    def extract_label(
        self,
        image_bytes: bytes,
        content_type: str | None = None,
    ) -> ExtractedLabel:
        self.calls.append(
            VisionServiceCall(image_bytes=image_bytes, content_type=content_type)
        )
        if self.error is not None:
            raise self.error
        return self.label


def extracted_label_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(EXTRACTED_LABEL_FIELDS),
        "properties": {
            field_name: _extracted_label_field_schema(field_name)
            for field_name in EXTRACTED_LABEL_FIELDS
        },
    }


def _extracted_label_field_schema(field_name: str) -> dict[str, Any]:
    if field_name == "extraction_confidence":
        return {"type": ["number", "null"], "minimum": 0, "maximum": 1}
    return {"type": ["string", "null"]}


def preprocess_label_image(
    image_bytes: bytes,
    *,
    max_long_side: int | None = None,
    jpeg_quality: int | None = None,
) -> ProcessedImage:
    if not image_bytes:
        raise VisionInvalidImageError("Image upload is empty.")

    max_long_side = max_long_side or _int_from_env(
        IMAGE_MAX_LONG_SIDE_ENV,
        default=MAX_IMAGE_LONG_SIDE,
        minimum=640,
        maximum=2000,
    )
    jpeg_quality = jpeg_quality or _int_from_env(
        IMAGE_JPEG_QUALITY_ENV,
        default=JPEG_QUALITY,
        minimum=60,
        maximum=95,
    )

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image = ImageOps.exif_transpose(image)
            original_width, original_height = image.size
            image = _to_rgb_on_white(image)
            image.thumbnail(
                (max_long_side, max_long_side),
                Image.Resampling.LANCZOS,
            )

            output = BytesIO()
            image.save(
                output,
                format="JPEG",
                quality=jpeg_quality,
                optimize=True,
            )
    except (UnidentifiedImageError, OSError) as exc:
        raise VisionInvalidImageError("Image upload is not a readable image.") from exc

    data = output.getvalue()
    data_url = (
        "data:image/jpeg;base64,"
        f"{base64.b64encode(data).decode('ascii')}"
    )
    return ProcessedImage(
        data=data,
        content_type="image/jpeg",
        data_url=data_url,
        original_width=original_width,
        original_height=original_height,
        width=image.width,
        height=image.height,
    )


def _to_rgb_on_white(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return image.convert("RGB")


def _structured_output_format() -> dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": "extracted_label",
            "strict": True,
            "schema": extracted_label_json_schema(),
        }
    }


class OpenAIVisionService:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        transport: Any | None = None,
        model_transport: Any | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model or os.getenv(OPENAI_MODEL_ENV, DEFAULT_VISION_MODEL)
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else _openai_timeout_seconds_from_env()
        )
        self._transport = transport
        self._model_transport = model_transport

    def extract_label(
        self,
        image_bytes: bytes,
        content_type: str | None = None,
    ) -> ExtractedLabel:
        start = time.perf_counter()
        processed_image = preprocess_label_image(image_bytes)
        preprocess_ms = _elapsed_ms(start)
        provider_start = time.perf_counter()
        try:
            response = self._create_response(processed_image)
        except VisionTimeoutError:
            logger.warning(
                "vision extraction timed out model=%s timeout_seconds=%s "
                "input_bytes=%s processed_bytes=%s original_size=%sx%s "
                "processed_size=%sx%s preprocess_ms=%s total_ms=%s",
                self.model,
                self.timeout_seconds,
                len(image_bytes),
                len(processed_image.data),
                processed_image.original_width,
                processed_image.original_height,
                processed_image.width,
                processed_image.height,
                preprocess_ms,
                _elapsed_ms(start),
            )
            raise
        except (VisionProviderError, VisionRateLimitError) as exc:
            logger.warning(
                "vision extraction provider failed model=%s input_bytes=%s "
                "processed_bytes=%s original_size=%sx%s processed_size=%sx%s "
                "preprocess_ms=%s total_ms=%s detail=%s",
                self.model,
                len(image_bytes),
                len(processed_image.data),
                processed_image.original_width,
                processed_image.original_height,
                processed_image.width,
                processed_image.height,
                preprocess_ms,
                _elapsed_ms(start),
                exc,
            )
            raise
        provider_ms = _elapsed_ms(provider_start)
        parse_start = time.perf_counter()
        label = _parse_extracted_label_response(response)
        parse_ms = _elapsed_ms(parse_start)
        logger.info(
            "vision extraction completed model=%s input_bytes=%s processed_bytes=%s "
            "original_size=%sx%s processed_size=%sx%s preprocess_ms=%s provider_ms=%s "
            "parse_ms=%s total_ms=%s",
            self.model,
            len(image_bytes),
            len(processed_image.data),
            processed_image.original_width,
            processed_image.original_height,
            processed_image.width,
            processed_image.height,
            preprocess_ms,
            provider_ms,
            parse_ms,
            _elapsed_ms(start),
        )
        return label

    def check_model(self) -> dict[str, Any]:
        try:
            if self._model_transport is not None:
                return self._model_transport(self.timeout_seconds, self.model)
            return self._get_openai_model()
        except VisionServiceError:
            raise
        except Exception as exc:
            if _is_timeout_error(exc):
                raise VisionTimeoutError("Vision model smoke check timed out.") from exc
            raise VisionProviderError("Vision model smoke check failed.") from exc

    def _create_response(self, processed_image: ProcessedImage) -> Any:
        request_body = _build_openai_request_body(processed_image, model=self.model)
        try:
            if self._transport is not None:
                return self._transport(request_body, self.timeout_seconds, self.model)
            return self._post_openai_request(request_body)
        except VisionServiceError:
            raise
        except Exception as exc:
            if _is_timeout_error(exc):
                raise VisionTimeoutError("Vision model request timed out.") from exc
            raise VisionProviderError("Vision model request failed.") from exc

    def _post_openai_request(self, request_body: dict[str, Any]) -> Any:
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise VisionConfigurationError("OPENAI_API_KEY is not configured.")

        request = Request(
            OPENAI_RESPONSES_ENDPOINT,
            data=json.dumps(request_body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429:
                raise VisionRateLimitError(
                    f"OpenAI API quota or rate limit was reached: {detail}"
                ) from exc
            raise VisionProviderError(f"OpenAI API request failed: {detail}") from exc
        except json.JSONDecodeError as exc:
            raise VisionParseError("OpenAI API response was not valid JSON.") from exc

    def _get_openai_model(self) -> dict[str, Any]:
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise VisionConfigurationError("OPENAI_API_KEY is not configured.")

        request = Request(
            OPENAI_MODELS_ENDPOINT.format(model=quote(self.model, safe="")),
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            method="GET",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429:
                raise VisionRateLimitError(
                    f"OpenAI API quota or rate limit was reached: {detail}"
                ) from exc
            raise VisionProviderError(f"OpenAI model check failed: {detail}") from exc
        except json.JSONDecodeError as exc:
            raise VisionParseError("OpenAI model check response was not valid JSON.") from exc


def _build_openai_request_body(
    processed_image: ProcessedImage,
    *,
    model: str,
) -> dict[str, Any]:
    return {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": VISION_EXTRACTION_PROMPT},
                    {
                        "type": "input_image",
                        "image_url": processed_image.data_url,
                        "detail": "low",
                    },
                ],
            }
        ],
        "text": _structured_output_format(),
        "temperature": 0,
        "max_output_tokens": 900,
        "store": False,
    }


def _is_timeout_error(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError | socket.timeout):
        return True
    if isinstance(exc, URLError) and isinstance(exc.reason, TimeoutError | socket.timeout):
        return True
    return exc.__class__.__name__ in {
        "APITimeoutError",
        "ReadTimeout",
        "Timeout",
        "TimeoutError",
    }


def _elapsed_ms(start: float) -> int:
    return max(0, int((time.perf_counter() - start) * 1000))


def _openai_timeout_seconds_from_env() -> float:
    raw_timeout = os.getenv(OPENAI_TIMEOUT_ENV)
    if raw_timeout is None or not raw_timeout.strip():
        return DEFAULT_OPENAI_TIMEOUT_SECONDS

    try:
        timeout_seconds = float(raw_timeout)
    except ValueError:
        return DEFAULT_OPENAI_TIMEOUT_SECONDS

    if timeout_seconds <= 0:
        return DEFAULT_OPENAI_TIMEOUT_SECONDS

    return timeout_seconds


def _int_from_env(
    env_name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw_value = os.getenv(env_name)
    if raw_value is None or not raw_value.strip():
        return default

    try:
        value = int(raw_value)
    except ValueError:
        return default

    if value < minimum or value > maximum:
        return default

    return value


def _parse_extracted_label_response(response: Any) -> ExtractedLabel:
    payload = _extract_structured_payload(response)
    return _parse_extracted_label_payload(payload)


def _extract_structured_payload(response: Any) -> Any:
    direct_payload = _get_value(response, "output_parsed")
    if direct_payload is not None:
        return direct_payload

    output_text = _get_value(response, "output_text")
    if output_text:
        return output_text

    candidates = _get_value(response, "candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            content = _get_value(candidate, "content")
            parts = _get_value(content, "parts")
            if not isinstance(parts, list):
                continue
            for part in parts:
                text = _get_value(part, "text")
                if text:
                    return text

    output = _get_value(response, "output")
    if isinstance(output, list):
        for output_item in output:
            content = _get_value(output_item, "content")
            if not isinstance(content, list):
                continue
            for content_item in content:
                parsed = _get_value(content_item, "parsed")
                if parsed is not None:
                    return parsed
                text = _get_value(content_item, "text")
                if text:
                    return text

    text = _get_value(response, "text")
    if isinstance(text, str) and text:
        return text

    raise VisionParseError("Vision response did not contain structured output.")


def _get_value(source: Any, key: str) -> Any:
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


def _parse_extracted_label_payload(payload: Any) -> ExtractedLabel:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise VisionParseError("Vision response was not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise VisionParseError("Vision response JSON was not an object.")

    expected_fields = set(EXTRACTED_LABEL_FIELDS)
    actual_fields = set(payload)
    if actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields)
        extra = sorted(actual_fields - expected_fields)
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if extra:
            details.append(f"extra fields: {', '.join(extra)}")
        raise VisionParseError("; ".join(details))

    try:
        return ExtractedLabel.model_validate(payload)
    except ValidationError as exc:
        raise VisionParseError("Vision response did not match ExtractedLabel.") from exc
