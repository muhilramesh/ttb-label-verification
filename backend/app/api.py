from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

import anyio
from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from backend.app.comparison import verify_label
from backend.app.models import (
    ApplicationData,
    BatchItemError,
    BatchItemResult,
    BatchItemStatus,
    BatchSummary,
    BatchVerificationResult,
    VerificationResult,
    VerificationVerdict,
)
from backend.app.vision import (
    OpenAIVisionService,
    VisionConfigurationError,
    VisionInvalidImageError,
    VisionParseError,
    VisionProviderError,
    VisionRateLimitError,
    VisionService,
    VisionTimeoutError,
)


logger = logging.getLogger("backend.app.verify")

router = APIRouter()

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_BATCH_LABELS = 10
DEFAULT_BATCH_CONCURRENCY = 3
BATCH_CONCURRENCY_ENV = "BATCH_CONCURRENCY"


def get_vision_service() -> VisionService:
    return OpenAIVisionService()


@router.post("/verify", response_model=VerificationResult)
async def verify(
    image: UploadFile | None = File(default=None),
    application_data: str | None = Form(default=None),
    vision_service: VisionService = Depends(get_vision_service),
) -> VerificationResult | JSONResponse:
    start = time.perf_counter()

    image_error = _validate_image_metadata(image)
    if image_error is not None:
        return _error_response(*image_error, start=start)

    assert image is not None
    image_read_start = time.perf_counter()
    image_bytes = await image.read(MAX_IMAGE_BYTES + 1)
    image_read_ms = _latency_ms(image_read_start)
    if not image_bytes:
        return _error_response(
            400,
            "empty_image",
            "Upload a non-empty label image.",
            start=start,
        )
    if len(image_bytes) > MAX_IMAGE_BYTES:
        return _error_response(
            413,
            "image_too_large",
            "Upload a label image smaller than 8 MB.",
            start=start,
        )

    application = _parse_application_data(application_data)
    if isinstance(application, JSONResponse):
        return _with_latency(application, _latency_ms(start))

    try:
        extracted_label = vision_service.extract_label(
            image_bytes,
            content_type=image.content_type,
        )
        comparison_start = time.perf_counter()
        result = verify_label(application, extracted_label)
        comparison_ms = _latency_ms(comparison_start)
        result.extracted_label = extracted_label
        result.latency_ms = _latency_ms(start)
        logger.info(
            "verify completed verdict=%s latency_ms=%s image_read_ms=%s "
            "comparison_ms=%s upload_bytes=%s",
            result.overall_verdict,
            result.latency_ms,
            image_read_ms,
            comparison_ms,
            len(image_bytes),
        )
        return result
    except VisionInvalidImageError:
        return _error_response(
            400,
            "invalid_image",
            "Upload a readable label image.",
            start=start,
        )
    except VisionTimeoutError:
        return _error_response(
            504,
            "vision_timeout",
            "The label image took too long to process. Try a clearer or smaller image.",
            start=start,
        )
    except VisionConfigurationError:
        return _error_response(
            503,
            "vision_not_configured",
            "Vision service is not configured.",
            start=start,
        )
    except VisionParseError:
        return _error_response(
            502,
            "vision_parse_error",
            "The vision service returned an unreadable extraction result.",
            start=start,
        )
    except VisionRateLimitError:
        return _error_response(
            503,
            "vision_rate_limited",
            "The vision service is temporarily busy. Try again in a minute.",
            start=start,
        )
    except VisionProviderError:
        return _error_response(
            502,
            "vision_provider_error",
            "The vision service could not process the label image.",
            start=start,
        )
    except Exception:
        logger.exception("verify failed unexpectedly latency_ms=%s", _latency_ms(start))
        return _error_response(
            500,
            "verify_failed",
            "Verification failed unexpectedly.",
            start=start,
            log_error=False,
        )


@router.post("/verify/batch", response_model=BatchVerificationResult)
async def verify_batch(
    images: list[UploadFile] | None = File(default=None),
    image_ids: str | None = Form(default=None),
    application_data: str | None = Form(default=None),
    vision_service: VisionService = Depends(get_vision_service),
) -> BatchVerificationResult | JSONResponse:
    start = time.perf_counter()

    parsed_items = _parse_batch_application_data(application_data)
    if isinstance(parsed_items, JSONResponse):
        return _with_latency(parsed_items, _latency_ms(start))

    if len(parsed_items) > MAX_BATCH_LABELS:
        return _error_response(
            413,
            "batch_too_large",
            f"Check at most {MAX_BATCH_LABELS} labels at a time.",
            start=start,
        )

    parsed_image_ids = _parse_batch_image_ids(image_ids, len(images or []))
    if isinstance(parsed_image_ids, JSONResponse):
        return _with_latency(parsed_image_ids, _latency_ms(start))

    uploads = await _read_batch_uploads(images or [], parsed_image_ids)
    semaphore = asyncio.Semaphore(_batch_concurrency_from_env())
    item_results = await asyncio.gather(
        *[
            _process_batch_item(
                raw_item,
                index=index,
                uploads=uploads,
                semaphore=semaphore,
                vision_service=vision_service,
            )
            for index, raw_item in enumerate(parsed_items)
        ]
    )

    result = _batch_result(item_results, start=start)
    logger.info(
        "verify batch completed total=%s passed=%s needs_review=%s errors=%s latency_ms=%s",
        result.summary.total,
        result.summary.passed,
        result.summary.needs_review,
        result.summary.errors,
        result.summary.latency_ms,
    )
    return result


def _validate_image_metadata(
    image: UploadFile | None,
) -> tuple[int, str, str] | None:
    if image is None:
        return 400, "missing_image", "Upload a label image."

    if image.content_type not in ALLOWED_IMAGE_TYPES:
        return (
            400,
            "invalid_image_type",
            "Upload a JPEG, PNG, or WebP label image.",
        )

    return None


def _parse_application_data(application_data: str | None) -> ApplicationData | JSONResponse:
    if application_data is None:
        return _plain_error_response(
            400,
            "missing_application_data",
            "Include application data for this label.",
        )

    try:
        payload = json.loads(application_data)
    except json.JSONDecodeError:
        return _plain_error_response(
            400,
            "invalid_application_data",
            "Application data must be valid JSON.",
        )

    try:
        application = ApplicationData.model_validate(payload)
    except ValidationError:
        return _plain_error_response(
            400,
            "invalid_application_data",
            "Application data must include exactly the required label fields.",
        )

    empty_fields = [
        field_name
        for field_name, value in application.model_dump().items()
        if isinstance(value, str) and not value.strip()
    ]
    if empty_fields:
        return _plain_error_response(
            400,
            "invalid_application_data",
            f"Application data has empty required fields: {', '.join(empty_fields)}.",
        )

    return application


def _parse_batch_application_data(application_data: str | None) -> list[Any] | JSONResponse:
    if application_data is None:
        return _plain_error_response(
            400,
            "missing_application_data",
            "Include application data for this batch.",
        )

    try:
        payload = json.loads(application_data)
    except json.JSONDecodeError:
        return _plain_error_response(
            400,
            "invalid_application_data",
            "Batch application data must be valid JSON.",
        )

    if not isinstance(payload, list):
        return _plain_error_response(
            400,
            "invalid_application_data",
            "Batch application data must be a list.",
        )

    if not payload:
        return _plain_error_response(
            400,
            "empty_batch",
            "Add at least one label to check.",
        )

    item_ids = [
        item.get("id").strip()
        for item in payload
        if isinstance(item, dict)
        and isinstance(item.get("id"), str)
        and item.get("id").strip()
    ]
    if len(item_ids) != len(set(item_ids)):
        return _plain_error_response(
            400,
            "duplicate_application_ids",
            "Batch application IDs must be unique.",
        )

    return payload


def _parse_batch_image_ids(
    image_ids: str | None,
    image_count: int,
) -> list[str] | JSONResponse:
    if image_count > MAX_BATCH_LABELS:
        return _plain_error_response(
            413,
            "batch_too_large",
            f"Check at most {MAX_BATCH_LABELS} labels at a time.",
        )

    if image_ids is None:
        return _plain_error_response(
            400,
            "missing_image_ids",
            "Include image IDs for this batch.",
        )

    try:
        payload = json.loads(image_ids)
    except json.JSONDecodeError:
        return _plain_error_response(
            400,
            "invalid_image_ids",
            "Batch image IDs must be valid JSON.",
        )

    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        return _plain_error_response(
            400,
            "invalid_image_ids",
            "Batch image IDs must be a list of text IDs.",
        )

    if len(payload) != image_count:
        return _plain_error_response(
            400,
            "invalid_image_ids",
            "Batch image IDs must match the uploaded images.",
        )

    normalized = [item.strip() for item in payload]
    if any(not item for item in normalized):
        return _plain_error_response(
            400,
            "invalid_image_ids",
            "Batch image IDs cannot be empty.",
        )

    if len(set(normalized)) != len(normalized):
        return _plain_error_response(
            400,
            "duplicate_image_ids",
            "Batch image IDs must be unique.",
        )

    return normalized


class BatchUpload:
    def __init__(
        self,
        *,
        filename: str | None,
        content_type: str | None,
        image_bytes: bytes | None = None,
        error: BatchItemError | None = None,
    ) -> None:
        self.filename = filename
        self.content_type = content_type
        self.image_bytes = image_bytes
        self.error = error


async def _read_batch_uploads(
    images: list[UploadFile],
    image_ids: list[str],
) -> dict[str, BatchUpload]:
    uploads: dict[str, BatchUpload] = {}
    for image_id, image in zip(image_ids, images, strict=True):
        uploads[image_id] = await _read_upload(image)
    return uploads


async def _read_upload(image: UploadFile) -> BatchUpload:
    metadata_error = _validate_image_metadata(image)
    if metadata_error is not None:
        _, code, message = metadata_error
        return BatchUpload(
            filename=image.filename,
            content_type=image.content_type,
            error=BatchItemError(code=code, message=message),
        )

    image_bytes = await image.read(MAX_IMAGE_BYTES + 1)
    if not image_bytes:
        return BatchUpload(
            filename=image.filename,
            content_type=image.content_type,
            error=BatchItemError(
                code="empty_image",
                message="Upload a non-empty label image.",
            ),
        )
    if len(image_bytes) > MAX_IMAGE_BYTES:
        return BatchUpload(
            filename=image.filename,
            content_type=image.content_type,
            error=BatchItemError(
                code="image_too_large",
                message="Upload a label image smaller than 8 MB.",
            ),
        )

    return BatchUpload(
        filename=image.filename,
        content_type=image.content_type,
        image_bytes=image_bytes,
    )


def _status_for_upload_error(code: str) -> int:
    if code == "image_too_large":
        return 413
    return 400


async def _process_batch_item(
    raw_item: Any,
    *,
    index: int,
    uploads: dict[str, BatchUpload],
    semaphore: asyncio.Semaphore,
    vision_service: VisionService,
) -> BatchItemResult:
    start = time.perf_counter()
    item_id = _batch_item_id(raw_item, index)

    if not isinstance(raw_item, dict):
        return _batch_item_error(
            item_id,
            None,
            "invalid_application_data",
            "Each batch item must be an object.",
            start=start,
        )

    application = _batch_item_application(raw_item)
    if isinstance(application, BatchItemError):
        return _batch_item_error(
            item_id,
            None,
            application.code,
            application.message,
            start=start,
        )

    upload = uploads.get(item_id)
    if upload is None:
        return _batch_item_error(
            item_id,
            None,
            "missing_image",
            "Upload a label image for this item.",
            start=start,
        )

    if upload.error is not None:
        return _batch_item_error(
            item_id,
            upload.filename,
            upload.error.code,
            upload.error.message,
            start=start,
        )

    assert upload.image_bytes is not None
    try:
        async with semaphore:
            extracted_label = await anyio.to_thread.run_sync(
                vision_service.extract_label,
                upload.image_bytes,
                upload.content_type,
            )
        verification = verify_label(application, extracted_label)
        verification.extracted_label = extracted_label
        verification.latency_ms = _latency_ms(start)
        return BatchItemResult(
            id=item_id,
            filename=upload.filename,
            status=(
                BatchItemStatus.APPROVED
                if verification.overall_verdict == VerificationVerdict.APPROVED
                else BatchItemStatus.NEEDS_REVIEW
            ),
            result=verification,
            error=None,
            latency_ms=_latency_ms(start),
        )
    except VisionInvalidImageError:
        return _batch_item_error(
            item_id,
            upload.filename,
            "invalid_image",
            "Upload a readable label image.",
            start=start,
        )
    except VisionTimeoutError:
        return _batch_item_error(
            item_id,
            upload.filename,
            "vision_timeout",
            "The label image took too long to process. Try a clearer or smaller image.",
            start=start,
        )
    except VisionConfigurationError:
        return _batch_item_error(
            item_id,
            upload.filename,
            "vision_not_configured",
            "Vision service is not configured.",
            start=start,
        )
    except VisionParseError:
        return _batch_item_error(
            item_id,
            upload.filename,
            "vision_parse_error",
            "The vision service returned an unreadable extraction result.",
            start=start,
        )
    except VisionRateLimitError:
        return _batch_item_error(
            item_id,
            upload.filename,
            "vision_rate_limited",
            "The vision service is temporarily busy. Try again in a minute.",
            start=start,
        )
    except VisionProviderError:
        return _batch_item_error(
            item_id,
            upload.filename,
            "vision_provider_error",
            "The vision service could not process the label image.",
            start=start,
        )
    except Exception:
        logger.exception("batch item failed unexpectedly id=%s", item_id)
        return _batch_item_error(
            item_id,
            upload.filename,
            "verify_failed",
            "Verification failed unexpectedly.",
            start=start,
        )


def _batch_item_id(raw_item: Any, index: int) -> str:
    if isinstance(raw_item, dict):
        item_id = raw_item.get("id")
        if isinstance(item_id, str) and item_id.strip():
            return item_id.strip()
    return f"item-{index + 1}"


def _batch_item_application(raw_item: dict[str, Any]) -> ApplicationData | BatchItemError:
    item_id = raw_item.get("id")
    if not isinstance(item_id, str) or not item_id.strip():
        return BatchItemError(
            code="invalid_application_data",
            message="Each batch item must include a non-empty id.",
        )

    payload = {key: value for key, value in raw_item.items() if key != "id"}
    try:
        application = ApplicationData.model_validate(payload)
    except ValidationError:
        return BatchItemError(
            code="invalid_application_data",
            message="Application data must include exactly the required label fields.",
        )

    empty_fields = [
        field_name
        for field_name, value in application.model_dump().items()
        if isinstance(value, str) and not value.strip()
    ]
    if empty_fields:
        return BatchItemError(
            code="invalid_application_data",
            message=(
                "Application data has empty required fields: "
                f"{', '.join(empty_fields)}."
            ),
        )

    return application


def _batch_item_error(
    item_id: str,
    filename: str | None,
    code: str,
    message: str,
    *,
    start: float,
) -> BatchItemResult:
    return BatchItemResult(
        id=item_id,
        filename=filename,
        status=BatchItemStatus.ERROR,
        result=None,
        error=BatchItemError(code=code, message=message),
        latency_ms=_latency_ms(start),
    )


def _batch_result(
    item_results: list[BatchItemResult],
    *,
    start: float,
) -> BatchVerificationResult:
    passed = sum(1 for item in item_results if item.status == BatchItemStatus.APPROVED)
    needs_review = sum(
        1 for item in item_results if item.status == BatchItemStatus.NEEDS_REVIEW
    )
    errors = sum(1 for item in item_results if item.status == BatchItemStatus.ERROR)
    return BatchVerificationResult(
        items=item_results,
        summary=BatchSummary(
            passed=passed,
            needs_review=needs_review,
            total=len(item_results),
            errors=errors,
            latency_ms=_latency_ms(start),
        ),
    )


def _batch_concurrency_from_env() -> int:
    raw_concurrency = os.getenv(BATCH_CONCURRENCY_ENV)
    if raw_concurrency is None or not raw_concurrency.strip():
        return DEFAULT_BATCH_CONCURRENCY

    try:
        concurrency = int(raw_concurrency)
    except ValueError:
        return DEFAULT_BATCH_CONCURRENCY

    if concurrency <= 0:
        return DEFAULT_BATCH_CONCURRENCY
    return min(concurrency, MAX_BATCH_LABELS)


def _error_response(
    status_code: int,
    code: str,
    message: str,
    *,
    start: float,
    log_error: bool = True,
) -> JSONResponse:
    latency_ms = _latency_ms(start)
    response = _plain_error_response(status_code, code, message)
    response = _with_latency(response, latency_ms)
    if log_error:
        logger.warning(
            "verify failed code=%s status_code=%s latency_ms=%s",
            code,
            status_code,
            latency_ms,
        )
    return response


def _plain_error_response(
    status_code: int,
    code: str,
    message: str,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
            }
        },
    )


def _with_latency(response: JSONResponse, latency_ms: int | float) -> JSONResponse:
    body = json.loads(response.body)
    body["latency_ms"] = int(latency_ms)
    return JSONResponse(status_code=response.status_code, content=body)


def _latency_ms(start: float) -> int:
    return max(0, int((time.perf_counter() - start) * 1000))
