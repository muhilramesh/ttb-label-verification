import asyncio
import logging
import os
import time
from typing import Any

import anyio
from fastapi import UploadFile
from pydantic import ValidationError

from backend.app.comparison import verify_label
from backend.app.config import batch_max_labels
from backend.app.errors import elapsed_ms
from backend.app.models import (
    ApplicationData, BatchItemError, BatchItemResult, BatchItemStatus,
    BatchSummary, BatchVerificationResult, VerificationVerdict,
)
from backend.app.validation import validate_image_metadata
from backend.app.vision import (
    VisionConfigurationError, VisionInvalidImageError, VisionParseError,
    VisionProviderError, VisionRateLimitError, VisionService, VisionTimeoutError,
)


logger = logging.getLogger("backend.app.verify")
MAX_IMAGE_BYTES = 8 * 1024 * 1024
DEFAULT_BATCH_CONCURRENCY = 3
BATCH_CONCURRENCY_ENV = "BATCH_CONCURRENCY"


class BatchUpload:
    def __init__(self, *, filename: str | None, content_type: str | None, image_bytes: bytes | None = None, error: BatchItemError | None = None) -> None:
        self.filename = filename
        self.content_type = content_type
        self.image_bytes = image_bytes
        self.error = error


async def process_batch(items: list[Any], images: list[UploadFile], image_ids: list[str], vision_service: VisionService, *, start: float) -> BatchVerificationResult:
    uploads = {image_id: await _read_upload(image) for image_id, image in zip(image_ids, images, strict=True)}
    semaphore = asyncio.Semaphore(_batch_concurrency())
    results = await asyncio.gather(*[
        _process_item(item, index=index, uploads=uploads, semaphore=semaphore, vision_service=vision_service)
        for index, item in enumerate(items)
    ])
    return _result(results, start)


async def _read_upload(image: UploadFile) -> BatchUpload:
    metadata_error = validate_image_metadata(image)
    if metadata_error:
        _, code, message = metadata_error
        return BatchUpload(filename=image.filename, content_type=image.content_type, error=BatchItemError(code=code, message=message))
    data = await image.read(MAX_IMAGE_BYTES + 1)
    if not data:
        return BatchUpload(filename=image.filename, content_type=image.content_type, error=BatchItemError(code="empty_image", message="Upload a non-empty label image."))
    if len(data) > MAX_IMAGE_BYTES:
        return BatchUpload(filename=image.filename, content_type=image.content_type, error=BatchItemError(code="image_too_large", message="Upload a label image smaller than 8 MB."))
    return BatchUpload(filename=image.filename, content_type=image.content_type, image_bytes=data)


async def _process_item(raw: Any, *, index: int, uploads: dict[str, BatchUpload], semaphore: asyncio.Semaphore, vision_service: VisionService) -> BatchItemResult:
    start = time.perf_counter()
    item_id = _item_id(raw, index)
    if not isinstance(raw, dict):
        return _error(item_id, None, "invalid_application_data", "Each batch item must be an object.", start)
    application = _application(raw)
    if isinstance(application, BatchItemError):
        return _error(item_id, None, application.code, application.message, start)
    upload = uploads.get(item_id)
    if upload is None:
        return _error(item_id, None, "missing_image", "Upload a label image for this item.", start)
    if upload.error:
        return _error(item_id, upload.filename, upload.error.code, upload.error.message, start)
    assert upload.image_bytes is not None
    try:
        async with semaphore:
            extracted = await anyio.to_thread.run_sync(vision_service.extract_label, upload.image_bytes, upload.content_type)
        verification = verify_label(application, extracted)
        verification.extracted_label = extracted
        verification.latency_ms = elapsed_ms(start)
        status = BatchItemStatus.APPROVED if verification.overall_verdict == VerificationVerdict.APPROVED else BatchItemStatus.NEEDS_REVIEW
        return BatchItemResult(id=item_id, filename=upload.filename, status=status, result=verification, error=None, latency_ms=elapsed_ms(start))
    except VisionInvalidImageError:
        return _error(item_id, upload.filename, "invalid_image", "Upload a readable label image.", start)
    except VisionTimeoutError:
        return _error(item_id, upload.filename, "vision_timeout", "The label image took too long to process. Try a clearer or smaller image.", start)
    except VisionConfigurationError:
        return _error(item_id, upload.filename, "vision_not_configured", "Vision service is not configured.", start)
    except VisionParseError:
        return _error(item_id, upload.filename, "vision_parse_error", "The vision service returned an unreadable extraction result.", start)
    except VisionRateLimitError:
        return _error(item_id, upload.filename, "vision_rate_limited", "The vision service is temporarily busy. Try again in a minute.", start)
    except VisionProviderError:
        return _error(item_id, upload.filename, "vision_provider_error", "The vision service could not process the label image.", start)
    except Exception:
        logger.exception("batch item failed unexpectedly id=%s", item_id)
        return _error(item_id, upload.filename, "verify_failed", "Verification failed unexpectedly.", start)


def _item_id(raw: Any, index: int) -> str:
    value = raw.get("id") if isinstance(raw, dict) else None
    return value.strip() if isinstance(value, str) and value.strip() else f"item-{index + 1}"


def _application(raw: dict[str, Any]) -> ApplicationData | BatchItemError:
    item_id = raw.get("id")
    if not isinstance(item_id, str) or not item_id.strip():
        return BatchItemError(code="invalid_application_data", message="Each batch item must include a non-empty id.")
    try:
        application = ApplicationData.model_validate({key: value for key, value in raw.items() if key != "id"})
    except ValidationError:
        return BatchItemError(code="invalid_application_data", message="Application data must include exactly the required label fields.")
    empty = [name for name, value in application.model_dump().items() if isinstance(value, str) and not value.strip()]
    if empty:
        return BatchItemError(code="invalid_application_data", message=f"Application data has empty required fields: {', '.join(empty)}.")
    return application


def _error(item_id: str, filename: str | None, code: str, message: str, start: float) -> BatchItemResult:
    return BatchItemResult(id=item_id, filename=filename, status=BatchItemStatus.ERROR, result=None, error=BatchItemError(code=code, message=message), latency_ms=elapsed_ms(start))


def _result(items: list[BatchItemResult], start: float) -> BatchVerificationResult:
    return BatchVerificationResult(items=items, summary=BatchSummary(
        passed=sum(item.status == BatchItemStatus.APPROVED for item in items),
        needs_review=sum(item.status == BatchItemStatus.NEEDS_REVIEW for item in items),
        errors=sum(item.status == BatchItemStatus.ERROR for item in items),
        total=len(items), latency_ms=elapsed_ms(start),
    ))


def _batch_concurrency() -> int:
    try:
        value = int(os.getenv(BATCH_CONCURRENCY_ENV, ""))
    except ValueError:
        return DEFAULT_BATCH_CONCURRENCY
    return min(value, batch_max_labels()) if value > 0 else DEFAULT_BATCH_CONCURRENCY
