import logging
import time

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import JSONResponse

from backend.app.batch import process_batch
from backend.app.comparison import verify_label
from backend.app.config import DEFAULT_BATCH_MAX_LABELS, batch_max_labels
from backend.app.errors import elapsed_ms, error_response, with_latency
from backend.app.models import BatchVerificationResult, VerificationResult
from backend.app.validation import (
    parse_application_data,
    parse_batch_application_data,
    parse_batch_image_ids,
    validate_image_metadata,
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
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_BATCH_LABELS = DEFAULT_BATCH_MAX_LABELS


def get_vision_service() -> VisionService:
    return OpenAIVisionService()


@router.post("/verify", response_model=VerificationResult)
async def verify(
    image: UploadFile | None = File(default=None),
    application_data: str | None = Form(default=None),
    vision_service: VisionService = Depends(get_vision_service),
) -> VerificationResult | JSONResponse:
    start = time.perf_counter()
    metadata_error = validate_image_metadata(image)
    if metadata_error:
        return error_response(*metadata_error, start=start)
    assert image is not None
    image_read_start = time.perf_counter()
    image_bytes = await image.read(MAX_IMAGE_BYTES + 1)
    image_read_ms = elapsed_ms(image_read_start)
    if not image_bytes:
        return error_response(400, "empty_image", "Upload a non-empty label image.", start=start)
    if len(image_bytes) > MAX_IMAGE_BYTES:
        return error_response(413, "image_too_large", "Upload a label image smaller than 8 MB.", start=start)
    application = parse_application_data(application_data)
    if isinstance(application, JSONResponse):
        return with_latency(application, elapsed_ms(start))
    try:
        extracted = vision_service.extract_label(image_bytes, content_type=image.content_type)
        comparison_start = time.perf_counter()
        result = verify_label(application, extracted)
        result.extracted_label = extracted
        result.latency_ms = elapsed_ms(start)
        logger.info(
            "verify completed verdict=%s latency_ms=%s image_read_ms=%s comparison_ms=%s upload_bytes=%s",
            result.overall_verdict, result.latency_ms, image_read_ms,
            elapsed_ms(comparison_start), len(image_bytes),
        )
        return result
    except VisionInvalidImageError:
        return error_response(400, "invalid_image", "Upload a readable label image.", start=start)
    except VisionTimeoutError:
        return error_response(504, "vision_timeout", "The label image took too long to process. Try a clearer or smaller image.", start=start)
    except VisionConfigurationError:
        return error_response(503, "vision_not_configured", "Vision service is not configured.", start=start)
    except VisionParseError:
        return error_response(502, "vision_parse_error", "The vision service returned an unreadable extraction result.", start=start)
    except VisionRateLimitError:
        return error_response(503, "vision_rate_limited", "The vision service is temporarily busy. Try again in a minute.", start=start)
    except VisionProviderError:
        return error_response(502, "vision_provider_error", "The vision service could not process the label image.", start=start)
    except Exception:
        logger.exception("verify failed unexpectedly latency_ms=%s", elapsed_ms(start))
        return error_response(500, "verify_failed", "Verification failed unexpectedly.", start=start, log_error=False)


@router.post("/verify/batch", response_model=BatchVerificationResult)
async def verify_batch(
    images: list[UploadFile] | None = File(default=None),
    image_ids: str | None = Form(default=None),
    application_data: str | None = Form(default=None),
    vision_service: VisionService = Depends(get_vision_service),
) -> BatchVerificationResult | JSONResponse:
    start = time.perf_counter()
    items = parse_batch_application_data(application_data)
    if isinstance(items, JSONResponse):
        return with_latency(items, elapsed_ms(start))
    maximum = batch_max_labels()
    if len(items) > maximum:
        return error_response(413, "batch_too_large", f"Check at most {maximum} labels at a time.", start=start)
    parsed_ids = parse_batch_image_ids(image_ids, len(images or []))
    if isinstance(parsed_ids, JSONResponse):
        return with_latency(parsed_ids, elapsed_ms(start))
    result = await process_batch(items, images or [], parsed_ids, vision_service, start=start)
    logger.info(
        "verify batch completed total=%s passed=%s needs_review=%s errors=%s latency_ms=%s",
        result.summary.total, result.summary.passed, result.summary.needs_review,
        result.summary.errors, result.summary.latency_ms,
    )
    return result
