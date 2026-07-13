import json
from typing import Any

from fastapi import UploadFile
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from backend.app.config import batch_max_labels
from backend.app.errors import plain_error_response
from backend.app.models import ApplicationData


ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}


def validate_image_metadata(image: UploadFile | None) -> tuple[int, str, str] | None:
    if image is None:
        return 400, "missing_image", "Upload a label image."
    if image.content_type not in ALLOWED_IMAGE_TYPES:
        return 400, "invalid_image_type", "Upload a JPEG, PNG, WebP, HEIC, or HEIF label image."
    return None


def parse_application_data(value: str | None) -> ApplicationData | JSONResponse:
    if value is None:
        return plain_error_response(400, "missing_application_data", "Include application data for this label.")
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return plain_error_response(400, "invalid_application_data", "Application data must be valid JSON.")
    try:
        application = ApplicationData.model_validate(payload)
    except ValidationError:
        return plain_error_response(400, "invalid_application_data", "Application data must include exactly the required label fields.")
    empty = [name for name, item in application.model_dump().items() if isinstance(item, str) and not item.strip()]
    if empty:
        return plain_error_response(400, "invalid_application_data", f"Application data has empty required fields: {', '.join(empty)}.")
    return application


def parse_batch_application_data(value: str | None) -> list[Any] | JSONResponse:
    if value is None:
        return plain_error_response(400, "missing_application_data", "Include application data for this batch.")
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return plain_error_response(400, "invalid_application_data", "Batch application data must be valid JSON.")
    if not isinstance(payload, list):
        return plain_error_response(400, "invalid_application_data", "Batch application data must be a list.")
    if not payload:
        return plain_error_response(400, "empty_batch", "Add at least one label to check.")
    ids = [item.get("id").strip() for item in payload if isinstance(item, dict) and isinstance(item.get("id"), str) and item.get("id").strip()]
    if len(ids) != len(set(ids)):
        return plain_error_response(400, "duplicate_application_ids", "Batch application IDs must be unique.")
    return payload


def parse_batch_image_ids(value: str | None, image_count: int) -> list[str] | JSONResponse:
    maximum = batch_max_labels()
    if image_count > maximum:
        return plain_error_response(413, "batch_too_large", f"Check at most {maximum} labels at a time.")
    if value is None:
        return plain_error_response(400, "missing_image_ids", "Include image IDs for this batch.")
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return plain_error_response(400, "invalid_image_ids", "Batch image IDs must be valid JSON.")
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        return plain_error_response(400, "invalid_image_ids", "Batch image IDs must be a list of text IDs.")
    if len(payload) != image_count:
        return plain_error_response(400, "invalid_image_ids", "Batch image IDs must match the uploaded images.")
    normalized = [item.strip() for item in payload]
    if any(not item for item in normalized):
        return plain_error_response(400, "invalid_image_ids", "Batch image IDs cannot be empty.")
    if len(set(normalized)) != len(normalized):
        return plain_error_response(400, "duplicate_image_ids", "Batch image IDs must be unique.")
    return normalized
