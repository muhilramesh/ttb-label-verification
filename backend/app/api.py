"""Compatibility exports for API routes and dependency overrides."""

from backend.app.routes import MAX_BATCH_LABELS, MAX_IMAGE_BYTES, get_vision_service, router

__all__ = ["MAX_BATCH_LABELS", "MAX_IMAGE_BYTES", "get_vision_service", "router"]
