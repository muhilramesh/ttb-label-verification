import os
import logging
from pathlib import Path
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api import router as api_router
from backend.app.config import batch_max_labels
from backend.app.vision import (
    DEFAULT_VISION_MODEL,
    OPENAI_MODEL_ENV,
    OpenAIVisionService,
    VisionConfigurationError,
    VisionParseError,
    VisionProviderError,
    VisionRateLimitError,
    VisionTimeoutError,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT_DIR / "frontend"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(levelname)s:%(name)s:%(message)s",
)

app = FastAPI(title="TTB Label Verification", version="0.1.0")
app.include_router(api_router)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/health")
def health() -> dict[str, str | int]:
    return {
        "status": "ok",
        "service": "ttb-label-verification",
        "environment": os.getenv("APP_ENV", "local"),
        "batch_max_labels": batch_max_labels(),
    }


@app.get("/health/deep", response_model=None)
def health_deep() -> dict[str, str | int] | JSONResponse:
    start = time.perf_counter()
    model = os.getenv(OPENAI_MODEL_ENV, DEFAULT_VISION_MODEL)
    service = OpenAIVisionService(model=model)

    try:
        model_info = service.check_model()
    except VisionConfigurationError:
        return _deep_health_error(
            "not_configured",
            model,
            start,
            "OpenAI API key is not configured.",
        )
    except VisionTimeoutError:
        return _deep_health_error(
            "timeout",
            model,
            start,
            "OpenAI model check timed out.",
        )
    except VisionRateLimitError:
        return _deep_health_error(
            "rate_limited",
            model,
            start,
            "OpenAI model check was rate limited.",
        )
    except VisionProviderError:
        return _deep_health_error(
            "model_unavailable",
            model,
            start,
            "OpenAI model check failed.",
        )
    except VisionParseError:
        return _deep_health_error(
            "provider_parse_error",
            model,
            start,
            "OpenAI model check returned an unreadable response.",
        )

    return {
        "status": "ok",
        "service": "ttb-label-verification",
        "provider": "openai",
        "model": str(model_info.get("id", model)),
        "latency_ms": _latency_ms(start),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


def _deep_health_error(
    code: str,
    model: str,
    start: float,
    message: str,
) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "status": "error",
            "service": "ttb-label-verification",
            "provider": "openai",
            "model": model,
            "error": {
                "code": code,
                "message": message,
            },
            "latency_ms": _latency_ms(start),
        },
    )


def _latency_ms(start: float) -> int:
    return max(0, int((time.perf_counter() - start) * 1000))
