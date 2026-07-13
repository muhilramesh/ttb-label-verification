import json
import logging
import time

from fastapi.responses import JSONResponse


logger = logging.getLogger("backend.app.verify")


def error_response(status_code: int, code: str, message: str, *, start: float, log_error: bool = True) -> JSONResponse:
    latency_ms = elapsed_ms(start)
    response = with_latency(plain_error_response(status_code, code, message), latency_ms)
    if log_error:
        logger.warning("verify failed code=%s status_code=%s latency_ms=%s", code, status_code, latency_ms)
    return response


def plain_error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": {"code": code, "message": message}})


def with_latency(response: JSONResponse, latency_ms: int | float) -> JSONResponse:
    body = json.loads(response.body)
    body["latency_ms"] = int(latency_ms)
    return JSONResponse(status_code=response.status_code, content=body)


def elapsed_ms(start: float) -> int:
    return max(0, int((time.perf_counter() - start) * 1000))
