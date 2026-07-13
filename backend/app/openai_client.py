import json
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from backend.app.vision_errors import VisionParseError, VisionProviderError, VisionRateLimitError


RESPONSES_ENDPOINT = "https://api.openai.com/v1/responses"
MODELS_ENDPOINT = "https://api.openai.com/v1/models/{model}"


def create_response(body: dict, api_key: str, timeout: float) -> dict:
    request = Request(RESPONSES_ENDPOINT, data=json.dumps(body).encode("utf-8"), headers=_headers(api_key, json_body=True), method="POST")
    return _send(request, timeout, operation="API request")


def get_model(model: str, api_key: str, timeout: float) -> dict:
    request = Request(MODELS_ENDPOINT.format(model=quote(model, safe="")), headers=_headers(api_key), method="GET")
    return _send(request, timeout, operation="model check")


def _headers(api_key: str, *, json_body: bool = False) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {api_key}"}
    if json_body:
        headers["Content-Type"] = "application/json"
    return headers


def _send(request: Request, timeout: float, *, operation: str) -> dict:
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 429:
            raise VisionRateLimitError(f"OpenAI API quota or rate limit was reached: {detail}") from exc
        raise VisionProviderError(f"OpenAI {operation} failed: {detail}") from exc
    except json.JSONDecodeError as exc:
        raise VisionParseError(f"OpenAI {operation} response was not valid JSON.") from exc
