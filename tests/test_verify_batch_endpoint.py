import json
import threading
import time

from fastapi.testclient import TestClient

from backend.app.api import MAX_BATCH_LABELS, get_vision_service
from backend.app.main import app
from backend.app.models import ExtractedLabel
from backend.app.vision import VisionRateLimitError, VisionTimeoutError


WARNING = "GOVERNMENT WARNING: EXACTLY AS PRINTED"


def application_item(item_id: str, **overrides) -> dict:
    payload = {
        "id": item_id,
        "brand_name": "Sunset Ridge",
        "class_type": "Cabernet Sauvignon",
        "producer": "North Valley Estate Winery LLC",
        "country_of_origin": "USA",
        "abv": "45%",
        "net_contents": "750 mL",
        "government_warning": WARNING,
    }
    payload.update(overrides)
    return payload


def extracted_label(**overrides) -> ExtractedLabel:
    payload = {
        "brand_name": "SUNSET RIDGE",
        "class_type": "Sauvignon Cabernet",
        "producer": "North Valley Estate Winery, LLC",
        "country_of_origin": "USA",
        "abv": "45% Alc./Vol. (90 Proof)",
        "net_contents": "750ml",
        "government_warning": WARNING,
        "government_warning_heading_bold": True,
        "raw_text": "SUNSET RIDGE\nGOVERNMENT WARNING: EXACTLY AS PRINTED",
        "extraction_confidence": 0.96,
    }
    payload.update(overrides)
    return ExtractedLabel(**payload)


class MappingVisionService:
    def __init__(self, responses: dict[bytes, ExtractedLabel | Exception]) -> None:
        self.responses = responses
        self.calls = []

    def extract_label(
        self,
        image_bytes: bytes,
        content_type: str | None = None,
    ) -> ExtractedLabel:
        self.calls.append((image_bytes, content_type))
        response = self.responses[image_bytes]
        if isinstance(response, Exception):
            raise response
        return response


class SlowVisionService:
    def __init__(self, delay_seconds: float = 0.12) -> None:
        self.delay_seconds = delay_seconds
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def extract_label(
        self,
        image_bytes: bytes,
        content_type: str | None = None,
    ) -> ExtractedLabel:
        with self.lock:
            self.calls.append((image_bytes, content_type))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay_seconds)
            return extracted_label()
        finally:
            with self.lock:
                self.active -= 1


def client_for(service):
    app.dependency_overrides[get_vision_service] = lambda: service
    return TestClient(app)


def post_batch(
    client: TestClient,
    *,
    items: list[dict],
    image_ids: list[str],
    files: list[tuple[str, bytes, str]] | None = None,
):
    upload_files = []
    for filename, image_bytes, content_type in files or []:
        upload_files.append(("images", (filename, image_bytes, content_type)))

    return client.post(
        "/verify/batch",
        data={
            "application_data": json.dumps(items),
            "image_ids": json.dumps(image_ids),
        },
        files=upload_files,
    )


def test_batch_returns_summary_for_two_passing_labels() -> None:
    service = MappingVisionService(
        {
            b"image-1": extracted_label(),
            b"image-2": extracted_label(),
        }
    )
    with client_for(service) as client:
        response = post_batch(
            client,
            items=[application_item("row-1"), application_item("row-2")],
            image_ids=["row-1", "row-2"],
            files=[
                ("one.jpg", b"image-1", "image/jpeg"),
                ("two.jpg", b"image-2", "image/jpeg"),
            ],
        )

    assert response.status_code == 200
    body = response.json()
    assert "total" not in body
    assert "passed" not in body
    assert body["summary"] == {
        "passed": 2,
        "needs_review": 0,
        "total": 2,
        "errors": 0,
        "latency_ms": body["summary"]["latency_ms"],
    }
    assert isinstance(body["summary"]["latency_ms"], int)
    assert [item["id"] for item in body["items"]] == ["row-1", "row-2"]
    assert all(item["status"] == "APPROVED" for item in body["items"])
    assert len(service.calls) == 2
    app.dependency_overrides.clear()


def test_batch_summary_counts_pass_needs_review_and_error() -> None:
    service = MappingVisionService(
        {
            b"image-1": extracted_label(),
            b"image-2": extracted_label(government_warning="Government Warning"),
            b"image-3": VisionTimeoutError("timeout"),
        }
    )
    with client_for(service) as client:
        response = post_batch(
            client,
            items=[
                application_item("pass-row"),
                application_item("review-row"),
                application_item("error-row"),
            ],
            image_ids=["pass-row", "review-row", "error-row"],
            files=[
                ("pass.jpg", b"image-1", "image/jpeg"),
                ("review.jpg", b"image-2", "image/jpeg"),
                ("error.jpg", b"image-3", "image/jpeg"),
            ],
        )

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["total"] == 3
    assert body["summary"]["passed"] == 1
    assert body["summary"]["needs_review"] == 1
    assert body["summary"]["errors"] == 1

    review_item = body["items"][1]
    assert review_item["status"] == "NEEDS_REVIEW"
    assert review_item["result"]["extracted_label"]["government_warning"] == "Government Warning"
    assert review_item["result"]["extracted_label"]["raw_text"].startswith("SUNSET RIDGE")
    warning_field = next(
        field for field in review_item["result"]["results"]
        if field["field"] == "government_warning"
    )
    assert warning_field["found"] == "Government Warning"

    error_item = body["items"][2]
    assert error_item["status"] == "ERROR"
    assert error_item["error"] == {
        "code": "vision_timeout",
        "message": "The label image took too long to process. Try a clearer or smaller image.",
    }
    app.dependency_overrides.clear()


def test_batch_invalid_one_item_application_data_is_item_error() -> None:
    service = MappingVisionService({b"image-1": extracted_label()})
    invalid_item = application_item("bad-row")
    invalid_item.pop("government_warning")

    with client_for(service) as client:
        response = post_batch(
            client,
            items=[application_item("good-row"), invalid_item],
            image_ids=["good-row", "bad-row"],
            files=[
                ("good.jpg", b"image-1", "image/jpeg"),
                ("bad.jpg", b"image-2", "image/jpeg"),
            ],
        )

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["passed"] == 1
    assert body["summary"]["errors"] == 1
    assert body["items"][1]["status"] == "ERROR"
    assert body["items"][1]["error"]["code"] == "invalid_application_data"
    assert len(service.calls) == 1
    app.dependency_overrides.clear()


def test_batch_missing_one_item_image_is_item_error() -> None:
    service = MappingVisionService({b"image-1": extracted_label()})
    with client_for(service) as client:
        response = post_batch(
            client,
            items=[application_item("row-1"), application_item("row-2")],
            image_ids=["row-1"],
            files=[("one.jpg", b"image-1", "image/jpeg")],
        )

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["passed"] == 1
    assert body["summary"]["errors"] == 1
    assert body["items"][1]["status"] == "ERROR"
    assert body["items"][1]["error"] == {
        "code": "missing_image",
        "message": "Upload a label image for this item.",
    }
    app.dependency_overrides.clear()


def test_batch_invalid_file_type_is_item_error() -> None:
    service = MappingVisionService({b"image-1": extracted_label()})
    with client_for(service) as client:
        response = post_batch(
            client,
            items=[application_item("row-1"), application_item("row-2")],
            image_ids=["row-1", "row-2"],
            files=[
                ("one.jpg", b"image-1", "image/jpeg"),
                ("two.txt", b"image-2", "text/plain"),
            ],
        )

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["passed"] == 1
    assert body["summary"]["errors"] == 1
    assert body["items"][1]["error"] == {
        "code": "invalid_image_type",
            "message": "Upload a JPEG, PNG, WebP, HEIC, or HEIF label image.",
    }
    assert len(service.calls) == 1
    app.dependency_overrides.clear()


def test_batch_rate_limited_item_does_not_fail_whole_batch() -> None:
    service = MappingVisionService(
        {
            b"image-1": extracted_label(),
            b"image-2": VisionRateLimitError("quota reached"),
        }
    )
    with client_for(service) as client:
        response = post_batch(
            client,
            items=[application_item("row-1"), application_item("row-2")],
            image_ids=["row-1", "row-2"],
            files=[
                ("one.jpg", b"image-1", "image/jpeg"),
                ("two.jpg", b"image-2", "image/jpeg"),
            ],
        )

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["passed"] == 1
    assert body["summary"]["errors"] == 1
    assert body["items"][1]["error"] == {
        "code": "vision_rate_limited",
        "message": "The vision service is temporarily busy. Try again in a minute.",
    }
    app.dependency_overrides.clear()


def test_batch_rejects_request_level_invalid_json() -> None:
    service = MappingVisionService({})
    with client_for(service) as client:
        response = client.post(
            "/verify/batch",
            data={
                "application_data": "{bad json",
                "image_ids": json.dumps([]),
            },
        )

    assert_batch_error(
        response,
        400,
        "invalid_application_data",
        "Batch application data must be valid JSON.",
    )
    app.dependency_overrides.clear()


def test_batch_rejects_duplicate_application_ids() -> None:
    service = MappingVisionService({})
    with client_for(service) as client:
        response = post_batch(
            client,
            items=[application_item("row-1"), application_item("row-1")],
            image_ids=["row-1", "row-2"],
            files=[
                ("one.jpg", b"image-1", "image/jpeg"),
                ("two.jpg", b"image-2", "image/jpeg"),
            ],
        )

    assert_batch_error(
        response,
        400,
        "duplicate_application_ids",
        "Batch application IDs must be unique.",
    )
    app.dependency_overrides.clear()


def test_batch_rejects_too_many_labels() -> None:
    service = MappingVisionService({})
    items = [application_item(f"row-{index}") for index in range(MAX_BATCH_LABELS + 1)]
    with client_for(service) as client:
        response = post_batch(
            client,
            items=items,
            image_ids=[],
            files=[],
        )

    assert_batch_error(
        response,
        413,
        "batch_too_large",
        f"Check at most {MAX_BATCH_LABELS} labels at a time.",
    )
    app.dependency_overrides.clear()


def test_batch_processes_labels_concurrently_with_bounded_cap(monkeypatch) -> None:
    monkeypatch.setenv("BATCH_CONCURRENCY", "2")
    service = SlowVisionService()
    started = time.perf_counter()
    with client_for(service) as client:
        response = post_batch(
            client,
            items=[
                application_item("row-1"),
                application_item("row-2"),
                application_item("row-3"),
            ],
            image_ids=["row-1", "row-2", "row-3"],
            files=[
                ("one.jpg", b"image-1", "image/jpeg"),
                ("two.jpg", b"image-2", "image/jpeg"),
                ("three.jpg", b"image-3", "image/jpeg"),
            ],
        )
    elapsed = time.perf_counter() - started

    assert response.status_code == 200
    assert response.json()["summary"]["passed"] == 3
    assert service.max_active == 2
    assert elapsed < 0.5
    app.dependency_overrides.clear()


def assert_batch_error(response, status_code: int, code: str, message: str) -> None:
    assert response.status_code == status_code
    body = response.json()
    assert body["error"] == {
        "code": code,
        "message": message,
    }
    assert isinstance(body["latency_ms"], int)
    assert "Traceback" not in response.text
