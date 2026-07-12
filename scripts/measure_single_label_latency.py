from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from run_vision_sample import create_sample_label


DEFAULT_SAMPLE_PATH = Path("samples/sample_label.jpg")
DEFAULT_APPLICATION_DATA = {
    "brand_name": "SUNSET RIDGE",
    "class_type": "CABERNET SAUVIGNON",
    "producer": "North Valley Estate Winery LLC",
    "country_of_origin": "USA",
    "abv": "45%",
    "net_contents": "750 mL",
    "government_warning": "GOVERNMENT WARNING: EXACTLY AS PRINTED",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure single-label /verify latency against a deployed URL."
    )
    parser.add_argument(
        "--url",
        required=True,
        help="Base URL or full /verify URL, for example https://example.up.railway.app.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of sequential verification requests. Defaults to 20.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_SAMPLE_PATH,
        help="Path to a JPEG sample label. Defaults to samples/sample_label.jpg.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=6.0,
        help="Client timeout per request. Defaults to 6 seconds.",
    )
    args = parser.parse_args()

    if args.samples <= 0:
        print("--samples must be positive.", file=sys.stderr)
        return 2

    if not args.image.exists():
        create_sample_label(args.image)

    verify_url = _verify_url(args.url)
    latencies_ms: list[int] = []
    failures = 0

    for index in range(args.samples):
        started = time.perf_counter()
        try:
            status_code, body = _post_verify(
                verify_url,
                image_bytes=args.image.read_bytes(),
                timeout_seconds=args.timeout_seconds,
            )
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            if status_code != 200:
                failures += 1
                print(f"{index + 1}: HTTP {status_code} {body[:180]}")
                continue
            latencies_ms.append(elapsed_ms)
            print(f"{index + 1}: {elapsed_ms} ms")
        except (HTTPError, URLError, TimeoutError) as exc:
            failures += 1
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            print(f"{index + 1}: failed after {elapsed_ms} ms: {exc}")

    if not latencies_ms:
        print("No successful samples.", file=sys.stderr)
        return 1

    summary = {
        "url": verify_url,
        "samples": args.samples,
        "successful": len(latencies_ms),
        "failures": failures,
        "p50_ms": _percentile(latencies_ms, 50),
        "p95_ms": _percentile(latencies_ms, 95),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
    }
    print(json.dumps(summary, indent=2))
    return 0 if failures == 0 else 1


def _verify_url(url: str) -> str:
    normalized = url.rstrip("/")
    if normalized.endswith("/verify"):
        return normalized
    return f"{normalized}/verify"


def _post_verify(
    url: str,
    *,
    image_bytes: bytes,
    timeout_seconds: float,
) -> tuple[int, str]:
    boundary = f"----ttb-latency-{time.time_ns()}"
    body = _multipart_body(boundary, image_bytes=image_bytes)
    request = Request(
        url,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
        method="POST",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        return response.status, response.read().decode("utf-8", errors="replace")


def _multipart_body(boundary: str, *, image_bytes: bytes) -> bytes:
    application_json = json.dumps(DEFAULT_APPLICATION_DATA)
    parts = [
        _form_field(boundary, "application_data", application_json),
        _file_field(boundary, "image", "sample_label.jpg", "image/jpeg", image_bytes),
        f"--{boundary}--\r\n".encode("utf-8"),
    ]
    return b"".join(parts)


def _form_field(boundary: str, name: str, value: str) -> bytes:
    return (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
        f"{value}\r\n"
    ).encode("utf-8")


def _file_field(
    boundary: str,
    name: str,
    filename: str,
    content_type: str,
    data: bytes,
) -> bytes:
    return (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8") + data + b"\r\n"


def _percentile(values: list[int], percentile: int) -> int:
    if len(values) == 1:
        return values[0]
    return int(statistics.quantiles(values, n=100, method="inclusive")[percentile - 1])


if __name__ == "__main__":
    raise SystemExit(main())
