from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from measure_single_label_latency import _post_verify, _verify_url
from run_vision_sample import create_sample_label


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one live end-to-end label verification.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--image", type=Path, default=Path("samples/sample_label.jpg"))
    parser.add_argument("--timeout-seconds", type=float, default=6.0)
    args = parser.parse_args()
    if not args.image.exists():
        create_sample_label(args.image)
    started = time.perf_counter()
    try:
        status, raw_body = _post_verify(_verify_url(args.url), image_bytes=args.image.read_bytes(), timeout_seconds=args.timeout_seconds)
        body = json.loads(raw_body)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    latency_ms = int((time.perf_counter() - started) * 1000)
    verdict = body.get("overall_verdict")
    if status < 200 or status >= 300 or verdict != "APPROVED":
        print(f"FAIL: HTTP {status}, verdict={verdict}, latency_ms={latency_ms}", file=sys.stderr)
        return 1
    print(f"PASS: HTTP {status}, verdict={verdict}, latency_ms={latency_ms}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
