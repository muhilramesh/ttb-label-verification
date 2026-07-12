from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.vision import OpenAIVisionService, VisionServiceError


DEFAULT_SAMPLE_PATH = Path("samples/sample_label.jpg")

SAMPLE_LABEL_LINES = [
    "SUNSET RIDGE",
    "CABERNET SAUVIGNON",
    "North Valley Estate Winery LLC",
    "Product of USA",
    "Alc. 45% by Vol. (90 Proof)",
    "750 mL",
    "GOVERNMENT WARNING: EXACTLY AS PRINTED",
]


def create_sample_label(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1200, 1600), "white")
    draw = ImageDraw.Draw(image)

    try:
        font_large = ImageFont.truetype("Arial.ttf", 92)
        font_medium = ImageFont.truetype("Arial.ttf", 62)
        font_small = ImageFont.truetype("Arial.ttf", 46)
    except OSError:
        font_large = ImageFont.load_default(size=92)
        font_medium = ImageFont.load_default(size=62)
        font_small = ImageFont.load_default(size=46)

    y = 150
    for index, line in enumerate(SAMPLE_LABEL_LINES):
        font = font_large if index == 0 else font_medium
        if line.startswith("GOVERNMENT WARNING"):
            font = font_small
        draw.text((90, y), line, fill="black", font=font)
        y += 145 if index == 0 else 120

    image.save(path, format="JPEG", quality=92, optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the OpenAI vision extractor against one label image."
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        type=Path,
        default=DEFAULT_SAMPLE_PATH,
        help="Path to a label image. Defaults to samples/sample_label.jpg.",
    )
    parser.add_argument(
        "--create-sample-only",
        action="store_true",
        help="Create the default sample image without calling the model.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=15.0,
        help="Model timeout for this manual sample run. Defaults to 15 seconds.",
    )
    args = parser.parse_args()

    if not args.image_path.exists():
        create_sample_label(args.image_path)
        print(f"Created sample image: {args.image_path}")

    if args.create_sample_only:
        return 0

    service = OpenAIVisionService(timeout_seconds=args.timeout_seconds)
    try:
        label = service.extract_label(
            args.image_path.read_bytes(),
            content_type="image/jpeg",
        )
    except VisionServiceError as exc:
        print(f"Vision extraction failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(label.model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
