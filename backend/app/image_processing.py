import base64
from dataclasses import dataclass
from io import BytesIO
import os

from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener

from backend.app.vision_errors import VisionInvalidImageError


IMAGE_MAX_LONG_SIDE_ENV = "IMAGE_MAX_LONG_SIDE"
IMAGE_JPEG_QUALITY_ENV = "IMAGE_JPEG_QUALITY"
MAX_IMAGE_LONG_SIDE = 768
JPEG_QUALITY = 70

register_heif_opener()


@dataclass(frozen=True)
class ProcessedImage:
    data: bytes
    content_type: str
    data_url: str
    original_width: int
    original_height: int
    width: int
    height: int


def preprocess_label_image(image_bytes: bytes, *, max_long_side: int | None = None, jpeg_quality: int | None = None) -> ProcessedImage:
    if not image_bytes:
        raise VisionInvalidImageError("Image upload is empty.")
    max_long_side = max_long_side or _int_from_env(IMAGE_MAX_LONG_SIDE_ENV, MAX_IMAGE_LONG_SIDE, 640, 2000)
    jpeg_quality = jpeg_quality or _int_from_env(IMAGE_JPEG_QUALITY_ENV, JPEG_QUALITY, 60, 95)
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image = ImageOps.exif_transpose(image)
            original_width, original_height = image.size
            image = _to_rgb_on_white(image)
            image.thumbnail((max_long_side, max_long_side), Image.Resampling.LANCZOS)
            output = BytesIO()
            image.save(output, format="JPEG", quality=jpeg_quality, optimize=True)
    except (UnidentifiedImageError, OSError) as exc:
        raise VisionInvalidImageError("Image upload is not a readable image.") from exc
    data = output.getvalue()
    return ProcessedImage(data, "image/jpeg", "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii"), original_width, original_height, image.width, image.height)


def _to_rgb_on_white(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return image.convert("RGB")


def _int_from_env(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, ""))
    except ValueError:
        return default
    return value if minimum <= value <= maximum else default
