"""Bounded image decoding with coordinates in the supplied, unrotated frame."""

import math
import warnings
from io import BytesIO

from PIL import Image, UnidentifiedImageError

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 24_000_000


def validate_image(image: bytes, bbox=None):
    if not isinstance(image, bytes) or not image or len(image) > MAX_IMAGE_BYTES:
        raise ValueError("Image must contain 1–20 MiB of JPEG or PNG data")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(image)) as frame:
                if frame.format not in {"JPEG", "PNG"} or getattr(frame, "n_frames", 1) != 1:
                    raise ValueError("Only single-frame JPEG and PNG are supported")
                width, height = frame.size
                if width * height > MAX_IMAGE_PIXELS:
                    raise ValueError("Image exceeds 24 megapixels")
                frame.verify()
            with Image.open(BytesIO(image)) as frame:
                if frame.getexif().get(274, 1) != 1:
                    raise ValueError("Normalize EXIF orientation and bounding-box coordinates before upload")
                frame.load()
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError("Invalid or oversized image") from exc
    if bbox is None:
        box = (0, 0, width, height)
    else:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("BBox must be [x, y, width, height]")
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or int(v) != v
            for v in bbox
        ):
            raise ValueError("BBox coordinates must be finite integers")
        x, y, w, h = map(int, bbox)
        if w <= 0 or h <= 0:
            raise ValueError("BBox width and height must be positive")
        x0, y0, x1, y1 = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("BBox does not intersect the image")
        box = (x0, y0, x1 - x0, y1 - y0)
    # Preserve original pixels and encoding: re-encoding JPEG would change research inputs.
    return image, box, width, height


def crop_image(image: bytes, bbox=None) -> bytes:
    original, (x, y, w, h), _, _ = validate_image(image, bbox)
    with Image.open(BytesIO(original)) as frame:
        crop = frame.convert("RGB").crop((x, y, x + w, y + h))
        output = BytesIO()
        crop.save(output, format="PNG")
        return output.getvalue()
