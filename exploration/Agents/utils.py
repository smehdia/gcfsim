"""Shared image prep and coordinate conversion for UI-TARS and MAI-UI agents."""

from __future__ import annotations

import base64
import math
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

ImageInput = Union[str, Path, Image.Image, np.ndarray]

MAI_SCALE_FACTOR = 999

IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


@dataclass
class ResizeMeta:
    orig_w: int
    orig_h: int
    sent_w: int
    sent_h: int


def action_with_resize_dims(parsed: Any, meta: ResizeMeta) -> Tuple[Any, int, int]:
    """Return ParsedAction with the smart_resize width and height sent to the model."""
    return parsed, meta.sent_w, meta.sent_h


def load_image(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return Image.fromarray(image).convert("RGB")
        if image.ndim == 3 and image.shape[2] == 3:
            return Image.fromarray(image[:, :, ::-1]).convert("RGB")
        if image.ndim == 3 and image.shape[2] == 4:
            return Image.fromarray(image[:, :, [2, 1, 0, 3]]).convert("RGB")
        raise ValueError(f"Unsupported numpy image shape: {image.shape}")
    return Image.open(image).convert("RGB")


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def linear_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    if width * height > max_pixels:
        resize_factor = math.sqrt(max_pixels / (width * height))
        width, height = int(width * resize_factor), int(height * resize_factor)
    if width * height < min_pixels:
        resize_factor = math.sqrt(min_pixels / (width * height))
        width, height = math.ceil(width * resize_factor), math.ceil(height * resize_factor)
    return height, width


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """
    Rescale so both sides are divisible by factor, pixel count is in
    [min_pixels, max_pixels], and aspect ratio is preserved as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got "
            f"{max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def _prepare_agent_image(image: ImageInput) -> Tuple[Image.Image, ResizeMeta]:
    """Shared Qwen-VL smart_resize prep for MAI-UI and UI-TARS."""
    img = load_image(image)
    ow, oh = img.size
    sent_h, sent_w = smart_resize(oh, ow)
    if sent_w == ow and sent_h == oh:
        return img, ResizeMeta(orig_w=ow, orig_h=oh, sent_w=sent_w, sent_h=sent_h)
    sent = img.resize((sent_w, sent_h), Image.Resampling.LANCZOS)
    return sent, ResizeMeta(orig_w=ow, orig_h=oh, sent_w=sent_w, sent_h=sent_h)


def prepare_mai_ui_image(image: ImageInput) -> Tuple[Image.Image, ResizeMeta]:
    """Resize for MAI-UI using Qwen-VL smart_resize."""
    return _prepare_agent_image(image)


def prepare_ui_tars_image(image: ImageInput) -> Tuple[Image.Image, ResizeMeta]:
    """Resize for UI-TARS using Qwen-VL smart_resize."""
    return _prepare_agent_image(image)


def image_to_base64(image: Image.Image, image_format: str = "PNG") -> str:
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def image_to_data_url(image: Image.Image, image_format: str = "PNG") -> str:
    mime = image_format.lower()
    if mime == "jpg":
        mime = "jpeg"
    return f"data:image/{mime};base64,{image_to_base64(image, image_format=image_format)}"


def prepare_ui_tars_jpeg_data_url(
    image: ImageInput,
    jpeg_quality: int = 90,
) -> Tuple[str, ResizeMeta]:
    sent, meta = prepare_ui_tars_image(image)
    buffer = BytesIO()
    sent.save(buffer, format="JPEG", quality=jpeg_quality)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}", meta


def safe_pil_to_bytes(image: Union[Image.Image, bytes]) -> bytes:
    if isinstance(image, Image.Image):
        buf = BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
    if isinstance(image, bytes):
        return image
    raise TypeError(f"Expected PIL Image or bytes, got {type(image)}")


def pil_to_base64(image: Image.Image) -> str:
    return image_to_base64(image, image_format="PNG")


def ui_tars_extract_coords_from_text(
    text: str,
    meta: ResizeMeta,
) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, Tuple[int, int]]]:
    """Parse box/point args from raw Action text using the actual sent frame size."""
    action_part = text.split("Action:")[-1] if "Action:" in text else text
    sent_coords: Dict[str, Tuple[int, int]] = {}
    orig_coords: Dict[str, Tuple[int, int]] = {}
    for key in ("start_box", "end_box", "start_point", "end_point", "point"):
        match = re.search(rf"{key}='((?:\\'|[^'])*)'", action_part)
        if not match:
            continue
        sent_xy, orig_xy = ui_tars_box_to_orig(match.group(1).replace("\\'", "'"), meta)
        if key in ("start_box", "start_point", "point"):
            out_key = "point"
        elif key == "end_box":
            out_key = "end_point"
        else:
            out_key = key
        if sent_xy:
            sent_coords[out_key] = sent_xy
        if orig_xy:
            orig_coords[out_key] = orig_xy
    return sent_coords, orig_coords


def ui_tars_box_to_orig(
    box_str: str,
    meta: ResizeMeta,
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """Map UI-TARS box/point string to sent-frame and original screenshot pixels."""
    try:
        nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", box_str)]
    except Exception:
        return None, None
    if len(nums) < 2:
        return None, None
    if len(nums) >= 4:
        cx = (nums[0] + nums[2]) / 2.0
        cy = (nums[1] + nums[3]) / 2.0
    else:
        cx, cy = nums[0], nums[1]
    if cx > 1.5 or cy > 1.5:
        sx = max(0, min(meta.sent_w - 1, int(round(cx))))
        sy = max(0, min(meta.sent_h - 1, int(round(cy))))
        ox = max(0, min(meta.orig_w - 1, int(round(cx * meta.orig_w / max(meta.sent_w, 1)))))
        oy = max(0, min(meta.orig_h - 1, int(round(cy * meta.orig_h / max(meta.sent_h, 1)))))
    else:
        sx = max(0, min(meta.sent_w - 1, int(round(cx * meta.sent_w))))
        sy = max(0, min(meta.sent_h - 1, int(round(cy * meta.sent_h))))
        ox = max(0, min(meta.orig_w - 1, int(round(cx * meta.orig_w))))
        oy = max(0, min(meta.orig_h - 1, int(round(cy * meta.orig_h))))
    return (sx, sy), (ox, oy)


def mai_coord_to_orig(
    coord: Optional[List[float]],
    meta: ResizeMeta,
    scale_factor: int = MAI_SCALE_FACTOR,
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """Map MAI-UI normalized or 0..999 coords to sent and original pixels."""
    if not coord or len(coord) < 2:
        return None, None
    x, y = float(coord[0]), float(coord[1])
    if 0 <= x <= 1.05 and 0 <= y <= 1.05:
        nx, ny = x, y
    elif 0 <= x <= scale_factor and 0 <= y <= scale_factor:
        nx, ny = x / scale_factor, y / scale_factor
    else:
        return None, None
    sent = (
        max(0, min(meta.sent_w - 1, int(round(nx * meta.sent_w)))),
        max(0, min(meta.sent_h - 1, int(round(ny * meta.sent_h)))),
    )
    orig = (
        max(0, min(meta.orig_w - 1, int(round(nx * meta.orig_w)))),
        max(0, min(meta.orig_h - 1, int(round(ny * meta.orig_h)))),
    )
    return sent, orig


def reverse_mai_swipe_direction(direction: str) -> str:
    d = str(direction or "up").strip().lower()
    if d == "up":
        return "down"
    if d == "down":
        return "up"
    return d if d in ("left", "right") else "down"
