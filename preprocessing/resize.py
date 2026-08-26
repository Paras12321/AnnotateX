"""
resize.py — Image resize logic.

Responsibilities:
    - Resize images only when needed (very large images) to keep
      inference fast. Does NOT resize unconditionally.
    - Preserves aspect ratio.
    - Returns the (possibly unchanged) file path and final dimensions.

Owner: Member A
"""

import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

# Maximum dimension (width or height) before we resize.
# 1920px is a safe upper bound for YOLOv8 inference speed.
MAX_DIMENSION = 1920


def resize_if_needed(
    file_path: str, width: int, height: int
) -> tuple[str, int, int]:
    """Resize the image if either dimension exceeds MAX_DIMENSION.

    Args:
        file_path: Path to the image file.
        width: Current image width.
        height: Current image height.

    Returns:
        Tuple of (file_path, final_width, final_height).
        If no resize was needed, returns the original values unchanged.
    """
    if width <= MAX_DIMENSION and height <= MAX_DIMENSION:
        return file_path, width, height

    # Calculate new dimensions preserving aspect ratio
    scale = MAX_DIMENSION / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    logger.info(
        "Resizing %s from %dx%d to %dx%d",
        file_path, width, height, new_width, new_height,
    )

    try:
        with Image.open(file_path) as img:
            resized = img.resize((new_width, new_height), Image.LANCZOS)
            resized.save(file_path)
    except Exception:
        logger.exception("Failed to resize %s, using original", file_path)
        return file_path, width, height

    return file_path, new_width, new_height
