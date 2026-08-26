"""
image_utils.py — Shared image helper functions.

Responsibilities:
    - Common image loading and drawing utilities used across
      preprocessing (Member A) and UI (Member D).
    - Draw bounding boxes on images for annotated previews.

Owner: Shared (Members A and D)
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Color palette for bounding box drawing (BGR for OpenCV)
COLORS = {
    "ACCEPT": (0, 200, 0),     # green
    "FLAG": (0, 200, 255),     # orange
    "REJECT": (0, 0, 200),     # red
    "default": (255, 200, 0),  # cyan
}


def load_image_cv2(file_path: str) -> np.ndarray | None:
    """Load an image via OpenCV. Returns None on failure."""
    img = cv2.imread(file_path)
    if img is None:
        logger.warning("cv2.imread returned None for %s", file_path)
    return img


def load_image_pil(file_path: str) -> Image.Image | None:
    """Load an image via Pillow. Returns None on failure."""
    try:
        img = Image.open(file_path)
        img.load()  # Force full load
        return img
    except Exception:
        logger.warning("Failed to load %s with Pillow", file_path, exc_info=True)
        return None


def draw_bboxes(
    image: np.ndarray,
    detections: list,
    decision_map: dict | None = None,
) -> np.ndarray:
    """Draw bounding boxes on an image (OpenCV BGR array).

    Args:
        image: BGR image array.
        detections: List of Detection objects.
        decision_map: Optional dict mapping Detection -> decision string
            ("ACCEPT"/"FLAG"/"REJECT") for color coding.

    Returns:
        Copy of the image with bounding boxes drawn.
    """
    canvas = image.copy()

    for det in detections:
        decision = "default"
        if decision_map and det in decision_map:
            decision = decision_map[det]

        color = COLORS.get(decision, COLORS["default"])
        x1, y1, x2, y2 = [int(c) for c in det.bbox]

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        label = f"{det.class_name} {det.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(
            canvas, label, (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

    return canvas
