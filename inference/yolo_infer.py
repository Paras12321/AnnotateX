"""
yolo_infer.py — YOLO inference engine.

Responsibilities:
    - Load a pretrained YOLO model (yolov8n.pt) once (lazy singleton).
    - run_inference(image_path) -> list[Detection]
        Run inference on a single image. Returns Detection objects with
        bbox as [x1, y1, x2, y2] in absolute pixel coordinates.

Owner: Member A
"""

import os
from pathlib import Path

from ultralytics import YOLO

from models.contracts import Detection

# ---------------------------------------------------------------------------
# Lazy-loaded singleton model
# ---------------------------------------------------------------------------
_model: YOLO | None = None


def _get_model() -> YOLO:
    """Return the shared YOLO model, loading it on first call.

    Raises RuntimeError if the model cannot be loaded.
    """
    global _model
    if _model is None:
        try:
            _model = YOLO("yolov8n.pt")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load YOLO model (yolov8n.pt): {exc}"
            ) from exc
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_inference(image_path: str) -> list[Detection]:
    """Run YOLO inference on a single image.

    Args:
        image_path: Path to an image file (jpg, png, bmp, etc.).

    Returns:
        list[Detection] — one Detection per detected object, with
        image_id set to the filename stem and bbox in absolute pixel
        coordinates [x1, y1, x2, y2].

    Raises:
        FileNotFoundError: If *image_path* does not exist.
        RuntimeError: If the YOLO model fails to load.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    model = _get_model()
    results = model(str(path), verbose=False)

    detections: list[Detection] = []
    image_id = path.stem

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            xyxy = box.xyxy[0].tolist()           # [x1, y1, x2, y2]
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]

            detections.append(
                Detection(
                    image_id=image_id,
                    bbox=xyxy,
                    class_id=cls_id,
                    class_name=cls_name,
                    conf=round(conf, 4),
                )
            )

    return detections
