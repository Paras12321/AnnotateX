"""
yolo_infer.py — YOLO inference engine.

Responsibilities:
    - Load a pretrained YOLO model (yolov8n.pt) once (lazy singleton).
    - run_inference(image_path) -> list[Detection]
        Run inference on a single image. Returns Detection objects with
        bbox as [x1, y1, x2, y2] in absolute pixel coordinates.
    - run_inference_batch(images) -> dict[str, list[Detection]]
        Run inference on a batch of ImageInput objects, skipping
        non-ok images with logging.

Owner: Member A
"""

import logging
import os
from pathlib import Path

from ultralytics import YOLO

from models.contracts import Detection, ImageInput

logger = logging.getLogger(__name__)

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


def run_inference_batch(
    images: list[ImageInput],
) -> dict[str, list[Detection]]:
    """Run inference on a batch of preprocessed images.

    Only processes images with status=="ok". Skipped images are logged
    (not silently dropped) and do NOT appear in the returned dict.

    Args:
        images: List of ImageInput objects from preprocess_batch.

    Returns:
        Dict mapping image_id -> list[Detection] for successfully
        processed images only.
    """
    results: dict[str, list[Detection]] = {}

    for img in images:
        if img.status != "ok":
            logger.warning(
                "Skipping image %s (status=%s, path=%s)",
                img.image_id, img.status, img.file_path,
            )
            continue

        try:
            detections = run_inference(img.file_path)
            results[img.image_id] = detections
            logger.info(
                "Inference on %s: %d detections",
                img.image_id, len(detections),
            )
        except Exception:
            logger.exception(
                "Inference failed for %s, skipping", img.image_id
            )

    return results

