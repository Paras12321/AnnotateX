"""
orchestrator.py — Pipeline orchestration.

Responsibilities:
    - Provide run_pipeline(file_paths, config) -> BatchResult.
    - Calls each stage in order:
        1. preprocessing.preprocess_batch(file_paths) -> list[ImageInput]
        2. inference.run_inference_batch(images) -> dict[str, list[Detection]]
        3. quality.evaluate_batch(...) -> dict[str, list[QualityResult]]
        4. routing.build_processing_result(...) per image
        5. validation + export via export_pipeline(...)
    - Assemble ProcessingResult per image and a final BatchResult.
    - Per-image error handling: one bad image does not kill the batch.

Owner: Member D (integration), shared with all members.
"""

import logging
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from models.contracts import (
    BatchResult,
    Detection,
    ImageInput,
    ProcessingResult,
    QualityResult,
)
from preprocessing.validate import preprocess_batch
from inference.yolo_infer import run_inference_batch
from quality.engine import evaluate_batch
from routing.router import build_processing_result
from validation.validate_export import export_pipeline
from utils.image_utils import load_image_cv2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loading — loaded once at module level
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
_config: dict | None = None


def _load_config(path: Path | None = None) -> dict:
    """Load and cache the YAML configuration file."""
    global _config
    if _config is not None:
        return _config

    config_path = path or _CONFIG_PATH
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            _config = yaml.safe_load(f)
        logger.info("Loaded config from %s", config_path)
    except Exception:
        logger.warning(
            "Failed to load config from %s, using defaults", config_path,
            exc_info=True,
        )
        _config = {
            "conf_threshold": 0.5,
            "min_area_px": 400,
            "iou_threshold": 0.9,
            "class_names": {0: "person"},
            "output_dir": "outputs",
        }
    return _config


def get_config() -> dict:
    """Return the cached config, loading it if necessary."""
    return _load_config()


# ---------------------------------------------------------------------------
# Annotated image helper
# ---------------------------------------------------------------------------


def annotate_image(
    file_path: str,
    detections: list[Detection],
    quality_results: list[QualityResult],
) -> np.ndarray | None:
    """Load an image and draw bounding boxes color-coded by decision.

    Note: Detection contains a list field (bbox) making it unhashable,
    so we cannot use Detection as a dict key for decision_map. Instead
    we build a list-indexed decision lookup from quality_results.

    Args:
        file_path: Path to the source image file.
        detections: Detections for this image.
        quality_results: Quality results (parallel to detections).

    Returns:
        RGB numpy array with boxes drawn, or None on failure.
    """
    # Color palette (BGR for OpenCV) — matches utils/image_utils.COLORS
    COLORS = {
        "ACCEPT": (0, 200, 0),     # green
        "FLAG": (0, 200, 255),     # orange
        "REJECT": (0, 0, 200),     # red
        "default": (255, 200, 0),  # cyan
    }

    try:
        img = load_image_cv2(file_path)
        if img is None:
            return None

        canvas = img.copy()

        # Build decision lookup: map detection index -> decision string
        # QualityResults parallel the detections list; also match by
        # identity using id() as fallback for reordered lists.
        det_id_to_decision: dict[int, str] = {}
        for qr in quality_results:
            det_id_to_decision[id(qr.detection)] = qr.decision

        for det in detections:
            decision = det_id_to_decision.get(id(det), "default")
            color = COLORS.get(decision, COLORS["default"])

            x1, y1, x2, y2 = [int(c) for c in det.bbox]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            label = f"{det.class_name} {det.conf:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                canvas, (x1, y1 - th - 6), (x1 + tw, y1), color, -1
            )
            cv2.putText(
                canvas, label, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

        # Convert BGR (OpenCV) -> RGB (Gradio/PIL)
        annotated_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return annotated_rgb
    except Exception:
        logger.warning("Failed to annotate image %s", file_path, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    file_paths: list[str],
    config: dict | None = None,
) -> BatchResult:
    """Run the full annotation pipeline on a batch of images.

    Pipeline stages:
        1. preprocessing.preprocess_batch(file_paths) -> list[ImageInput]
        2. inference.run_inference_batch(ok_images) -> detections per image
        3. quality.evaluate_batch(...) -> quality results per image
        4. routing.build_processing_result(...) per image
        5. validation.export_pipeline(accepted, ...) -> export files

    Each image is wrapped in its own try/except so one bad image
    does not kill the entire batch.

    Args:
        file_paths: List of image file paths to process.
        config: Optional config dict. Loaded from config.yaml if None.

    Returns:
        BatchResult with per-image results, totals, and export paths.
    """
    if config is None:
        config = get_config()

    if not file_paths:
        return BatchResult(
            results=[],
            total_images=0,
            total_detections=0,
            total_accepted=0,
            total_flagged=0,
            total_rejected=0,
            export_paths={},
        )

    # ------------------------------------------------------------------
    # Stage 1: Preprocessing — validate and load all images
    # ------------------------------------------------------------------
    images: list[ImageInput] = preprocess_batch(file_paths)

    # Build lookup maps
    image_map: dict[str, ImageInput] = {img.image_id: img for img in images}
    image_dims: dict[str, tuple[int, int]] = {
        img.image_id: (img.width, img.height)
        for img in images
        if img.status == "ok"
    }

    # ------------------------------------------------------------------
    # Stage 2: Inference — run YOLO on all ok images
    # ------------------------------------------------------------------
    ok_images = [img for img in images if img.status == "ok"]
    detections_by_image: dict[str, list[Detection]] = {}

    if ok_images:
        try:
            detections_by_image = run_inference_batch(ok_images)
        except Exception:
            logger.error("Batch inference failed", exc_info=True)
            # Will proceed with empty detections

    # ------------------------------------------------------------------
    # Stage 3: Quality evaluation — evaluate all detections
    # ------------------------------------------------------------------
    quality_by_image: dict[str, list[QualityResult]] = {}

    if detections_by_image:
        try:
            quality_by_image = evaluate_batch(
                detections_by_image, image_dims, config
            )
        except Exception:
            logger.error("Batch quality evaluation failed", exc_info=True)

    # ------------------------------------------------------------------
    # Stage 4-5: Per-image routing and result assembly
    # ------------------------------------------------------------------
    results: list[ProcessingResult] = []
    all_accepted: list[Detection] = []

    total_detections = 0
    total_accepted = 0
    total_flagged = 0
    total_rejected = 0

    for img in images:
        start = time.perf_counter()
        try:
            if img.status != "ok":
                # Non-ok image: empty processing result
                elapsed_ms = (time.perf_counter() - start) * 1000
                results.append(
                    ProcessingResult(
                        image_id=img.image_id,
                        detections=[],
                        quality_results=[],
                        accepted=[],
                        flagged=[],
                        rejected=[],
                        processing_time_ms=round(elapsed_ms, 2),
                    )
                )
                logger.info(
                    "Skipped image %s (status=%s)", img.image_id, img.status
                )
                continue

            dets = detections_by_image.get(img.image_id, [])
            qrs = quality_by_image.get(img.image_id, [])

            elapsed_ms = (time.perf_counter() - start) * 1000
            pr = build_processing_result(
                image_id=img.image_id,
                detections=dets,
                quality_results=qrs,
                processing_time_ms=round(elapsed_ms, 2),
            )
            results.append(pr)

            total_detections += len(pr.detections)
            total_accepted += len(pr.accepted)
            total_flagged += len(pr.flagged)
            total_rejected += len(pr.rejected)
            all_accepted.extend(pr.accepted)

        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "Failed processing image %s", img.image_id, exc_info=True
            )
            results.append(
                ProcessingResult(
                    image_id=img.image_id,
                    detections=[],
                    quality_results=[],
                    accepted=[],
                    flagged=[],
                    rejected=[],
                    processing_time_ms=round(elapsed_ms, 2),
                )
            )

    # ------------------------------------------------------------------
    # Stage 6: Validation + Export — only if there are accepted detections
    # ------------------------------------------------------------------
    export_paths: dict[str, str] = {}

    if all_accepted:
        try:
            class_names_raw = config.get("class_names", {})
            # Ensure keys are ints (YAML may parse them as ints already)
            class_names: dict[int, str] = {
                int(k): str(v) for k, v in class_names_raw.items()
            }
            output_dir = config.get("output_dir", "outputs")

            export_paths = export_pipeline(
                accepted_annotations=all_accepted,
                image_dims=image_dims,
                class_names=class_names,
                output_dir=output_dir,
            )
            logger.info("Export complete: %s", export_paths)
        except Exception:
            logger.error("Export pipeline failed", exc_info=True)

    # ------------------------------------------------------------------
    # Stage 7: Assemble BatchResult
    # ------------------------------------------------------------------
    return BatchResult(
        results=results,
        total_images=len(file_paths),
        total_detections=total_detections,
        total_accepted=total_accepted,
        total_flagged=total_flagged,
        total_rejected=total_rejected,
        export_paths=export_paths,
    )
