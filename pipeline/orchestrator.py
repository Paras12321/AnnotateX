"""
orchestrator.py — Pipeline orchestration.

Responsibilities:
    - Provide run_pipeline(file_paths) -> BatchResult.
    - Day 1: returns hand-built dummy BatchResult matching the contract shape.
    - Day 2+: will call each stage in order:
        1. preprocessing.preprocess_batch(file_paths) -> list[ImageInput]
        2. inference.run_inference(image)              -> list[Detection]
        3. quality.evaluate(...)                       -> list[QualityResult]
        4. routing.route(...) per image                -> accepted / flagged / rejected
        5. validation + export                         -> export files
    - Assemble ProcessingResult per image and a final BatchResult.

Owner: Member D (integration), shared with all members.
"""

import time
from pathlib import Path

from models.contracts import (
    BatchResult,
    Detection,
    ProcessingResult,
    QualityResult,
)


def run_pipeline(file_paths: list[str]) -> BatchResult:
    """Run the full annotation pipeline on a batch of images.

    Day 1 stub: returns a hand-built dummy BatchResult so the UI can
    render results without real inference/quality/export.

    Args:
        file_paths: List of image file paths to process.

    Returns:
        BatchResult with dummy data matching the locked contract shape.
    """
    results: list[ProcessingResult] = []
    total_detections = 0
    total_accepted = 0
    total_flagged = 0
    total_rejected = 0

    for fp in file_paths:
        image_id = Path(fp).stem
        start = time.perf_counter()

        # --- Dummy detections ---
        dummy_detections = [
            Detection(
                image_id=image_id,
                bbox=[50.0, 30.0, 200.0, 180.0],
                class_id=0,
                class_name="person",
                conf=0.92,
            ),
            Detection(
                image_id=image_id,
                bbox=[220.0, 50.0, 400.0, 300.0],
                class_id=2,
                class_name="car",
                conf=0.78,
            ),
            Detection(
                image_id=image_id,
                bbox=[10.0, 10.0, 25.0, 25.0],
                class_id=15,
                class_name="cat",
                conf=0.35,
            ),
        ]

        # --- Dummy quality results ---
        dummy_quality = [
            QualityResult(
                detection=dummy_detections[0],
                passed_rules=["confidence", "valid_bounds", "not_tiny", "no_duplicate"],
                failed_rules=[],
                decision="ACCEPT",
                reason="All rules passed.",
            ),
            QualityResult(
                detection=dummy_detections[1],
                passed_rules=["confidence", "valid_bounds", "not_tiny"],
                failed_rules=["no_duplicate"],
                decision="FLAG",
                reason="Possible duplicate (IoU > 0.9 with another car detection).",
            ),
            QualityResult(
                detection=dummy_detections[2],
                passed_rules=["valid_bounds"],
                failed_rules=["confidence", "not_tiny"],
                decision="REJECT",
                reason="Low confidence (0.35 < 0.5) and too small (225px < 400px).",
            ),
        ]

        accepted = [dummy_detections[0]]
        flagged = [dummy_detections[1]]
        rejected = [dummy_detections[2]]

        elapsed_ms = (time.perf_counter() - start) * 1000

        results.append(
            ProcessingResult(
                image_id=image_id,
                detections=dummy_detections,
                quality_results=dummy_quality,
                accepted=accepted,
                flagged=flagged,
                rejected=rejected,
                processing_time_ms=round(elapsed_ms, 2),
            )
        )

        total_detections += len(dummy_detections)
        total_accepted += len(accepted)
        total_flagged += len(flagged)
        total_rejected += len(rejected)

    return BatchResult(
        results=results,
        total_images=len(file_paths),
        total_detections=total_detections,
        total_accepted=total_accepted,
        total_flagged=total_flagged,
        total_rejected=total_rejected,
        export_paths={},  # No exports on Day 1
    )
