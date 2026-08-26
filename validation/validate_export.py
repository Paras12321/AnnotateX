"""
validate_export.py — Pre-export structural and format-safety checks.

Responsibilities:
    - validate_annotation(detection, image_dims, valid_class_ids) -> ValidationResult
        Checks: class_id in valid set, bbox has 4 numeric values with x2>x1 and y2>y1,
        coordinates fit within image dimensions, width/height > 0.
    - validate_batch(annotations, image_dims, valid_class_ids) -> list[ValidationResult]
        Runs validate_annotation on every annotation.
    - export_pipeline(accepted, image_dims, class_names, output_dir) -> dict[str, str]
        Validates, filters to is_valid==True only, then calls YOLO + COCO exporters.
        Invalid annotations are excluded (never silently exported) and errors are logged.

This layer is independent of the Quality Engine — an annotation can pass Quality (ACCEPT)
but still fail Validation (e.g. class_id not in config). Validation is the last safety net
before anything touches disk.

Owner: Member C

NOTE: Day 1 stub only. Full validation logic is Day 2 scope.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.contracts import Detection, ValidationResult


def validate_annotation(
    detection: Detection,
    image_dims: dict[str, tuple[int, int]],
    valid_class_ids: set[int],
) -> ValidationResult:
    """Validate a single annotation before export.

    Day 1 stub: returns is_valid=True for all annotations.
    Day 2: will implement bbox range checks, class_id validation, etc.
    """
    # TODO(Day 2): Implement full structural validation
    return ValidationResult(annotation=detection, is_valid=True, errors=[])


def validate_batch(
    annotations: list[Detection],
    image_dims: dict[str, tuple[int, int]],
    valid_class_ids: set[int],
) -> list[ValidationResult]:
    """Validate a batch of annotations.

    Day 1 stub: delegates to validate_annotation for each item.
    """
    return [
        validate_annotation(det, image_dims, valid_class_ids)
        for det in annotations
    ]
