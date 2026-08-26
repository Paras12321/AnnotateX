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
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.contracts import Detection, ValidationResult
from export.yolo_writer import export_yolo
from export.coco_writer import export_coco


def validate_annotation(
    detection: Detection,
    image_dims: tuple[int, int],
    valid_class_ids: set[int],
) -> ValidationResult:
    """Validate a single annotation before export.
    Checks bbox boundaries and class_id validity.
    """
    errors = []
    
    if detection.class_id not in valid_class_ids:
        errors.append(f"Invalid class_id: {detection.class_id}")
        
    if not isinstance(detection.bbox, list) or len(detection.bbox) != 4:
        errors.append("bbox must be a list of 4 numeric values")
    else:
        x1, y1, x2, y2 = detection.bbox
        if not all(isinstance(v, (int, float)) for v in [x1, y1, x2, y2]):
            errors.append("bbox values must be numeric")
        else:
            if x2 <= x1:
                errors.append("bbox x2 must be greater than x1")
            if y2 <= y1:
                errors.append("bbox y2 must be greater than y1")
                
            img_w, img_h = image_dims
            if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
                errors.append("bbox coordinates must fit within image dimensions")

    is_valid = len(errors) == 0
    return ValidationResult(annotation=detection, is_valid=is_valid, errors=errors)


def validate_batch(
    annotations: list[Detection],
    image_dims: dict[str, tuple[int, int]],
    valid_class_ids: set[int],
) -> list[ValidationResult]:
    """Validate a batch of annotations."""
    results = []
    for det in annotations:
        if det.image_id not in image_dims:
            results.append(
                ValidationResult(
                    annotation=det, 
                    is_valid=False, 
                    errors=[f"Missing image_dims for image_id: {det.image_id}"]
                )
            )
        else:
            results.append(validate_annotation(det, image_dims[det.image_id], valid_class_ids))
    return results


def export_pipeline(
    accepted_annotations: list[Detection],
    image_dims: dict[str, tuple[int, int]],
    class_names: dict[int, str],
    output_dir: str,
) -> dict[str, str]:
    """Runs validate_batch, filters valid, calls exporters."""
    valid_class_ids = set(class_names.keys())
    validation_results = validate_batch(accepted_annotations, image_dims, valid_class_ids)
    
    valid_annotations = [res.annotation for res in validation_results if res.is_valid]
    
    yolo_dir = os.path.join(output_dir, "yolo")
    yolo_paths = export_yolo(valid_annotations, image_dims, yolo_dir)
    
    coco_json_path = os.path.join(output_dir, "coco", "annotations.json")
    coco_path = export_coco(valid_annotations, image_dims, class_names, coco_json_path)
    
    return {
        "yolo_dir": yolo_dir,
        "coco_json": coco_path
    }
