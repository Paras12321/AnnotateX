"""
rules.py — Individual quality rule functions.

Each rule is a pure function (no side effects, no I/O) that evaluates
a single Detection and returns True (pass) or False (fail).

Rules:
    - rule_confidence:  conf >= threshold
    - rule_valid_box:   bbox within image bounds, x2>x1, y2>y1
    - rule_not_tiny:    bbox area >= min_area_px
    - rule_no_duplicate: no same-class detection overlaps above IoU threshold

Owner: Member B
"""

from models.contracts import Detection
from utils.iou import compute_iou


def rule_confidence(detection: Detection, threshold: float) -> bool:
    """True if detection.conf >= threshold.

    Args:
        detection: The detection to evaluate.
        threshold: Minimum confidence value (0.0–1.0).

    Returns:
        True if confidence meets or exceeds the threshold.
    """
    if detection.conf is None:
        return False
    return detection.conf >= threshold


def rule_valid_box(detection: Detection, image_width: int, image_height: int) -> bool:
    """True if bbox is structurally valid and within image bounds.

    Checks:
        - image_width and image_height are positive (> 0)
        - bbox has exactly 4 values
        - x2 > x1 and y2 > y1 (positive area)
        - bbox is within image bounds (x1 >= 0, y1 >= 0, x2 <= width, y2 <= height)

    If image_width or image_height is 0 or missing, the detection is
    treated as failing (cannot verify bounds) rather than crashing.

    Args:
        detection: The detection to evaluate.
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        True if the bounding box is valid and within bounds.
    """
    # Cannot verify bounds without valid image dimensions
    if not image_width or image_width <= 0 or not image_height or image_height <= 0:
        return False

    bbox = detection.bbox
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False

    x1, y1, x2, y2 = bbox

    # Check ordering: must have positive width and height
    if x2 <= x1 or y2 <= y1:
        return False

    # Check bounds: bbox must be within image
    if x1 < 0 or y1 < 0:
        return False
    if x2 > image_width or y2 > image_height:
        return False

    return True


def rule_not_tiny(detection: Detection, min_area_px: int = 400) -> bool:
    """True if bbox area >= min_area_px.

    A tiny bounding box is typically noise, not a real object. This rule
    filters out detections that are too small to be meaningful.

    Args:
        detection: The detection to evaluate.
        min_area_px: Minimum area in pixels (default 400).

    Returns:
        True if the bounding box area meets the minimum.
    """
    bbox = detection.bbox
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False

    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        return False

    return (width * height) >= min_area_px


def rule_no_duplicate(
    detection: Detection,
    others: list[Detection],
    iou_threshold: float = 0.9,
) -> bool:
    """True if no other same-class detection has IoU >= iou_threshold.

    Compares the detection against all *other* detections in the same image
    with the same class_id. A detection is not compared against itself
    (identity check by object reference).

    Args:
        detection: The detection to evaluate.
        others: All detections in the same image (may include this detection).
        iou_threshold: IoU threshold above which two detections are
            considered duplicates (default 0.9).

    Returns:
        True if this detection has no duplicates (is unique).
    """
    for other in others:
        # Skip self (identity check)
        if other is detection:
            continue

        # Only check same-class detections
        if other.class_id != detection.class_id:
            continue

        iou = compute_iou(detection.bbox, other.bbox)
        if iou >= iou_threshold:
            # We have a duplicate. Only one should survive.
            # Handle None confidence gracefully by treating it as -1.0
            other_conf = other.conf if other.conf is not None else -1.0
            det_conf = detection.conf if detection.conf is not None else -1.0

            if other_conf > det_conf:
                return False
            elif other_conf == det_conf:
                # Tie-breaker: deterministic based on list order.
                # The one appearing LATER in the list survives.
                idx_other = -1
                idx_det = -1
                for i, item in enumerate(others):
                    if item is other:
                        idx_other = i
                    if item is detection:
                        idx_det = i
                
                if idx_other > idx_det:
                    return False

    return True
