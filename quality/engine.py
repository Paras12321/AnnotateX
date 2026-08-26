"""
engine.py — Quality Engine: combines rules into a QualityResult.

Responsibilities:
    - evaluate_quality(detection, all_detections, image_w, image_h, config) -> QualityResult
        Runs all four rules, builds passed_rules/failed_rules lists, and sets decision:
            REJECT  — rule_valid_box fails (structurally broken)
            ACCEPT  — all rules pass and conf >= threshold
            FLAG    — borderline confidence, tiny box, or duplicate
    - evaluate_batch(detections_by_image, image_dims, config) -> dict[str, list[QualityResult]]
        Runs evaluate_quality for every detection in every image.

Owner: Member B
"""

from models.contracts import Detection, QualityResult
from quality.rules import rule_confidence, rule_valid_box, rule_not_tiny, rule_no_duplicate


# Rule names used in passed_rules / failed_rules — keep consistent
_RULE_CONFIDENCE = "confidence"
_RULE_VALID_BOX = "valid_box"
_RULE_NOT_TINY = "not_tiny"
_RULE_NO_DUPLICATE = "no_duplicate"


def evaluate_quality(
    detection: Detection,
    all_detections_same_image: list[Detection],
    image_width: int,
    image_height: int,
    config: dict,
) -> QualityResult:
    """Evaluate a single detection against all four quality rules.

    Decision logic:
        REJECT — rule_valid_box fails (structurally broken, never worth reviewing)
        ACCEPT — all rules pass and conf >= threshold
        FLAG   — any non-bounds rule fails (borderline confidence, tiny box, or duplicate)

    Args:
        detection: The detection to evaluate.
        all_detections_same_image: All detections for this image (for duplicate check).
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        config: Dict with keys: "conf_threshold" (float), optionally
            "min_area_px" (int, default 400), "iou_threshold" (float, default 0.9).

    Returns:
        QualityResult with passed/failed rules, decision, and human-readable reason.
    """
    conf_threshold = config.get("conf_threshold", 0.5)
    min_area_px = config.get("min_area_px", 400)
    iou_threshold = config.get("iou_threshold", 0.9)

    passed_rules: list[str] = []
    failed_rules: list[str] = []

    # --- Run each rule ---

    # Rule 1: Confidence
    if rule_confidence(detection, conf_threshold):
        passed_rules.append(_RULE_CONFIDENCE)
    else:
        failed_rules.append(_RULE_CONFIDENCE)

    # Rule 2: Valid bounding box (bounds + geometry)
    if rule_valid_box(detection, image_width, image_height):
        passed_rules.append(_RULE_VALID_BOX)
    else:
        failed_rules.append(_RULE_VALID_BOX)

    # Rule 3: Not tiny
    if rule_not_tiny(detection, min_area_px):
        passed_rules.append(_RULE_NOT_TINY)
    else:
        failed_rules.append(_RULE_NOT_TINY)

    # Rule 4: No duplicate
    if rule_no_duplicate(detection, all_detections_same_image, iou_threshold):
        passed_rules.append(_RULE_NO_DUPLICATE)
    else:
        failed_rules.append(_RULE_NO_DUPLICATE)

    # --- Determine decision ---

    if _RULE_VALID_BOX in failed_rules:
        # Structurally invalid — never worth reviewing
        decision = "REJECT"
        reason = f"rejected: {', '.join(failed_rules)}"
    elif len(failed_rules) == 0:
        # All rules passed
        decision = "ACCEPT"
        reason = "accepted: all rules passed"
    else:
        # Some non-bounds rules failed — needs human review
        decision = "FLAG"
        reason = f"flagged: {', '.join(failed_rules)}"

    return QualityResult(
        detection=detection,
        passed_rules=passed_rules,
        failed_rules=failed_rules,
        decision=decision,
        reason=reason,
    )


import logging

def evaluate_batch(
    detections_by_image: dict[str, list[Detection]],
    image_dims: dict[str, tuple[int, int]],
    config: dict,
) -> dict[str, list[QualityResult]]:
    """Runs evaluate_quality for every detection in every image.

    Args:
        detections_by_image: Dictionary mapping image_id to list of Detections.
        image_dims: Dictionary mapping image_id to (width, height).
        config: Configuration dict with thresholds.

    Returns:
        Dictionary mapping image_id to list of QualityResult objects.
        If an image has zero detections, returns an empty list for that image_id.
    """
    results: dict[str, list[QualityResult]] = {}
    
    for image_id, detections in detections_by_image.items():
        if image_id not in image_dims:
            logging.warning(f"Image dimensions missing for image_id: {image_id}. Skipping quality evaluation.")
            continue
            
        width, height = image_dims[image_id]
        
        image_quality_results = []
        for detection in detections:
            qr = evaluate_quality(
                detection=detection,
                all_detections_same_image=detections,
                image_width=width,
                image_height=height,
                config=config,
            )
            image_quality_results.append(qr)
            
        results[image_id] = image_quality_results
        
    return results
