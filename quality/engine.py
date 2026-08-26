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
