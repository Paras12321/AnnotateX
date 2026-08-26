"""
iou.py — Intersection over Union (IoU) calculation.

Responsibilities:
    - compute_iou(box_a, box_b) -> float
        Given two bounding boxes as [x1, y1, x2, y2], compute IoU (0.0 to 1.0).
        Used by the Quality Engine's duplicate detection rule.

Kept generic and reusable — no dependency on quality/ or any other pipeline module.

Owner: Shared (primarily used by Member B)
"""
