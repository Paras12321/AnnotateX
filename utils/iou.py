"""
iou.py — Intersection over Union (IoU) calculation.

Responsibilities:
    - compute_iou(box_a, box_b) -> float
        Given two bounding boxes as [x1, y1, x2, y2], compute IoU (0.0 to 1.0).
        Used by the Quality Engine's duplicate detection rule.

Kept generic and reusable — no dependency on quality/ or any other pipeline module.

Owner: Shared (primarily used by Member B)
"""


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute Intersection over Union for two bounding boxes.

    Args:
        box_a: [x1, y1, x2, y2] in pixel coordinates.
        box_b: [x1, y1, x2, y2] in pixel coordinates.

    Returns:
        IoU value between 0.0 and 1.0. Returns 0.0 if either box has
        zero or negative area, or if the boxes don't overlap.
    """
    # Intersection rectangle
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])

    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # Individual areas
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])

    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area
