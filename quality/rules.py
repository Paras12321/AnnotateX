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
