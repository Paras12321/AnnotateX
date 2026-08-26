"""
test_quality.py — Tests for Quality Engine (quality/rules.py, quality/engine.py).

Test cases to implement:
    - Each rule tested independently with at least one pass and one fail case.
    - evaluate_quality tested for all three decision outcomes (ACCEPT/FLAG/REJECT).
    - IoU function: identical boxes -> 1.0, non-overlapping boxes -> 0.0.
    - evaluate_batch with real-shaped multi-image dict.
    - None conf value handled gracefully (doesn't crash).
    - Three-way duplicate overlap case.

Owner: Member B
"""
