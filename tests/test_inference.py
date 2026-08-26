"""
test_inference.py — Tests for YOLO inference (inference/yolo_infer.py).

Test cases to implement:
    - run_inference() on a known sample image returns a non-empty list of Detection objects.
    - Each Detection has valid types: bbox is list of 4 floats, conf between 0 and 1.
    - run_inference() on a nonexistent path raises FileNotFoundError.
    - run_inference_batch() skips corrupt ImageInput objects without raising.
    - Empty batch input returns empty dict, no error.
    - Image with zero detections returns an entry with an empty list (not omitted).

Owner: Member A
"""
