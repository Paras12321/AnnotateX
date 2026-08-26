"""
test_inference.py — Tests for YOLO inference (inference/yolo_infer.py).

Test cases:
    - run_inference() on a known sample image returns a non-empty list of Detection objects.
    - Each Detection has valid types: bbox is list of 4 floats, conf between 0 and 1.
    - run_inference() on a nonexistent path raises FileNotFoundError.

Owner: Member A
"""

import os
import sys
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.yolo_infer import run_inference
from models.contracts import Detection

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"


def _find_sample_image() -> str | None:
    """Return the path to the first usable image in sample_data/, or None."""
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        matches = list(SAMPLE_DIR.glob(ext))
        if matches:
            return str(matches[0])
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunInference:
    """Tests for the run_inference() public API."""

    def test_nonexistent_path_raises(self):
        """run_inference on a path that doesn't exist must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            run_inference("/tmp/does_not_exist_abc123.jpg")

    def test_returns_list_of_detections(self):
        """run_inference on a real image returns a non-empty list[Detection]."""
        img = _find_sample_image()
        if img is None:
            pytest.skip("No sample image found in sample_data/")

        detections = run_inference(img)

        assert isinstance(detections, list), "Expected list return type"
        # A real YOLO model on a typical photo should detect *something*
        assert len(detections) > 0, "Expected at least one detection"

        for det in detections:
            assert isinstance(det, Detection), f"Expected Detection, got {type(det)}"

    def test_detection_field_types(self):
        """Every Detection should have correct field types."""
        img = _find_sample_image()
        if img is None:
            pytest.skip("No sample image found in sample_data/")

        detections = run_inference(img)
        assert len(detections) > 0, "Need at least one detection to validate fields"

        for det in detections:
            # image_id is a string
            assert isinstance(det.image_id, str)
            assert len(det.image_id) > 0

            # bbox is a list of 4 numbers (floats or ints)
            assert isinstance(det.bbox, list), f"bbox should be list, got {type(det.bbox)}"
            assert len(det.bbox) == 4, f"bbox should have 4 elements, got {len(det.bbox)}"
            for coord in det.bbox:
                assert isinstance(coord, (int, float)), f"bbox coord should be numeric, got {type(coord)}"

            # class_id is int
            assert isinstance(det.class_id, int)

            # class_name is non-empty string
            assert isinstance(det.class_name, str)
            assert len(det.class_name) > 0

            # conf is float in [0, 1]
            assert isinstance(det.conf, float)
            assert 0.0 <= det.conf <= 1.0, f"conf out of range: {det.conf}"

    def test_bbox_absolute_pixel_coords(self):
        """Bounding box values should be in absolute pixel coords (not normalized 0-1)."""
        img = _find_sample_image()
        if img is None:
            pytest.skip("No sample image found in sample_data/")

        detections = run_inference(img)
        if len(detections) == 0:
            pytest.skip("No detections to check")

        for det in detections:
            # At least one coordinate should be > 1 (pixel coords, not normalized)
            assert any(
                c > 1.0 for c in det.bbox
            ), f"bbox looks normalized, expected pixel coords: {det.bbox}"

    def test_image_id_is_filename_stem(self):
        """image_id should equal the filename stem (no extension)."""
        img = _find_sample_image()
        if img is None:
            pytest.skip("No sample image found in sample_data/")

        detections = run_inference(img)
        expected_stem = Path(img).stem

        for det in detections:
            assert det.image_id == expected_stem, (
                f"image_id mismatch: expected '{expected_stem}', got '{det.image_id}'"
            )
