"""
test_inference.py — Tests for YOLO inference (inference/yolo_infer.py).

Test cases:
    - run_inference() on a known sample image returns a non-empty list of Detection objects.
    - Each Detection has valid types: bbox is list of 4 floats, conf between 0 and 1.
    - run_inference() on a nonexistent path raises FileNotFoundError.
    - run_inference_batch() skips corrupt ImageInput objects without raising.
    - Empty batch input returns empty dict, no error.
    - Batch with mixed ok/corrupt returns only ok entries.

Owner: Member A
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.yolo_infer import run_inference, run_inference_batch
from models.contracts import Detection, ImageInput

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"


def _find_sample_image() -> str | None:
    """Return the path to the first usable image in sample_data/, or None."""
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        matches = list(SAMPLE_DIR.glob(ext))
        for m in matches:
            if "corrupt" not in m.name:
                return str(m)
    return None


# ---------------------------------------------------------------------------
# run_inference tests (Day 1, preserved)
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
            assert isinstance(det.image_id, str)
            assert len(det.image_id) > 0
            assert isinstance(det.bbox, list)
            assert len(det.bbox) == 4
            for coord in det.bbox:
                assert isinstance(coord, (int, float))
            assert isinstance(det.class_id, int)
            assert isinstance(det.class_name, str)
            assert len(det.class_name) > 0
            assert isinstance(det.conf, float)
            assert 0.0 <= det.conf <= 1.0

    def test_bbox_absolute_pixel_coords(self):
        """Bounding box values should be in absolute pixel coords (not normalized 0-1)."""
        img = _find_sample_image()
        if img is None:
            pytest.skip("No sample image found in sample_data/")

        detections = run_inference(img)
        if len(detections) == 0:
            pytest.skip("No detections to check")

        for det in detections:
            assert any(c > 1.0 for c in det.bbox), (
                f"bbox looks normalized, expected pixel coords: {det.bbox}"
            )

    def test_image_id_is_filename_stem(self):
        """image_id should equal the filename stem (no extension)."""
        img = _find_sample_image()
        if img is None:
            pytest.skip("No sample image found in sample_data/")

        detections = run_inference(img)
        expected_stem = Path(img).stem

        for det in detections:
            assert det.image_id == expected_stem


# ---------------------------------------------------------------------------
# run_inference_batch tests (Day 2)
# ---------------------------------------------------------------------------


class TestRunInferenceBatch:
    """Tests for the run_inference_batch() API."""

    def test_empty_batch_returns_empty_dict(self):
        """Empty input list -> empty dict, no error."""
        result = run_inference_batch([])
        assert result == {}
        assert isinstance(result, dict)

    def test_skips_corrupt_images(self):
        """Corrupt ImageInput objects are skipped, not in the returned dict."""
        corrupt = ImageInput(
            image_id="bad",
            file_path="/fake/corrupt.jpg",
            status="corrupt",
        )
        result = run_inference_batch([corrupt])
        assert "bad" not in result
        assert len(result) == 0

    def test_skips_unsupported_images(self):
        """Unsupported ImageInput objects are skipped."""
        unsupported = ImageInput(
            image_id="doc",
            file_path="/fake/file.txt",
            status="unsupported_format",
        )
        result = run_inference_batch([unsupported])
        assert "doc" not in result

    def test_processes_ok_images(self):
        """Images with status='ok' are processed and appear in the dict."""
        img = _find_sample_image()
        if img is None:
            pytest.skip("No sample image in sample_data/")

        ok_image = ImageInput(
            image_id=Path(img).stem,
            file_path=img,
            width=640,
            height=480,
            status="ok",
        )
        result = run_inference_batch([ok_image])

        assert Path(img).stem in result
        detections = result[Path(img).stem]
        assert isinstance(detections, list)
        assert len(detections) > 0
        for det in detections:
            assert isinstance(det, Detection)

    def test_mixed_batch_only_returns_ok(self):
        """Batch with ok + corrupt + unsupported only processes ok images."""
        img = _find_sample_image()
        if img is None:
            pytest.skip("No sample image in sample_data/")

        images = [
            ImageInput(image_id="corrupt_one", file_path="/fake/bad.jpg", status="corrupt"),
            ImageInput(
                image_id=Path(img).stem,
                file_path=img,
                width=640, height=480,
                status="ok",
            ),
            ImageInput(image_id="doc", file_path="/fake/a.txt", status="unsupported_format"),
        ]

        result = run_inference_batch(images)

        # Only the ok image should be in results
        assert len(result) == 1
        assert Path(img).stem in result
        assert "corrupt_one" not in result
        assert "doc" not in result

    def test_batch_never_raises_on_bad_input(self):
        """run_inference_batch with only bad inputs should not raise."""
        images = [
            ImageInput(image_id="c1", file_path="/fake/c1.jpg", status="corrupt"),
            ImageInput(image_id="c2", file_path="/fake/c2.png", status="corrupt"),
        ]
        result = run_inference_batch(images)  # Must not raise
        assert isinstance(result, dict)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Day 3: Edge case hardening tests
# ---------------------------------------------------------------------------


class TestDay3InferenceEdgeCases:
    """Day 3 — stabilization edge cases for inference."""

    def test_zero_detection_image_represented_not_dropped(self):
        """An image with zero YOLO detections must appear in the dict with [].

        We create a tiny 1x1 white image that YOLO will find nothing on.
        The key must exist in the returned dict with an empty list.
        """
        import tempfile
        from PIL import Image as PILImage

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tiny = PILImage.new("RGB", (1, 1), color=(255, 255, 255))
            tiny.save(f.name)
            tiny_path = f.name

        try:
            img = ImageInput(
                image_id="tiny_white",
                file_path=tiny_path,
                width=1, height=1,
                status="ok",
            )
            result = run_inference_batch([img])

            # The key MUST exist (not dropped), value is an empty list
            assert "tiny_white" in result, (
                "Zero-detection image must appear in result dict"
            )
            assert result["tiny_white"] == [], (
                f"Expected empty list, got {result['tiny_white']}"
            )
        finally:
            import os
            os.unlink(tiny_path)

    def test_induced_inference_failure_batch_continues(self):
        """If one image causes an inference error, the batch continues.

        We give an ok-status ImageInput pointing to a nonexistent file.
        run_inference will raise FileNotFoundError, which
        run_inference_batch should catch and continue.
        """
        img = _find_sample_image()
        if img is None:
            pytest.skip("No sample image")

        images = [
            # This will cause FileNotFoundError inside run_inference
            ImageInput(
                image_id="will_fail",
                file_path="/tmp/this_does_not_exist_xyz.jpg",
                status="ok",
            ),
            # This should succeed
            ImageInput(
                image_id=Path(img).stem,
                file_path=img,
                width=640, height=480,
                status="ok",
            ),
        ]

        result = run_inference_batch(images)  # Must NOT raise

        # The failed image should NOT be in results
        assert "will_fail" not in result
        # The good image SHOULD be in results
        assert Path(img).stem in result
        assert len(result[Path(img).stem]) > 0

    def test_empty_batch_returns_empty_dict(self):
        """run_inference_batch([]) -> {}, no error (explicit Day 3 re-test)."""
        result = run_inference_batch([])
        assert result == {}
        assert isinstance(result, dict)

    def test_all_fail_batch_returns_empty_dict(self):
        """If every image fails, we get an empty dict, no crash."""
        images = [
            ImageInput(
                image_id="fail1",
                file_path="/tmp/nope1.jpg",
                status="ok",
            ),
            ImageInput(
                image_id="fail2",
                file_path="/tmp/nope2.jpg",
                status="ok",
            ),
        ]
        result = run_inference_batch(images)
        assert isinstance(result, dict)
        assert len(result) == 0

