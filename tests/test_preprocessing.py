"""
test_preprocessing.py — Tests for preprocessing (validate.py, resize.py).

Test cases:
    - Valid image -> ImageInput with status "ok" and correct width/height.
    - Corrupt file -> ImageInput with status "corrupt".
    - Unsupported file extension -> ImageInput with status "unsupported_format".
    - Nonexistent file -> ImageInput with status "corrupt".
    - preprocess_batch with empty list returns [], no error.
    - preprocess_batch with mixed inputs never raises.
    - Resize only triggers for oversized images.

Owner: Member A
"""

import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.contracts import ImageInput
from preprocessing.validate import validate_and_load, preprocess_batch
from preprocessing.resize import resize_if_needed, MAX_DIMENSION

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_data"


def _find_sample_image() -> str | None:
    """Return the path to the first usable image in sample_data/."""
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        matches = list(SAMPLE_DIR.glob(ext))
        for m in matches:
            if "corrupt" not in m.name:
                return str(m)
    return None


# ---------------------------------------------------------------------------
# validate_and_load tests
# ---------------------------------------------------------------------------


class TestValidateAndLoad:
    """Tests for validate_and_load()."""

    def test_valid_image_returns_ok(self):
        """A real image should return status='ok' with width/height > 0."""
        img = _find_sample_image()
        if img is None:
            pytest.skip("No sample image in sample_data/")

        result = validate_and_load(img)

        assert isinstance(result, ImageInput)
        assert result.status == "ok"
        assert result.width > 0
        assert result.height > 0
        assert result.image_id == Path(img).stem
        assert result.file_path == img

    def test_corrupt_file_returns_corrupt(self):
        """A file with a valid extension but invalid content -> 'corrupt'."""
        corrupt_path = str(SAMPLE_DIR / "corrupt.jpg")
        result = validate_and_load(corrupt_path)

        assert isinstance(result, ImageInput)
        assert result.status == "corrupt"
        assert result.image_id == "corrupt"

    def test_unsupported_extension_returns_unsupported(self):
        """A file with unsupported extension -> 'unsupported_format'."""
        txt_path = str(SAMPLE_DIR / "test.txt")
        result = validate_and_load(txt_path)

        assert isinstance(result, ImageInput)
        assert result.status == "unsupported_format"

    def test_nonexistent_file_returns_corrupt(self):
        """A path that doesn't exist -> status='corrupt'."""
        result = validate_and_load("/nonexistent/path/fake.jpg")

        assert isinstance(result, ImageInput)
        assert result.status == "corrupt"

    def test_never_raises(self):
        """validate_and_load should never raise, even with garbage input."""
        bad_inputs = [
            "",
            "/dev/null",
            "/tmp/definitely_not_here.jpg",
            str(SAMPLE_DIR / "corrupt.jpg"),
            str(SAMPLE_DIR / "test.txt"),
        ]
        for path in bad_inputs:
            result = validate_and_load(path)  # Must not raise
            assert isinstance(result, ImageInput)
            assert result.status in ("ok", "corrupt", "unsupported_format")


# ---------------------------------------------------------------------------
# preprocess_batch tests
# ---------------------------------------------------------------------------


class TestPreprocessBatch:
    """Tests for preprocess_batch()."""

    def test_empty_list(self):
        """Empty input -> empty output, no error."""
        result = preprocess_batch([])
        assert result == []

    def test_never_drops_entries(self):
        """Output list length must equal input list length."""
        inputs = [
            str(SAMPLE_DIR / "corrupt.jpg"),
            str(SAMPLE_DIR / "test.txt"),
            "/nonexistent/file.jpg",
        ]
        img = _find_sample_image()
        if img:
            inputs.insert(0, img)

        results = preprocess_batch(inputs)
        assert len(results) == len(inputs)

    def test_mixed_inputs_never_raises(self):
        """Batch with valid, corrupt, and unsupported files never raises."""
        inputs = [
            str(SAMPLE_DIR / "corrupt.jpg"),
            str(SAMPLE_DIR / "test.txt"),
        ]
        img = _find_sample_image()
        if img:
            inputs.append(img)

        results = preprocess_batch(inputs)  # Must not raise

        statuses = {r.status for r in results}
        # We should see at least corrupt and unsupported
        assert "corrupt" in statuses
        assert "unsupported_format" in statuses

    def test_all_results_are_image_inputs(self):
        """Every result should be an ImageInput instance."""
        inputs = [str(SAMPLE_DIR / "corrupt.jpg")]
        img = _find_sample_image()
        if img:
            inputs.append(img)

        for result in preprocess_batch(inputs):
            assert isinstance(result, ImageInput)


# ---------------------------------------------------------------------------
# resize_if_needed tests
# ---------------------------------------------------------------------------


class TestResizeIfNeeded:
    """Tests for resize_if_needed()."""

    def test_no_resize_when_small(self):
        """Images within MAX_DIMENSION are not resized."""
        path, w, h = resize_if_needed("/fake/path.jpg", 640, 480)
        assert w == 640
        assert h == 480

    def test_resize_when_too_wide(self):
        """Images wider than MAX_DIMENSION get scaled down."""
        img = _find_sample_image()
        if img is None:
            pytest.skip("No sample image")

        # Simulate oversized dimensions (don't actually resize the test file)
        _, new_w, new_h = resize_if_needed(img, 4000, 3000)

        assert new_w <= MAX_DIMENSION
        assert new_h <= MAX_DIMENSION
        # Aspect ratio preserved (roughly)
        assert abs(new_w / new_h - 4000 / 3000) < 0.05


# ---------------------------------------------------------------------------
# Day 3: Edge case hardening tests
# ---------------------------------------------------------------------------


class TestDay3EdgeCases:
    """Day 3 — stabilization edge cases."""

    def test_empty_batch_returns_empty_list(self):
        """preprocess_batch([]) -> [], no error."""
        result = preprocess_batch([])
        assert result == []
        assert isinstance(result, list)

    def test_empty_string_path_never_raises(self):
        """validate_and_load('') should not raise."""
        result = validate_and_load("")
        assert isinstance(result, ImageInput)
        assert result.status in ("corrupt", "unsupported_format")

    def test_none_like_paths_in_batch(self):
        """Batch with pathological inputs never raises."""
        bad_inputs = [
            "",
            "   ",
            "/dev/null",
            "/proc/cpuinfo",
            str(SAMPLE_DIR / "nonexistent.jpg"),
        ]
        results = preprocess_batch(bad_inputs)
        assert len(results) == len(bad_inputs)
        for r in results:
            assert isinstance(r, ImageInput)
            assert r.status in ("ok", "corrupt", "unsupported_format")

    def test_valid_image_has_positive_dimensions(self):
        """A valid image must have width > 0 and height > 0 after loading."""
        img = _find_sample_image()
        if img is None:
            pytest.skip("No sample image")

        result = validate_and_load(img)
        assert result.status == "ok"
        assert result.width > 0, "width must be positive"
        assert result.height > 0, "height must be positive"

    def test_corrupt_image_has_zero_dimensions(self):
        """A corrupt image should have default 0 dimensions."""
        result = validate_and_load(str(SAMPLE_DIR / "corrupt.jpg"))
        assert result.status == "corrupt"
        assert result.width == 0
        assert result.height == 0

