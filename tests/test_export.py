"""
test_export.py — Tests for export (export/yolo_writer.py, coco_writer.py) and validation.

Test cases implemented:
    - export_yolo produces correctly formatted normalized values for a known bbox/image size.
    - export_coco produces valid JSON with correct top-level keys and pixel-format bbox.
    - Missing image_dims entry raises ValueError.
    - Zero-annotation image still gets an (empty) .txt file — not applicable (YOLO only writes
      files for images that have annotations; empty-input produces empty dict).
    - Fully empty batch produces valid empty outputs, no crash.
    - Structural verification: YOLO lines have 5 numeric fields; COCO JSON is well-formed.

Owner: Member C
"""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.contracts import Detection
from export.yolo_writer import export_yolo
from export.coco_writer import export_coco


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_detection():
    """The canonical example from the spec."""
    return Detection(
        image_id="img1",
        bbox=[100, 50, 300, 250],
        class_id=0,
        class_name="person",
        conf=0.9,
    )


@pytest.fixture
def image_dims():
    return {"img1": (640, 480)}


@pytest.fixture
def class_names():
    return {0: "person"}


# ---------------------------------------------------------------------------
# YOLO exporter tests
# ---------------------------------------------------------------------------

class TestExportYolo:

    def test_hand_calculated_values(self, sample_detection, image_dims, tmp_path):
        """Verify normalized YOLO line matches hand calculation.

        bbox = [100, 50, 300, 250],  image = 640×480
        x_center = (100+300)/2 / 640 = 200/640 = 0.3125
        y_center = (50+250)/2 / 480  = 150/480 = 0.3125
        width    = (300-100) / 640    = 200/640 = 0.3125
        height   = (250-50)  / 480    = 200/480 ≈ 0.4167
        """
        result = export_yolo([sample_detection], image_dims, str(tmp_path))

        assert "img1" in result
        file_path = result["img1"]
        assert os.path.isfile(file_path)

        with open(file_path) as f:
            content = f.read().strip()

        parts = content.split()
        assert len(parts) == 5, f"Expected 5 fields, got {len(parts)}: {content}"

        class_id = int(parts[0])
        x_c, y_c, w, h = [float(v) for v in parts[1:]]

        assert class_id == 0
        assert abs(x_c - 0.3125) < 1e-4
        assert abs(y_c - 0.3125) < 1e-4
        assert abs(w - 0.3125) < 1e-4
        assert abs(h - 0.4167) < 1e-4

    def test_multiple_detections_same_image(self, image_dims, tmp_path):
        """Two detections for the same image should produce two lines in one file."""
        dets = [
            Detection("img1", [0, 0, 64, 48], 0, "person", 0.8),
            Detection("img1", [320, 240, 640, 480], 1, "car", 0.7),
        ]
        result = export_yolo(dets, image_dims, str(tmp_path))

        assert len(result) == 1  # single file
        with open(result["img1"]) as f:
            lines = [l for l in f.read().strip().split("\n") if l]
        assert len(lines) == 2

    def test_multiple_images(self, tmp_path):
        """Different image_ids produce separate files."""
        dims = {"img1": (640, 480), "img2": (1920, 1080)}
        dets = [
            Detection("img1", [100, 50, 300, 250], 0, "person", 0.9),
            Detection("img2", [0, 0, 960, 540], 1, "car", 0.8),
        ]
        result = export_yolo(dets, dims, str(tmp_path))

        assert set(result.keys()) == {"img1", "img2"}
        assert os.path.isfile(result["img1"])
        assert os.path.isfile(result["img2"])

    def test_yolo_line_has_five_numeric_fields(self, sample_detection, image_dims, tmp_path):
        """Structural check: every YOLO line should have exactly 5 fields, all parseable."""
        export_yolo([sample_detection], image_dims, str(tmp_path))
        with open(os.path.join(str(tmp_path), "img1.txt")) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                assert len(parts) == 5
                int(parts[0])  # class_id must be int-parseable
                for v in parts[1:]:
                    float(v)   # coords must be float-parseable

    def test_missing_image_dims_raises(self, sample_detection, tmp_path):
        """Missing image_dims entry must raise a clear ValueError."""
        with pytest.raises(ValueError, match="Missing image_dims"):
            export_yolo([sample_detection], {}, str(tmp_path))

    def test_empty_annotations_produces_empty_result(self, tmp_path):
        """No annotations -> empty dict, no crash."""
        result = export_yolo([], {}, str(tmp_path))
        assert result == {}

    def test_output_dir_created(self, sample_detection, image_dims, tmp_path):
        """output_dir should be created automatically if missing."""
        new_dir = os.path.join(str(tmp_path), "nested", "output")
        export_yolo([sample_detection], image_dims, new_dir)
        assert os.path.isdir(new_dir)


# ---------------------------------------------------------------------------
# COCO exporter tests
# ---------------------------------------------------------------------------

class TestExportCoco:

    def test_valid_json_structure(self, sample_detection, image_dims, class_names, tmp_path):
        """Output must be valid JSON with the four required COCO top-level keys."""
        out_path = os.path.join(str(tmp_path), "coco.json")
        export_coco([sample_detection], image_dims, class_names, out_path)

        with open(out_path) as f:
            coco = json.load(f)

        assert "info" in coco
        assert "images" in coco
        assert "annotations" in coco
        assert "categories" in coco

    def test_pixel_space_bbox(self, sample_detection, image_dims, class_names, tmp_path):
        """COCO bbox must be [x_min, y_min, width, height] in PIXELS.

        bbox input = [100, 50, 300, 250]
        Expected COCO bbox = [100, 50, 200, 200]
        """
        out_path = os.path.join(str(tmp_path), "coco.json")
        export_coco([sample_detection], image_dims, class_names, out_path)

        with open(out_path) as f:
            coco = json.load(f)

        assert len(coco["annotations"]) == 1
        bbox = coco["annotations"][0]["bbox"]
        assert bbox == [100, 50, 200, 200], (
            f"COCO bbox should be pixel-space [x_min, y_min, w, h], got {bbox}"
        )

    def test_area_computed(self, sample_detection, image_dims, class_names, tmp_path):
        """COCO annotation area should equal width * height of the bbox."""
        out_path = os.path.join(str(tmp_path), "coco.json")
        export_coco([sample_detection], image_dims, class_names, out_path)

        with open(out_path) as f:
            coco = json.load(f)

        ann = coco["annotations"][0]
        expected_area = 200 * 200  # w=200, h=200
        assert ann["area"] == expected_area

    def test_images_section(self, sample_detection, image_dims, class_names, tmp_path):
        """Images section should contain correct dimensions."""
        out_path = os.path.join(str(tmp_path), "coco.json")
        export_coco([sample_detection], image_dims, class_names, out_path)

        with open(out_path) as f:
            coco = json.load(f)

        assert len(coco["images"]) == 1
        img = coco["images"][0]
        assert img["width"] == 640
        assert img["height"] == 480

    def test_categories_section(self, sample_detection, image_dims, class_names, tmp_path):
        """Categories should contain all entries from class_names."""
        out_path = os.path.join(str(tmp_path), "coco.json")
        export_coco([sample_detection], image_dims, class_names, out_path)

        with open(out_path) as f:
            coco = json.load(f)

        assert len(coco["categories"]) == 1
        assert coco["categories"][0]["id"] == 0
        assert coco["categories"][0]["name"] == "person"

    def test_missing_image_dims_raises(self, sample_detection, class_names, tmp_path):
        """Missing image_dims entry must raise ValueError."""
        out_path = os.path.join(str(tmp_path), "coco.json")
        with pytest.raises(ValueError, match="Missing image_dims"):
            export_coco([sample_detection], {}, class_names, out_path)

    def test_empty_annotations_produces_valid_json(self, tmp_path):
        """Empty annotation list should still produce valid COCO JSON."""
        out_path = os.path.join(str(tmp_path), "coco.json")
        export_coco([], {}, {}, out_path)

        with open(out_path) as f:
            coco = json.load(f)

        assert coco["annotations"] == []
        assert coco["images"] == []
        assert coco["categories"] == []

    def test_output_dir_created(self, sample_detection, image_dims, class_names, tmp_path):
        """Parent directory of output_path should be created if missing."""
        out_path = os.path.join(str(tmp_path), "nested", "dir", "coco.json")
        export_coco([sample_detection], image_dims, class_names, out_path)
        assert os.path.isfile(out_path)

    def test_score_preserved(self, sample_detection, image_dims, class_names, tmp_path):
        """Confidence score from Detection should be preserved in COCO annotation."""
        out_path = os.path.join(str(tmp_path), "coco.json")
        export_coco([sample_detection], image_dims, class_names, out_path)

        with open(out_path) as f:
            coco = json.load(f)

        assert coco["annotations"][0]["score"] == 0.9
