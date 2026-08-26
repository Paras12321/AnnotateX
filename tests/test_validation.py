"""
test_validation.py — Tests for pre-export validation (validation/validate_export.py).

Test cases implemented:
    - Valid annotation passes validation.
    - Out-of-range class_id fails validation.
    - Bbox outside image dimensions fails validation.
    - Empty annotations list -> empty results, no crash.

Owner: Member C
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.contracts import Detection
from validation.validate_export import validate_annotation, validate_batch


@pytest.fixture
def valid_class_ids():
    return {0, 1, 2}

@pytest.fixture
def image_dims_single():
    return (640, 480)

@pytest.fixture
def image_dims_batch():
    return {"img1": (640, 480)}

def test_validate_annotation_valid(valid_class_ids, image_dims_single):
    det = Detection("img1", [10, 10, 100, 100], 0, "person", 0.9)
    res = validate_annotation(det, image_dims_single, valid_class_ids)
    assert res.is_valid
    assert len(res.errors) == 0

def test_validate_annotation_invalid_class_id(valid_class_ids, image_dims_single):
    det = Detection("img1", [10, 10, 100, 100], 99, "unknown", 0.9)
    res = validate_annotation(det, image_dims_single, valid_class_ids)
    assert not res.is_valid
    assert any("Invalid class_id" in e for e in res.errors)

def test_validate_annotation_bbox_out_of_bounds(valid_class_ids, image_dims_single):
    det = Detection("img1", [10, 10, 700, 100], 0, "person", 0.9)
    res = validate_annotation(det, image_dims_single, valid_class_ids)
    assert not res.is_valid
    assert any("fit within image dimensions" in e for e in res.errors)

def test_validate_annotation_bbox_invalid_coords(valid_class_ids, image_dims_single):
    det = Detection("img1", [100, 100, 10, 10], 0, "person", 0.9)
    res = validate_annotation(det, image_dims_single, valid_class_ids)
    assert not res.is_valid
    assert any("greater than" in e for e in res.errors)

def test_validate_annotation_bbox_format(valid_class_ids, image_dims_single):
    det = Detection("img1", [10, 10, 100], 0, "person", 0.9)
    res = validate_annotation(det, image_dims_single, valid_class_ids)
    assert not res.is_valid
    assert any("4 numeric values" in e for e in res.errors)

def test_validate_batch(valid_class_ids, image_dims_batch):
    dets = [
        Detection("img1", [10, 10, 100, 100], 0, "person", 0.9),
        Detection("img1", [10, 10, 100, 100], 99, "unknown", 0.9)
    ]
    results = validate_batch(dets, image_dims_batch, valid_class_ids)
    assert len(results) == 2
    assert results[0].is_valid
    assert not results[1].is_valid

def test_validate_batch_empty_list(valid_class_ids, image_dims_batch):
    results = validate_batch([], image_dims_batch, valid_class_ids)
    assert results == []
