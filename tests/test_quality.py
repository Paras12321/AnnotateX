"""
test_quality.py — Tests for Quality Engine (quality/rules.py, quality/engine.py)
and IoU utility (utils/iou.py).

Covers:
    - Each rule tested independently with at least one pass and one fail case.
    - evaluate_quality tested for all three decision outcomes (ACCEPT/FLAG/REJECT).
    - IoU function: identical boxes -> 1.0, non-overlapping boxes -> 0.0.

Owner: Member B
"""

import pytest
from models.contracts import Detection, QualityResult
from utils.iou import compute_iou
from quality.rules import rule_confidence, rule_valid_box, rule_not_tiny, rule_no_duplicate
from quality.engine import evaluate_quality
from routing.router import route, build_processing_result


# ---------------------------------------------------------------------------
# Helper: create a Detection with sensible defaults
# ---------------------------------------------------------------------------

def _det(
    bbox=None, conf=0.85, class_id=0, class_name="person", image_id="img1"
) -> Detection:
    """Shorthand factory for test Detection objects."""
    if bbox is None:
        bbox = [100.0, 50.0, 300.0, 250.0]  # 200x200 = 40000 px area
    return Detection(
        image_id=image_id,
        bbox=bbox,
        class_id=class_id,
        class_name=class_name,
        conf=conf,
    )


# ===== IoU Tests =====

class TestComputeIoU:
    """Tests for utils/iou.py — compute_iou."""

    def test_identical_boxes_iou_is_one(self):
        box = [0.0, 0.0, 100.0, 100.0]
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_non_overlapping_boxes_iou_is_zero(self):
        box_a = [0.0, 0.0, 50.0, 50.0]
        box_b = [100.0, 100.0, 200.0, 200.0]
        assert compute_iou(box_a, box_b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        box_a = [0.0, 0.0, 100.0, 100.0]    # area = 10000
        box_b = [50.0, 50.0, 150.0, 150.0]  # area = 10000
        # Intersection: [50,50] to [100,100] = 50*50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        expected = 2500.0 / 17500.0
        assert compute_iou(box_a, box_b) == pytest.approx(expected)

    def test_one_box_inside_another(self):
        outer = [0.0, 0.0, 200.0, 200.0]   # area = 40000
        inner = [50.0, 50.0, 100.0, 100.0]  # area = 2500
        # Intersection = inner area = 2500
        # Union = 40000 + 2500 - 2500 = 40000
        expected = 2500.0 / 40000.0
        assert compute_iou(outer, inner) == pytest.approx(expected)

    def test_zero_area_box_returns_zero(self):
        box_a = [10.0, 10.0, 10.0, 10.0]  # zero area (point)
        box_b = [0.0, 0.0, 100.0, 100.0]
        assert compute_iou(box_a, box_b) == pytest.approx(0.0)


# ===== Rule: Confidence =====

class TestRuleConfidence:
    """Tests for rule_confidence."""

    def test_pass_above_threshold(self):
        det = _det(conf=0.8)
        assert rule_confidence(det, threshold=0.5) is True

    def test_pass_at_threshold(self):
        det = _det(conf=0.5)
        assert rule_confidence(det, threshold=0.5) is True

    def test_fail_below_threshold(self):
        det = _det(conf=0.3)
        assert rule_confidence(det, threshold=0.5) is False

    def test_fail_none_conf(self):
        det = _det(conf=None)
        assert rule_confidence(det, threshold=0.5) is False


# ===== Rule: Valid Box =====

class TestRuleValidBox:
    """Tests for rule_valid_box."""

    def test_pass_valid_box(self):
        det = _det(bbox=[10.0, 10.0, 200.0, 200.0])
        assert rule_valid_box(det, image_width=640, image_height=480) is True

    def test_fail_inverted_x(self):
        det = _det(bbox=[200.0, 10.0, 100.0, 200.0])  # x2 < x1
        assert rule_valid_box(det, image_width=640, image_height=480) is False

    def test_fail_inverted_y(self):
        det = _det(bbox=[10.0, 200.0, 100.0, 100.0])  # y2 < y1
        assert rule_valid_box(det, image_width=640, image_height=480) is False

    def test_fail_out_of_bounds(self):
        det = _det(bbox=[10.0, 10.0, 700.0, 200.0])  # x2 > image_width
        assert rule_valid_box(det, image_width=640, image_height=480) is False

    def test_fail_negative_coords(self):
        det = _det(bbox=[-10.0, 10.0, 100.0, 200.0])
        assert rule_valid_box(det, image_width=640, image_height=480) is False

    def test_fail_zero_image_width(self):
        """Missing/zero image dimensions should fail, not crash."""
        det = _det(bbox=[10.0, 10.0, 100.0, 200.0])
        assert rule_valid_box(det, image_width=0, image_height=480) is False

    def test_fail_zero_image_height(self):
        det = _det(bbox=[10.0, 10.0, 100.0, 200.0])
        assert rule_valid_box(det, image_width=640, image_height=0) is False

    def test_fail_equal_x1_x2(self):
        """Zero-width box should fail."""
        det = _det(bbox=[100.0, 10.0, 100.0, 200.0])
        assert rule_valid_box(det, image_width=640, image_height=480) is False


# ===== Rule: Not Tiny =====

class TestRuleNotTiny:
    """Tests for rule_not_tiny."""

    def test_pass_large_box(self):
        det = _det(bbox=[0.0, 0.0, 100.0, 100.0])  # area = 10000
        assert rule_not_tiny(det, min_area_px=400) is True

    def test_pass_exactly_at_minimum(self):
        det = _det(bbox=[0.0, 0.0, 20.0, 20.0])  # area = 400
        assert rule_not_tiny(det, min_area_px=400) is True

    def test_fail_tiny_box(self):
        det = _det(bbox=[0.0, 0.0, 10.0, 10.0])  # area = 100
        assert rule_not_tiny(det, min_area_px=400) is False

    def test_fail_zero_area(self):
        det = _det(bbox=[10.0, 10.0, 10.0, 50.0])  # width = 0
        assert rule_not_tiny(det, min_area_px=400) is False


# ===== Rule: No Duplicate =====

class TestRuleNoDuplicate:
    """Tests for rule_no_duplicate."""

    def test_pass_no_other_detections(self):
        det = _det()
        assert rule_no_duplicate(det, [det], iou_threshold=0.9) is True

    def test_pass_different_classes(self):
        det_a = _det(bbox=[10.0, 10.0, 200.0, 200.0], class_id=0)
        det_b = _det(bbox=[10.0, 10.0, 200.0, 200.0], class_id=1)
        assert rule_no_duplicate(det_a, [det_a, det_b], iou_threshold=0.9) is True

    def test_fail_exact_duplicate_same_class(self):
        det_a = _det(bbox=[10.0, 10.0, 200.0, 200.0], class_id=0)
        det_b = _det(bbox=[10.0, 10.0, 200.0, 200.0], class_id=0)
        assert rule_no_duplicate(det_a, [det_a, det_b], iou_threshold=0.9) is False

    def test_pass_low_overlap_same_class(self):
        det_a = _det(bbox=[0.0, 0.0, 100.0, 100.0], class_id=0)
        det_b = _det(bbox=[90.0, 90.0, 200.0, 200.0], class_id=0)
        # IoU is small — should pass
        assert rule_no_duplicate(det_a, [det_a, det_b], iou_threshold=0.9) is True

    def test_does_not_compare_against_self(self):
        """A detection in the list should not flag itself as a duplicate."""
        det = _det()
        assert rule_no_duplicate(det, [det], iou_threshold=0.0) is True

    def test_three_way_duplicate_only_one_survives(self):
        """Three identical boxes, only one should survive (highest conf/id)."""
        det1 = _det(bbox=[10.0, 10.0, 200.0, 200.0], conf=0.8, class_id=0)
        det2 = _det(bbox=[10.0, 10.0, 200.0, 200.0], conf=0.9, class_id=0)
        det3 = _det(bbox=[10.0, 10.0, 200.0, 200.0], conf=0.85, class_id=0)
        
        # det2 has highest conf, it should survive
        assert rule_no_duplicate(det1, [det1, det2, det3], iou_threshold=0.9) is False
        assert rule_no_duplicate(det2, [det1, det2, det3], iou_threshold=0.9) is True
        assert rule_no_duplicate(det3, [det1, det2, det3], iou_threshold=0.9) is False

    def test_tie_breaker_equal_confidence(self):
        """Two identical boxes, equal conf -> tiebreaker by object ID."""
        det1 = _det(bbox=[10.0, 10.0, 200.0, 200.0], conf=0.9, class_id=0)
        det2 = _det(bbox=[10.0, 10.0, 200.0, 200.0], conf=0.9, class_id=0)
        
        survives1 = rule_no_duplicate(det1, [det1, det2], iou_threshold=0.9)
        survives2 = rule_no_duplicate(det2, [det1, det2], iou_threshold=0.9)
        
        assert survives1 != survives2
        assert survives1 or survives2


# ===== Quality Engine: evaluate_quality =====

class TestEvaluateQuality:
    """Tests for quality/engine.py — evaluate_quality."""

    DEFAULT_CONFIG = {"conf_threshold": 0.5, "min_area_px": 400, "iou_threshold": 0.9}

    def test_accept_all_rules_pass(self):
        """High confidence, valid box, not tiny, no duplicate → ACCEPT."""
        det = _det(bbox=[10.0, 10.0, 200.0, 200.0], conf=0.9)
        result = evaluate_quality(det, [det], 640, 480, self.DEFAULT_CONFIG)

        assert isinstance(result, QualityResult)
        assert result.decision == "ACCEPT"
        assert len(result.failed_rules) == 0
        assert "confidence" in result.passed_rules
        assert "valid_box" in result.passed_rules
        assert "not_tiny" in result.passed_rules
        assert "no_duplicate" in result.passed_rules
        assert "accepted" in result.reason.lower()

    def test_flag_low_confidence(self):
        """Low confidence, but valid box → FLAG, not REJECT."""
        det = _det(bbox=[10.0, 10.0, 200.0, 200.0], conf=0.3)
        result = evaluate_quality(det, [det], 640, 480, self.DEFAULT_CONFIG)

        assert result.decision == "FLAG"
        assert "confidence" in result.failed_rules
        assert "valid_box" in result.passed_rules

    def test_flag_tiny_box(self):
        """Good confidence but tiny box → FLAG."""
        det = _det(bbox=[10.0, 10.0, 15.0, 15.0], conf=0.9)
        # bbox is inside image but area = 25 < 400
        result = evaluate_quality(det, [det], 640, 480, self.DEFAULT_CONFIG)

        assert result.decision == "FLAG"
        assert "not_tiny" in result.failed_rules

    def test_flag_duplicate(self):
        """Good confidence, big box, duplicate -> highest conf ACCEPT, others FLAG."""
        det_a = _det(bbox=[10.0, 10.0, 200.0, 200.0], conf=0.9, class_id=0)
        det_b = _det(bbox=[10.0, 10.0, 200.0, 200.0], conf=0.85, class_id=0)
        
        result_a = evaluate_quality(det_a, [det_a, det_b], 640, 480, self.DEFAULT_CONFIG)
        result_b = evaluate_quality(det_b, [det_a, det_b], 640, 480, self.DEFAULT_CONFIG)

        assert result_a.decision == "ACCEPT"
        assert result_b.decision == "FLAG"
        assert "no_duplicate" in result_b.failed_rules

    def test_reject_invalid_box(self):
        """Invalid bounding box (x2 < x1) → REJECT regardless of confidence."""
        det = _det(bbox=[200.0, 10.0, 100.0, 200.0], conf=0.99)
        result = evaluate_quality(det, [det], 640, 480, self.DEFAULT_CONFIG)

        assert result.decision == "REJECT"
        assert "valid_box" in result.failed_rules

    def test_reject_out_of_bounds(self):
        """Bounding box outside image → REJECT."""
        det = _det(bbox=[10.0, 10.0, 700.0, 500.0], conf=0.95)
        result = evaluate_quality(det, [det], 640, 480, self.DEFAULT_CONFIG)

        assert result.decision == "REJECT"
        assert "valid_box" in result.failed_rules

    def test_reject_zero_image_dims(self):
        """Zero image dimensions → REJECT (valid_box can't verify)."""
        det = _det(conf=0.9)
        result = evaluate_quality(det, [det], 0, 0, self.DEFAULT_CONFIG)

        assert result.decision == "REJECT"

    def test_flag_multiple_failures(self):
        """Low confidence + tiny box → FLAG with both in failed_rules."""
        det = _det(bbox=[10.0, 10.0, 15.0, 15.0], conf=0.3)
        result = evaluate_quality(det, [det], 640, 480, self.DEFAULT_CONFIG)

        assert result.decision == "FLAG"
        assert "confidence" in result.failed_rules
        assert "not_tiny" in result.failed_rules

    def test_reason_is_human_readable(self):
        """Reason string should name the failed rules."""
        det = _det(bbox=[10.0, 10.0, 200.0, 200.0], conf=0.3)
        result = evaluate_quality(det, [det], 640, 480, self.DEFAULT_CONFIG)

        assert "confidence" in result.reason

    def test_config_thresholds_are_used(self):
        """Custom config thresholds should be respected."""
        det = _det(bbox=[10.0, 10.0, 200.0, 200.0], conf=0.3)

        # With a low threshold, this should ACCEPT
        lenient_config = {"conf_threshold": 0.1, "min_area_px": 100, "iou_threshold": 0.9}
        result = evaluate_quality(det, [det], 640, 480, lenient_config)
        assert result.decision == "ACCEPT"


# ===== Router Tests =====

class TestRoute:
    """Tests for routing/router.py — route()."""

    def test_splits_correctly(self):
        """route() should split by decision string."""
        det_a = _det(conf=0.9)
        det_f = _det(conf=0.3)
        det_r = _det(conf=0.1)

        qr_accept = QualityResult(det_a, ["confidence", "valid_box"], [], "ACCEPT", "ok")
        qr_flag = QualityResult(det_f, ["valid_box"], ["confidence"], "FLAG", "low conf")
        qr_reject = QualityResult(det_r, [], ["valid_box"], "REJECT", "bad box")

        accepted, flagged, rejected = route([qr_accept, qr_flag, qr_reject])

        assert len(accepted) == 1
        assert len(flagged) == 1
        assert len(rejected) == 1
        assert accepted[0] is det_a
        assert flagged[0] is det_f
        assert rejected[0] is det_r

    def test_empty_input(self):
        accepted, flagged, rejected = route([])
        assert accepted == []
        assert flagged == []
        assert rejected == []

    def test_all_same_decision(self):
        det = _det()
        qr = QualityResult(det, ["valid_box"], ["confidence"], "FLAG", "low conf")
        accepted, flagged, rejected = route([qr, qr, qr])

        assert len(accepted) == 0
        assert len(flagged) == 3
        assert len(rejected) == 0


# ===== build_processing_result =====

class TestBuildProcessingResult:
    """Tests for routing/router.py — build_processing_result()."""

    def test_assembles_contract(self):
        det_a = _det(bbox=[10.0, 10.0, 200.0, 200.0], conf=0.9)
        det_f = _det(bbox=[10.0, 10.0, 15.0, 15.0], conf=0.9)

        config = {"conf_threshold": 0.5, "min_area_px": 400, "iou_threshold": 0.9}
        qr_a = evaluate_quality(det_a, [det_a, det_f], 640, 480, config)
        qr_f = evaluate_quality(det_f, [det_a, det_f], 640, 480, config)

        result = build_processing_result(
            image_id="img1",
            detections=[det_a, det_f],
            quality_results=[qr_a, qr_f],
            processing_time_ms=42.5,
        )

        assert result.image_id == "img1"
        assert len(result.detections) == 2
        assert len(result.quality_results) == 2
        assert result.processing_time_ms == 42.5
        # accepted + flagged + rejected should cover all detections
        total_routed = len(result.accepted) + len(result.flagged) + len(result.rejected)
        total_routed = len(result.accepted) + len(result.flagged) + len(result.rejected)
        assert total_routed == 2


# ===== Quality Engine: evaluate_batch =====

from quality.engine import evaluate_batch
import logging

class TestEvaluateBatch:
    """Tests for quality/engine.py — evaluate_batch."""

    DEFAULT_CONFIG = {"conf_threshold": 0.5, "min_area_px": 400, "iou_threshold": 0.9}

    def test_evaluate_batch_mixed_images(self):
        det1 = _det(image_id="img1", bbox=[10.0, 10.0, 200.0, 200.0], conf=0.9)  # ACCEPT
        det2 = _det(image_id="img2", bbox=[10.0, 10.0, 200.0, 200.0], conf=0.3)  # FLAG
        
        detections_by_image = {
            "img1": [det1],
            "img2": [det2]
        }
        image_dims = {
            "img1": (640, 480),
            "img2": (800, 600)
        }
        
        results = evaluate_batch(detections_by_image, image_dims, self.DEFAULT_CONFIG)
        
        assert "img1" in results
        assert "img2" in results
        assert len(results["img1"]) == 1
        assert len(results["img2"]) == 1
        
        assert results["img1"][0].decision == "ACCEPT"
        assert results["img2"][0].decision == "FLAG"

    def test_evaluate_batch_zero_detections(self):
        detections_by_image = {
            "img1": []
        }
        image_dims = {
            "img1": (640, 480)
        }
        
        results = evaluate_batch(detections_by_image, image_dims, self.DEFAULT_CONFIG)
        
        assert "img1" in results
        assert len(results["img1"]) == 0

    def test_evaluate_batch_missing_dims_skips(self, caplog):
        det1 = _det(image_id="img1", bbox=[10.0, 10.0, 200.0, 200.0], conf=0.9)
        detections_by_image = {
            "img1": [det1]
        }
        image_dims = {}  # Missing img1
        
        with caplog.at_level(logging.WARNING):
            results = evaluate_batch(detections_by_image, image_dims, self.DEFAULT_CONFIG)
        
        assert "img1" not in results
        assert "Image dimensions missing for image_id: img1" in caplog.text
