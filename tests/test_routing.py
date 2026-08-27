"""
test_routing.py — Tests for routing logic (routing/router.py).

Test cases:
    - route() correctly splits a mixed list of QualityResults into accepted/flagged/rejected.
    - All-low-confidence image routes entirely to flagged (never over-routes to REJECT).
    - Empty quality_results -> three empty lists, no error.

Owner: Member B
"""

import pytest
from models.contracts import Detection, QualityResult
from routing.router import route

def _det(
    bbox=None, conf=0.85, class_id=0, class_name="person", image_id="img1"
) -> Detection:
    if bbox is None:
        bbox = [100.0, 50.0, 300.0, 250.0]
    return Detection(
        image_id=image_id,
        bbox=bbox,
        class_id=class_id,
        class_name=class_name,
        conf=conf,
    )

class TestRoute:
    def test_splits_correctly(self):
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

    def test_all_low_confidence(self):
        """All-low-confidence image routes entirely to flagged (never over-routes to REJECT)."""
        det1 = _det(conf=0.1)
        det2 = _det(conf=0.2)
        
        qr1 = QualityResult(det1, ["valid_box"], ["confidence"], "FLAG", "low conf")
        # det2 failed valid_box AND confidence
        qr2 = QualityResult(det2, [], ["confidence", "valid_box"], "REJECT", "bad box and low conf")
        
        accepted, flagged, rejected = route([qr1, qr2])
        
        assert len(accepted) == 0
        assert len(flagged) == 2, "Both should be flagged if entire batch is low confidence"
        assert len(rejected) == 0, "No detection should be rejected in an all-low-confidence batch"
