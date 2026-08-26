"""
test_routing.py — Tests for routing logic (routing/router.py).

Test cases to implement:
    - route() correctly splits a mixed list of QualityResults into accepted/flagged/rejected.
    - All-low-confidence image routes entirely to flagged (never over-routes to REJECT).
    - Empty quality_results -> three empty lists, no error.

Owner: Member B
"""
