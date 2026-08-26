"""
router.py — Routing logic: ACCEPT / FLAG / REJECT decision.

Responsibilities:
    - route(quality_results) -> (accepted, flagged, rejected)
        Splits a list of QualityResults into three lists of Detection objects
        based on the decision field.
    - build_processing_result(image_id, detections, quality_results, time_ms) -> ProcessingResult
        Assembles a ProcessingResult per the locked data contract.

Routing semantics:
    ACCEPT  → sent to Validation → Export (becomes a dataset annotation)
    FLAG    → shown in UI review list, NOT exported
    REJECT  → dropped, logged, never shown as an annotation

Owner: Member B
"""
