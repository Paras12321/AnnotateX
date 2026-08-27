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

NOTE: Day 1 stub — route() and build_processing_result() interfaces are
defined. Full batch-level routing comes in Day 2.
"""

from models.contracts import Detection, QualityResult, ProcessingResult


def route(quality_results: list[QualityResult]) -> tuple[list[Detection], list[Detection], list[Detection]]:
    """Split quality results into accepted, flagged, and rejected detections.

    Args:
        quality_results: List of QualityResult objects from the engine.

    Returns:
        Tuple of (accepted, flagged, rejected) — each a list of Detection objects.
    """
    accepted: list[Detection] = []
    flagged: list[Detection] = []
    rejected: list[Detection] = []

    is_all_low_conf = len(quality_results) > 0 and all(
        "confidence" in qr.failed_rules for qr in quality_results
    )

    for qr in quality_results:
        if is_all_low_conf:
            flagged.append(qr.detection)
        else:
            if qr.decision == "ACCEPT":
                accepted.append(qr.detection)
            elif qr.decision == "FLAG":
                flagged.append(qr.detection)
            elif qr.decision == "REJECT":
                rejected.append(qr.detection)

    return accepted, flagged, rejected


def build_processing_result(
    image_id: str,
    detections: list[Detection],
    quality_results: list[QualityResult],
    processing_time_ms: float,
) -> ProcessingResult:
    """Assemble a ProcessingResult from quality-evaluated detections.

    Args:
        image_id: The image identifier.
        detections: All raw detections for this image.
        quality_results: QualityResult for each detection.
        processing_time_ms: Time taken to process this image.

    Returns:
        A fully populated ProcessingResult per the locked contract.
    """
    accepted, flagged, rejected = route(quality_results)

    return ProcessingResult(
        image_id=image_id,
        detections=detections,
        quality_results=quality_results,
        accepted=accepted,
        flagged=flagged,
        rejected=rejected,
        processing_time_ms=processing_time_ms,
    )
