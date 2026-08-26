"""
orchestrator.py — Pipeline orchestration.

Responsibilities:
    - Provide run_pipeline(file_paths, config) -> BatchResult.
    - Call each stage in order:
        1. preprocessing.preprocess_batch(file_paths) -> list[ImageInput]
        2. inference.run_inference_batch(images)      -> dict[image_id, list[Detection]]
        3. quality.evaluate_batch(...)                -> dict[image_id, list[QualityResult]]
        4. routing.route(...) per image               -> accepted / flagged / rejected
        5. validation + export                        -> export files
    - Assemble ProcessingResult per image and a final BatchResult.
    - Handle per-image failures gracefully (skip the image, continue the batch).

Owner: Member D (integration), shared with all members.
"""
