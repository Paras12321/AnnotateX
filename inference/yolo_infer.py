"""
yolo_infer.py — YOLO inference engine.

Responsibilities:
    - Load a pretrained YOLO model (e.g. yolov8n.pt) once (singleton/lazy load).
    - run_inference(image_path) -> list[Detection]
        Run inference on a single image. Returns Detection objects with
        bbox as [x1, y1, x2, y2] in absolute pixel coordinates.
    - run_inference_batch(images: list[ImageInput]) -> dict[str, list[Detection]]
        Run inference on a batch. Only processes images with status=="ok".
        Skipped images are logged, not silently dropped.

Owner: Member A
"""
