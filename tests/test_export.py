"""
test_export.py — Tests for export (export/yolo_writer.py, coco_writer.py) and validation.

Test cases to implement:
    - export_yolo produces correctly formatted normalized values for a known bbox/image size.
    - export_coco produces valid JSON with correct top-level keys and pixel-format bbox.
    - Missing image_dims entry raises ValueError.
    - export_pipeline excludes invalid annotations while exporting valid ones.
    - Zero-annotation image still gets an (empty) .txt file.
    - Fully empty batch produces valid empty outputs, no crash.
    - Structural verification: YOLO lines have 5 numeric fields; COCO JSON is well-formed.

Owner: Member C
"""
