"""
coco_writer.py — COCO format exporter.

Responsibilities:
    - export_coco(annotations, image_dims, class_names, output_path) -> str
        Writes a single COCO-format JSON file with top-level keys:
            info, images, annotations, categories
        COCO bbox format: [x_min, y_min, width, height] in PIXEL coordinates
        (NOT normalized — different convention from YOLO, handled explicitly).

Owner: Member C
"""
