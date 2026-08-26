"""
yolo_writer.py — YOLO format exporter.

Responsibilities:
    - export_yolo(annotations, image_dims, output_dir) -> dict[str, str]
        Writes one .txt file per image_id. Each line:
            class_id  x_center  y_center  width  height
        All four coordinate values normalized to [0, 1].
        Normalization formulas:
            x_center = ((x1 + x2) / 2) / image_width
            y_center = ((y1 + y2) / 2) / image_height
            width    = (x2 - x1) / image_width
            height   = (y2 - y1) / image_height

Owner: Member C
"""
