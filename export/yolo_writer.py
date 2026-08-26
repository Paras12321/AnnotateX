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

import os
from collections import defaultdict

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.contracts import Detection


def export_yolo(
    annotations: list[Detection],
    image_dims: dict[str, tuple[int, int]],
    output_dir: str,
) -> dict[str, str]:
    """Export annotations to YOLO .txt format.

    Writes one .txt file per image_id.  Each line contains:
        class_id  x_center  y_center  width  height
    All coordinate values are normalized to [0, 1].

    Args:
        annotations: List of Detection objects to export.
        image_dims: Mapping of image_id -> (width, height) in pixels.
        output_dir: Directory to write .txt files into (created if missing).

    Returns:
        Mapping of image_id -> absolute file path of the written .txt file.

    Raises:
        ValueError: If an annotation references an image_id not present
            in *image_dims*.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Group annotations by image_id
    grouped: dict[str, list[Detection]] = defaultdict(list)
    for det in annotations:
        if det.image_id not in image_dims:
            raise ValueError(
                f"Missing image_dims entry for image_id '{det.image_id}'. "
                f"Cannot compute normalized coordinates without image dimensions."
            )
        grouped[det.image_id].append(det)

    written_files: dict[str, str] = {}

    for image_id, dets in grouped.items():
        img_w, img_h = image_dims[image_id]
        lines: list[str] = []

        for det in dets:
            x1, y1, x2, y2 = det.bbox

            # YOLO normalization formulas
            x_center = ((x1 + x2) / 2) / img_w
            y_center = ((y1 + y2) / 2) / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h

            lines.append(
                f"{det.class_id} {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f}"
            )

        file_path = os.path.join(output_dir, f"{image_id}.txt")
        with open(file_path, "w") as f:
            f.write("\n".join(lines) + "\n" if lines else "")

        written_files[image_id] = file_path

    return written_files
