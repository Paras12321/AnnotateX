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

import json
import os
from collections import defaultdict
from datetime import datetime

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.contracts import Detection


def export_coco(
    annotations: list[Detection],
    image_dims: dict[str, tuple[int, int]],
    class_names: dict[int, str],
    output_path: str,
) -> str:
    """Export annotations to a single COCO-format JSON file.

    COCO bbox convention: [x_min, y_min, width, height] in **pixel**
    coordinates (not normalized).

    ⚠️  FLAG FOR MEMBER D (dashboard/UI):
        COCO bbox values are in pixel space.  If the dashboard renders
        overlay boxes from COCO JSON, ensure the coordinate system matches
        the displayed image resolution.

    Args:
        annotations: List of Detection objects to export.
        image_dims: Mapping of image_id -> (width, height) in pixels.
        class_names: Mapping of class_id -> human-readable class name.
        output_path: File path for the output JSON (parent dir created
            if missing).

    Returns:
        The *output_path* that was written.

    Raises:
        ValueError: If an annotation references an image_id not present
            in *image_dims*.
    """
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # -- Validate all image_ids up-front before writing anything ----------
    for det in annotations:
        if det.image_id not in image_dims:
            raise ValueError(
                f"Missing image_dims entry for image_id '{det.image_id}'. "
                f"Cannot build COCO image record without dimensions."
            )

    # -- info section -----------------------------------------------------
    coco: dict = {
        "info": {
            "description": "AnnotateX export",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat(),
        },
        "images": [],
        "annotations": [],
        "categories": [],
    }

    # -- images section ---------------------------------------------------
    # Collect unique image_ids referenced by image_dims
    image_ids_seen: set[str] = set(image_dims.keys())
    image_id_to_int: dict[str, int] = {}

    for idx, image_id in enumerate(sorted(image_ids_seen), start=1):
        img_w, img_h = image_dims[image_id]
        image_id_to_int[image_id] = idx
        coco["images"].append({
            "id": idx,
            "file_name": f"{image_id}.jpg",
            "width": img_w,
            "height": img_h,
        })

    # -- categories section -----------------------------------------------
    for class_id in sorted(class_names.keys()):
        coco["categories"].append({
            "id": class_id,
            "name": class_names[class_id],
        })

    # -- annotations section ----------------------------------------------
    for ann_idx, det in enumerate(annotations, start=1):
        x1, y1, x2, y2 = det.bbox

        # COCO bbox: [x_min, y_min, width, height] in PIXELS
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        coco["annotations"].append({
            "id": ann_idx,
            "image_id": image_id_to_int[det.image_id],
            "category_id": det.class_id,
            "bbox": [x1, y1, bbox_w, bbox_h],
            "area": bbox_w * bbox_h,
            "iscrowd": 0,
            "score": det.conf,
        })

    # -- write ------------------------------------------------------------
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    return output_path
