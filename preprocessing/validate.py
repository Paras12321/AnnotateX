"""
validate.py — Image format and integrity validation.

Responsibilities:
    - validate_and_load(file_path) -> ImageInput
        Check file format (jpg/jpeg/png/bmp), attempt to load with Pillow.
        Return ImageInput with status="ok" + width/height on success, or
        status="corrupt" / "unsupported_format" on failure.
        NEVER raises for bad input — status field communicates the problem.
    - preprocess_batch(file_paths) -> list[ImageInput]
        Run validate_and_load on each path. Returns the full list
        including corrupt/unsupported entries — never drops silently.

Owner: Member A
"""

import logging
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from models.contracts import ImageInput
from preprocessing.resize import resize_if_needed

logger = logging.getLogger(__name__)

# Supported image extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def validate_and_load(file_path: str) -> ImageInput:
    """Validate an image file and return an ImageInput.

    Checks:
        1. File exists
        2. Extension is in SUPPORTED_EXTENSIONS
        3. Pillow can open and verify the file (catches truncated/corrupt)
        4. Populates width, height from the loaded image

    Returns ImageInput with status="ok" on success, or
    "corrupt"/"unsupported_format" on failure. Never raises.
    """
    path = Path(file_path)
    image_id = path.stem

    # --- Check file existence ---
    if not path.is_file():
        logger.warning("File not found: %s", file_path)
        return ImageInput(
            image_id=image_id,
            file_path=str(path),
            status="corrupt",
        )

    # --- Check extension ---
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        logger.info("Unsupported format (%s): %s", path.suffix, file_path)
        return ImageInput(
            image_id=image_id,
            file_path=str(path),
            status="unsupported_format",
        )

    # --- Try to load with Pillow ---
    try:
        with Image.open(path) as img:
            img.verify()  # Checks for truncation/corruption

        # Re-open after verify() (verify closes the file pointer)
        with Image.open(path) as img:
            width, height = img.size

    except Exception as exc:
        logger.warning("Corrupt image (%s): %s", exc, file_path)
        return ImageInput(
            image_id=image_id,
            file_path=str(path),
            status="corrupt",
        )

    # --- Resize if needed (updates file in place, returns new dims) ---
    final_path, width, height = resize_if_needed(str(path), width, height)

    return ImageInput(
        image_id=image_id,
        file_path=final_path,
        width=width,
        height=height,
        status="ok",
    )


def preprocess_batch(file_paths: list[str]) -> list[ImageInput]:
    """Run validate_and_load on each path. Never drops entries.

    Returns the full list including corrupt/unsupported entries so
    downstream stages can log what was skipped and why.
    Never raises — one broken file cannot crash the batch.
    """
    results = []
    for fp in file_paths:
        try:
            result = validate_and_load(fp)
        except Exception:
            # Absolute last resort — validate_and_load should never raise,
            # but if it somehow does, we still don't drop the entry.
            logger.exception("Unexpected error in validate_and_load for %s", fp)
            result = ImageInput(
                image_id=Path(fp).stem if fp else "unknown",
                file_path=str(fp),
                status="corrupt",
            )
        logger.info(
            "Preprocessed %s -> status=%s", result.image_id, result.status
        )
        results.append(result)
    return results
