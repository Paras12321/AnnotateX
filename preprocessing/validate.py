"""
validate.py — Image format and integrity validation (stub).

Responsibilities:
    - validate_and_load(file_path) -> ImageInput
        Check file format (jpg/jpeg/png/bmp), attempt to load.
        Return ImageInput with status="ok" on success, or
        status="corrupt" / "unsupported_format" on failure.
        Never raises on bad input — status field communicates the problem.
    - preprocess_batch(file_paths) -> list[ImageInput]
        Run validate_and_load on each path. Returns the full list
        including corrupt/unsupported entries (never drops silently).

Owner: Member A
Status: STUB — full implementation coming Day 2.
"""

from pathlib import Path

from models.contracts import ImageInput

# Supported image extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def validate_and_load(file_path: str) -> ImageInput:
    """Validate an image file and return an ImageInput.

    Stub implementation — checks extension and file existence only.
    Full format/corruption validation coming Day 2.
    """
    path = Path(file_path)
    image_id = path.stem

    if not path.is_file():
        return ImageInput(
            image_id=image_id,
            file_path=str(path),
            status="corrupt",
        )

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return ImageInput(
            image_id=image_id,
            file_path=str(path),
            status="unsupported_format",
        )

    # Stub: assume ok if file exists and extension is supported.
    # Day 2 will add Pillow-based integrity check + width/height.
    return ImageInput(
        image_id=image_id,
        file_path=str(path),
        status="ok",
    )


def preprocess_batch(file_paths: list[str]) -> list[ImageInput]:
    """Run validate_and_load on each path. Never drops entries."""
    return [validate_and_load(fp) for fp in file_paths]
