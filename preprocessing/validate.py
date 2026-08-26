"""
validate.py — Image format and integrity validation.

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
"""
