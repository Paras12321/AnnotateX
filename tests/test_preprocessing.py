"""
test_preprocessing.py — Tests for preprocessing (preprocessing/validate.py, resize.py).

Test cases to implement:
    - Valid image -> ImageInput with status "ok" and correct width/height.
    - Corrupt file -> ImageInput with status "corrupt".
    - Unsupported file extension -> ImageInput with status "unsupported_format".
    - preprocess_batch with empty list returns [], no error.
    - preprocess_batch never raises on bad input.

Owner: Member A
"""
