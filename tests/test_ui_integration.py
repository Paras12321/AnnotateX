"""
test_ui_integration.py — Smoke tests and integration tests for the Gradio UI.

Test cases to implement:
    - app/ui.py builds a Gradio Blocks object without raising.
    - run_pipeline stub (Day 1) / real pipeline (Day 2+) returns a well-formed BatchResult.
    - Full pipeline on sample_data produces a valid BatchResult with non-empty export_paths.
    - One corrupt file in a batch doesn't stop the rest.
    - Dashboard metric calculation matches hand-computed values.

Owner: Member D
"""
