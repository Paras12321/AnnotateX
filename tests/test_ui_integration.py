"""
test_ui_integration.py — Smoke tests for the Gradio UI and orchestrator stub.

Test cases:
    - app/ui.py builds a Gradio Blocks object without raising.
    - run_pipeline stub returns a well-formed dummy BatchResult.
    - run_pipeline on empty list returns zeroed BatchResult.
    - process_images with no files returns a warning message.

Owner: Member D
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.contracts import (
    BatchResult,
    Detection,
    ProcessingResult,
    QualityResult,
)
from pipeline.orchestrator import run_pipeline


# ---------------------------------------------------------------------------
# Orchestrator stub tests
# ---------------------------------------------------------------------------


class TestRunPipelineStub:
    """Tests for the Day 1 run_pipeline stub."""

    def test_returns_batch_result(self):
        """run_pipeline returns a BatchResult instance."""
        result = run_pipeline(["fake/image1.jpg", "fake/image2.png"])
        assert isinstance(result, BatchResult)

    def test_batch_result_shape(self):
        """BatchResult has correct totals for dummy data."""
        result = run_pipeline(["a.jpg", "b.jpg"])

        assert result.total_images == 2
        assert result.total_detections > 0
        assert result.total_accepted > 0
        assert result.total_flagged > 0
        assert result.total_rejected > 0
        assert isinstance(result.export_paths, dict)

    def test_per_image_processing_results(self):
        """Each image produces a ProcessingResult with correct fields."""
        result = run_pipeline(["test.jpg"])

        assert len(result.results) == 1
        pr = result.results[0]
        assert isinstance(pr, ProcessingResult)
        assert pr.image_id == "test"
        assert isinstance(pr.detections, list)
        assert isinstance(pr.quality_results, list)
        assert isinstance(pr.accepted, list)
        assert isinstance(pr.flagged, list)
        assert isinstance(pr.rejected, list)
        assert isinstance(pr.processing_time_ms, float)
        assert pr.processing_time_ms >= 0

    def test_detections_are_detection_objects(self):
        """Detections list contains Detection dataclass instances."""
        result = run_pipeline(["img.jpg"])
        pr = result.results[0]

        for det in pr.detections:
            assert isinstance(det, Detection)
            assert isinstance(det.image_id, str)
            assert isinstance(det.bbox, list)
            assert len(det.bbox) == 4
            assert isinstance(det.class_id, int)
            assert isinstance(det.class_name, str)
            assert isinstance(det.conf, float)
            assert 0.0 <= det.conf <= 1.0

    def test_quality_results_are_quality_result_objects(self):
        """Quality results contain QualityResult dataclass instances."""
        result = run_pipeline(["img.jpg"])
        pr = result.results[0]

        for qr in pr.quality_results:
            assert isinstance(qr, QualityResult)
            assert isinstance(qr.detection, Detection)
            assert isinstance(qr.passed_rules, list)
            assert isinstance(qr.failed_rules, list)
            assert qr.decision in ("ACCEPT", "FLAG", "REJECT")
            assert isinstance(qr.reason, str)
            assert len(qr.reason) > 0

    def test_empty_input_returns_zeroed_batch(self):
        """Empty file list returns a BatchResult with all zeros."""
        result = run_pipeline([])

        assert isinstance(result, BatchResult)
        assert result.total_images == 0
        assert result.total_detections == 0
        assert result.total_accepted == 0
        assert result.total_flagged == 0
        assert result.total_rejected == 0
        assert result.results == []

    def test_image_id_is_filename_stem(self):
        """image_id should match the filename stem, not the full path."""
        result = run_pipeline(["/some/path/to/photo.jpg"])
        pr = result.results[0]
        assert pr.image_id == "photo"
        for det in pr.detections:
            assert det.image_id == "photo"


# ---------------------------------------------------------------------------
# Gradio app build test
# ---------------------------------------------------------------------------


class TestGradioApp:
    """Smoke tests for the Gradio UI."""

    def test_build_app_returns_blocks(self):
        """build_app() should return a gr.Blocks instance without error."""
        import gradio as gr
        from app.ui import build_app

        app = build_app()
        assert isinstance(app, gr.Blocks)

    def test_process_images_no_files(self):
        """process_images with None shows a warning, no crash."""
        from app.ui import process_images

        results_md, dashboard_md = process_images(None)
        assert "No files uploaded" in results_md
        assert isinstance(dashboard_md, str)

    def test_process_images_empty_list(self):
        """process_images with empty list shows a warning, no crash."""
        from app.ui import process_images

        results_md, dashboard_md = process_images([])
        assert "No files uploaded" in results_md
        assert isinstance(dashboard_md, str)
