"""
test_ui_integration.py — Integration tests for the pipeline + Gradio UI.

Test cases:
    - Gradio app builds without raising.
    - process_images with no files returns a warning message.
    - run_pipeline on real sample_data produces valid BatchResult.
    - Corrupt image mixed in doesn't kill the batch.
    - annotate_image produces annotated array for real image.

Owner: Member D
"""

import os
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
from pipeline.orchestrator import run_pipeline, annotate_image

# Resolve sample data paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DOG = str(PROJECT_ROOT / "sample_data" / "dog.jpg")
SAMPLE_CORRUPT = str(PROJECT_ROOT / "sample_data" / "corrupt.jpg")
SAMPLE_TXT = str(PROJECT_ROOT / "sample_data" / "test.txt")


# ---------------------------------------------------------------------------
# Real pipeline integration tests
# ---------------------------------------------------------------------------


class TestRunPipelineReal:
    """Integration tests for the real Day 2 pipeline."""

    def test_real_pipeline_valid_image(self):
        """run_pipeline on a real image produces a BatchResult with detections
        and non-empty export_paths."""
        assert os.path.isfile(SAMPLE_DOG), f"Missing sample: {SAMPLE_DOG}"

        result = run_pipeline([SAMPLE_DOG])

        assert isinstance(result, BatchResult)
        assert result.total_images == 1
        assert result.total_detections > 0
        assert isinstance(result.export_paths, dict)
        # Should have export paths since there are real detections
        if result.total_accepted > 0:
            assert "yolo_dir" in result.export_paths or "coco_json" in result.export_paths

    def test_real_pipeline_contract_shape(self):
        """Each ProcessingResult has correct field types and Detections
        and QualityResults are proper dataclass instances."""
        assert os.path.isfile(SAMPLE_DOG), f"Missing sample: {SAMPLE_DOG}"

        result = run_pipeline([SAMPLE_DOG])
        assert len(result.results) == 1

        pr = result.results[0]
        assert isinstance(pr, ProcessingResult)
        assert pr.image_id == "dog"
        assert isinstance(pr.detections, list)
        assert isinstance(pr.quality_results, list)
        assert isinstance(pr.accepted, list)
        assert isinstance(pr.flagged, list)
        assert isinstance(pr.rejected, list)
        assert isinstance(pr.processing_time_ms, float)
        assert pr.processing_time_ms >= 0

        # Verify detection objects
        for det in pr.detections:
            assert isinstance(det, Detection)
            assert isinstance(det.image_id, str)
            assert isinstance(det.bbox, list)
            assert len(det.bbox) == 4
            assert isinstance(det.class_id, int)
            assert isinstance(det.class_name, str)
            assert isinstance(det.conf, float)
            assert 0.0 <= det.conf <= 1.0

        # Verify quality result objects
        for qr in pr.quality_results:
            assert isinstance(qr, QualityResult)
            assert isinstance(qr.detection, Detection)
            assert isinstance(qr.passed_rules, list)
            assert isinstance(qr.failed_rules, list)
            assert qr.decision in ("ACCEPT", "FLAG", "REJECT")
            assert isinstance(qr.reason, str)
            assert len(qr.reason) > 0

    def test_corrupt_image_doesnt_kill_batch(self):
        """A corrupt image mixed with a valid image doesn't stop the batch.
        The valid image should still produce detections."""
        assert os.path.isfile(SAMPLE_DOG), f"Missing sample: {SAMPLE_DOG}"
        assert os.path.isfile(SAMPLE_CORRUPT), f"Missing sample: {SAMPLE_CORRUPT}"

        result = run_pipeline([SAMPLE_DOG, SAMPLE_CORRUPT])

        assert isinstance(result, BatchResult)
        assert result.total_images == 2
        assert len(result.results) == 2

        # The valid image (dog) should still have detections
        dog_result = None
        for pr in result.results:
            if pr.image_id == "dog":
                dog_result = pr
                break

        assert dog_result is not None
        assert len(dog_result.detections) > 0

        # Corrupt image should have zero detections (gracefully handled)
        corrupt_result = None
        for pr in result.results:
            if pr.image_id == "corrupt":
                corrupt_result = pr
                break

        assert corrupt_result is not None
        assert len(corrupt_result.detections) == 0

    def test_unsupported_format_handled(self):
        """A non-image file mixed in doesn't kill the batch."""
        assert os.path.isfile(SAMPLE_DOG), f"Missing sample: {SAMPLE_DOG}"
        assert os.path.isfile(SAMPLE_TXT), f"Missing sample: {SAMPLE_TXT}"

        result = run_pipeline([SAMPLE_DOG, SAMPLE_TXT])

        assert isinstance(result, BatchResult)
        assert result.total_images == 2
        assert len(result.results) == 2

        # Dog should still work fine
        assert result.total_detections > 0

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
        assert result.export_paths == {}


# ---------------------------------------------------------------------------
# Annotated image tests
# ---------------------------------------------------------------------------


class TestAnnotateImage:
    """Tests for the annotate_image helper."""

    def test_annotate_valid_image(self):
        """annotate_image on a real image with detections returns an RGB array."""
        import numpy as np

        assert os.path.isfile(SAMPLE_DOG), f"Missing sample: {SAMPLE_DOG}"

        # Get real detections first
        result = run_pipeline([SAMPLE_DOG])
        pr = result.results[0]

        if len(pr.detections) == 0:
            pytest.skip("No detections found on sample image")

        annotated = annotate_image(
            SAMPLE_DOG, pr.detections, pr.quality_results
        )

        assert annotated is not None
        assert isinstance(annotated, np.ndarray)
        assert len(annotated.shape) == 3
        assert annotated.shape[2] == 3  # RGB

    def test_annotate_nonexistent_file_returns_none(self):
        """annotate_image on a non-existent file returns None gracefully."""
        dummy_det = Detection(
            image_id="fake", bbox=[0, 0, 10, 10],
            class_id=0, class_name="person", conf=0.9,
        )
        dummy_qr = QualityResult(
            detection=dummy_det,
            passed_rules=["confidence"],
            failed_rules=[],
            decision="ACCEPT",
            reason="ok",
        )

        result = annotate_image("nonexistent.jpg", [dummy_det], [dummy_qr])
        assert result is None


# ---------------------------------------------------------------------------
# Gradio app smoke tests (preserved from Day 1)
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

        outputs = process_images(None)
        # process_images now returns 4 outputs
        assert len(outputs) == 4
        results_md = outputs[0]
        assert "No files uploaded" in results_md

    def test_process_images_empty_list(self):
        """process_images with empty list shows a warning, no crash."""
        from app.ui import process_images

        outputs = process_images([])
        assert len(outputs) == 4
        results_md = outputs[0]
        assert "No files uploaded" in results_md


# ---------------------------------------------------------------------------
# Dashboard metrics tests
# ---------------------------------------------------------------------------


class TestDashboardMetrics:
    """Tests for dashboard metrics calculation."""

    def test_dashboard_metrics_calculation(self):
        """Dashboard metric calculation matches hand-computed values."""
        from app.ui import _format_dashboard
        
        # Create dummy BatchResult
        det1 = Detection(image_id="img1", bbox=[0,0,10,10], class_id=0, class_name="cat", conf=0.8)
        det2 = Detection(image_id="img1", bbox=[0,0,10,10], class_id=0, class_name="cat", conf=0.6)
        det3 = Detection(image_id="img2", bbox=[0,0,10,10], class_id=0, class_name="dog", conf=0.9)
        
        pr1 = ProcessingResult(
            image_id="img1",
            detections=[det1, det2],
            quality_results=[],
            accepted=[det1],
            flagged=[det2],
            rejected=[],
            processing_time_ms=100.0,
        )
        pr2 = ProcessingResult(
            image_id="img2",
            detections=[det3],
            quality_results=[],
            accepted=[det3],
            flagged=[],
            rejected=[],
            processing_time_ms=200.0,
        )
        
        batch = BatchResult(
            results=[pr1, pr2],
            total_images=2,
            total_detections=3,
            total_accepted=2,
            total_flagged=1,
            total_rejected=0,
            export_paths={},
        )
        
        dashboard_md = _format_dashboard(batch)
        
        assert "Images processed | 2" in dashboard_md
        assert "Total detections | 3" in dashboard_md
        assert "Accepted | 2" in dashboard_md
        assert "Flagged | 1" in dashboard_md
        assert "Rejected | 0" in dashboard_md
        
        # Acceptance %: 2 / 3 = 66.666% -> 66.7%
        assert "Acceptance % | 66.7%" in dashboard_md
        # Flag rate %: 1 / 3 = 33.333% -> 33.3%
        assert "Flag rate % | 33.3%" in dashboard_md
        # Avg conf: (0.8 + 0.6 + 0.9) / 3 = 2.3 / 3 = 0.7666 -> 0.77
        assert "Average confidence | 0.77" in dashboard_md
        # Avg processing time: (100 + 200) / 2 = 150 -> 150.0 ms
        assert "Average processing time | 150.0 ms" in dashboard_md
