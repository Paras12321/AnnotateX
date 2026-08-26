"""
ui.py — Gradio application entry point.

Responsibilities:
    - Build the Gradio Blocks layout (upload, process button, results, dashboard, downloads).
    - Handle user interactions (file upload, "Process" click).
    - Call pipeline/orchestrator.run_pipeline() and render the returned BatchResult.
    - Display per-image annotated previews (original vs. boxes drawn) and aggregate dashboard metrics.
    - Provide "Download YOLO" and "Download COCO" buttons linked to exported files.

Owner: Member D
"""
