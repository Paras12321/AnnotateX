"""
ui.py — Gradio dashboard application for AnnotateX.

Responsibilities:
    - Premium dark-theme Gradio Blocks app with fixed sidebar navigation.
    - Page-based layout: Dashboard, Upload, Results, Quality Engine, Exports, Settings, About.
    - Connects to run_pipeline() from pipeline/orchestrator.py.
    - Displays 100% REAL data from BatchResult.
    - Preserves backward-compatible process_images (4 outputs) and _format_dashboard for tests.

Owner: Member D
"""

import os
import sys
import logging
from pathlib import Path

import gradio as gr

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.orchestrator import run_pipeline, annotate_image, get_config
from models.contracts import BatchResult
from app.styles import CUSTOM_CSS
from app.components import (
    SIDEBAR_JS,
    render_sidebar,
    render_header,
    render_metric_cards,
    create_donut_chart,
    create_confidence_distribution_chart,
    render_performance_card,
    render_pipeline_philosophy,
    render_pipeline_tracker,
    render_quality_rules_cards,
    render_rule_performance_table,
    render_quality_breakdown_table,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (backward-compatible, used by tests)
# ---------------------------------------------------------------------------

def _format_dashboard(batch: BatchResult) -> str:
    """Format aggregate dashboard metrics as markdown (test-facing)."""
    if batch.total_images == 0:
        return "*No data yet.*"

    accept_rate = (
        (batch.total_accepted / batch.total_detections * 100)
        if batch.total_detections > 0 else 0
    )
    flag_rate = (
        (batch.total_flagged / batch.total_detections * 100)
        if batch.total_detections > 0 else 0
    )

    total_conf = 0.0
    for pr in batch.results:
        for det in pr.detections:
            total_conf += det.conf
    avg_conf = (total_conf / batch.total_detections) if batch.total_detections > 0 else 0.0

    total_time = sum(pr.processing_time_ms for pr in batch.results)
    avg_time = (total_time / len(batch.results)) if batch.results else 0.0

    lines = [
        "## 📊 Dashboard",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| 🖼️ Images processed | {batch.total_images} |",
        f"| 🎯 Total detections | {batch.total_detections} |",
        f"| ✅ Accepted | {batch.total_accepted} |",
        f"| ⚠️ Flagged | {batch.total_flagged} |",
        f"| ❌ Rejected | {batch.total_rejected} |",
        f"| 📈 Acceptance % | {accept_rate:.1f}% |",
        f"| 🚩 Flag rate % | {flag_rate:.1f}% |",
        f"| 🧠 Average confidence | {avg_conf:.2f} |",
        f"| ⏱️ Average processing time | {avg_time:.1f} ms |",
    ]

    if batch.export_paths:
        lines.append("")
        lines.append("### Export Files Ready")
        for fmt, path in batch.export_paths.items():
            lines.append(f"- **{fmt}**: `{path}`")
    else:
        lines.append("")
        lines.append("*No annotations accepted — nothing to export.*")

    return "\n".join(lines)


def _collect_export_files(batch: BatchResult) -> list[str]:
    """Collect all downloadable export file paths from BatchResult."""
    files: list[str] = []
    for key, path in batch.export_paths.items():
        if key == "yolo_dir":
            yolo_dir = Path(path)
            if yolo_dir.is_dir():
                for txt_file in sorted(yolo_dir.glob("*.txt")):
                    files.append(str(txt_file))
        elif key == "coco_json":
            if os.path.isfile(path):
                files.append(path)
    return files


# ---------------------------------------------------------------------------
# Processing Callback  (4 outputs for test compatibility)
# ---------------------------------------------------------------------------

def process_images(files) -> tuple[str, str, list | None, list[str] | None]:
    """Execute the full AnnotateX pipeline on uploaded images.

    Returns exactly 4 outputs to satisfy test_ui_integration.py.
    """
    if files is None or len(files) == 0:
        msg = (
            "⚠️ **No files uploaded.** "
            "Please upload one or more images before clicking Process."
        )
        return msg, "*No data yet.*", None, None

    file_paths = [f.name if hasattr(f, "name") else str(f) for f in files]
    batch = run_pipeline(file_paths)

    # Per-image markdown
    result_lines: list[str] = []
    for pr in batch.results:
        result_lines.append(f"### 🖼️ `{pr.image_id}`")
        result_lines.append(f"- **Detections:** {len(pr.detections)}")
        result_lines.append(f"- ✅ Accepted: {len(pr.accepted)}")
        result_lines.append(f"- ⚠️ Flagged: {len(pr.flagged)}")
        result_lines.append(f"- ❌ Rejected: {len(pr.rejected)}")
        result_lines.append(f"- ⏱️ Processing time: {pr.processing_time_ms:.1f} ms")
        result_lines.append("")
        for qr in pr.quality_results:
            det = qr.detection
            emoji = {"ACCEPT": "✅", "FLAG": "⚠️", "REJECT": "❌"}.get(qr.decision, "❓")
            result_lines.append(
                f"  {emoji} **{det.class_name}** (conf={det.conf:.2f}) → {qr.decision}: {qr.reason}"
            )
        result_lines.append("")
        result_lines.append("---")
        result_lines.append("")
    results_md = "\n".join(result_lines) if result_lines else "*No results.*"

    dashboard_md = _format_dashboard(batch)

    # Gallery
    gallery_images = []
    id_to_path = {Path(fp).stem: fp for fp in file_paths}
    for pr in batch.results:
        fp = id_to_path.get(pr.image_id)
        if fp and pr.detections:
            annotated = annotate_image(fp, pr.detections, pr.quality_results)
            if annotated is not None:
                gallery_images.append((fp, f"{pr.image_id} — Original"))
                caption = f"{pr.image_id} — {len(pr.accepted)}✅ {len(pr.flagged)}⚠️ {len(pr.rejected)}❌"
                gallery_images.append((annotated, caption))

    download_files = _collect_export_files(batch)

    return (
        results_md,
        dashboard_md,
        gallery_images if gallery_images else None,
        download_files if download_files else None,
    )


# ---------------------------------------------------------------------------
# Extended callback for new dashboard components (updates everything)
# ---------------------------------------------------------------------------

def _process_extended(files):
    """Run pipeline and return updates for ALL new dashboard components.

    Returns 10 values:
        status_msg, metrics_html, donut_fig, hist_fig, perf_html,
        tracker_html, gallery_items, rule_perf_html, breakdown_html,
        export_files
    """
    if files is None or len(files) == 0:
        warning = (
            "⚠️ **No files uploaded.** "
            "Please upload one or more images before clicking Process."
        )
        return (
            warning,
            render_metric_cards(None),
            create_donut_chart(None),
            create_confidence_distribution_chart(None),
            render_performance_card(None),
            render_pipeline_tracker(None),
            [],
            render_rule_performance_table(None),
            render_quality_breakdown_table(None),
            None,
        )

    file_paths = [f.name if hasattr(f, "name") else str(f) for f in files]
    batch = run_pipeline(file_paths)

    # Gallery
    gallery_items = []
    path_map = {Path(fp).stem: fp for fp in file_paths}
    for pr in batch.results:
        fp = path_map.get(pr.image_id)
        if fp and os.path.isfile(fp):
            gallery_items.append((fp, f"Original: {pr.image_id}"))
            if pr.detections:
                annotated = annotate_image(fp, pr.detections, pr.quality_results)
                if annotated is not None:
                    gallery_items.append((
                        annotated,
                        f"Annotated: {len(pr.accepted)} Acc • {len(pr.flagged)} Flg • {len(pr.rejected)} Rej"
                    ))
            else:
                gallery_items.append((fp, f"{pr.image_id} (No Objects)"))

    export_files = _collect_export_files(batch)
    status = (
        f"✅ **Batch Processed Successfully**: {batch.total_images} images, "
        f"{batch.total_detections} detections."
    )

    return (
        status,
        render_metric_cards(batch),
        create_donut_chart(batch),
        create_confidence_distribution_chart(batch),
        render_performance_card(batch),
        render_pipeline_tracker(batch),
        gallery_items,
        render_rule_performance_table(batch),
        render_quality_breakdown_table(batch),
        export_files if export_files else None,
    )


# ---------------------------------------------------------------------------
# Gradio Layout
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    """Build and return the Gradio Blocks application."""
    config = get_config()

    with gr.Blocks(
        title="AnnotateX — Data Annotation Pipeline",
    ) as app:

        # ---- App Shell ----
        with gr.Row(elem_id="app-shell"):

            # ---- Sidebar ----
            with gr.Column(elem_id="sidebar-col"):
                gr.HTML(render_sidebar())

            # ---- Content ----
            with gr.Column(elem_id="content-col"):

                # =========================================================
                # PAGE: Dashboard
                # =========================================================
                with gr.Column(elem_id="page-dashboard", elem_classes="dashboard-page"):
                    gr.HTML(render_header("Pipeline Overview", "Real-time metrics and quality evaluation", show_upload_btn=True))
                    metric_cards_out = gr.HTML(render_metric_cards(None))

                    with gr.Row(elem_classes="gap compact"):
                        with gr.Column(scale=1):
                            donut_out = gr.Plot(value=create_donut_chart(None), label="Detection Distribution")
                        with gr.Column(scale=1):
                            hist_out = gr.Plot(value=create_confidence_distribution_chart(None), label="Confidence")

                    with gr.Row(elem_classes="gap compact"):
                        with gr.Column(scale=1):
                            perf_out = gr.HTML(render_performance_card(None))
                        with gr.Column(scale=1):
                            gr.HTML(render_pipeline_philosophy())

                # =========================================================
                # PAGE: Upload & Process
                # =========================================================
                with gr.Column(elem_id="page-upload", elem_classes="dashboard-page"):
                    gr.HTML(render_header("Upload & Process", "Batch ingestion and pipeline execution"))
                    with gr.Row():
                        with gr.Column(scale=3):
                            file_input = gr.File(
                                label="Drag & Drop Images",
                                file_count="multiple",
                                file_types=["image"],
                                type="filepath",
                                elem_classes="upload-zone",
                            )
                            with gr.Row():
                                process_btn = gr.Button("🚀 Process Batch", variant="primary", size="lg", elem_classes="primary-btn")
                                clear_btn = gr.ClearButton(components=[file_input], value="🔄 Clear", size="lg")
                            status_out = gr.Markdown(value="*Ready for upload. Select images and click Process.*")
                        with gr.Column(scale=2):
                            tracker_out = gr.HTML(render_pipeline_tracker(None))

                # =========================================================
                # PAGE: Results
                # =========================================================
                with gr.Column(elem_id="page-results", elem_classes="dashboard-page"):
                    gr.HTML(render_header("Inspection Results", "Side-by-side original vs annotated preview"))
                    gr.HTML("""
<div class="color-legend">
    <div class="legend-item"><div class="legend-dot" style="background:#10b981;"></div> ACCEPT</div>
    <div class="legend-item"><div class="legend-dot" style="background:#f59e0b;"></div> FLAG</div>
    <div class="legend-item"><div class="legend-dot" style="background:#ef4444;"></div> REJECT</div>
</div>""")
                    gallery_out = gr.Gallery(
                        label="Original & Annotated Pairs",
                        columns=2, height="auto", object_fit="contain", preview=True,
                    )

                # =========================================================
                # PAGE: Quality Engine
                # =========================================================
                with gr.Column(elem_id="page-quality", elem_classes="dashboard-page"):
                    gr.HTML(render_header("Quality Engine", "Deterministic rules and auditable execution log"))
                    gr.HTML(render_quality_rules_cards(config))
                    rule_perf_out = gr.HTML(render_rule_performance_table(None))
                    breakdown_out = gr.HTML(render_quality_breakdown_table(None))

                # =========================================================
                # PAGE: Exports
                # =========================================================
                with gr.Column(elem_id="page-exports", elem_classes="dashboard-page"):
                    gr.HTML(render_header("Dataset Exports", "Download pre-validated YOLO and COCO datasets"))
                    gr.HTML("""
<div class="export-grid">
    <div class="export-card">
        <div class="export-card-icon" style="background:rgba(139,92,246,0.15);color:#a78bfa;">📄</div>
        <div class="export-card-title">YOLO Format (.txt)</div>
        <div class="export-card-desc">One .txt per image with normalized coords. Only ACCEPTED boxes.</div>
        <div class="export-card-status export-available">Ready to Export</div>
    </div>
    <div class="export-card">
        <div class="export-card-icon" style="background:rgba(16,185,129,0.15);color:#34d399;">🗂️</div>
        <div class="export-card-title">COCO Dataset (.json)</div>
        <div class="export-card-desc">Standard JSON schema with absolute pixel coords.</div>
        <div class="export-card-status export-available">Ready to Export</div>
    </div>
</div>""")
                    download_out = gr.File(
                        label="Download Exported Artifacts",
                        file_count="multiple", interactive=False,
                        elem_classes="file-component",
                    )

                # =========================================================
                # PAGE: Settings
                # =========================================================
                with gr.Column(elem_id="page-settings", elem_classes="dashboard-page"):
                    gr.HTML(render_header("System Settings", "Configuration parameters (read-only)"))
                    gr.HTML(f"""
<div class="settings-grid">
    <div class="setting-item"><div class="setting-label">Inference Engine</div><div class="setting-value">YOLOv8 Nano</div></div>
    <div class="setting-item"><div class="setting-label">Confidence Threshold</div><div class="setting-value">{config.get("conf_threshold", 0.5)}</div></div>
    <div class="setting-item"><div class="setting-label">Min Area</div><div class="setting-value">{config.get("min_area_px", 400)} px²</div></div>
    <div class="setting-item"><div class="setting-label">IoU Limit</div><div class="setting-value">{config.get("iou_threshold", 0.9)}</div></div>
    <div class="setting-item"><div class="setting-label">Output Dir</div><div class="setting-value">./{config.get("output_dir", "outputs")}</div></div>
</div>""")

                # =========================================================
                # PAGE: About
                # =========================================================
                with gr.Column(elem_id="page-about", elem_classes="dashboard-page"):
                    gr.HTML(render_header("About AnnotateX", "Pipeline architecture details"))
                    gr.HTML("""
<div class="dashboard-panel about-content">
    <h3>🔄 Processing Sequence</h3>
    <code>IMAGE → PREPROCESS → YOLO → QUALITY ENGINE → ROUTING → VALIDATE → EXPORT</code>
    <br><br>
    <ol>
        <li><strong>Preprocessing:</strong> Format validation, corrupt file quarantine, Lanczos resizing.</li>
        <li><strong>Inference:</strong> YOLOv8 object detection generating candidate boxes.</li>
        <li><strong>Quality Engine:</strong> 4 deterministic rules (Confidence, Validity, Size, Overlap).</li>
        <li><strong>Routing:</strong> ACCEPT / FLAG / REJECT decision triage.</li>
        <li><strong>Validation:</strong> Schema safety checks for coordinates and class IDs.</li>
        <li><strong>Export:</strong> YOLO .txt and COCO .json generation.</li>
    </ol>
</div>""")

        # ---- Wire Process Button ----
        process_btn.click(
            fn=_process_extended,
            inputs=[file_input],
            outputs=[
                status_out,
                metric_cards_out,
                donut_out,
                hist_out,
                perf_out,
                tracker_out,
                gallery_out,
                rule_perf_out,
                breakdown_out,
                download_out,
            ],
        )

    return app


# ---------------------------------------------------------------------------
# Module-level app instance + direct launch support
# ---------------------------------------------------------------------------

app = build_app()

if __name__ == "__main__":
    theme = gr.themes.Base(
        primary_hue="purple",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    )
    app.launch(theme=theme, css=CUSTOM_CSS, head=SIDEBAR_JS)
