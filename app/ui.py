"""
ui.py — Gradio application entry point.

Responsibilities:
    - Build the Gradio Blocks layout (upload, process button, results,
      annotated image gallery, dashboard, downloads).
    - Handle user interactions (file upload, "Process" click).
    - Call pipeline/orchestrator.run_pipeline() and render the returned
      BatchResult.
    - Display per-image results, annotated bounding-box previews,
      and aggregate dashboard metrics.
    - Provide "Download YOLO" and "Download COCO" buttons wired to
      real export files.

Owner: Member D
"""

import sys
import os
import logging
from pathlib import Path

import gradio as gr

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.orchestrator import run_pipeline, annotate_image, get_config
from models.contracts import BatchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_per_image_results(batch: BatchResult) -> str:
    """Format per-image results as a readable markdown string."""
    if not batch.results:
        return "*No results.*"

    lines: list[str] = []
    for pr in batch.results:
        lines.append(f"### 🖼️ `{pr.image_id}`")
        lines.append(f"- **Detections:** {len(pr.detections)}")
        lines.append(f"- ✅ Accepted: {len(pr.accepted)}")
        lines.append(f"- ⚠️ Flagged: {len(pr.flagged)}")
        lines.append(f"- ❌ Rejected: {len(pr.rejected)}")
        lines.append(f"- ⏱️ Processing time: {pr.processing_time_ms:.1f} ms")
        lines.append("")

        # Detail each detection's quality decision
        for qr in pr.quality_results:
            det = qr.detection
            emoji = {"ACCEPT": "✅", "FLAG": "⚠️", "REJECT": "❌"}.get(
                qr.decision, "❓"
            )
            lines.append(
                f"  {emoji} **{det.class_name}** (conf={det.conf:.2f}) "
                f"→ {qr.decision}: {qr.reason}"
            )
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def _format_dashboard(batch: BatchResult) -> str:
    """Format aggregate dashboard metrics as markdown."""
    if batch.total_images == 0:
        return "*No data yet.*"

    accept_rate = (
        (batch.total_accepted / batch.total_detections * 100)
        if batch.total_detections > 0
        else 0
    )

    lines = [
        "## 📊 Dashboard",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Images | {batch.total_images} |",
        f"| Total Detections | {batch.total_detections} |",
        f"| ✅ Accepted | {batch.total_accepted} |",
        f"| ⚠️ Flagged | {batch.total_flagged} |",
        f"| ❌ Rejected | {batch.total_rejected} |",
        f"| Acceptance Rate | {accept_rate:.1f}% |",
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
            # Collect all .txt files from the YOLO output directory
            yolo_dir = Path(path)
            if yolo_dir.is_dir():
                for txt_file in sorted(yolo_dir.glob("*.txt")):
                    files.append(str(txt_file))
        elif key == "coco_json":
            if os.path.isfile(path):
                files.append(path)
    return files


# ---------------------------------------------------------------------------
# Processing callback
# ---------------------------------------------------------------------------

def process_images(files) -> tuple[str, str, list | None, list[str] | None]:
    """Callback for the Process button.

    Args:
        files: List of file paths from gr.File (type="filepath"), or None.

    Returns:
        Tuple of (per_image_results_md, dashboard_md, gallery_images,
                  download_files).
    """
    if files is None or len(files) == 0:
        msg = (
            "⚠️ **No files uploaded.** "
            "Please upload one or more images before clicking Process."
        )
        return msg, "*No data yet.*", None, None

    # Gradio File objects expose a .name attribute with the temp path,
    # or may be plain strings when type="filepath"
    file_paths = [f.name if hasattr(f, "name") else str(f) for f in files]

    batch = run_pipeline(file_paths)

    results_md = _format_per_image_results(batch)
    dashboard_md = _format_dashboard(batch)

    # --- Build annotated image gallery ---
    gallery_images = []
    # Build a mapping of image_id -> file_path for lookup
    image_id_to_path: dict[str, str] = {}
    for fp in file_paths:
        image_id_to_path[Path(fp).stem] = fp

    for pr in batch.results:
        fp = image_id_to_path.get(pr.image_id)
        if fp and pr.detections:
            annotated = annotate_image(fp, pr.detections, pr.quality_results)
            if annotated is not None:
                caption = (
                    f"{pr.image_id}: "
                    f"{len(pr.accepted)}✅ {len(pr.flagged)}⚠️ {len(pr.rejected)}❌"
                )
                gallery_images.append((annotated, caption))

    # --- Collect export files for download ---
    download_files = _collect_export_files(batch)

    return (
        results_md,
        dashboard_md,
        gallery_images if gallery_images else None,
        download_files if download_files else None,
    )


# ---------------------------------------------------------------------------
# Gradio Blocks layout
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    """Build and return the Gradio Blocks application (does not launch)."""

    with gr.Blocks(
        title="AnnotateX — Intelligent Annotation Pipeline",
    ) as app:

        # --- Header ---
        gr.Markdown(
            "# 🏷️ AnnotateX\n"
            "**Intelligent Automatic Data Annotation Pipeline**\n\n"
            "Upload images → Process → Review quality-filtered "
            "annotations → Download clean datasets."
        )

        # --- Upload + Process ---
        with gr.Row():
            with gr.Column(scale=3):
                file_input = gr.File(
                    label="Upload Images",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                )
            with gr.Column(scale=1, min_width=160):
                process_btn = gr.Button(
                    "🚀 Process", variant="primary", size="lg"
                )

        # --- Results area ---
        gr.Markdown("## Results")

        with gr.Row():
            with gr.Column(scale=2):
                results_output = gr.Markdown(
                    value="*Upload images and click Process to see results.*",
                    label="Per-Image Results",
                )
            with gr.Column(scale=1):
                dashboard_output = gr.Markdown(
                    value="*No data yet.*",
                    label="Dashboard",
                )

        # --- Annotated image preview gallery ---
        gr.Markdown("## 🔍 Annotated Preview")
        gr.Markdown(
            "*Bounding boxes color-coded: "
            "🟢 green = accepted, 🟠 orange = flagged, 🔴 red = rejected*"
        )

        gallery_output = gr.Gallery(
            label="Annotated Images",
            columns=2,
            height="auto",
            object_fit="contain",
        )

        # --- Download section ---
        gr.Markdown("## 📥 Export Downloads")

        download_output = gr.File(
            label="Download Exported Annotations (YOLO .txt + COCO .json)",
            file_count="multiple",
            interactive=False,
        )

        # --- Wire up the Process button ---
        process_btn.click(
            fn=process_images,
            inputs=[file_input],
            outputs=[results_output, dashboard_output, gallery_output, download_output],
        )

    return app


# ---------------------------------------------------------------------------
# Module-level app instance + direct launch support
# ---------------------------------------------------------------------------

app = build_app()

if __name__ == "__main__":
    app.launch()
