"""
components.py — UI Component generators for AnnotateX.

Provides modular builders for:
    - Sidebar Navigation
    - Page Headers
    - Metric Cards
    - Plotly Visualizations (Donut Chart, Confidence Histogram)
    - Performance Panel
    - Pipeline Tracker
    - Quality Rules Cards & Tables
    - Detection Breakdown Table

Owner: Member D
"""

import plotly.graph_objects as go
from models.contracts import BatchResult


# ---------------------------------------------------------------------------
# JavaScript for sidebar page switching (injected via gr.Blocks head= param)
# ---------------------------------------------------------------------------

SIDEBAR_JS = """
<script>
window.switchPage = function(pageId) {
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    var activeNav = document.getElementById('nav-' + pageId);
    if(activeNav) activeNav.classList.add('active');
    var pages = ['page-dashboard','page-upload','page-results','page-quality','page-exports','page-settings','page-about'];
    pages.forEach(function(p) {
        var el = document.getElementById(p);
        if(el) el.style.display = 'none';
    });
    var sel = document.getElementById('page-' + pageId);
    if(sel) sel.style.display = 'block';
}
</script>
"""


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> str:
    """Render the fixed left sidebar with navigation."""
    return """
<div class="sidebar">
    <div class="sidebar-brand">
        <div class="sidebar-logo">⚡</div>
        <div>
            <div class="sidebar-brand-name">AnnotateX</div>
            <div class="sidebar-brand-tagline">QUALITY PIPELINE v1.0</div>
        </div>
    </div>
    <div class="sidebar-nav">
        <div class="nav-section-label">Pipeline</div>
        <div class="nav-item active" id="nav-dashboard" onclick="switchPage('dashboard')">
            <div class="nav-icon">📊</div><div>Dashboard</div>
        </div>
        <div class="nav-item" id="nav-upload" onclick="switchPage('upload')">
            <div class="nav-icon">⬆️</div><div>Upload & Process</div>
        </div>
        <div class="nav-item" id="nav-results" onclick="switchPage('results')">
            <div class="nav-icon">🖼️</div><div>Results</div>
        </div>
        <div class="nav-section-label">Configuration</div>
        <div class="nav-item" id="nav-quality" onclick="switchPage('quality')">
            <div class="nav-icon">⚙️</div><div>Quality Engine</div>
        </div>
        <div class="nav-item" id="nav-exports" onclick="switchPage('exports')">
            <div class="nav-icon">📥</div><div>Exports</div>
        </div>
        <div class="nav-item" id="nav-settings" onclick="switchPage('settings')">
            <div class="nav-icon">🔧</div><div>Settings</div>
        </div>
        <div class="nav-section-label">System</div>
        <div class="nav-item" id="nav-about" onclick="switchPage('about')">
            <div class="nav-icon">ℹ️</div><div>About</div>
        </div>
    </div>
    <div class="sidebar-footer">
        <div class="sidebar-status">
            <div class="status-dot"></div>
            <span>All Systems Operational</span>
        </div>
        <div class="sidebar-version">Engine v1.0.0 &bull; YOLOv8n</div>
    </div>
</div>
"""


# ---------------------------------------------------------------------------
# Page Header
# ---------------------------------------------------------------------------

def render_header(title: str, subtitle: str, show_upload_btn: bool = False) -> str:
    """Render a page header with optional New Upload button."""
    btn = ""
    if show_upload_btn:
        btn = '<button class="btn-new-upload" onclick="switchPage(\'upload\')">+ New Upload</button>'
    return f"""
<div class="page-header">
    <div class="page-header-left">
        <h2>{title}</h2>
        <p>{subtitle}</p>
    </div>
    <div>{btn}</div>
</div>
"""


# ---------------------------------------------------------------------------
# Metric Cards
# ---------------------------------------------------------------------------

def render_metric_cards(batch: BatchResult | None) -> str:
    """Render 5 compact metric cards."""
    if batch is None or batch.total_images == 0:
        return """
<div class="metric-grid">
    <div class="metric-card card-blue">
        <div class="metric-header"><span class="metric-title">Images</span><div class="metric-icon">🖼️</div></div>
        <div class="metric-value">0</div><div class="metric-subtitle">Images processed</div>
    </div>
    <div class="metric-card card-purple">
        <div class="metric-header"><span class="metric-title">Detections</span><div class="metric-icon">🎯</div></div>
        <div class="metric-value">0</div><div class="metric-subtitle">Total predictions</div>
    </div>
    <div class="metric-card card-green">
        <div class="metric-header"><span class="metric-title">Accepted</span><div class="metric-icon">✅</div></div>
        <div class="metric-value">0</div><div class="metric-subtitle">0.0% acceptance</div>
    </div>
    <div class="metric-card card-amber">
        <div class="metric-header"><span class="metric-title">Flagged</span><div class="metric-icon">⚠️</div></div>
        <div class="metric-value">0</div><div class="metric-subtitle">0.0% flagged</div>
    </div>
    <div class="metric-card card-red">
        <div class="metric-header"><span class="metric-title">Rejected</span><div class="metric-icon">❌</div></div>
        <div class="metric-value">0</div><div class="metric-subtitle">0.0% rejected</div>
    </div>
</div>"""

    tot = batch.total_detections
    acc_pct = (batch.total_accepted / tot * 100) if tot > 0 else 0.0
    flg_pct = (batch.total_flagged / tot * 100) if tot > 0 else 0.0
    rej_pct = (batch.total_rejected / tot * 100) if tot > 0 else 0.0

    return f"""
<div class="metric-grid">
    <div class="metric-card card-blue">
        <div class="metric-header"><span class="metric-title">Images</span><div class="metric-icon">🖼️</div></div>
        <div class="metric-value">{batch.total_images}</div><div class="metric-subtitle">Images processed</div>
    </div>
    <div class="metric-card card-purple">
        <div class="metric-header"><span class="metric-title">Detections</span><div class="metric-icon">🎯</div></div>
        <div class="metric-value">{batch.total_detections}</div><div class="metric-subtitle">Total predictions</div>
    </div>
    <div class="metric-card card-green">
        <div class="metric-header"><span class="metric-title">Accepted</span><div class="metric-icon">✅</div></div>
        <div class="metric-value">{batch.total_accepted}</div>
        <div class="metric-subtitle"><span class="badge-accept">{acc_pct:.1f}% acceptance</span></div>
    </div>
    <div class="metric-card card-amber">
        <div class="metric-header"><span class="metric-title">Flagged</span><div class="metric-icon">⚠️</div></div>
        <div class="metric-value">{batch.total_flagged}</div>
        <div class="metric-subtitle"><span class="badge-flag">{flg_pct:.1f}% flagged</span></div>
    </div>
    <div class="metric-card card-red">
        <div class="metric-header"><span class="metric-title">Rejected</span><div class="metric-icon">❌</div></div>
        <div class="metric-value">{batch.total_rejected}</div>
        <div class="metric-subtitle"><span class="badge-reject">{rej_pct:.1f}% rejected</span></div>
    </div>
</div>"""


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def create_donut_chart(batch: BatchResult | None) -> go.Figure:
    """Dark-themed donut chart for detection decision distribution."""
    fig = go.Figure()
    if batch is None or batch.total_detections == 0:
        labels, values, colors = ["No Detections"], [1], ["#22314e"]
        center = "0<br><span style='font-size:12px;color:#94a3b8;'>Detections</span>"
    else:
        labels = ["Accepted", "Flagged", "Rejected"]
        values = [batch.total_accepted, batch.total_flagged, batch.total_rejected]
        colors = ["#10b981", "#f59e0b", "#ef4444"]
        center = f"<b>{batch.total_detections}</b><br><span style='font-size:12px;color:#94a3b8;'>Total</span>"

    fig.add_trace(go.Pie(
        labels=labels, values=values, hole=0.68,
        marker=dict(colors=colors, line=dict(color="#131b2e", width=3)),
        textinfo="percent" if (batch and batch.total_detections > 0) else "none",
        textposition="outside", hoverinfo="label+value+percent", showlegend=True,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"), margin=dict(t=20, b=20, l=20, r=20), height=260,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, font=dict(size=12, color="#94a3b8")),
        annotations=[dict(text=center, x=0.5, y=0.5, font_size=20, font_color="#ffffff", showarrow=False)],
    )
    return fig


def create_confidence_distribution_chart(batch: BatchResult | None) -> go.Figure:
    """Dark-themed confidence distribution bar chart."""
    fig = go.Figure()
    bin_labels = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
    counts = [0, 0, 0, 0, 0]
    if batch and batch.results:
        for pr in batch.results:
            for det in pr.detections:
                c = det.conf if det.conf is not None else 0.0
                if c < 0.2: counts[0] += 1
                elif c < 0.4: counts[1] += 1
                elif c < 0.6: counts[2] += 1
                elif c < 0.8: counts[3] += 1
                else: counts[4] += 1

    fig.add_trace(go.Bar(
        x=bin_labels, y=counts,
        marker=dict(color=["#ef4444", "#f59e0b", "#a78bfa", "#8b5cf6", "#10b981"], line=dict(color="#131b2e", width=1), opacity=0.9),
        text=counts if (batch and batch.total_detections > 0) else None, textposition="auto",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"), margin=dict(t=20, b=20, l=30, r=20), height=260,
        xaxis=dict(title="Confidence Range", gridcolor="#1e2d4a", zeroline=False, color="#94a3b8"),
        yaxis=dict(title="Detections", gridcolor="#1e2d4a", zeroline=False, color="#94a3b8"),
    )
    return fig


# ---------------------------------------------------------------------------
# Performance Card
# ---------------------------------------------------------------------------

def render_performance_card(batch: BatchResult | None) -> str:
    """Render performance panel with trust bar."""
    if batch is None or batch.total_images == 0:
        return """
<div class="dashboard-panel">
    <div class="panel-title">⚡ Pipeline Performance</div>
    <div class="panel-desc">Throughput and latency metrics</div>
    <div class="perf-grid">
        <div class="perf-stat"><div class="perf-stat-label">Avg Latency</div><div class="perf-stat-value">0.0 ms</div></div>
        <div class="perf-stat"><div class="perf-stat-label">Throughput</div><div class="perf-stat-value">0 img/s</div></div>
        <div class="perf-stat"><div class="perf-stat-label">Avg Conf</div><div class="perf-stat-value" style="color:#a78bfa;">0.00</div></div>
        <div class="perf-stat"><div class="perf-stat-label">Yield</div><div class="perf-stat-value" style="color:#10b981;">0.0%</div></div>
    </div>
    <div class="trust-bar-container">
        <div class="trust-bar-header"><span>Dataset Trust Score</span><span style="font-weight:700;color:#10b981;">0.0%</span></div>
        <div class="trust-bar-track"><div class="trust-bar-fill" style="width:0%;"></div></div>
    </div>
</div>"""

    total_time = sum(pr.processing_time_ms for pr in batch.results)
    avg_time = total_time / len(batch.results) if batch.results else 0.0
    throughput = (1000.0 / avg_time) if avg_time > 0 else 0.0
    all_confs = [det.conf for pr in batch.results for det in pr.detections if det.conf is not None]
    avg_conf = (sum(all_confs) / len(all_confs)) if all_confs else 0.0
    acc_rate = (batch.total_accepted / batch.total_detections * 100) if batch.total_detections > 0 else 0.0

    return f"""
<div class="dashboard-panel">
    <div class="panel-title">⚡ Pipeline Performance</div>
    <div class="panel-desc">Throughput and latency metrics</div>
    <div class="perf-grid">
        <div class="perf-stat"><div class="perf-stat-label">Avg Latency</div><div class="perf-stat-value">{avg_time:.1f} ms</div></div>
        <div class="perf-stat"><div class="perf-stat-label">Throughput</div><div class="perf-stat-value">{throughput:.1f} i/s</div></div>
        <div class="perf-stat"><div class="perf-stat-label">Avg Conf</div><div class="perf-stat-value" style="color:#a78bfa;">{avg_conf:.2f}</div></div>
        <div class="perf-stat"><div class="perf-stat-label">Yield</div><div class="perf-stat-value" style="color:#10b981;">{acc_rate:.1f}%</div></div>
    </div>
    <div class="trust-bar-container">
        <div class="trust-bar-header"><span>Dataset Trust Score</span><span style="font-weight:700;color:#10b981;">{acc_rate:.1f}%</span></div>
        <div class="trust-bar-track"><div class="trust-bar-fill" style="width:{acc_rate}%;"></div></div>
    </div>
</div>"""


# ---------------------------------------------------------------------------
# Pipeline Philosophy & Tracker
# ---------------------------------------------------------------------------

def render_pipeline_philosophy() -> str:
    """Render the Quality-First Architecture panel."""
    return """
<div class="dashboard-panel">
    <div class="panel-title">🧠 Quality-First Architecture</div>
    <div class="panel-desc">How AnnotateX ensures dataset integrity</div>
    <div class="pipeline-flow">
        <div class="flow-step"><span class="flow-step-icon">📸</span> Input</div>
        <div class="flow-arrow">&rarr;</div>
        <div class="flow-step"><span class="flow-step-icon">🎯</span> YOLO</div>
        <div class="flow-arrow">&rarr;</div>
        <div class="flow-step" style="background:var(--accent-purple-bg);border-color:var(--accent-purple);color:var(--text-primary);">
            <span class="flow-step-icon">🔬</span> Quality Engine
        </div>
        <div class="flow-arrow">&rarr;</div>
        <div class="flow-step"><span class="flow-step-icon">✅</span> Export</div>
    </div>
    <div style="font-size:12px;color:var(--text-secondary);margin-top:12px;line-height:1.5;">
        Every bounding box must pass 4 deterministic rules (Confidence, Validity, Size, Overlap)
        before being ACCEPTED. This prevents bad data from reaching your training set.
    </div>
</div>"""


def render_pipeline_tracker(batch: BatchResult | None) -> str:
    """Render 6-stage pipeline tracker."""
    done = batch is not None and batch.total_images > 0
    cls = "step-badge-done" if done else "step-badge-pending"
    txt = "Completed" if done else "Pending"
    ico = '✅' if done else '⚪'
    steps = [
        "1. Image Integrity & Format",
        "2. Dynamic Resizing",
        "3. YOLOv8 Detection",
        "4. Quality Engine Routing",
        "5. Structural Validation",
        "6. Export Generation",
    ]
    rows = "\n".join(f"""
    <div class="pipeline-step">
        <div class="step-label"><span>{ico}</span><span>{s}</span></div>
        <span class="{cls}">{txt}</span>
    </div>""" for s in steps)

    return f"""
<div class="dashboard-panel">
    <div class="panel-title">🔄 Processing Status</div>
    <div class="panel-desc">Real-time pipeline progression</div>
    {rows}
</div>"""


# ---------------------------------------------------------------------------
# Quality Engine
# ---------------------------------------------------------------------------

def render_quality_rules_cards(config: dict) -> str:
    """Render quality rule config cards."""
    conf = config.get("conf_threshold", 0.5)
    area = config.get("min_area_px", 400)
    iou = config.get("iou_threshold", 0.9)
    return f"""
<div class="rules-grid">
    <div class="rule-card">
        <div class="rule-card-header"><span class="rule-card-title" style="color:#a78bfa;">R1: Confidence</span><span class="badge-active">Active</span></div>
        <div class="rule-card-desc">Min Detection Confidence</div>
        <div class="rule-card-value">&ge; {conf:.2f}</div>
    </div>
    <div class="rule-card">
        <div class="rule-card-header"><span class="rule-card-title" style="color:#60a5fa;">R2: Box Validity</span><span class="badge-active">Active</span></div>
        <div class="rule-card-desc">Boundary Check</div>
        <div class="rule-card-value">x2&gt;x1, y2&gt;y1</div>
    </div>
    <div class="rule-card">
        <div class="rule-card-header"><span class="rule-card-title" style="color:#34d399;">R3: Size Filter</span><span class="badge-active">Active</span></div>
        <div class="rule-card-desc">Min Box Area</div>
        <div class="rule-card-value">&ge; {area} px²</div>
    </div>
    <div class="rule-card">
        <div class="rule-card-header"><span class="rule-card-title" style="color:#fbbf24;">R4: Duplicate</span><span class="badge-active">Active</span></div>
        <div class="rule-card-desc">IoU Overlap Threshold</div>
        <div class="rule-card-value">&lt; {iou:.2f} IoU</div>
    </div>
</div>"""


def render_rule_performance_table(batch: BatchResult | None) -> str:
    """Render rule performance stats table."""
    if batch is None or batch.total_detections == 0:
        return """
<div class="dashboard-panel">
    <div class="panel-title">📊 Rule Performance</div>
    <div class="panel-desc">Pass / fail rates across evaluated detections</div>
    <div class="empty-state">
        <div class="empty-state-icon">⚙️</div>
        <div class="empty-state-title">No Rules Evaluated</div>
        <div class="empty-state-desc">Upload images and click Process to see rule metrics.</div>
    </div>
</div>"""

    stats = {
        "confidence": {"name": "R1: Confidence Filter", "p": 0, "f": 0},
        "valid_box": {"name": "R2: Box Validity", "p": 0, "f": 0},
        "not_tiny": {"name": "R3: Size Filter", "p": 0, "f": 0},
        "no_duplicate": {"name": "R4: Duplicate Check", "p": 0, "f": 0},
    }
    for pr in batch.results:
        for qr in pr.quality_results:
            for p in qr.passed_rules:
                if p in stats: stats[p]["p"] += 1
            for f in qr.failed_rules:
                if f in stats: stats[f]["f"] += 1

    rows = []
    for v in stats.values():
        total = v["p"] + v["f"]
        rate = (v["p"] / total * 100) if total > 0 else 0.0
        rows.append(f'<tr><td style="font-weight:600;">{v["name"]}</td>'
                     f'<td style="color:#34d399;">{v["p"]}</td>'
                     f'<td style="color:#f87171;">{v["f"]}</td>'
                     f'<td><span class="badge-accept">{rate:.1f}%</span></td></tr>')

    return f"""
<div class="dashboard-panel">
    <div class="panel-title">📊 Rule Performance</div>
    <div class="panel-desc">Pass / fail rates across evaluated detections</div>
    <table class="custom-table">
        <thead><tr><th>Rule</th><th>Passed</th><th>Failed</th><th>Pass Rate</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
    </table>
</div>"""


def render_quality_breakdown_table(batch: BatchResult | None) -> str:
    """Render per-detection audit table."""
    if batch is None or batch.total_detections == 0:
        return """
<div class="dashboard-panel">
    <div class="panel-title">🔍 Detection Audit</div>
    <div class="panel-desc">Per-detection quality evaluation log</div>
    <div class="empty-state">
        <div class="empty-state-icon">📋</div>
        <div class="empty-state-title">Audit Log Empty</div>
        <div class="empty-state-desc">Upload images to generate detection logs.</div>
    </div>
</div>"""

    rows = []
    idx = 1
    for pr in batch.results:
        for qr in pr.quality_results:
            det = qr.detection
            checks = [
                ("confidence", "✓" if "confidence" in qr.passed_rules else "✕"),
                ("valid_box", "✓" if "valid_box" in qr.passed_rules else "✕"),
                ("not_tiny", "✓" if "not_tiny" in qr.passed_rules else "✕"),
                ("no_duplicate", "✓" if "no_duplicate" in qr.passed_rules else "✕"),
            ]
            badge = {"ACCEPT": "badge-accept", "FLAG": "badge-flag", "REJECT": "badge-reject"}.get(qr.decision, "badge-flag")
            rule_cells = "".join(
                f'<td style="color:{"#34d399" if v == "✓" else "#f87171"};text-align:center;font-weight:700;">{v}</td>'
                for _, v in checks
            )
            rows.append(
                f'<tr><td>#{idx}</td><td><code>{pr.image_id}</code></td>'
                f'<td><b>{det.class_name}</b></td><td>{det.conf:.2f}</td>'
                f'{rule_cells}'
                f'<td><span class="{badge}">{qr.decision}</span></td>'
                f'<td style="color:var(--text-secondary);font-size:12px;">{qr.reason}</td></tr>'
            )
            idx += 1

    return f"""
<div class="dashboard-panel">
    <div class="panel-title">🔍 Detection Audit</div>
    <div class="panel-desc">Auditable rule verification per detection</div>
    <div style="overflow-x:auto;">
        <table class="custom-table">
            <thead><tr>
                <th>#</th><th>Image</th><th>Class</th><th>Conf</th>
                <th style="text-align:center;">R1</th><th style="text-align:center;">R2</th>
                <th style="text-align:center;">R3</th><th style="text-align:center;">R4</th>
                <th>Decision</th><th>Explanation</th>
            </tr></thead>
            <tbody>{''.join(rows)}</tbody>
        </table>
    </div>
</div>"""
