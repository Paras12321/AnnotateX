"""
styles.py — Dark theme CSS for AnnotateX premium dashboard.

Provides comprehensive CSS for:
    - Fixed sidebar navigation
    - Page-based layout with show/hide switching
    - Compact metric cards, dashboard panels, badges
    - Quality tables, pipeline tracker, export cards
    - Dark navy/black backgrounds with purple accents
"""

CUSTOM_CSS = """
/* ==========================================================================
   AnnotateX Premium Dark Dashboard Theme
   ========================================================================== */

/* ---------- CSS Variables ---------- */
:root {
    --bg-main: #0b0f19;
    --bg-card: #131b2e;
    --bg-card-hover: #182238;
    --bg-sidebar: #0d1117;
    --bg-input: #0f1626;
    --border-color: #1e2d4a;
    --border-light: #2e4168;

    --accent-purple: #8b5cf6;
    --accent-purple-dark: #7c3aed;
    --accent-purple-glow: rgba(139, 92, 246, 0.25);
    --accent-purple-bg: rgba(139, 92, 246, 0.12);

    --color-accept: #10b981;
    --color-accept-bg: rgba(16, 185, 129, 0.12);
    --color-accept-border: rgba(16, 185, 129, 0.30);

    --color-flag: #f59e0b;
    --color-flag-bg: rgba(245, 158, 11, 0.12);
    --color-flag-border: rgba(245, 158, 11, 0.30);

    --color-reject: #ef4444;
    --color-reject-bg: rgba(239, 68, 68, 0.12);
    --color-reject-border: rgba(239, 68, 68, 0.30);

    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;

    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;

    --shadow-card: 0 2px 12px -2px rgba(0, 0, 0, 0.40);
    --sidebar-width: 250px;
    --transition-fast: 0.15s ease;
}

/* ---------- Global ---------- */
body, .gradio-container {
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}
.gradio-container {
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}
footer { display: none !important; }

/* ---------- App Shell ---------- */
#app-shell {
    gap: 0 !important;
    flex-wrap: nowrap !important;
    min-height: 100vh;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    background: none !important;
    box-shadow: none !important;
}
#app-shell > .block { background: none !important; box-shadow: none !important; border: none !important; }

/* ---------- Sidebar Column ---------- */
#sidebar-col {
    max-width: var(--sidebar-width) !important;
    min-width: var(--sidebar-width) !important;
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border-color) !important;
    padding: 0 !important;
    overflow-y: auto;
    overflow-x: hidden;
    height: 100vh;
    position: sticky;
    top: 0;
}
#sidebar-col > .block { background: none !important; box-shadow: none !important; border: none !important; padding: 0 !important; }

/* ---------- Content Column ---------- */
#content-col {
    flex: 1 !important;
    min-width: 0 !important;
    padding: 0 !important;
    background: var(--bg-main) !important;
    overflow-y: auto;
    height: 100vh;
}
#content-col > .block { background: none !important; box-shadow: none !important; border: none !important; padding: 0 24px !important; }

/* ---------- Sidebar Styles ---------- */
.sidebar {
    display: flex;
    flex-direction: column;
    height: 100vh;
}
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 20px 20px 16px;
    border-bottom: 1px solid var(--border-color);
}
.sidebar-logo {
    width: 36px; height: 36px;
    border-radius: 10px;
    background: linear-gradient(135deg, var(--accent-purple) 0%, #6d28d9 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; flex-shrink: 0;
    box-shadow: 0 2px 10px var(--accent-purple-glow);
}
.sidebar-brand-name {
    font-size: 16px; font-weight: 800;
    letter-spacing: 0.05em; color: var(--text-primary);
}
.sidebar-brand-tagline {
    font-size: 10px; color: var(--text-muted);
    margin-top: 1px; letter-spacing: 0.02em;
}

/* Navigation */
.sidebar-nav {
    flex: 1; padding: 12px 10px;
    display: flex; flex-direction: column; gap: 2px;
}
.nav-section-label {
    font-size: 10px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--text-muted); padding: 14px 12px 6px;
}
.nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 9px 14px; border-radius: var(--radius-sm);
    color: var(--text-secondary); font-size: 13px; font-weight: 500;
    cursor: pointer; transition: all var(--transition-fast);
    user-select: none; position: relative;
}
.nav-item:hover {
    background: rgba(255, 255, 255, 0.04); color: var(--text-primary);
}
.nav-item.active {
    background: var(--accent-purple-bg);
    color: var(--accent-purple); font-weight: 600;
}
.nav-item.active::before {
    content: ''; position: absolute;
    left: 0; top: 6px; bottom: 6px; width: 3px;
    background: var(--accent-purple); border-radius: 0 3px 3px 0;
}
.nav-icon { font-size: 16px; width: 20px; text-align: center; flex-shrink: 0; }

/* Sidebar Footer */
.sidebar-footer {
    padding: 14px 16px;
    border-top: 1px solid var(--border-color);
}
.sidebar-status {
    display: flex; align-items: center; gap: 8px;
    font-size: 11px; color: var(--text-secondary); margin-bottom: 6px;
}
.status-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--color-accept);
    box-shadow: 0 0 6px var(--color-accept); flex-shrink: 0;
}
.sidebar-version { font-size: 10px; color: var(--text-muted); }

/* ---------- Page Header ---------- */
.page-header {
    display: flex; align-items: center;
    justify-content: space-between;
    padding: 20px 0 16px; margin-bottom: 4px;
}
.page-header-left h2 {
    font-size: 20px; font-weight: 700;
    color: var(--text-primary); margin: 0; line-height: 1.3;
}
.page-header-left p {
    font-size: 12px; color: var(--text-muted); margin: 2px 0 0;
}
.btn-new-upload {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 8px 18px;
    background: linear-gradient(135deg, var(--accent-purple) 0%, #6d28d9 100%);
    color: #fff; border: none; border-radius: var(--radius-sm);
    font-size: 13px; font-weight: 600; cursor: pointer;
    box-shadow: 0 3px 12px rgba(109, 40, 217, 0.35);
    transition: all 0.25s ease;
}
.btn-new-upload:hover {
    transform: translateY(-1px);
    box-shadow: 0 5px 18px rgba(109, 40, 217, 0.50);
}

/* ---------- Metric Cards ---------- */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px; margin-bottom: 18px;
}
@media (max-width: 1200px) { .metric-grid { grid-template-columns: repeat(3, 1fr); } }
@media (max-width: 800px) { .metric-grid { grid-template-columns: repeat(2, 1fr); } }

.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    padding: 14px 16px;
    box-shadow: var(--shadow-card);
    transition: transform var(--transition-fast), border-color var(--transition-fast);
    position: relative; overflow: hidden;
}
.metric-card:hover { border-color: var(--border-light); transform: translateY(-1px); }
.metric-card::after {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
}
.card-purple::after { background: var(--accent-purple); }
.card-blue::after { background: #3b82f6; }
.card-green::after { background: var(--color-accept); }
.card-amber::after { background: var(--color-flag); }
.card-red::after { background: var(--color-reject); }

.metric-header {
    display: flex; align-items: center;
    justify-content: space-between; margin-bottom: 8px;
}
.metric-title {
    font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--text-secondary);
}
.metric-icon {
    width: 32px; height: 32px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 15px;
}
.card-purple .metric-icon { background: rgba(139, 92, 246, 0.15); }
.card-blue .metric-icon { background: rgba(59, 130, 246, 0.15); }
.card-green .metric-icon { background: rgba(16, 185, 129, 0.15); }
.card-amber .metric-icon { background: rgba(245, 158, 11, 0.15); }
.card-red .metric-icon { background: rgba(239, 68, 68, 0.15); }

.metric-value {
    font-size: 26px; font-weight: 700;
    color: var(--text-primary); line-height: 1.1; margin: 2px 0;
}
.metric-subtitle { font-size: 11px; color: var(--text-muted); margin-top: 4px; }

/* ---------- Badges ---------- */
.badge-accept {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 2px 8px; border-radius: 9999px;
    font-size: 10px; font-weight: 600;
    background: var(--color-accept-bg);
    color: var(--color-accept);
    border: 1px solid var(--color-accept-border);
}
.badge-flag {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 2px 8px; border-radius: 9999px;
    font-size: 10px; font-weight: 600;
    background: var(--color-flag-bg);
    color: var(--color-flag);
    border: 1px solid var(--color-flag-border);
}
.badge-reject {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 2px 8px; border-radius: 9999px;
    font-size: 10px; font-weight: 600;
    background: var(--color-reject-bg);
    color: var(--color-reject);
    border: 1px solid var(--color-reject-border);
}
.badge-active {
    display: inline-flex; padding: 2px 8px;
    border-radius: 9999px; font-size: 10px; font-weight: 600;
    background: var(--color-accept-bg);
    color: var(--color-accept);
    border: 1px solid var(--color-accept-border);
}

/* ---------- Dashboard Panels ---------- */
.dashboard-panel {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    padding: 16px 18px; margin-bottom: 14px;
    box-shadow: var(--shadow-card);
}
.panel-title {
    font-size: 14px; font-weight: 700;
    color: var(--text-primary); margin-bottom: 2px;
    display: flex; align-items: center; gap: 8px;
}
.panel-desc {
    font-size: 11px; color: var(--text-muted); margin-bottom: 12px;
}

/* Performance stats */
.perf-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px; margin-top: 10px;
}
.perf-stat {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-sm); padding: 10px; text-align: center;
}
.perf-stat-label {
    font-size: 10px; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.05em;
}
.perf-stat-value {
    font-size: 18px; font-weight: 700;
    color: var(--text-primary); margin-top: 3px;
}

/* Trust Bar */
.trust-bar-container { margin: 12px 0 8px; }
.trust-bar-header {
    display: flex; justify-content: space-between;
    font-size: 11px; color: var(--text-secondary); margin-bottom: 5px;
}
.trust-bar-track {
    width: 100%; height: 6px; background: #1e2d4a;
    border-radius: 9999px; overflow: hidden;
}
.trust-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent-purple), var(--color-accept));
    border-radius: 9999px; transition: width 0.6s ease;
}

/* Pipeline Flow */
.pipeline-flow {
    display: flex; align-items: center;
    gap: 0; flex-wrap: wrap; margin: 10px 0;
}
.flow-step {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 12px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--border-color);
    border-radius: 6px; font-size: 11px; font-weight: 500;
    color: var(--text-secondary); white-space: nowrap;
}
.flow-arrow { color: var(--text-muted); font-size: 14px; margin: 0 4px; flex-shrink: 0; }
.flow-step-icon { font-size: 14px; }

/* Pipeline Tracker */
.pipeline-step {
    display: flex; align-items: center;
    justify-content: space-between;
    padding: 8px 12px; border-radius: 6px;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(30, 45, 74, 0.5); margin-bottom: 6px;
}
.step-label { display: flex; align-items: center; gap: 8px; font-size: 12px; font-weight: 500; }
.step-badge-done {
    color: #34d399; font-size: 10px; font-weight: 600;
    background: rgba(16, 185, 129, 0.12);
    padding: 2px 8px; border-radius: 9999px;
}
.step-badge-pending {
    color: var(--text-muted); font-size: 10px;
    background: rgba(255, 255, 255, 0.04);
    padding: 2px 8px; border-radius: 9999px;
}

/* ---------- Quality Tables ---------- */
.custom-table {
    width: 100%; border-collapse: separate; border-spacing: 0;
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md); overflow: hidden;
    background: var(--bg-card); font-size: 12px;
}
.custom-table th {
    background: #0d1117; color: var(--text-secondary);
    font-weight: 600; text-align: left;
    padding: 10px 14px;
    border-bottom: 1px solid var(--border-color);
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em;
}
.custom-table td {
    padding: 9px 14px;
    border-bottom: 1px solid rgba(30, 45, 74, 0.4);
    color: var(--text-primary);
}
.custom-table tr:last-child td { border-bottom: none; }
.custom-table tr:hover td { background: rgba(255, 255, 255, 0.02); }

/* Rules Grid */
.rules-grid {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 10px; margin-bottom: 14px;
}
@media (max-width: 1000px) { .rules-grid { grid-template-columns: repeat(2, 1fr); } }
.rule-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md); padding: 14px;
}
.rule-card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.rule-card-title { font-size: 12px; font-weight: 700; }
.rule-card-desc { font-size: 10px; color: var(--text-muted); margin-bottom: 6px; }
.rule-card-value { font-size: 20px; font-weight: 700; color: #fff; }

/* ---------- Upload Zone ---------- */
.upload-zone { border: 2px dashed var(--border-light) !important; border-radius: var(--radius-lg) !important; background: rgba(255, 255, 255, 0.015) !important; }
.upload-zone:hover { border-color: var(--accent-purple) !important; background: var(--accent-purple-bg) !important; }

/* ---------- Export Cards ---------- */
.export-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 14px; margin-bottom: 16px;
}
.export-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md); padding: 18px;
    display: flex; flex-direction: column;
}
.export-card:hover { border-color: var(--border-light); }
.export-card-icon {
    width: 40px; height: 40px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; margin-bottom: 12px;
}
.export-card-title { font-size: 15px; font-weight: 700; color: var(--text-primary); margin-bottom: 4px; }
.export-card-desc { font-size: 12px; color: var(--text-secondary); line-height: 1.5; margin-bottom: 14px; flex: 1; }
.export-card-status { font-size: 11px; padding: 6px 12px; border-radius: var(--radius-sm); text-align: center; font-weight: 600; }
.export-available { background: var(--color-accept-bg); color: var(--color-accept); border: 1px solid var(--color-accept-border); }

/* ---------- Color Legend ---------- */
.color-legend { display: flex; gap: 16px; margin-bottom: 12px; font-size: 11px; font-weight: 600; color: var(--text-secondary); }
.legend-item { display: flex; align-items: center; gap: 6px; }
.legend-dot { width: 8px; height: 8px; border-radius: 2px; flex-shrink: 0; }

/* ---------- Empty States ---------- */
.empty-state {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 40px 20px; text-align: center;
}
.empty-state-icon { font-size: 40px; margin-bottom: 14px; opacity: 0.4; }
.empty-state-title { font-size: 15px; font-weight: 600; color: var(--text-secondary); margin-bottom: 6px; }
.empty-state-desc { font-size: 12px; color: var(--text-muted); max-width: 320px; }

/* ---------- Settings ---------- */
.settings-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; }
.setting-item { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: var(--radius-md); padding: 16px; }
.setting-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); font-weight: 600; margin-bottom: 6px; }
.setting-value { font-size: 16px; font-weight: 700; color: var(--text-primary); }

/* About */
.about-content { font-size: 13px; color: var(--text-secondary); line-height: 1.7; }
.about-content h3 { font-size: 16px; color: var(--text-primary); margin-top: 18px; margin-bottom: 8px; }
.about-content code { background: var(--bg-input); padding: 2px 6px; border-radius: 4px; font-size: 12px; color: var(--accent-purple); }

/* ---------- Buttons ---------- */
button.primary-btn, .primary-btn button, button.primary {
    background: linear-gradient(135deg, var(--accent-purple) 0%, #6d28d9 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(167, 139, 250, 0.3) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important; padding: 9px 22px !important;
    box-shadow: 0 3px 12px rgba(109, 40, 217, 0.35) !important;
}
button.primary-btn:hover, .primary-btn button:hover, button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 5px 18px rgba(109, 40, 217, 0.50) !important;
}

/* ---------- Gradio Overrides ---------- */
.block { border: none !important; background: none !important; box-shadow: none !important; }
.block.padded { padding: 0 !important; }
.plot-container { background: transparent !important; }
.prose { color: var(--text-secondary) !important; font-size: 13px !important; }
.prose h2, .prose h3 { color: var(--text-primary) !important; }
.prose strong { color: var(--text-primary) !important; }
.file-component { background: var(--bg-card) !important; border: 1px solid var(--border-color) !important; border-radius: var(--radius-md) !important; }
.gap.compact { gap: 10px !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--border-light); }

/* Page visibility */
.dashboard-page { display: none; }
#page-dashboard { display: block; }
"""
