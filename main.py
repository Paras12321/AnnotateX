"""
main.py — CLI entry point for AnnotateX.

Usage:
    python main.py                      # Launch the Gradio web app
    python main.py --help               # Show help

This file calls pipeline/orchestrator.py and app/ui.py.
It is the single entry point for running the application.
"""

import sys
from pathlib import Path

import gradio as gr

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.ui import app
from app.styles import CUSTOM_CSS
from app.components import SIDEBAR_JS

if __name__ == "__main__":
    theme = gr.themes.Base(
        primary_hue="purple",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    )
    app.launch(theme=theme, css=CUSTOM_CSS, head=SIDEBAR_JS)
