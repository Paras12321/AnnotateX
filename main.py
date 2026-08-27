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

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.ui import app


if __name__ == "__main__":
    app.launch()
