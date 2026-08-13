"""Convenient launcher for the Reunion YOLO head-detector workflow.

Run from the PCV repository root:
    python latih_reunion.py --prepare-annotation
"""

from pathlib import Path
import runpy


runpy.run_path(
    str(
        Path(__file__).resolve().parent
        / ".worktrees"
        / "turtle-reid-prototype"
        / "eksperimen"
        / "latih_detektor_reunion.py"
    ),
    run_name="__main__",
)
