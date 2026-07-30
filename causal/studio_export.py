"""Build a portable audit bundle containing the executable notebook."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

from nbformat.v4 import new_markdown_cell

from .notebook import build as build_notebook
from .notebook import to_json


RUNTIME_FILES = ("__init__.py", "lanes.py", "estimate.py", "prep.py", "profile.py")


def build_bundle(state: dict[str, Any]) -> bytes:
    """Return notebook + data + policy + minimal causal runtime as one zip."""
    notebook = build_notebook(
        csv_path="analysis_data.csv",
        question=state["context"]["question"],
        context=state["context"]["description"],
        lane=state["lane"],
        kwargs=state["kwargs"],
        estimate=state["estimate"],
        strength=state["strength"],
        source=state.get("source", "uploaded dataset"),
    )
    notebook.cells.insert(1, new_markdown_cell(state.get("report", "")))
    notebook.cells.insert(2, new_markdown_cell(
        "## Policy snapshot\n\n```json\n"
        + json.dumps(state.get("policy", {}), indent=2, default=str)
        + "\n```"
    ))

    root = Path(__file__).parent
    requirements = """numpy
pandas
scipy
statsmodels
scikit-learn
rdrobust
nbformat
"""
    readme = """CAUSAL STUDIO AUDIT BUNDLE

1. Create a Python environment and run: pip install -r requirements.txt
2. Start Jupyter from this extracted directory.
3. Open analysis.ipynb and run every cell.

The notebook recomputes the estimate from analysis_data.csv and asserts that
it matches the application result. data_version.json binds those bytes to the
approved repair/cohort manifest. policy.json and run.json are immutable
snapshots of the decision and trace shown in the UI.
"""

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("analysis.ipynb", to_json(notebook))
        archive.write(state["csv_path"], "analysis_data.csv")
        archive.writestr("policy.json", json.dumps(state.get("policy", {}), indent=2, default=str))
        archive.writestr("design_contract.json", json.dumps(state.get("design_contract", {}), indent=2, default=str))
        archive.writestr("data_version.json", json.dumps(state.get("data_version", {}), indent=2, default=str))
        archive.writestr("preflight.json", json.dumps(state.get("preflight", []), indent=2, default=str))
        archive.writestr("run.json", json.dumps(state, indent=2, default=str))
        archive.writestr("monitoring.json", json.dumps({
            "summary": state.get("monitoring", {}),
            "alerts": state.get("monitoring_alerts", []),
        }, indent=2, default=str))
        archive.writestr("report.md", state.get("report", ""))
        archive.writestr("requirements.txt", requirements)
        archive.writestr("README.txt", readme)
        for name in RUNTIME_FILES:
            archive.write(root / name, f"causal/{name}")
    return output.getvalue()
