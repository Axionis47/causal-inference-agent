"""The 13-section notebook, built to rerun from clean state.

The notebook's cwd at execution is the job's analysis dir. Display
sections read the artifact files when they exist and render an explicit
"this stage did not run" placeholder when they do not; the essentials
(spec, plan, recorded estimates) are embedded as frozen literals so the
verification cell always works. That cell re-runs the exact lane on the
saved data and asserts the recorded estimate, so a notebook that drifts
from its artifacts fails execution instead of lying.

The per-section cell bodies live in ``sections.py``; this module frames
the ordered ``Section`` list with the title cell and emits the notebook.
``build_notebook`` accepts an explicit ``narrative`` so the agentic S10
path can supply its own ordering; with ``None`` it falls back to the
deterministic spine order.
"""
from __future__ import annotations

import nbformat
from nbformat.v4 import new_markdown_cell, new_notebook

from src.analysis_v2.core import AnalysisRunState

from .sections import Section, ordered_narrative

SECTIONS = [
    "Load dataset and config",
    "Dataset profile",
    "User causal question",
    "Structured causal spec",
    "Detected design and plan",
    "Base and targeted EDA",
    "Method lane execution",
    "Main estimate",
    "Diagnostics",
    "Sensitivity checks",
    "Claim critique and final interpretation",
    "Limitations",
    "Artifact index",
]


def build_notebook(
    run: AnalysisRunState, narrative: list[Section] | None = None
) -> nbformat.NotebookNode:
    sections = narrative if narrative is not None else ordered_narrative(run)
    cells: list[nbformat.NotebookNode] = [
        new_markdown_cell(
            f"# Causal analysis notebook\n\n**Question.** {run.causal_question}\n\n"
            f"Claim strength: **{run.claim_critique.strength.value}**. "
            "This notebook reruns from the saved dataset, config, and artifacts; "
            "it has no hidden state from the web app."
        ),
    ]
    for section in sections:
        cells.extend(section.cells)
    return new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
    )
