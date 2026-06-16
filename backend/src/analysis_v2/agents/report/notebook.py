"""Frame the study-shaped notebook: the title cell plus the ordered sections.

The notebook's cwd at execution is the job's analysis dir. The per-section
cell bodies live in ``study.py``; this module seeds the title cell and
flattens the ordered ``Section`` list into one notebook. ``build_notebook``
accepts an explicit ``narrative`` so the agentic S10 path can supply prose;
with ``None`` it uses the deterministic study spine.
"""
from __future__ import annotations

import nbformat
from nbformat.v4 import new_markdown_cell, new_notebook

from src.analysis_v2.core import AnalysisRunState

from .study import Section, ordered_study

SECTIONS = [
    "Question and setup",
    "Data",
    "Identification strategy",
    "Estimation",
    "Covariate balance",
    "Diagnostics",
    "Robustness and sensitivity",
    "Conclusion",
    "Limitations",
]


def build_notebook(
    run: AnalysisRunState, narrative: list[Section] | None = None
) -> nbformat.NotebookNode:
    sections = narrative if narrative is not None else ordered_study(run)
    cells: list[nbformat.NotebookNode] = [
        new_markdown_cell(
            f"# Causal analysis notebook\n\n**Question.** {run.causal_question}\n\n"
            f"Claim strength: **{run.claim_critique.strength.value}**. "
            "Every figure and number below is recomputed from the saved dataset "
            "and analysis inputs; the notebook has no hidden state from the web app."
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
