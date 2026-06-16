"""The study-shaped spine of the generated notebook.

The notebook reads like a causal study: question, data, identification,
estimation, balance, diagnostics, robustness, conclusion, limitations. Each
section is a markdown header (the agentic S10 path swaps in the model's prose,
keeping the code cells) followed by the cells that recompute and show that part
of the analysis inline, from the same dataset and inputs the app used. There is
no embedded JSON, no self-check assert, and no artifact-registry dump; every
number and figure is recomputed here, so the notebook mirrors what the app
shows the user without re-deriving a separate story.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

from src.analysis_v2.core import AnalysisRunState

from . import plot_cells as P
from . import recompute_cells as R
from .load_cell import build_load_cells
from .prompt import dag_identification_lines

# id -> display title, the study spine in order.
SECTION_TITLES: dict[str, str] = {
    "question": "Question and setup",
    "data": "Data",
    "identification": "Identification strategy",
    "estimation": "Estimation",
    "balance": "Covariate balance",
    "diagnostics": "Diagnostics",
    "robustness": "Robustness and sensitivity",
    "conclusion": "Conclusion",
    "limitations": "Limitations",
}

# Sections the agentic loop may narrate; data and limitations stay deterministic
# so the profile table and the honest limitation bullets are preserved verbatim.
NARRATIVE_SECTIONS: tuple[str, ...] = (
    "question", "identification", "estimation", "balance",
    "diagnostics", "robustness", "conclusion",
)


@dataclass
class Section:
    """One study section: a stable id, a title, and the cells that render it.
    The first cell is always the markdown header the agentic path may replace."""

    section_id: str
    title: str
    cells: list[nbformat.NotebookNode] = field(default_factory=list)


def _section(section_id: str, lead: str, *sources: str) -> Section:
    cells = [new_markdown_cell(f"## {SECTION_TITLES[section_id]}\n\n{lead}")]
    cells.extend(new_code_cell(src) for src in sources)
    return Section(section_id, SECTION_TITLES[section_id], cells)


def _question(run: AnalysisRunState) -> Section:
    spec = run.causal_spec
    treat = spec.treatment.column if spec and spec.treatment else "the treatment"
    out = spec.outcome.column if spec and spec.outcome else "the outcome"
    qtype = spec.question_type.value if spec else "unknown"
    conf = spec.confidence.value if spec else "unknown"
    lead = (
        f"> {run.causal_question}\n\n"
        f"We estimate the effect of `{treat}` on `{out}` "
        f"(question type `{qtype}`, confidence {conf})."
    )
    return _section("question", lead)


def _data(run: AnalysisRunState) -> Section:
    lead = "Load the saved dataset and config, then summarise the columns."
    cells = [new_markdown_cell(f"## {SECTION_TITLES['data']}\n\n{lead}")]
    cells.extend(build_load_cells(run))
    cells.append(new_code_cell(R.PROFILE))
    return Section("data", SECTION_TITLES["data"], cells)


def _identification(run: AnalysisRunState) -> Section:
    lead = "\n".join(dag_identification_lines(run))
    if run.causal_dag is None:
        return _section("identification", lead)
    return _section("identification", lead, P.DAG_FIGURE)


def _estimation(run: AnalysisRunState) -> Section:
    lead = (
        "Re-run the chosen estimator on the loaded data and read the effect "
        "off the table and the forest plot below."
    )
    return _section("estimation", lead, R.ESTIMATE, P.FOREST)


def _balance(run: AnalysisRunState) -> Section:
    lead = (
        "Standardized mean differences show whether the compared groups were "
        "alike on the covariates; smaller is better (0.1 is a common threshold)."
    )
    return _section("balance", lead, R.BALANCE, P.LOVE)


def _diagnostics(run: AnalysisRunState) -> Section:
    lead = "Re-run the design's assumption checks on the loaded data."
    return _section("diagnostics", lead, R.DIAGNOSTICS)


def _robustness(run: AnalysisRunState) -> Section:
    lead = "Perturb the analysis and read off the robustness verdict."
    return _section("robustness", lead, R.SENSITIVITY)


def _conclusion(run: AnalysisRunState) -> Section:
    critique = run.claim_critique
    if critique is None:
        return _section("conclusion", "No claim critique was produced for this run.")
    allowed = ", ".join(critique.allowed_language)
    lead = f"Claim strength: **{critique.strength.value}**.\n\n{critique.rationale}"
    if allowed:
        lead += f"\n\nFair phrasings: {allowed}."
    return _section("conclusion", lead)


def _limitations(run: AnalysisRunState) -> Section:
    critique = run.claim_critique
    limits = critique.limitations if critique else []
    bullets = "\n".join(f"- {limit}" for limit in limits) or "- none recorded"
    return _section("limitations", bullets)


def ordered_study(run: AnalysisRunState) -> list[Section]:
    """The fixed study-spine order, deterministic from run state."""
    return [
        _question(run),
        _data(run),
        _identification(run),
        _estimation(run),
        _balance(run),
        _diagnostics(run),
        _robustness(run),
        _conclusion(run),
        _limitations(run),
    ]
