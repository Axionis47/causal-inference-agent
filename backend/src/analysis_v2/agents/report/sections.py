"""Deterministic section builders for the analysis notebook.

Each builder returns the notebook cells for one report section, including
its own ``## N. Title`` header. ``ordered_narrative`` assembles them in the
fixed spine order; ``build_notebook`` (in ``notebook.py``) flattens the
result and frames it with the title cell. Holding the per-section bodies
here keeps ``notebook.py`` under the file cap and gives a single ordered
list the agentic S10 path can reorder without forking the cell templates.

Every section carries a stable ``section_id``: the agentic path records
which ids the model narrated and the post-loop floor-fill appends any
mandatory section the model omitted, so coverage is an invariant.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

from src.analysis_v2.core import AnalysisRunState

from .load_cell import build_load_cells

# id -> display title, the full spine in order.
SECTION_TITLES: dict[str, str] = {
    "load": "Load dataset and config",
    "profile": "Dataset profile",
    "question": "User causal question",
    "spec": "Structured causal spec",
    "design": "Detected design and plan",
    "eda": "Base and targeted EDA",
    "method": "Method lane execution",
    "estimate": "Main estimate",
    "diagnostics": "Diagnostics",
    "sensitivity": "Sensitivity checks",
    "claim": "Claim critique and final interpretation",
    "limitations": "Limitations",
    "artifact_index": "Artifact index",
}

# Sections that must appear whenever their upstream slot exists: a report may
# not silently drop a finding. The floor-fill enforces this after the loop.
MANDATORY_SECTIONS: frozenset[str] = frozenset(
    {"estimate", "method", "claim", "limitations"}
)

# Sections the model may narrate. The rest (load, profile, spec, limitations,
# artifact_index) stay deterministic so their tables and the limitation bullets
# are preserved verbatim. The code cells always run in their fixed dependency
# order; the model supplies prose and emphasis, not execution order.
NARRATIVE_SECTIONS: tuple[str, ...] = (
    "question", "design", "eda", "method", "estimate", "diagnostics",
    "sensitivity", "claim",
)


@dataclass
class Section:
    """One report section: a stable id, a title, and the cells that render it."""

    section_id: str
    title: str
    cells: list[nbformat.NotebookNode] = field(default_factory=list)


def _literal(payload: dict) -> str:
    return json.dumps(payload, indent=1, default=str)


def _placeholder(section: str, artifact: str) -> nbformat.NotebookNode:
    return new_markdown_cell(
        f"*{section}: skipped — artifact `{artifact}` was not produced in "
        "this run (the stage did not execute), so there is nothing to show.*"
    )


def _load_section(run: AnalysisRunState) -> Section:
    return Section("load", "Load dataset and config", build_load_cells(run))


def _profile_section(have: set[str]) -> Section:
    cells = [new_markdown_cell("## 2. Dataset profile")]
    if "profiling/dataset_profile" in have:
        cells.append(
            new_code_cell(
                "profile = json.loads(Path('profiling/dataset_profile.json').read_text())\n"
                "print(f\"{profile['n_rows']:,} rows, {profile['n_columns']} columns; \"\n"
                "      f\"duplicates: {profile['duplicate_row_count']}\")\n"
                "pd.DataFrame(profile['columns'])[['name', 'semantic_type', 'missing_fraction', 'n_unique']]"
            )
        )
    else:
        cells.append(_placeholder("Dataset profile", "profiling/dataset_profile"))
    return Section("profile", "Dataset profile", cells)


def _question_section(run: AnalysisRunState) -> Section:
    return Section(
        "question",
        "User causal question",
        [
            new_markdown_cell(
                f"## 3. User causal question\n\n> {run.causal_question}\n\n"
                f"Question type: `{run.causal_spec.question_type.value}` "
                f"(confidence {run.causal_spec.confidence.value})."
            ),
        ],
    )


def _spec_section(spec_json: str) -> Section:
    # The spec is bound for the verification cell; the visible output is a
    # field/value frame, never a raw dict dump.
    return Section(
        "spec",
        "Structured causal spec",
        [
            new_markdown_cell("## 4. Structured causal spec"),
            new_code_cell(
                f"spec = json.loads('''{spec_json}''')\n"
                "pd.DataFrame(\n"
                "    [(k, v) for k, v in spec.items() if v not in (None, [], {})],\n"
                "    columns=['field', 'value'],\n"
                ")"
            ),
        ],
    )


def _design_section(plan_json: str, have: set[str]) -> Section:
    eligibility = (
        "eligibility = json.loads(Path('design/tool_eligibility.json').read_text())\n"
        "pd.DataFrame(eligibility['lanes'])[['lane', 'state', 'reason']]"
        if "design/tool_eligibility" in have
        else "print('lane eligibility table not produced in this run')"
    )
    return Section(
        "design",
        "Detected design and plan",
        [
            new_markdown_cell("## 5. Detected design and plan"),
            new_code_cell(
                f"plan = json.loads('''{plan_json}''')\n"
                "print('Lane:', plan['lane'], '| estimator:', plan['estimator'],\n"
                "      '| estimand:', plan['estimand'])\n" + eligibility
            ),
        ],
    )


def _eda_section(have: set[str]) -> Section:
    cells = [new_markdown_cell("## 6. Base and targeted EDA")]
    if "eda/summary" in have or "eda/plan" in have:
        cells.append(
            new_code_cell(
                "from IPython.display import Image, display, Markdown\n"
                "eda = json.loads(Path('eda/eda_summary.json').read_text())\n"
                "display(Markdown(eda['story']))\n"
                "pd.DataFrame([\n"
                "    {'check': c['name'], 'status': c['status'], 'detail': c['detail']}\n"
                "    for c in eda['checks']\n"
                "])"
            )
        )
        cells.append(
            new_code_cell(
                "plots = sorted(Path('eda/plots').glob('*.png')) if Path('eda/plots').exists() else []\n"
                "for png in plots:\n"
                "    display(Image(filename=str(png)))"
            )
        )
    else:
        cells.append(_placeholder("EDA", "eda/summary"))
    return Section("eda", "Base and targeted EDA", cells)


def _method_section(recorded_json: str) -> Section:
    return Section(
        "method",
        "Method lane execution",
        [
            new_markdown_cell("## 7. Method lane execution"),
            new_code_cell(
                f"recorded = json.loads('''{recorded_json}''')\n"
                "print('Estimator:', recorded['estimator'], 'on', recorded['n_rows_used'], 'rows')\n"
                "pd.DataFrame(recorded['effects'])"
            ),
            new_markdown_cell(
                "### Verification: re-run the lane from the saved data\n\n"
                "The cell below re-executes the exact lane with the frozen plan and "
                "asserts the primary estimate matches the recorded one."
            ),
            new_code_cell(
                "from src.analysis_v2.agents.method_lane.lanes import LANES\n"
                "from src.analysis_v2.spec import CausalSpec, MethodLane, MethodPlan\n\n"
                "plan_model = MethodPlan.model_validate(plan)\n"
                "spec_model = CausalSpec.model_validate(spec)\n"
                "rerun = LANES[MethodLane(plan['lane'])](df, plan_model, spec_model)\n"
                "fresh = rerun.result.primary.estimate\n"
                "stored = recorded['effects'][0]['estimate']\n"
                "tolerance = max(abs(stored) * 0.01, 1e-9)\n"
                "assert abs(fresh - stored) <= tolerance, (\n"
                "    f'recomputed {fresh} differs from recorded {stored}')\n"
                "print(f'verified: recomputed {fresh:.6g} matches recorded {stored:.6g}')"
            ),
        ],
    )


def _estimate_section() -> Section:
    return Section(
        "estimate",
        "Main estimate",
        [
            new_markdown_cell("## 8. Main estimate"),
            new_code_cell(
                "primary = recorded['effects'][0]\n"
                "interval = ''\n"
                "if primary.get('ci_lower') is not None:\n"
                "    interval = f\" (95% interval {primary['ci_lower']:.4g} to {primary['ci_upper']:.4g})\"\n"
                "print(f\"{primary['estimand']}: {primary['estimate']:.4g}{interval}\")"
            ),
        ],
    )


def _diagnostics_section(have: set[str]) -> Section:
    cells = [new_markdown_cell("## 9. Diagnostics")]
    if "diagnostics/report" in have:
        cells.append(
            new_code_cell(
                "diag = json.loads(Path('diagnostics/diagnostics_report.json').read_text())\n"
                "pd.DataFrame([\n"
                "    {'check': c['name'], 'status': c['status'], 'detail': c['detail']}\n"
                "    for c in diag['checks']\n"
                "])"
            )
        )
    else:
        cells.append(_placeholder("Diagnostics", "diagnostics/report"))
    return Section("diagnostics", "Diagnostics", cells)


def _sensitivity_section(have: set[str]) -> Section:
    cells = [new_markdown_cell("## 10. Sensitivity checks")]
    if "diagnostics/sensitivity" in have:
        cells.append(
            new_code_cell(
                "sens = json.loads(Path('diagnostics/sensitivity_report.json').read_text())\n"
                "print('Robustness:', sens['robustness'], '-', sens['confidence_reason'])\n"
                "pd.DataFrame([\n"
                "    {'check': c['name'], 'status': c['status'], 'detail': c['detail']}\n"
                "    for c in sens['checks']\n"
                "]) if sens['checks'] else None"
            )
        )
    else:
        cells.append(_placeholder("Sensitivity checks", "diagnostics/sensitivity"))
    return Section("sensitivity", "Sensitivity checks", cells)


def _claim_section(run: AnalysisRunState) -> Section:
    return Section(
        "claim",
        "Claim critique and final interpretation",
        [
            new_markdown_cell(
                "## 11. Claim critique and final interpretation\n\n"
                f"Claim strength: **{run.claim_critique.strength.value}**.\n\n"
                f"{run.claim_critique.rationale}\n\n"
                "Allowed phrasings: "
                + ", ".join(f"“{p}”" for p in run.claim_critique.allowed_language)
                + "."
            ),
        ],
    )


def _limitations_section(run: AnalysisRunState) -> Section:
    return Section(
        "limitations",
        "Limitations",
        [
            new_markdown_cell(
                "## 12. Limitations\n\n"
                + "\n".join(f"- {limit}" for limit in run.claim_critique.limitations)
            ),
        ],
    )


def _artifact_index_section() -> Section:
    return Section(
        "artifact_index",
        "Artifact index",
        [
            new_markdown_cell("## 13. Artifact index"),
            new_code_cell(
                "index = json.loads(Path('report/artifacts_index.json').read_text())\n"
                "pd.DataFrame(index['artifacts'])[['artifact_id', 'kind', 'agent', 'title', 'path']]"
            ),
        ],
    )


def ordered_narrative(run: AnalysisRunState) -> list[Section]:
    """The fixed spine order of report sections, deterministic from run state."""
    have = set(run.artifact_registry.ids())
    spec_json = _literal(run.causal_spec.model_dump(mode="json"))
    plan_json = _literal(run.method_plan.model_dump(mode="json"))
    recorded_json = _literal(run.estimate_result.model_dump(mode="json"))
    return [
        _load_section(run),
        _profile_section(have),
        _question_section(run),
        _spec_section(spec_json),
        _design_section(plan_json, have),
        _eda_section(have),
        _method_section(recorded_json),
        _estimate_section(),
        _diagnostics_section(have),
        _sensitivity_section(have),
        _claim_section(run),
        _limitations_section(run),
        _artifact_index_section(),
    ]
