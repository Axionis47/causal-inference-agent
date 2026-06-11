"""The 13-section notebook, built to rerun from clean state.

The notebook's cwd at execution is the job's analysis dir. Display
sections read the artifact files when they exist and render an explicit
"this stage did not run" placeholder when they do not; the essentials
(spec, plan, recorded estimates) are embedded as frozen literals so the
verification cell always works. That cell re-runs the exact lane on the
saved data and asserts the recorded estimate, so a notebook that drifts
from its artifacts fails execution instead of lying.
"""
from __future__ import annotations

import json

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from src.analysis_v2.core import AnalysisRunState

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


def _literal(payload: dict) -> str:
    return json.dumps(payload, indent=1, default=str)


def _placeholder(section: str, artifact: str) -> nbformat.NotebookNode:
    return new_markdown_cell(
        f"*{section}: skipped — artifact `{artifact}` was not produced in "
        "this run (the stage did not execute), so there is nothing to show.*"
    )


def build_notebook(run: AnalysisRunState) -> nbformat.NotebookNode:
    have = set(run.artifact_registry.ids())
    spec_json = _literal(run.causal_spec.model_dump(mode="json"))
    plan_json = _literal(run.method_plan.model_dump(mode="json"))
    recorded_json = _literal(run.estimate_result.model_dump(mode="json"))

    cells = [
        new_markdown_cell(
            f"# Causal analysis notebook\n\n**Question.** {run.causal_question}\n\n"
            f"Claim strength: **{run.claim_critique.strength.value}**. "
            "This notebook reruns from the saved dataset, config, and artifacts; "
            "it has no hidden state from the web app."
        ),
        new_markdown_cell("## 1. Load dataset and config"),
        new_code_cell(
            "import json, sys\n"
            "from pathlib import Path\n"
            "import pandas as pd\n\n"
            "CONFIG = json.loads(Path('notebook/notebook_config.json').read_text())\n"
            "sys.path.insert(0, CONFIG['backend_dir'])\n"
            "from src.analysis_v2.runner.loader import coerce_bool_columns\n\n"
            "df = coerce_bool_columns(pd.read_parquet(CONFIG['dataset_path']))\n"
            "for col in CONFIG['ignored_columns']:\n"
            "    if col in df.columns:\n"
            "        df = df.drop(columns=[col])\n"
            "print(f'{len(df):,} rows x {len(df.columns)} columns')\n"
            "df.head()"
        ),
        new_markdown_cell("## 2. Dataset profile"),
    ]
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

    cells += [
        new_markdown_cell(
            f"## 3. User causal question\n\n> {run.causal_question}\n\n"
            f"Question type: `{run.causal_spec.question_type.value}` "
            f"(confidence {run.causal_spec.confidence.value})."
        ),
        new_markdown_cell("## 4. Structured causal spec"),
        new_code_cell(
            f"spec = json.loads('''{spec_json}''')\n"
            "{k: v for k, v in spec.items() if v not in (None, [], {})}"
        ),
        new_markdown_cell("## 5. Detected design and plan"),
        new_code_cell(
            f"plan = json.loads('''{plan_json}''')\n"
            "print('Lane:', plan['lane'], '| estimator:', plan['estimator'],\n"
            "      '| estimand:', plan['estimand'])\n"
            + (
                "eligibility = json.loads(Path('design/tool_eligibility.json').read_text())\n"
                "pd.DataFrame(eligibility['lanes'])[['lane', 'state', 'reason']]"
                if "design/tool_eligibility" in have
                else "print('lane eligibility table not produced in this run')"
            )
        ),
        new_markdown_cell("## 6. Base and targeted EDA"),
    ]
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

    cells += [
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
        new_markdown_cell("## 8. Main estimate"),
        new_code_cell(
            "primary = recorded['effects'][0]\n"
            "interval = ''\n"
            "if primary.get('ci_lower') is not None:\n"
            "    interval = f\" (95% interval {primary['ci_lower']:.4g} to {primary['ci_upper']:.4g})\"\n"
            "print(f\"{primary['estimand']}: {primary['estimate']:.4g}{interval}\")"
        ),
        new_markdown_cell("## 9. Diagnostics"),
    ]
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
    cells.append(new_markdown_cell("## 10. Sensitivity checks"))
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

    cells += [
        new_markdown_cell(
            "## 11. Claim critique and final interpretation\n\n"
            f"Claim strength: **{run.claim_critique.strength.value}**.\n\n"
            f"{run.claim_critique.rationale}\n\n"
            "Allowed phrasings: "
            + ", ".join(f"“{p}”" for p in run.claim_critique.allowed_language)
            + "."
        ),
        new_markdown_cell(
            "## 12. Limitations\n\n"
            + "\n".join(f"- {limit}" for limit in run.claim_critique.limitations)
        ),
        new_markdown_cell("## 13. Artifact index"),
        new_code_cell(
            "index = json.loads(Path('report/artifacts_index.json').read_text())\n"
            "pd.DataFrame(index['artifacts'])[['artifact_id', 'kind', 'agent', 'title', 'path']]"
        ),
    ]
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
