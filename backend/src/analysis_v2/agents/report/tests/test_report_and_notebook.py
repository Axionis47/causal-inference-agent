"""Report language guard, dashboard payload, and notebook structure."""
from __future__ import annotations

import nbformat
import pytest

from src.analysis_v2.agents.report.agent import ReportNotebookAgent
from src.analysis_v2.agents.report.notebook import SECTIONS, build_notebook
from src.analysis_v2.agents.report.report import (
    build_dashboard_payload,
    build_report_markdown,
)
from src.analysis_v2.core import (
    AgentRun,
    AgentRunStatus,
    AnalysisRunState,
    AnalysisStage,
)
from src.analysis_v2.spec import (
    ClaimCritique,
    ClaimStrength,
    EffectEstimate,
    EstimateResult,
    MethodLane,
    MethodPlan,
    CausalSpec,
    NotebookBuildResult,
    QuestionType,
    ReportToolCall,
    ReportToolTrace,
    RobustnessStatus,
    SensitivityResult,
    VariableRef,
)


def _run_state() -> AnalysisRunState:
    run = AnalysisRunState(
        job_id="job-report",
        causal_question="Does the program raise earnings?",
    )
    run.causal_spec = CausalSpec(
        question_type=QuestionType.BINARY_TREATMENT,
        outcome=VariableRef(column="re78"),
        treatment=VariableRef(column="treat"),
    )
    run.method_plan = MethodPlan(
        lane=MethodLane.OBSERVATIONAL, estimator="regression_adjustment",
        estimand="ate", outcome="re78", treatment="treat",
    )
    run.estimate_result = EstimateResult(
        lane=MethodLane.OBSERVATIONAL, estimator="regression_adjustment",
        effects=[EffectEstimate(estimand="ate", estimate=1548.2, std_error=750.0,
                                ci_lower=78.2, ci_upper=3018.2, p_value=0.04,
                                interpretation="adjusted association")],
        n_rows_used=614, outcome="re78", treatment="treat",
        covariates_used=["age", "re74"],
    )
    run.sensitivity_result = SensitivityResult(
        robustness=RobustnessStatus.FRAGILE, confidence_reason="e-value 1.4"
    )
    run.claim_critique = ClaimCritique(
        strength=ClaimStrength.WEAK,
        allowed_language=["suggests", "consistent with"],
        forbidden_language=["proves", "definitely causes"],
        limitations=["unmeasured confounding can bias the estimate"],
        rationale="observational design, fragile sensitivity",
    )
    record = AgentRun(agent="method_lane", stage=AnalysisStage.S7_METHOD_EXECUTED)
    record.start()
    record.finish(AgentRunStatus.PASSED)
    run.add_agent_run(record)
    return run


def test_report_uses_guarded_language_and_lists_limitations():
    run = _run_state()
    text = build_report_markdown(run)
    lowered = text.lower()
    for forbidden in run.claim_critique.forbidden_language:
        assert forbidden not in lowered
    assert "weak" in lowered
    assert "unmeasured confounding" in text
    assert "1548" in text.replace(",", "")


def test_dashboard_payload_carries_headline_tiles_and_costs():
    payload = build_dashboard_payload(_run_state())
    assert payload["headline"]["claim_strength"] == "weak"
    assert payload["tiles"][0]["agent"] == "method_lane"
    assert "total_cost_usd" in payload["costs"]


def test_notebook_has_all_sections_and_placeholders_for_missing_artifacts():
    run = _run_state()  # registry is empty: profile/eda/diagnostics never ran
    notebook = build_notebook(run)
    nbformat.validate(notebook)
    markdown = "\n".join(
        c.source for c in notebook.cells if c.cell_type == "markdown"
    )
    for index, section in enumerate(SECTIONS, start=1):
        assert f"## {index}. {section}" in markdown, section
    assert markdown.count("skipped — artifact") >= 3  # honest placeholders
    code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert "LANES[MethodLane(plan['lane'])]" in code  # the verification cell
    assert "assert abs(fresh - stored)" in code


def test_no_notebook_code_cell_displays_a_raw_dict():
    # A report shows prose, clean frames, and figures, never a raw dict dump.
    notebook = build_notebook(_run_state())
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        last = [line for line in cell.source.splitlines() if line.strip()][-1]
        assert not last.lstrip().startswith("{"), cell.source


def test_report_tool_trace_round_trips_and_old_records_still_load():
    run = _run_state()
    run.report_tool_trace = ReportToolTrace(
        calls=[ReportToolCall(name="Main estimate", section_id="estimate",
                              kind="section", status="ok")]
    )
    payload = run.model_dump(mode="json")

    reloaded = AnalysisRunState.load(payload)
    assert reloaded.report_tool_trace.calls[0].section_id == "estimate"

    # a record persisted before this slot existed must still load (no bump)
    payload.pop("report_tool_trace")
    old = AnalysisRunState.load(payload)
    assert old.report_tool_trace is None
    assert old.schema_version == 1


def test_commit_homes_a_synthesized_trace_from_the_rendered_sections():
    run = _run_state()
    output = NotebookBuildResult(
        notebook_artifact_id="notebook/causal_analysis",
        report_artifact_id="report/final_report",
        sections=["Main estimate", "Diagnostics"],
    )
    ReportNotebookAgent().commit(run, output)
    assert run.notebook_build is output
    assert [c.section_id for c in run.report_tool_trace.calls] == [
        "Main estimate", "Diagnostics"
    ]


def test_report_guard_allows_causal_prose_the_eda_guard_would_reject():
    from src.analysis_v2.agents.report.guard import forbidden_hit, passes_report_guard
    from src.analysis_v2.agents.targeted_eda.agent import _no_causal_language

    critique = _run_state().claim_critique  # forbids "proves", "definitely causes"
    # A report at the permitted strength may name the causal effect; EDA may not.
    at_strength = "the program causes higher earnings under this design"
    assert passes_report_guard(at_strength, critique) is True
    assert _no_causal_language(at_strength) is False  # the EDA ban would gut it

    # The critic's forbidden phrasing is still caught.
    overclaim = "this proves the program lifts earnings"
    assert forbidden_hit(overclaim, critique) == "proves"
    assert passes_report_guard(overclaim, critique) is False


def test_deterministic_report_assert_routes_through_the_report_guard():
    import pytest

    run = _run_state()
    # Force a forbidden phrase into a slot the deterministic report renders.
    run.estimate_result.effects[0].interpretation = "this proves a causal effect"
    with pytest.raises(AssertionError, match="forbidden phrase in report: proves"):
        build_report_markdown(run)


def test_notebook_load_cell_coerces_bool_columns_like_the_pipeline_loader(
    tmp_path, monkeypatch
):
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    backend_dir = Path.cwd()
    pd.DataFrame(
        {"treatment": [True, False, True], "y_factual": [5.6, 6.9, 4.8]}
    ).to_parquet(tmp_path / "data.parquet", index=False)
    (tmp_path / "notebook").mkdir()
    (tmp_path / "notebook" / "notebook_config.json").write_text(
        json.dumps(
            {
                "backend_dir": str(backend_dir),
                "dataset_path": str(tmp_path / "data.parquet"),
                "ignored_columns": [],
            }
        )
    )
    run = _run_state()
    (tmp_path / "notebook" / "analysis_inputs.json").write_text(
        json.dumps(ReportNotebookAgent._analysis_inputs(run))
    )

    notebook = build_notebook(run)
    load_cell = next(
        c.source
        for c in notebook.cells
        if c.cell_type == "code" and "dataset_path" in c.source
    )
    monkeypatch.chdir(tmp_path)
    namespace: dict = {}
    exec(load_cell, namespace)

    df = namespace["df"]
    assert df["treatment"].dtype.kind in "iu"
    assert np.asarray(df).dtype != object
    assert namespace["PLAN"]["lane"] == "observational"  # frozen inputs bound
    assert namespace["SPEC"]["outcome"]["column"] == "re78"
