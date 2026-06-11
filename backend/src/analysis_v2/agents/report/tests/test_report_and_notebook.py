"""Report language guard, dashboard payload, and notebook structure."""
from __future__ import annotations

import nbformat
import pytest

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
    QuestionType,
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
