"""DesignDetectionAgent on manifest-grounded scenarios (deterministic eval).

Each case builds the spec a competent intake would produce for a
representative dataset (roles per the manifest's expected_fields), runs
the agent against the real fixture's profile, and pins the leading
candidate lane plus the disabled lanes.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analysis_v2.agents.base import AgentCtx
from src.analysis_v2.agents.design_detection import DesignDetectionAgent
from src.analysis_v2.agents.profiling.tools import build_profile_summary
from src.analysis_v2.core import AnalysisRunState, GateStatus
from src.analysis_v2.spec import (
    CausalSpec,
    EligibilityState,
    MethodLane,
    QuestionType,
    VariableRef,
)

DATA = Path(__file__).resolve().parents[3] / "evals" / "fixtures" / "data"


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path))
    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    yield tmp_path
    settings_mod.get_settings.cache_clear()


async def _run(spec: CausalSpec, frame: pd.DataFrame):
    run = AnalysisRunState(job_id="job-design", causal_question="q?")
    run.causal_spec = spec
    run.dataset_profile = build_profile_summary(frame)
    ctx = AgentCtx(job_id="job-design", run=run, frame=frame)
    agent = DesignDetectionAgent()
    result = await agent.execute(ctx)
    agent.commit(ctx.run, result.output)
    return result, ctx.run


async def test_lalonde_leads_observational_with_matching_and_others_disabled(data_dir):
    spec = CausalSpec(
        question_type=QuestionType.BINARY_TREATMENT,
        outcome=VariableRef(column="re78"),
        treatment=VariableRef(column="treat"),
        candidate_confounders=["age", "educ", "re74", "re75"],
    )
    result, run = await _run(spec, pd.read_csv(DATA / "lalonde.csv"))

    assert result.gate.status == GateStatus.ADVANCE
    lanes = [c.lane for c in run.design_candidates]
    assert lanes[0] == MethodLane.OBSERVATIONAL
    assert MethodLane.MATCHING in lanes
    for lane in (MethodLane.IV, MethodLane.RDD, MethodLane.SURVIVAL):
        assert run.tool_eligibility.for_lane(lane).state == EligibilityState.DISABLED
    # re74/re75/re78 are real yearly pre/post earnings, so DID may surface as
    # a reshape-conditional design; it must never be silently ENABLED.
    assert (
        run.tool_eligibility.for_lane(MethodLane.DID).state != EligibilityState.ENABLED
    )
    assert run.artifact_registry.get("design/tool_eligibility") is not None


async def test_bank_rdd_promotes_rdd_and_asks_for_the_cutoff(data_dir):
    spec = CausalSpec(
        question_type=QuestionType.RDD,
        outcome=VariableRef(column="actual_recovery_amount"),
        treatment=VariableRef(derived=True, clue="strategy assigned above threshold"),
        running_variable=VariableRef(column="expected_recovery_amount"),
    )
    result, run = await _run(spec, pd.read_csv(DATA / "bank_data.csv"))

    assert run.design_candidates[0].lane == MethodLane.RDD
    rdd = run.tool_eligibility.for_lane(MethodLane.RDD)
    assert rdd.state == EligibilityState.CONDITIONAL
    assert "cutoff_value" in rdd.missing_requirements


async def test_card_iv_enables_iv_and_keeps_the_observational_fallback(data_dir):
    spec = CausalSpec(
        question_type=QuestionType.IV,
        outcome=VariableRef(column="lwage"),
        treatment=VariableRef(column="educ"),
        instrument=VariableRef(column="nearc4"),
        candidate_confounders=["exper", "black", "south", "smsa"],
    )
    result, run = await _run(spec, pd.read_csv(DATA / "card_schooling_wages.csv"))

    lanes = [c.lane for c in run.design_candidates]
    assert lanes[0] == MethodLane.IV
    assert MethodLane.OBSERVATIONAL in lanes


async def test_heart_failure_enables_survival(data_dir):
    spec = CausalSpec(
        question_type=QuestionType.SURVIVAL,
        outcome=VariableRef(column="time"),
        treatment=VariableRef(column="high_blood_pressure"),
        duration_column=VariableRef(column="time"),
        event_column=VariableRef(column="DEATH_EVENT"),
    )
    result, run = await _run(spec, pd.read_csv(DATA / "heart_failure.csv"))

    assert run.design_candidates[0].lane == MethodLane.SURVIVAL
    assert (
        run.tool_eligibility.for_lane(MethodLane.SURVIVAL).state
        == EligibilityState.ENABLED
    )


async def test_wide_employment_did_is_offered_conditionally_not_silently_enabled(data_dir):
    spec = CausalSpec(
        question_type=QuestionType.DID,
        outcome=VariableRef(candidates=["total_emp_feb", "total_emp_nov"]),
        treatment=VariableRef(derived=True, clue="NJ stores after the wage rise"),
        group_column=VariableRef(column="state"),
    )
    result, run = await _run(spec, pd.read_csv(DATA / "employment.csv"))

    did = run.tool_eligibility.for_lane(MethodLane.DID)
    assert did.state == EligibilityState.CONDITIONAL
    assert any("reshape" in m for m in did.missing_requirements)
    assert run.design_candidates[0].lane == MethodLane.DID


async def test_sole_candidate_promotion_is_recorded_and_committed(data_dir):
    spec = CausalSpec(
        question_type=QuestionType.DOSE_RESPONSE,
        outcome=VariableRef(candidates=["Sales ($)"]),
        treatment=VariableRef(column="TV Ad Budget ($)"),
        treatment_is_continuous=False,  # wrong on purpose; resolve corrects it
    )
    result, run = await _run(spec, pd.read_csv(DATA / "advertising.csv"))

    assert run.causal_spec.outcome.column == "Sales ($)"
    assert run.causal_spec.treatment_is_continuous is True
    assert "promoted sole candidate" in result.public_summary
    matching = run.tool_eligibility.for_lane(MethodLane.MATCHING)
    assert matching.state == EligibilityState.DISABLED  # continuous treatment


async def test_missing_upstream_slots_fail_the_gate(data_dir):
    run = AnalysisRunState(job_id="job-design", causal_question="q?")
    ctx = AgentCtx(job_id="job-design", run=run, frame=pd.DataFrame({"a": [1]}))
    result = await DesignDetectionAgent().execute(ctx)
    assert result.gate.status == GateStatus.FAIL
