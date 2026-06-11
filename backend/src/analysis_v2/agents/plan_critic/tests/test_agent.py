"""Plan gate dispositions: auto-approve, ask, fail."""
from __future__ import annotations

import pandas as pd
import pytest

from src.analysis_v2.agents.base import AgentCtx
from src.analysis_v2.agents.plan_critic import PlanCriticAgent
from src.analysis_v2.agents.profiling.tools import build_profile_summary
from src.analysis_v2.core import AnalysisRunState, GateStatus
from src.analysis_v2.spec import (
    CausalSpec,
    Confidence,
    DesignCandidate,
    MethodLane,
    PlanGateStatus,
    QuestionType,
    VariableRef,
)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path))
    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    yield tmp_path
    settings_mod.get_settings.cache_clear()


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "treat": [0, 1] * 40,
            "re78": [float(i) for i in range(80)],
            "age": list(range(20, 100)),
        }
    )


def _candidate(lane=MethodLane.OBSERVATIONAL, confidence=Confidence.HIGH, missing=None):
    return DesignCandidate(
        lane=lane, design_label="label", confidence=confidence,
        rationale="r", missing_requirements=missing or [],
    )


async def _run(spec, candidates, frame=None):
    frame = frame if frame is not None else _frame()
    run = AnalysisRunState(job_id="job-plan", causal_question="q?")
    run.causal_spec = spec
    run.dataset_profile = build_profile_summary(frame)
    run.design_candidates = candidates
    ctx = AgentCtx(job_id="job-plan", run=run, frame=frame)
    agent = PlanCriticAgent()
    result = await agent.execute(ctx)
    if result.output is not None:
        agent.commit(ctx.run, result.output)
    return result, ctx.run


async def test_a_fully_resolved_high_confidence_plan_auto_approves(data_dir):
    spec = CausalSpec(
        question_type=QuestionType.BINARY_TREATMENT,
        confidence=Confidence.HIGH,
        outcome=VariableRef(column="re78"),
        treatment=VariableRef(column="treat"),
        candidate_confounders=["age"],
    )
    result, run = await _run(spec, [_candidate()])

    assert result.gate.status == GateStatus.ADVANCE
    assert run.plan_critique.status == PlanGateStatus.PASS_AUTO_APPROVED
    assert run.method_plan.estimator == "regression_adjustment"
    assert run.method_plan.settings["include_ipw"] is True
    assert run.selected_design.lane == MethodLane.OBSERVATIONAL


async def test_missing_rdd_cutoff_parks_with_a_card_item(data_dir):
    spec = CausalSpec(
        question_type=QuestionType.RDD,
        confidence=Confidence.HIGH,
        outcome=VariableRef(column="re78"),
        treatment=VariableRef(column="treat"),
        running_variable=VariableRef(column="age"),
    )
    result, run = await _run(
        spec, [_candidate(MethodLane.RDD, missing=["cutoff_value"])]
    )

    assert result.gate.status == GateStatus.NEEDS_USER
    critique = run.plan_critique
    assert critique.status == PlanGateStatus.NEEDS_USER_CONFIRMATION
    fields = [i.field for i in critique.confirmation_card.items]
    assert "cutoff_value" in fields
    assert run.artifact_registry.get("plan/confirmation_card") is not None


async def test_candidate_only_outcome_asks_with_options(data_dir):
    spec = CausalSpec(
        question_type=QuestionType.SIMPLE_EFFECT,
        confidence=Confidence.MEDIUM,
        outcome=VariableRef(candidates=["re78", "age"]),
        treatment=VariableRef(column="treat"),
    )
    result, run = await _run(spec, [_candidate(confidence=Confidence.MEDIUM)])

    assert result.gate.status == GateStatus.NEEDS_USER
    item = next(
        i for i in run.plan_critique.confirmation_card.items if i.field == "outcome"
    )
    assert item.options == ["re78", "age"]


async def test_derived_treatment_requires_an_explicit_acknowledgement(data_dir):
    spec = CausalSpec(
        question_type=QuestionType.DID,
        confidence=Confidence.HIGH,
        outcome=VariableRef(column="re78"),
        treatment=VariableRef(derived=True, clue="post-period indicator"),
        time_column=VariableRef(column="age"),
    )
    result, run = await _run(spec, [_candidate(MethodLane.DID)])

    assert result.gate.status == GateStatus.NEEDS_USER
    fields = [i.field for i in run.plan_critique.confirmation_card.items]
    assert "confirm_derived_treatment" in fields


async def test_hard_missing_outcome_fails_the_gate_with_the_field_list(data_dir):
    spec = CausalSpec(question_type=QuestionType.BINARY_TREATMENT)
    result, _ = await _run(spec, [])

    assert result.gate.status == GateStatus.FAIL
    assert any("outcome" in f for f in result.gate.hard_failures)
    assert any("no executable design" in f for f in result.gate.hard_failures)
