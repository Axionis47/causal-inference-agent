"""AnalysisRunState commit discipline and persistence round-trip."""
from __future__ import annotations

import pytest

from src.analysis_v2.core import (
    AnalysisRunState,
    AnalysisStage,
    Artifact,
    ArtifactKind,
    GateResult,
    RunStatus,
    StaleRunState,
    TokenUsage,
)
from src.analysis_v2.spec import (
    CausalSpec,
    Confidence,
    EffectEstimate,
    EstimateResult,
    MethodLane,
    QuestionType,
    VariableRef,
)


def _state() -> AnalysisRunState:
    return AnalysisRunState(
        job_id="job-42",
        causal_question="Did the minimum wage rise reduce fast-food employment?",
        user_context="NJ/PA stores surveyed before and after the 1992 change.",
    )


def _artifact(artifact_id: str, stage: AnalysisStage) -> Artifact:
    return Artifact(
        artifact_id=artifact_id,
        kind=ArtifactKind.JSON,
        stage=stage,
        agent="intake",
        title="Causal spec draft",
        path="intake/causal_spec.json",
        media_type="application/json",
    )


def test_record_transition_appends_event_and_advances_stage_and_totals():
    s = _state()
    event = s.record_transition(
        to_state=AnalysisStage.S1_INTAKE_PARSED,
        agent_name="intake",
        gate_result=GateResult.advance(),
        tokens=TokenUsage(input_tokens=900, output_tokens=150),
        cost_usd=0.012,
    )
    assert event.sequence == 0
    assert event.from_state == AnalysisStage.S0_DATASET_SAVED
    assert event.to_state == AnalysisStage.S1_INTAKE_PARSED
    assert s.current_state == AnalysisStage.S1_INTAKE_PARSED
    assert s.state_version == 1
    assert s.total_tokens.total == 1050
    assert s.total_cost_usd == pytest.approx(0.012)

    second = s.record_transition(
        to_state=AnalysisStage.S2_PROFILE_CREATED, agent_name="profiling"
    )
    assert second.sequence == 1
    assert s.state_version == 2


def test_back_transition_to_a_later_stage_is_rejected():
    s = _state()
    s.record_transition(to_state=AnalysisStage.S2_PROFILE_CREATED, agent_name="profiling")
    bad = GateResult.back(AnalysisStage.S7_METHOD_EXECUTED, ["invalid variable role"])
    with pytest.raises(ValueError):
        s.record_transition(
            to_state=AnalysisStage.S7_METHOD_EXECUTED,
            agent_name="method_lane",
            gate_result=bad,
        )
    # nothing committed on rejection
    assert s.current_state == AnalysisStage.S2_PROFILE_CREATED
    assert len(s.state_events) == 1


def test_register_artifact_rejects_duplicates():
    s = _state()
    s.register_artifact(_artifact("intake/spec", AnalysisStage.S1_INTAKE_PARSED))
    with pytest.raises(ValueError, match="duplicate"):
        s.register_artifact(_artifact("intake/spec", AnalysisStage.S1_INTAKE_PARSED))


def test_mark_failed_and_completed_set_status_and_completion_time():
    s = _state()
    s.mark_failed("method lane crashed")
    assert s.status == RunStatus.FAILED
    assert s.error_message == "method lane crashed"
    assert s.completed_at is not None

    s2 = _state()
    s2.mark_completed()
    assert s2.status == RunStatus.COMPLETED
    assert s2.completed_at is not None


def test_json_round_trip_preserves_slots_events_and_artifacts():
    s = _state()
    s.causal_spec = CausalSpec(
        question_type=QuestionType.DID,
        confidence=Confidence.HIGH,
        outcome=VariableRef(column="fte_employment"),
        treatment=VariableRef(column="state", clue="NJ raised its minimum wage"),
        time_column=VariableRef(column="period"),
        group_column=VariableRef(column="state"),
    )
    s.estimate_result = EstimateResult(
        lane=MethodLane.DID,
        estimator="two-way fixed effects",
        effects=[
            EffectEstimate(
                estimand="att",
                estimate=2.75,
                std_error=1.34,
                ci_lower=0.12,
                ci_upper=5.38,
                p_value=0.04,
                unit="full-time-equivalent employees",
            )
        ],
        n_rows_used=384,
        outcome="fte_employment",
        treatment="state",
    )
    s.register_artifact(_artifact("intake/spec", AnalysisStage.S1_INTAKE_PARSED))
    s.record_transition(
        to_state=AnalysisStage.S1_INTAKE_PARSED,
        agent_name="intake",
        gate_result=GateResult.advance(soft_warnings=["wide-format panel"]),
        output_artifacts=["intake/spec"],
        tokens=TokenUsage(input_tokens=10, output_tokens=5),
    )

    payload = s.model_dump(mode="json")
    loaded = AnalysisRunState.load(payload)

    assert loaded.causal_spec is not None
    assert loaded.causal_spec.question_type == QuestionType.DID
    assert loaded.estimate_result is not None
    assert loaded.estimate_result.primary.estimate == pytest.approx(2.75)
    assert loaded.artifact_registry.ids() == ["intake/spec"]
    assert len(loaded.state_events) == 1
    assert loaded.state_events[0].gate_result is not None
    assert loaded.state_events[0].gate_result.soft_warnings == ["wide-format panel"]
    assert loaded.current_state == AnalysisStage.S1_INTAKE_PARSED


def test_load_refuses_wrong_schema_version_and_garbage():
    s = _state()
    payload = s.model_dump(mode="json")
    payload["schema_version"] = 999
    with pytest.raises(StaleRunState):
        AnalysisRunState.load(payload)
    with pytest.raises(StaleRunState):
        AnalysisRunState.load({"job_id": "x", "schema_version": 1})  # missing question
