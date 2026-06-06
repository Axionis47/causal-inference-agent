"""Data-review gate: should_pause_for_approval + park_for_approval.

Pins the truth table the orchestrator honours (pause once the data is
profiled, before any analysis runs) and the data summary the SSE event
carries to the UI.
"""
from __future__ import annotations

import pytest

from src.analysis.agents.base import DataProfile
from src.analysis.agents.base.state import AnalysisState, DatasetInfo, FileEntry, JobStatus
from src.analysis.agents.eda.output import EDAResult
from src.analysis.agents.effect_estimator.output import TreatmentEffectResult
from src.analysis.orchestrator.base import (
    _build_gate_payload,
    park_for_approval,
    should_pause_for_approval,
)
from src.domain.approval import HumanApproval


def _profile(**overrides) -> DataProfile:
    base = dict(
        n_samples=614,
        n_features=11,
        feature_names=["treat", "age", "re78"],
        feature_types={"treat": "binary"},
        missing_values={"treat": 0},
        numeric_stats={},
        categorical_stats={},
        treatment_candidates=["treat"],
        outcome_candidates=["re78"],
        potential_confounders=["age"],
    )
    base.update(overrides)
    return DataProfile(**base)


def _state(**overrides) -> AnalysisState:
    state = AnalysisState(
        job_id="job-1",
        dataset_info=DatasetInfo(url="kaggle.com/x"),
        treatment_variable="t",
        outcome_variable="y",
    )
    for k, v in overrides.items():
        setattr(state, k, v)
    return state


# --- should_pause_for_approval truth table ---------------------------------


def test_no_profile_does_not_pause():
    # Data not profiled yet: nothing to review.
    assert should_pause_for_approval(_state()) is False


def test_profiled_pauses_when_no_approval():
    assert should_pause_for_approval(_state(data_profile=_profile())) is True


def test_prior_approved_decision_skips_pause():
    state = _state(data_profile=_profile(), human_approval=HumanApproval.approve())
    assert should_pause_for_approval(state) is False


def test_prior_rejected_decision_does_not_skip_pause():
    # Defensive: a REJECTED record with the job not yet failed must still hold
    # the gate, not silently proceed.
    state = _state(
        data_profile=_profile(),
        human_approval=HumanApproval.reject(reason="wrong data"),
    )
    assert should_pause_for_approval(state) is True


def test_eda_started_does_not_pause():
    # EDA landing means we have resumed past the data gate.
    state = _state(data_profile=_profile(), eda_result=EDAResult(data_quality_score=80.0))
    assert should_pause_for_approval(state) is False


def test_estimation_in_progress_does_not_pause():
    state = _state(
        data_profile=_profile(),
        treatment_effects=[
            TreatmentEffectResult(
                method="ols",
                estimand="ATE",
                estimate=1.0,
                std_error=0.1,
                ci_lower=0.8,
                ci_upper=1.2,
                p_value=0.01,
            )
        ],
    )
    assert should_pause_for_approval(state) is False


# --- _build_gate_payload shape --------------------------------------------


def test_gate_payload_carries_treatment_and_outcome():
    payload = _build_gate_payload(_state(data_profile=_profile()))
    assert payload["treatment_variable"] == "t"
    assert payload["outcome_variable"] == "y"


def test_gate_payload_data_summary_pulls_shape_and_candidates():
    payload = _build_gate_payload(_state(data_profile=_profile()))
    summary = payload["data_summary"]
    assert summary["n_samples"] == 614
    assert summary["n_features"] == 11
    assert summary["treatment_candidates"] == ["treat"]
    assert summary["outcome_candidates"] == ["re78"]


def test_gate_payload_lists_downloaded_files():
    state = _state(data_profile=_profile())
    state.dataset_info.files = [
        FileEntry(name="lalonde.csv", size_bytes=9400, format="csv", used=True),
        FileEntry(name="notes.txt", size_bytes=64, format="txt", used=False),
    ]
    payload = _build_gate_payload(state)
    assert [f["name"] for f in payload["files"]] == ["lalonde.csv", "notes.txt"]
    assert payload["files"][0]["used"] is True


def test_gate_payload_data_summary_none_before_profile():
    payload = _build_gate_payload(_state())
    assert payload["data_summary"] is None
    assert payload["files"] == []


# --- park_for_approval async ----------------------------------------------


@pytest.mark.asyncio
async def test_park_sets_status_emits_event_and_calls_callback():
    state = _state(data_profile=_profile())

    captured: list[AnalysisState] = []

    async def cb(s: AnalysisState) -> None:
        captured.append(s)

    returned = await park_for_approval(state, status_callback=cb)
    assert returned is state
    assert state.status == JobStatus.AWAITING_APPROVAL
    assert len(captured) == 1 and captured[0] is state
    assert any(e["event_type"] == "approval_required" for e in state.sse_events)
    gate_event = next(e for e in state.sse_events if e["event_type"] == "approval_required")
    assert gate_event["data"]["treatment_variable"] == "t"
    assert gate_event["data"]["data_summary"]["n_samples"] == 614


@pytest.mark.asyncio
async def test_park_without_callback_still_sets_status_and_emits_event():
    state = _state(data_profile=_profile())
    await park_for_approval(state, status_callback=None)
    assert state.status == JobStatus.AWAITING_APPROVAL
    assert any(e["event_type"] == "approval_required" for e in state.sse_events)
