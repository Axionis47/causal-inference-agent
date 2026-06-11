"""LocalStorageClient parked-state round-trip.

The data-review gate persists the full AnalysisState mid-flow, then reloads it
when the analyst edits inputs, confirms, or rejects. These tests pin that the
round-trip preserves the fields the gate reads back (question, profile,
approval), that delete is idempotent, and that the schema-version guard
refuses incompatible payloads.

Firestore round-trip lives in integration tests (needs emulator); the
local backend covers shape and idempotency.
"""
from __future__ import annotations

import pytest

from src.analysis_v2.state import (
    SCHEMA_VERSION,
    AnalysisState,
    DataProfile,
    DatasetInfo,
    JobStatus,
    StaleParkedState,
)
from src.domain.approval import HumanApproval
from src.storage.local_storage import LocalStorageClient


@pytest.fixture
def storage(tmp_path, monkeypatch) -> LocalStorageClient:
    """LocalStorageClient pointed at a temp directory so each test is isolated."""
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path))
    # Clear cached Settings so the new env var takes effect
    from src.config import settings as settings_mod
    settings_mod.get_settings.cache_clear()
    client = LocalStorageClient()
    yield client
    settings_mod.get_settings.cache_clear()


def _gate_state(job_id: str = "job-1") -> AnalysisState:
    return AnalysisState(
        job_id=job_id,
        dataset_info=DatasetInfo(
            url="https://www.kaggle.com/datasets/o/n",
            user_provided_context="initial notes",
        ),
        causal_question="Does training raise income?",
        status=JobStatus.AWAITING_APPROVAL,
        data_profile=DataProfile(
            n_samples=614,
            n_features=2,
            feature_names=["treat", "re78"],
            feature_types={"treat": "binary", "re78": "numeric"},
            missing_values={"treat": 0, "re78": 3},
            has_time_dimension=False,
        ),
    )


# --- save → load round-trip preserves the gate snapshot --------------------


@pytest.mark.asyncio
async def test_round_trip_preserves_question_profile_and_context(storage):
    original = _gate_state()
    await storage.save_parked_state(original)
    restored = await storage.load_parked_state("job-1")

    assert restored is not None
    assert restored.job_id == "job-1"
    assert restored.status == JobStatus.AWAITING_APPROVAL
    assert restored.causal_question == "Does training raise income?"
    assert restored.dataset_info.user_provided_context == "initial notes"
    assert restored.data_profile.feature_types == {
        "treat": "binary",
        "re78": "numeric",
    }
    assert restored.data_profile.missing_values["re78"] == 3


@pytest.mark.asyncio
async def test_round_trip_preserves_human_approval_when_set(storage):
    """Confirm case: the APPROVED decision is attached when the worker writes
    the confirmed record back, and must survive the reload."""
    state = _gate_state()
    state.human_approval = HumanApproval.approve(
        granted_by="r", appended_context="extra notes"
    )
    state.status = JobStatus.CONFIRMED
    await storage.save_parked_state(state)
    restored = await storage.load_parked_state("job-1")
    assert restored.human_approval is not None
    assert restored.human_approval.granted_by == "r"
    assert restored.human_approval.appended_context == "extra notes"
    assert restored.is_approved() is True


# --- load missing returns None --------------------------------------------


@pytest.mark.asyncio
async def test_load_missing_returns_none(storage):
    assert await storage.load_parked_state("does-not-exist") is None


# --- delete is idempotent --------------------------------------------------


@pytest.mark.asyncio
async def test_delete_returns_true_when_present_false_when_missing(storage):
    state = _gate_state()
    await storage.save_parked_state(state)
    assert await storage.delete_parked_state("job-1") is True
    assert await storage.delete_parked_state("job-1") is False
    assert await storage.load_parked_state("job-1") is None


# --- save is keyed by job_id (multiple jobs coexist) -----------------------


@pytest.mark.asyncio
async def test_two_jobs_coexist_in_store(storage):
    a = _gate_state(job_id="job-a")
    b = _gate_state(job_id="job-b")
    await storage.save_parked_state(a)
    await storage.save_parked_state(b)
    restored_a = await storage.load_parked_state("job-a")
    restored_b = await storage.load_parked_state("job-b")
    assert restored_a.job_id == "job-a"
    assert restored_b.job_id == "job-b"


# --- save overwrites last writer wins -------------------------------------


@pytest.mark.asyncio
async def test_resave_overwrites_existing_entry(storage):
    state = _gate_state()
    await storage.save_parked_state(state)
    state.causal_question = "Does training raise earnings instead?"
    await storage.save_parked_state(state)
    restored = await storage.load_parked_state("job-1")
    assert restored.causal_question == "Does training raise earnings instead?"


# --- schema_version guard --------------------------------------------------


@pytest.mark.asyncio
async def test_round_trip_carries_the_schema_version(storage):
    await storage.save_parked_state(_gate_state())
    restored = await storage.load_parked_state("job-1")
    assert restored.schema_version == SCHEMA_VERSION


def test_load_parked_rejects_an_incompatible_schema_version():
    payload = _gate_state().model_dump()
    payload["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(StaleParkedState):
        AnalysisState.load_parked(payload)


def test_load_parked_accepts_a_payload_missing_schema_version():
    # States parked before the version field existed default to the current
    # version and load cleanly (backward compatible).
    payload = _gate_state().model_dump()
    payload.pop("schema_version", None)
    restored = AnalysisState.load_parked(payload)
    assert restored.job_id == "job-1"
