"""Run-state records through the local storage client."""
from __future__ import annotations

import json

import pytest

from src.analysis_v2.core import (
    AnalysisRunState,
    AnalysisStage,
    Artifact,
    ArtifactKind,
    GateResult,
    StaleRunState,
)
from src.analysis_v2.persistence import delete_run, load_run, save_run
from src.analysis_v2.spec import CausalSpec, QuestionType, VariableRef


@pytest.fixture
def storage_dir(tmp_path, monkeypatch):
    """Local storage on tmp_path; reset client singletons around each test."""
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path))
    monkeypatch.setenv("USE_FIRESTORE", "false")
    from src.config import settings as settings_mod
    from src.storage.local_storage import reset_local_storage_client

    settings_mod.get_settings.cache_clear()
    reset_local_storage_client()
    yield tmp_path
    settings_mod.get_settings.cache_clear()
    reset_local_storage_client()


def _run_state(job_id: str = "job-9") -> AnalysisRunState:
    state = AnalysisRunState(
        job_id=job_id,
        causal_question="Does the training program raise 1978 earnings?",
    )
    state.causal_spec = CausalSpec(
        question_type=QuestionType.BINARY_TREATMENT,
        outcome=VariableRef(column="re78"),
        treatment=VariableRef(column="treat"),
    )
    state.register_artifact(
        Artifact(
            artifact_id="intake/spec",
            kind=ArtifactKind.JSON,
            stage=AnalysisStage.S1_INTAKE_PARSED,
            agent="intake",
            title="Causal spec draft",
            path="intake/causal_spec.json",
            media_type="application/json",
        )
    )
    state.record_transition(
        to_state=AnalysisStage.S1_INTAKE_PARSED,
        agent_name="intake",
        gate_result=GateResult.advance(),
        output_artifacts=["intake/spec"],
    )
    state.record_transition(
        to_state=AnalysisStage.S2_PROFILE_CREATED, agent_name="profiling"
    )
    return state


async def test_save_then_load_round_trips_a_populated_run(storage_dir):
    await save_run(_run_state())
    assert (storage_dir / "analysis_runs.json").exists()

    loaded = await load_run("job-9")
    assert loaded is not None
    assert loaded.causal_spec.question_type == QuestionType.BINARY_TREATMENT
    assert loaded.artifact_registry.ids() == ["intake/spec"]
    assert [e.to_state for e in loaded.state_events] == [
        AnalysisStage.S1_INTAKE_PARSED,
        AnalysisStage.S2_PROFILE_CREATED,
    ]
    assert loaded.state_version == 2


async def test_load_returns_none_for_unknown_job(storage_dir):
    assert await load_run("nope") is None


async def test_saving_twice_overwrites_the_record(storage_dir):
    first = _run_state()
    await save_run(first)
    first.mark_failed("method lane crashed")
    await save_run(first)

    loaded = await load_run("job-9")
    assert loaded.error_message == "method lane crashed"
    store = json.loads((storage_dir / "analysis_runs.json").read_text())
    assert list(store) == ["job-9"]


async def test_delete_returns_true_then_false(storage_dir):
    await save_run(_run_state())
    assert await delete_run("job-9") is True
    assert await delete_run("job-9") is False
    assert await load_run("job-9") is None


async def test_incompatible_schema_version_raises_stale_run_state(storage_dir):
    await save_run(_run_state())
    path = storage_dir / "analysis_runs.json"
    store = json.loads(path.read_text())
    store["job-9"]["schema_version"] = 999
    path.write_text(json.dumps(store))

    with pytest.raises(StaleRunState):
        await load_run("job-9")
