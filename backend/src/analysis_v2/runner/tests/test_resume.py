"""The plan gate round trip: park, edit, confirm or reject, resume."""
from __future__ import annotations

import asyncio

import pytest

from src.analysis_v2.agents.profiling.tools import build_profile_summary
from src.analysis_v2.core import AnalysisRunState, AnalysisStage, RunStatus
from src.analysis_v2.evals.fixtures import generators
from src.analysis_v2.persistence import load_run, save_run
from src.analysis_v2.runner.resume import (
    InvalidPlanEdits,
    NotWaitingForUser,
    apply_plan_decision,
)
from src.analysis_v2.spec import (
    CausalSpec,
    Confidence,
    ConfirmationCard,
    ConfirmationItem,
    DesignCandidate,
    MethodLane,
    PlanCritique,
    PlanGateStatus,
    QuestionType,
    VariableRef,
)
from src.analysis_v2.state import AnalysisState, DatasetInfo, JobStatus

JOB_ID = "job-resume"


@pytest.fixture
def storage_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path))
    from src.config import settings as settings_mod
    from src.storage.local_storage import reset_local_storage_client

    settings_mod.get_settings.cache_clear()
    reset_local_storage_client()
    yield tmp_path
    settings_mod.get_settings.cache_clear()
    reset_local_storage_client()


class StubManager:
    def __init__(self):
        from src.storage.local_storage import get_local_storage_client

        self.firestore = get_local_storage_client()
        self._jobs_lock = asyncio.Lock()
        self._running_jobs: dict = {}
        self._active_states: dict = {}


def _stage_dataset(frame) -> None:
    from src.domain.dataset_manifest import DatasetManifest, ManifestFile
    from src.storage.job_data import job_normalized_dir, job_raw_dir, write_manifest

    raw = job_raw_dir(JOB_ID)
    frame.to_csv(raw / "rdd.csv", index=False)
    frame.to_parquet(job_normalized_dir(JOB_ID) / "rdd.csv.parquet", index=False)
    write_manifest(
        JOB_ID,
        DatasetManifest(
            job_id=JOB_ID,
            kaggle_url="https://www.kaggle.com/datasets/x/y",
            raw_dir=str(raw),
            winner="rdd.csv",
            files=[
                ManifestFile(
                    name="rdd.csv", relative_path="raw/rdd.csv",
                    size_bytes=1, format="csv", sha256="0" * 64,
                    used=True, normalized_path="normalized/rdd.csv.parquet",
                    tabular=True,
                )
            ],
        ),
    )


async def _seed_parked_rdd() -> tuple[AnalysisRunState, AnalysisState, StubManager]:
    frame = generators.scholarship_rdd()
    _stage_dataset(frame)
    run = AnalysisRunState(
        job_id=JOB_ID,
        causal_question="Did crossing the scholarship cutoff raise outcomes?",
        status=RunStatus.WAITING_FOR_USER,
    )
    run.current_state = AnalysisStage.S5_PLAN_CRITIQUED
    run.causal_spec = CausalSpec(
        question_type=QuestionType.RDD,
        confidence=Confidence.HIGH,
        outcome=VariableRef(column="outcome_sharp"),
        treatment=VariableRef(column="scholarship_sharp"),
        running_variable=VariableRef(column="score"),
    )
    run.dataset_profile = build_profile_summary(frame)
    run.design_candidates = [
        DesignCandidate(
            lane=MethodLane.RDD, design_label="rdd", confidence=Confidence.MEDIUM,
            rationale="cutoff unconfirmed", missing_requirements=["cutoff_value"],
        )
    ]
    run.plan_critique = PlanCritique(
        status=PlanGateStatus.NEEDS_USER_CONFIRMATION,
        reasons=["cutoff unconfirmed"],
        confirmation_card=ConfirmationCard(
            headline="Confirm the rdd plan",
            plan_summary="rdd at an unconfirmed cutoff",
            items=[
                ConfirmationItem(
                    field="cutoff_value", label="Cutoff", required=True,
                    why="needed for assignment",
                )
            ],
        ),
    )
    await save_run(run)

    state = AnalysisState(
        job_id=JOB_ID,
        dataset_info=DatasetInfo(url="https://www.kaggle.com/datasets/x/y"),
        causal_question=run.causal_question,
        status=JobStatus.WAITING_FOR_USER,
    )
    manager = StubManager()
    await manager.firestore.create_job(state)
    await manager.firestore.save_parked_state(state)
    return run, state, manager


async def test_confirm_with_the_cutoff_records_s6_and_relaunches(storage_dir, monkeypatch):
    _, _, manager = await _seed_parked_rdd()

    result = await apply_plan_decision(
        JOB_ID, manager, decision="confirm", edits={"cutoff_value": "50"}
    )
    assert result["status"] == "running_analysis"

    task = manager._running_jobs.get(JOB_ID)
    assert task is not None
    await asyncio.wait_for(task, timeout=30)

    run = await load_run(JOB_ID)
    s6 = [e for e in run.state_events
          if e.to_state == AnalysisStage.S6_USER_CONFIRMED_OR_AUTO_APPROVED]
    assert len(s6) == 1
    assert "cutoff_value" in s6[0].warnings[0]
    assert run.causal_spec.cutoff_value == 50.0
    assert run.method_plan.settings["cutoff"] == 50.0
    assert run.selected_design.lane == MethodLane.RDD
    # the resumed spine ran the rdd lane and recovered the built-in jump
    assert run.estimate_result is not None
    jump = next(e for e in run.estimate_result.effects if e.estimand == "itt_jump")
    assert abs(jump.estimate - 8.0) < 2.0
    # then stopped honestly at the next frontier (S8)
    assert "s10_report_notebook_created" in (run.error_message or "")


async def test_reject_fails_the_run_with_the_reason(storage_dir):
    _, _, manager = await _seed_parked_rdd()

    result = await apply_plan_decision(
        JOB_ID, manager, decision="reject", reason="wrong outcome column"
    )
    assert result["status"] == "failed"
    run = await load_run(JOB_ID)
    assert run.status == RunStatus.FAILED
    assert "wrong outcome column" in run.error_message


async def test_invalid_edits_are_refused_with_a_reason(storage_dir):
    _, _, manager = await _seed_parked_rdd()

    with pytest.raises(InvalidPlanEdits, match="not a dataset column"):
        await apply_plan_decision(
            JOB_ID, manager, decision="confirm",
            edits={"outcome": "nope", "cutoff_value": "50"},
        )
    with pytest.raises(InvalidPlanEdits, match="unknown edit field"):
        await apply_plan_decision(
            JOB_ID, manager, decision="confirm",
            edits={"hacks": "1", "cutoff_value": "50"},
        )
    # confirming without the required cutoff is refused by the card itself
    with pytest.raises(InvalidPlanEdits, match="required items"):
        await apply_plan_decision(JOB_ID, manager, decision="confirm", edits={})
    with pytest.raises(InvalidPlanEdits, match="disabled"):
        await apply_plan_decision(
            JOB_ID, manager, decision="confirm",
            edits={"cutoff_value": "50", "lane": "iv"},
        )
    # nothing was committed by the refused attempts
    run = await load_run(JOB_ID)
    assert run.status == RunStatus.WAITING_FOR_USER
    assert run.causal_spec.outcome.column == "outcome_sharp"


async def test_a_run_not_parked_at_the_gate_is_a_conflict(storage_dir):
    run, _, manager = await _seed_parked_rdd()
    run.status = RunStatus.RUNNING
    await save_run(run)

    with pytest.raises(NotWaitingForUser):
        await apply_plan_decision(JOB_ID, manager, decision="confirm", edits={})
