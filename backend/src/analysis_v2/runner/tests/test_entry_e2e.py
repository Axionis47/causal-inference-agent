"""End-to-end spine run over the LaLonde fixture with a stubbed LLM.

This is the workflow harness: a real CONFIRMED record, a real manifest
with a normalized parquet, the real intake/profiling/design agents, and
the real persistence layer. Only the LLM is stubbed. The spine runs
S1 -> S2 -> S3 and stops honestly at the S4 frontier.
"""
from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.analysis_v2.agents.intake.schema import IntakeDraft
from src.analysis_v2.core import AnalysisStage
from src.analysis_v2.persistence import load_run
from src.analysis_v2.runner import start
from src.analysis_v2.spec import Confidence, MethodLane, QuestionType
from src.analysis_v2.state import AnalysisState, DatasetInfo, JobStatus
from src.domain.dataset_manifest import DatasetManifest, ManifestFile
from src.storage.job_data import job_normalized_dir, job_raw_dir, write_manifest

FIXTURE = (
    Path(__file__).resolve().parents[2] / "evals" / "fixtures" / "data" / "lalonde.csv"
)
JOB_ID = "job-e2e"


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


class StubLLM:
    async def generate_structured(self, prompt, response_schema, system_instruction=None):
        return IntakeDraft(
            question_type=QuestionType.BINARY_TREATMENT,
            confidence=Confidence.HIGH,
            outcome_column="re78",
            treatment_column="treat",
            candidate_confounders=["age", "educ", "re74", "re75"],
            reasoning_summary="Binary treatment question mapping treat to re78.",
        )

    async def generate(self, prompt, system_instruction=None, tools=None):
        class R:
            text = "Groups differ in size and prior earnings; adjustment looks necessary."

        return R()


def _stage_dataset(storage_dir: Path) -> None:
    raw = job_raw_dir(JOB_ID)
    shutil.copy(FIXTURE, raw / "lalonde.csv")
    frame = pd.read_csv(FIXTURE)
    normalized = job_normalized_dir(JOB_ID) / "lalonde.csv.parquet"
    frame.to_parquet(normalized, index=False)
    manifest = DatasetManifest(
        job_id=JOB_ID,
        kaggle_url="https://www.kaggle.com/datasets/samuelzakouri/lalonde",
        dataset_ref="samuelzakouri/lalonde",
        raw_dir=str(raw),
        winner="lalonde.csv",
        files=[
            ManifestFile(
                name="lalonde.csv",
                relative_path="raw/lalonde.csv",
                size_bytes=FIXTURE.stat().st_size,
                format="csv",
                sha256="0" * 64,
                n_rows=len(frame),
                n_columns=len(frame.columns),
                columns=[str(c) for c in frame.columns],
                used=True,
                normalized_path="normalized/lalonde.csv.parquet",
                tabular=True,
            )
        ],
    )
    write_manifest(JOB_ID, manifest)


def _confirmed_state() -> AnalysisState:
    return AnalysisState(
        job_id=JOB_ID,
        dataset_info=DatasetInfo(
            url="https://www.kaggle.com/datasets/samuelzakouri/lalonde",
            name="lalonde",
            user_provided_context="NSW job training study; re74/re75 are prior earnings.",
        ),
        causal_question="Does participating in the job training program increase 1978 earnings?",
        status=JobStatus.CONFIRMED,
        ignored_columns=["Unnamed: 0"],
    )


async def test_spine_runs_intake_profiling_design_and_stops_at_the_frontier(
    storage_dir, monkeypatch
):
    monkeypatch.setattr(
        "src.analysis_v2.agents.intake.agent.get_llm_client", lambda: StubLLM()
    )
    monkeypatch.setattr(
        "src.analysis_v2.agents.targeted_eda.agent.get_llm_client", lambda: StubLLM()
    )
    _stage_dataset(storage_dir)
    state = _confirmed_state()
    manager = StubManager()
    await manager.firestore.create_job(state)

    ack = await start(state, manager)
    assert ack == {"resumed": True, "status": "running_analysis"}

    task = manager._running_jobs[JOB_ID]
    await asyncio.wait_for(task, timeout=30)

    # the spine stopped at the S8 frontier and said so honestly
    run = await load_run(JOB_ID)
    assert run is not None
    assert run.current_state == AnalysisStage.S8_DIAGNOSTICS_SENSITIVITY_COMPLETE
    assert run.status.value == "failed"
    assert "s9_claim_critiqued" in run.error_message

    # the six stages each passed and committed their slots
    assert [r.agent for r in run.agent_runs] == [
        "intake", "profiling", "design_detection", "targeted_eda",
        "plan_critic", "method_lane", "diagnostics_sensitivity",
    ]
    assert all(r.status.value in ("passed", "warning") for r in run.agent_runs)
    assert run.causal_spec.outcome.column == "re78"
    assert run.dataset_profile.n_rows == 614
    # the human-ignored index column never reached the agents
    assert "Unnamed: 0" not in run.dataset_profile.column_names()
    assert run.design_candidates[0].lane == MethodLane.OBSERVATIONAL
    assert run.tool_eligibility is not None
    assert run.eda_summary is not None
    assert run.eda_summary.check("covariate_balance") is not None
    # the fully-resolved high-confidence plan auto-approved through S6
    assert run.plan_critique.status.value == "pass_auto_approved"
    assert run.method_plan.estimator == "regression_adjustment"
    # and the lane actually estimated: the adjusted LaLonde effect is positive
    assert run.estimate_result is not None
    assert 500 < run.estimate_result.primary.estimate < 3000
    assert run.sensitivity_result is not None
    assert len(run.state_events) == 8  # S1..S5, the orchestrator-owned S6, S7, S8
    assert run.state_version == 8

    # artifacts persisted on disk and registered
    for artifact in run.artifact_registry.artifacts:
        assert (storage_dir / JOB_ID / "analysis" / artifact.path).exists()

    # live wire: SSE buffer carries the analysis vocabulary with headlines
    kinds = [e["event_type"] for e in state.sse_events]
    assert "analysis_started" in kinds
    assert "analysis_stage_started" in kinds
    assert "analysis_agent_completed" in kinds
    assert "analysis_failed" in kinds  # frontier, until S4+ exist
    assert all("headline" in e["data"] for e in state.sse_events)

    # public job status reflects the frontier failure; live tables drained
    assert state.status == JobStatus.FAILED
    assert manager._active_states == {}
    assert manager._running_jobs == {} or JOB_ID not in manager._running_jobs
