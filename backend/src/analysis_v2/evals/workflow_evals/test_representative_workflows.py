"""Full-spine workflow evals over the representative fixtures.

Each case stages its dataset like the input slice would, stubs only the
LLM (a per-case intake draft from the manifest's expected fields), and
drives all ten agents through POST-run semantics: S1..S11 plus the
terminal S12, ending in a verified notebook. Assertions are structural
and truth-banded per the manifest; exact coefficients are asserted only
for synthetic ground truth. Hermetic: no Kaggle, no network.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from src.analysis_v2.agents.intake.schema import IntakeDraft
from src.analysis_v2.core import AnalysisStage, RunStatus
from src.analysis_v2.evals.fixtures import generators
from src.analysis_v2.persistence import load_run
from src.analysis_v2.runner import start
from src.analysis_v2.spec import Confidence, MethodLane, QuestionType
from src.analysis_v2.state import AnalysisState, DatasetInfo, JobStatus
from src.domain.dataset_manifest import DatasetManifest, ManifestFile
from src.storage.job_data import job_normalized_dir, job_raw_dir, write_manifest

DATA = Path(__file__).resolve().parents[1] / "fixtures" / "data"


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


def _stub_llm(monkeypatch, draft: IntakeDraft) -> None:
    class StubLLM:
        async def generate_structured(self, prompt, response_schema, system_instruction=None):
            return draft

        async def generate(self, prompt, system_instruction=None, tools=None):
            class R:
                text = "Descriptive patterns only; see the computed checks."

            return R()

    monkeypatch.setattr(
        "src.analysis_v2.agents.intake.agent.get_llm_client", lambda: StubLLM()
    )
    monkeypatch.setattr(
        "src.analysis_v2.agents.targeted_eda.agent.get_llm_client", lambda: StubLLM()
    )


def _stage(job_id: str, frame: pd.DataFrame) -> None:
    raw = job_raw_dir(job_id)
    frame.to_csv(raw / "data.csv", index=False)
    frame.to_parquet(job_normalized_dir(job_id) / "data.csv.parquet", index=False)
    write_manifest(
        job_id,
        DatasetManifest(
            job_id=job_id,
            kaggle_url="https://www.kaggle.com/datasets/eval/case",
            raw_dir=str(raw),
            winner="data.csv",
            files=[
                ManifestFile(
                    name="data.csv", relative_path="raw/data.csv", size_bytes=1,
                    format="csv", sha256="0" * 64, used=True,
                    normalized_path="normalized/data.csv.parquet", tabular=True,
                )
            ],
        ),
    )


async def _drive(job_id: str, frame, question: str) -> tuple:
    """Run the spine; when the plan gate parks, confirm with card defaults
    (the human-in-loop round trip) and let it resume."""
    _stage(job_id, frame)
    state = AnalysisState(
        job_id=job_id,
        dataset_info=DatasetInfo(url="https://www.kaggle.com/datasets/eval/case"),
        causal_question=question,
        status=JobStatus.CONFIRMED,
    )
    manager = StubManager()
    await manager.firestore.create_job(state)
    await start(state, manager)
    await asyncio.wait_for(manager._running_jobs[job_id], timeout=300)

    run = await load_run(job_id)
    if run.status == RunStatus.WAITING_FOR_USER:
        from src.analysis_v2.runner.resume import apply_plan_decision

        await apply_plan_decision(job_id, manager, decision="confirm", edits={})
        await asyncio.wait_for(manager._running_jobs[job_id], timeout=300)
        run = await load_run(job_id)
        record = await manager.firestore.get_job(job_id)
        state.status = JobStatus(record["status"])
    return run, state


@dataclass
class Case:
    case_id: str
    frame_fn: object
    question: str
    draft: IntakeDraft
    expected_lane: MethodLane
    estimand: str
    truth_band: tuple[float, float] | None = None
    extra_warning: str | None = None


CASES = [
    Case(
        case_id="synthetic-did-panel",
        frame_fn=generators.did_panel,
        question="Did the program raise outcomes for the treated group after period 4?",
        draft=IntakeDraft(
            question_type=QuestionType.DID, confidence=Confidence.HIGH,
            outcome_column="outcome",
            treatment_column=None, treatment_derived=True,
            treatment_clue="treated group after period 4",
            time_column="period", group_column="group",
            reasoning_summary="A policy-style group/time comparison.",
        ),
        expected_lane=MethodLane.DID,
        estimand="att",
        truth_band=(1.4, 2.6),
    ),
    Case(
        case_id="synthetic-iv-late",
        frame_fn=generators.synthetic_iv,
        question="Does x increase y, using z as an instrument?",
        draft=IntakeDraft(
            question_type=QuestionType.IV, confidence=Confidence.HIGH,
            outcome_column="y", treatment_column="x", instrument_column="z",
            candidate_confounders=["c1", "c2"],
            reasoning_summary="An instrumented effect question.",
        ),
        expected_lane=MethodLane.IV,
        estimand="late",
        truth_band=(1.65, 2.35),
    ),
    Case(
        case_id="synthetic-mediation",
        frame_fn=generators.mediation,
        question="Does the exercise program reduce disease risk through weight change?",
        draft=IntakeDraft(
            question_type=QuestionType.MEDIATION, confidence=Confidence.HIGH,
            outcome_column="disease_risk", treatment_column="exercise_program",
            mediator_column="weight_change",
            candidate_confounders=["baseline_health"],
            reasoning_summary="A pathway question through a mediator.",
        ),
        expected_lane=MethodLane.MEDIATION,
        estimand="total",
        truth_band=(0.45, 0.75),
        extra_warning="timing",
    ),
    Case(
        case_id="heart-failure-survival",
        frame_fn=lambda: pd.read_csv(DATA / "heart_failure.csv"),
        question="Does higher ejection fraction delay death in heart failure patients?",
        draft=IntakeDraft(
            question_type=QuestionType.SURVIVAL, confidence=Confidence.HIGH,
            outcome_column="time", treatment_column="ejection_fraction",
            duration_column="time", event_column="DEATH_EVENT",
            candidate_confounders=["age", "serum_creatinine"],
            reasoning_summary="A time-to-event question with censoring.",
        ),
        expected_lane=MethodLane.SURVIVAL,
        estimand="hazard_ratio",
        truth_band=(0.5, 0.99),  # protective: hr below 1
    ),
    Case(
        case_id="advertising-dose-response",
        frame_fn=lambda: pd.read_csv(DATA / "advertising.csv").drop(columns=["Unnamed: 0"]),
        question="Does more TV advertising spend increase sales?",
        draft=IntakeDraft(
            question_type=QuestionType.DOSE_RESPONSE, confidence=Confidence.HIGH,
            outcome_column="Sales ($)", treatment_column="TV Ad Budget ($)",
            treatment_is_continuous=True,
            candidate_confounders=["Radio Ad Budget ($)", "Newspaper Ad Budget ($)"],
            reasoning_summary="A continuous dose-response question.",
        ),
        expected_lane=MethodLane.OBSERVATIONAL,
        estimand="ate",
        truth_band=(0.035, 0.06),  # textbook 0.046
    ),
    Case(
        case_id="website-visitors-its-step",
        frame_fn=lambda: generators.website_visitors_step(
            DATA / "daily_website_visitors.csv"
        ),
        question="Did the site change on 2019-04-01 increase daily visits?",
        draft=IntakeDraft(
            question_type=QuestionType.TIME_SERIES_INTERVENTION,
            confidence=Confidence.HIGH,
            outcome_column="visits", time_column="Date",
            intervention_date="2019-04-01",
            treatment_derived=True, treatment_clue="post-change period",
            reasoning_summary="A single-series intervention question.",
        ),
        expected_lane=MethodLane.TIME_SERIES,
        estimand="level_shift",
        truth_band=(200.0, 1400.0),
        extra_warning="before/after",
    ),
]


@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
async def test_representative_case_completes_with_a_verified_notebook(
    case: Case, storage_dir, monkeypatch
):
    _stub_llm(monkeypatch, case.draft)
    run, state = await _drive(f"wf-{case.case_id[:12]}", case.frame_fn(), case.question)

    assert run.status == RunStatus.COMPLETED, run.error_message
    assert run.current_state == AnalysisStage.S12_JOB_COMPLETE
    assert state.status == JobStatus.COMPLETED

    # the right lane ran and the estimate lands in the truth band
    assert run.estimate_result.lane == case.expected_lane
    effect = next(
        e for e in run.estimate_result.effects if e.estimand == case.estimand
    )
    if case.truth_band is not None:
        lo, hi = case.truth_band
        assert lo <= effect.estimate <= hi, (case.case_id, effect.estimate)

    # honesty surfaces: claims bounded, limitations present
    assert run.claim_critique is not None
    assert run.claim_critique.limitations
    if case.extra_warning is not None:
        joined = " ".join(run.claim_critique.limitations).lower()
        assert case.extra_warning in joined

    # the notebook executed and the artifact files exist on disk
    assert run.notebook_verification.notebook_status.value == "verified_running"
    for artifact in run.artifact_registry.artifacts:
        assert (storage_dir / run.job_id / "analysis" / artifact.path).exists()


async def test_a_completed_job_reopens_through_the_api_with_full_tiles(
    storage_dir, monkeypatch
):
    case = CASES[0]
    _stub_llm(monkeypatch, case.draft)
    run, _ = await _drive("wf-reopen", case.frame_fn(), case.question)
    assert run.status == RunStatus.COMPLETED

    from fastapi.testclient import TestClient

    from src.api.main import app

    client = TestClient(app)
    view = client.get("/jobs/wf-reopen/analysis").json()
    assert view["status"] == "completed"
    assert view["current_state"] == "s12_job_complete"
    assert len(view["agents"]) == 10
    assert all(a["status"] in ("passed", "warning") for a in view["agents"])
    assert view["notebook"]["status"] == "verified_running"
    assert view["costs"]["total_tool_calls"] >= 0

    # artifact bytes serve through the same surface the tiles link to
    notebook_bytes = client.get(
        "/jobs/wf-reopen/analysis/artifacts/notebook/verified"
    )
    assert notebook_bytes.status_code == 200
    assert b"Causal analysis notebook" in notebook_bytes.content
