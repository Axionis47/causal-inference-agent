"""Shared full-spine eval harness: stage a dataset like the input slice,
stub only the LLM (a per-case intake draft), drive S1..S12 through the
production entry, and confirm the plan gate when the run parks. Hermetic:
no Kaggle, no network."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.analysis_v2.agents.intake.schema import IntakeDraft
from src.analysis_v2.core import RunStatus
from src.analysis_v2.persistence import load_run
from src.analysis_v2.runner import start
from src.analysis_v2.spec import MethodLane
from src.analysis_v2.state import AnalysisState, DatasetInfo, JobStatus
from src.domain.dataset_manifest import DatasetManifest, ManifestFile
from src.storage.job_data import job_normalized_dir, job_raw_dir, write_manifest

DATA = Path(__file__).resolve().parents[1] / "fixtures" / "data"


class StubManager:
    def __init__(self):
        from src.storage.local_storage import get_local_storage_client

        self.firestore = get_local_storage_client()
        self._jobs_lock = asyncio.Lock()
        self._running_jobs: dict = {}
        self._active_states: dict = {}


def stub_llm(monkeypatch, draft: IntakeDraft) -> None:
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
    # no chat_with_tools on the stub: the investigator takes its degraded path
    monkeypatch.setattr(
        "src.analysis_v2.agents.investigator.agent.get_llm_client", lambda: StubLLM()
    )


def stage(job_id: str, frame: pd.DataFrame) -> None:
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


async def drive(job_id: str, frame, question: str) -> tuple:
    """Run the spine; when the plan gate parks, confirm with card defaults
    (the human-in-loop round trip) and let it resume."""
    stage(job_id, frame)
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
