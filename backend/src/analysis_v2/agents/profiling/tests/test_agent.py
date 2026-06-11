"""ProfilingAgent end-to-end on the LaLonde fixture and the empty gate."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.analysis_v2.agents.base import AgentCtx
from src.analysis_v2.agents.profiling import ProfilingAgent
from src.analysis_v2.core import AnalysisRunState, GateStatus

FIXTURE = (
    Path(__file__).resolve().parents[3] / "evals" / "fixtures" / "data" / "lalonde.csv"
)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path))
    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    yield tmp_path
    settings_mod.get_settings.cache_clear()


def _ctx(frame: pd.DataFrame, job_id: str = "job-prof") -> AgentCtx:
    run = AnalysisRunState(
        job_id=job_id,
        causal_question="Does the training program raise 1978 earnings?",
    )
    return AgentCtx(job_id=job_id, run=run, frame=frame)


async def test_lalonde_profile_commits_summary_and_persists_artifacts(data_dir):
    ctx = _ctx(pd.read_csv(FIXTURE))
    agent = ProfilingAgent()

    result = await agent.execute(ctx)
    agent.commit(ctx.run, result.output)

    assert result.gate.status == GateStatus.ADVANCE
    summary = ctx.run.dataset_profile
    assert summary is not None
    assert summary.n_rows == 614
    assert "Unnamed: 0" in summary.id_like_columns
    assert summary.column("treat").semantic_type == "binary"
    assert summary.column("re78").semantic_type == "numeric"

    # every emitted artifact is registered AND exists on disk
    registered = ctx.run.artifact_registry
    assert set(result.artifact_ids).issubset(set(registered.ids()))
    analysis_dir = data_dir / "job-prof" / "analysis"
    for artifact in registered.artifacts:
        assert (analysis_dir / artifact.path).exists(), artifact.path

    profile_payload = json.loads(
        (analysis_dir / "profiling/dataset_profile.json").read_text()
    )
    assert profile_payload["n_rows"] == 614
    preview = pd.read_csv(analysis_dir / "profiling/preview_table.csv")
    assert len(preview) == 50
    png = (analysis_dir / "profiling/numeric_distributions.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


async def test_clean_data_advances_without_warnings(data_dir):
    frame = pd.DataFrame(
        {"x": [i + 0.5 for i in range(90)], "y": [i % 2 for i in range(90)]}
    )
    result = await ProfilingAgent().execute(_ctx(frame, "job-clean"))
    assert result.gate.status == GateStatus.ADVANCE
    assert result.warnings == []
    assert "No data quality concerns" in result.public_summary


async def test_empty_dataset_fails_the_gate(data_dir):
    result = await ProfilingAgent().execute(_ctx(pd.DataFrame(), "job-empty"))
    assert result.gate.status == GateStatus.FAIL
    assert result.gate.hard_failures == ["dataset has no rows or no columns"]
    assert result.output is None
