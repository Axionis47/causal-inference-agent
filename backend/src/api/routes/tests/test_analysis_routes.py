"""GET /jobs/{id}/analysis view and artifact bytes."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.analysis_v2.core import (
    AgentRun,
    AgentRunStatus,
    AnalysisRunState,
    AnalysisStage,
    Artifact,
    ArtifactKind,
    GateResult,
    RunStatus,
    TokenUsage,
)
from src.analysis_v2.persistence import save_run, write_artifact_text


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path))
    from src.config import settings as settings_mod
    from src.storage.local_storage import reset_local_storage_client

    settings_mod.get_settings.cache_clear()
    reset_local_storage_client()
    from src.api.main import app

    yield TestClient(app)
    settings_mod.get_settings.cache_clear()
    reset_local_storage_client()


async def _seed_run(job_id: str = "job-api") -> AnalysisRunState:
    run = AnalysisRunState(
        job_id=job_id,
        causal_question="Does treatment raise the outcome?",
        status=RunStatus.RUNNING,
    )
    record = AgentRun(agent="intake", stage=AnalysisStage.S1_INTAKE_PARSED)
    record.start()
    record.public_summary = "Mapped the question."
    record.tokens = TokenUsage(input_tokens=500, output_tokens=80)
    record.cost_usd = 0.004
    record.finish(AgentRunStatus.PASSED)
    run.add_agent_run(record)
    write_artifact_text(job_id, "intake/summary.md", "## Intake\nfine")
    run.register_artifact(
        Artifact(
            artifact_id="intake/summary",
            kind=ArtifactKind.MARKDOWN,
            stage=AnalysisStage.S1_INTAKE_PARSED,
            agent="intake",
            title="Intake summary",
            path="intake/summary.md",
            media_type="text/markdown",
        )
    )
    run.record_transition(
        to_state=AnalysisStage.S1_INTAKE_PARSED,
        agent_name="intake",
        gate_result=GateResult.advance(),
        tokens=record.tokens,
        cost_usd=record.cost_usd,
    )
    await save_run(run)
    return run


async def test_analysis_view_reconstructs_tiles_costs_and_events(client):
    await _seed_run()
    response = client.get("/jobs/job-api/analysis")
    assert response.status_code == 200
    body = response.json()
    assert body["current_state"] == "s1_intake_parsed"
    assert body["stage_index"] == 1 and body["total_stages"] == 14
    agent = body["agents"][0]
    assert agent["agent"] == "intake"
    assert agent["status"] == "passed"
    assert agent["tokens"] == {"input_tokens": 500, "output_tokens": 80}
    assert body["costs"]["total_cost_usd"] == pytest.approx(0.004)
    assert body["artifacts"][0]["artifact_id"] == "intake/summary"
    assert body["events"][0]["to_state"] == "s1_intake_parsed"


async def test_artifact_bytes_round_trip_with_media_type(client):
    await _seed_run()
    response = client.get("/jobs/job-api/analysis/artifacts/intake/summary")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert response.text.startswith("## Intake")


async def test_missing_run_and_unknown_artifact_return_404(client):
    assert client.get("/jobs/nope/analysis").status_code == 404
    await _seed_run()
    assert client.get("/jobs/job-api/analysis/artifacts/nope/missing").status_code == 404
