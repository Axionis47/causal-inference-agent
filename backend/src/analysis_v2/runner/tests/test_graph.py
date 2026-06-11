"""Spine control flow with fake agents: advance, fail, park, frontier."""
from __future__ import annotations

import pandas as pd
import pytest

from src.analysis_v2.agents.base import AgentCtx, AgentResult, AnalysisAgent
from src.analysis_v2.core import AnalysisRunState, AnalysisStage, GateResult, RunStatus
from src.analysis_v2.runner.graph import run_spine


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


class FakeAgent(AnalysisAgent):
    def __init__(self, stage: AnalysisStage, gate: GateResult):
        self.name = f"fake_{stage.value}"
        self.stage = stage
        self._gate = gate
        self.ran = False

    async def execute(self, ctx: AgentCtx) -> AgentResult:
        self.ran = True
        return AgentResult(gate=self._gate, output=None, public_summary=f"{self.name} ok")

    def commit(self, run, output) -> None:
        return None


def _ctx(events: list | None = None) -> AgentCtx:
    run = AnalysisRunState(
        job_id="job-spine", causal_question="q?", status=RunStatus.RUNNING
    )
    sink = events if events is not None else []
    return AgentCtx(
        job_id="job-spine",
        run=run,
        frame=pd.DataFrame({"a": [1]}),
        emit=lambda t, d: sink.append((t, d)),
    )


async def test_advancing_agents_walk_stages_in_order_and_stop_at_the_frontier(storage_dir):
    s1 = FakeAgent(AnalysisStage.S1_INTAKE_PARSED, GateResult.advance())
    s2 = FakeAgent(AnalysisStage.S2_PROFILE_CREATED, GateResult.advance())
    events: list = []
    ctx = _ctx(events)

    outcome = await run_spine(ctx, {a.stage: a for a in (s1, s2)})

    assert outcome == "frontier"  # S3 has no agent in this registry
    assert s1.ran and s2.ran
    assert ctx.run.current_state == AnalysisStage.S2_PROFILE_CREATED
    assert ctx.run.status == RunStatus.FAILED
    assert "no implementation yet" in ctx.run.error_message
    assert [e[0] for e in events][:2] == [
        "analysis_stage_started",
        "analysis_agent_completed",
    ]


async def test_a_hard_failure_stops_the_spine_and_marks_the_run(storage_dir):
    s1 = FakeAgent(
        AnalysisStage.S1_INTAKE_PARSED,
        GateResult.fail(["question is empty"]),
    )
    s2 = FakeAgent(AnalysisStage.S2_PROFILE_CREATED, GateResult.advance())
    ctx = _ctx()

    outcome = await run_spine(ctx, {a.stage: a for a in (s1, s2)})

    assert outcome == "failed"
    assert not s2.ran
    assert ctx.run.status == RunStatus.FAILED
    assert ctx.run.error_message == "question is empty"
    assert ctx.run.current_state == AnalysisStage.S0_DATASET_SAVED  # no transition


async def test_needs_user_parks_the_run(storage_dir):
    s1 = FakeAgent(
        AnalysisStage.S1_INTAKE_PARSED,
        GateResult.needs_user(["confirm the cutoff"]),
    )
    ctx = _ctx()

    outcome = await run_spine(ctx, {s1.stage: s1})

    assert outcome == "parked"
    assert ctx.run.status == RunStatus.WAITING_FOR_USER


async def test_a_crashing_agent_fails_the_run_with_its_error(storage_dir):
    class Crasher(FakeAgent):
        async def execute(self, ctx):
            raise RuntimeError("kaput")

    ctx = _ctx()
    outcome = await run_spine(
        ctx, {AnalysisStage.S1_INTAKE_PARSED: Crasher(
            AnalysisStage.S1_INTAKE_PARSED, GateResult.advance())}
    )

    assert outcome == "failed"
    assert "kaput" in ctx.run.error_message
    assert ctx.run.agent_runs[-1].status.value == "failed"
