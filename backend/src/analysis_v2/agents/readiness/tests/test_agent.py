"""ReadinessAgent gates the frozen plan on its lane's check_ready."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis_v2.agents.base import AgentCtx
from src.analysis_v2.agents.readiness import ReadinessAgent
from src.analysis_v2.core import AnalysisRunState, GateStatus
from src.analysis_v2.spec import (
    CausalSpec,
    MethodLane,
    MethodPlan,
    QuestionType,
    VariableRef,
)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path))
    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    yield tmp_path
    settings_mod.get_settings.cache_clear()


def _spec() -> CausalSpec:
    return CausalSpec(
        question_type=QuestionType.BINARY_TREATMENT,
        outcome=VariableRef(column="y"),
        treatment=VariableRef(column="t"),
    )


def _obs_plan() -> MethodPlan:
    return MethodPlan(
        lane=MethodLane.OBSERVATIONAL, estimator="regression_adjustment",
        estimand="ate", outcome="y", treatment="t", covariates=["x"], settings={},
    )


def _frame(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"y": rng.normal(size=n), "t": rng.integers(0, 2, n), "x": rng.normal(size=n)}
    )


async def _run(plan, spec, frame):
    run = AnalysisRunState(job_id="job-ready", causal_question="q?")
    run.causal_spec = spec
    run.method_plan = plan
    ctx = AgentCtx(job_id="job-ready", run=run, frame=frame)
    agent = ReadinessAgent()
    result = await agent.execute(ctx)
    if result.output is not None and result.gate.status == GateStatus.ADVANCE:
        agent.commit(run, result.output)
    return result, run


async def test_a_runnable_plan_advances_and_commits_the_result(data_dir):
    result, run = await _run(_obs_plan(), _spec(), _frame(60))
    assert result.gate.status == GateStatus.ADVANCE
    assert run.readiness is not None and run.readiness.ready is True
    assert run.readiness.blocking_reasons == []
    assert run.artifact_registry.get("readiness/result") is not None


async def test_an_unrunnable_plan_fails_here_with_the_lane_reason(data_dir):
    result, _ = await _run(_obs_plan(), _spec(), _frame(8))  # < 20 complete rows
    assert result.gate.status == GateStatus.FAIL
    assert "cannot run on this data" in result.gate.hard_failures[0]
    assert "complete rows across the design" in result.gate.hard_failures[0]


async def test_a_missing_plan_fails_the_gate(data_dir):
    run = AnalysisRunState(job_id="job-ready", causal_question="q?")
    ctx = AgentCtx(job_id="job-ready", run=run, frame=_frame(60))
    result = await ReadinessAgent().execute(ctx)
    assert result.gate.status == GateStatus.FAIL
