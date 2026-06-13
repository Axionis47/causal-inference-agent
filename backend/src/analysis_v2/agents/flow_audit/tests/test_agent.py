"""FlowAuditAgent: covariate sanction, honesty consistency, the open-question
ledger. Inconsistencies fail the run; unaccounted signals warn."""
from __future__ import annotations

import pandas as pd
import pytest

from src.analysis_v2.agents.base import AgentCtx
from src.analysis_v2.agents.flow_audit import FlowAuditAgent
from src.analysis_v2.core import AnalysisRunState, GateStatus
from src.analysis_v2.spec import (
    CausalDAG,
    CausalEdge,
    CausalNode,
    ClaimCritique,
    ClaimStrength,
    DatasetDossier,
    EffectEstimate,
    EstimateResult,
    MethodLane,
)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path))
    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    yield tmp_path
    settings_mod.get_settings.cache_clear()


def _dag(edges, latent=(), treatment="T", outcome="Y") -> CausalDAG:
    nodes = {c for e in edges for c in e}
    return CausalDAG(
        nodes=[CausalNode(name=n, observed=n not in latent) for n in nodes],
        edges=[CausalEdge(source=s, target=t, mechanism="m") for s, t in edges],
        treatment=treatment,
        outcome=outcome,
    )


def _estimate(covariates, estimator="regression_adjustment") -> EstimateResult:
    return EstimateResult(
        lane=MethodLane.OBSERVATIONAL, estimator=estimator,
        effects=[EffectEstimate(estimand="ate", estimate=1.0)],
        n_rows_used=100, outcome="Y", treatment="T", covariates_used=covariates,
    )


def _critique(strength=ClaimStrength.MODERATE) -> ClaimCritique:
    return ClaimCritique(strength=strength, rationale="r")


async def _execute(run: AnalysisRunState):
    ctx = AgentCtx(job_id="job-fa", run=run, frame=pd.DataFrame())
    agent = FlowAuditAgent()
    result = await agent.execute(ctx)
    if result.gate.status == GateStatus.ADVANCE and result.output is not None:
        agent.commit(run, result.output)
    return result, run


def _run(**kw) -> AnalysisRunState:
    run = AnalysisRunState(job_id="job-fa", causal_question="q?")
    for key, value in kw.items():
        setattr(run, key, value)
    return run


async def test_a_consistent_run_passes(data_dir):
    run = _run(
        causal_dag=_dag([("age", "T"), ("age", "Y"), ("T", "Y")]),
        estimate_result=_estimate(["age"]),
        claim_critique=_critique(ClaimStrength.MODERATE),
    )
    result, run = await _execute(run)
    assert result.gate.status == GateStatus.ADVANCE
    assert run.flow_audit is not None and run.flow_audit.passed is True


async def test_adjusting_for_an_unsanctioned_covariate_fails(data_dir):
    run = _run(
        causal_dag=_dag([("age", "T"), ("age", "Y"), ("T", "Y")]),
        estimate_result=_estimate(["age", "sneaky"]),  # 'sneaky' not in the DAG
        claim_critique=_critique(),
    )
    result, _ = await _execute(run)
    assert result.gate.status == GateStatus.FAIL
    assert "does not sanction" in result.gate.hard_failures[0]


async def test_a_non_adjustment_estimator_skips_the_covariate_check(data_dir):
    # multi-factor's joint_regression uses factors, not the adjustment set.
    run = _run(
        causal_dag=_dag([("T", "Y")]),  # empty adjustment set
        estimate_result=_estimate(["x1", "x2"], estimator="joint_regression"),
        claim_critique=_critique(),
    )
    result, _ = await _execute(run)
    assert result.gate.status == GateStatus.ADVANCE


async def test_a_strong_claim_on_an_unidentified_effect_fails(data_dir):
    run = _run(
        causal_dag=_dag([("U", "T"), ("U", "Y"), ("T", "Y")], latent=("U",)),
        estimate_result=_estimate([]),
        claim_critique=_critique(ClaimStrength.MODERATE),
    )
    result, _ = await _execute(run)
    assert result.gate.status == GateStatus.FAIL
    assert "not point-identified" in result.gate.hard_failures[0]


async def test_open_questions_surface_as_a_warning(data_dir):
    run = _run(
        causal_dag=_dag([("age", "T"), ("age", "Y"), ("T", "Y")]),
        estimate_result=_estimate(["age"]),
        claim_critique=_critique(ClaimStrength.MODERATE),
        dataset_dossier=DatasetDossier(
            provenance="t", summary="s",
            open_questions=["how were units assigned to treatment?"],
        ),
    )
    result, run = await _execute(run)
    assert result.gate.status == GateStatus.ADVANCE
    assert any("open question" in w for w in result.warnings)
    assert run.flow_audit.passed is True
