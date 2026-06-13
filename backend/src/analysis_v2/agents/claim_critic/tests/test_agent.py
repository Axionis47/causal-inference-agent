"""ClaimCriticAgent: a latent confounder caps a backdoor-adjusted claim, but
designs that identify despite backdoor confounding (IV) keep their strength."""
from __future__ import annotations

import pandas as pd
import pytest

from src.analysis_v2.agents.base import AgentCtx
from src.analysis_v2.agents.claim_critic import ClaimCriticAgent
from src.analysis_v2.core import AnalysisRunState, GateStatus
from src.analysis_v2.spec import (
    CausalDAG,
    CausalEdge,
    CausalNode,
    CausalSpec,
    ClaimStrength,
    DatasetDossier,
    EffectEstimate,
    EstimateResult,
    MethodLane,
    QuestionType,
    RobustnessStatus,
    SensitivityResult,
    VariableRef,
)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path))
    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    yield tmp_path
    settings_mod.get_settings.cache_clear()


def _dag_with_latent() -> CausalDAG:
    return CausalDAG(
        nodes=[
            CausalNode(name="T"), CausalNode(name="Y"),
            CausalNode(name="U", observed=False),
        ],
        edges=[
            CausalEdge(source="U", target="T", mechanism="m"),
            CausalEdge(source="U", target="Y", mechanism="m"),
            CausalEdge(source="T", target="Y", mechanism="m"),
        ],
        treatment="T", outcome="Y",
    )


def _run(estimator: str, lane: MethodLane) -> AnalysisRunState:
    run = AnalysisRunState(job_id="job-cc", causal_question="q?")
    run.causal_spec = CausalSpec(
        question_type=QuestionType.BINARY_TREATMENT,
        treatment=VariableRef(column="T"), outcome=VariableRef(column="Y"),
    )
    run.estimate_result = EstimateResult(
        lane=lane, estimator=estimator,
        effects=[EffectEstimate(estimand="ate", estimate=1.0, ci_lower=0.5, ci_upper=1.5)],
        n_rows_used=600, outcome="Y", treatment="T",
    )
    run.sensitivity_result = SensitivityResult(
        checks=[], robustness=RobustnessStatus.ROBUST, confidence_reason="r"
    )
    run.causal_dag = _dag_with_latent()
    run.dataset_dossier = DatasetDossier(
        provenance="t", summary="s", suspected_latent_confounders=["market size"]
    )
    return run


async def _execute(run: AnalysisRunState):
    ctx = AgentCtx(job_id="job-cc", run=run, frame=pd.DataFrame())
    agent = ClaimCriticAgent()
    result = await agent.execute(ctx)
    if result.output is not None and result.gate.status == GateStatus.ADVANCE:
        agent.commit(run, result.output)
    return result, run


async def test_latent_confounding_caps_an_observational_claim(data_dir):
    # observational base weak, robust -> moderate; the latent then caps to weak.
    result, run = await _execute(_run("regression_adjustment", MethodLane.OBSERVATIONAL))
    assert result.gate.status == GateStatus.ADVANCE
    assert run.claim_critique.strength == ClaimStrength.WEAK
    assert any("not point-identified" in lim for lim in run.claim_critique.limitations)
    assert any("market size" in lim for lim in run.claim_critique.limitations)


async def test_iv_keeps_its_strength_despite_latent_confounding(data_dir):
    # IV identifies despite backdoor confounding; the cap must not touch it.
    result, run = await _execute(_run("two_stage_least_squares", MethodLane.IV))
    assert run.claim_critique.strength == ClaimStrength.STRONG
    assert not any("not point-identified" in lim for lim in run.claim_critique.limitations)
