"""TargetedEDAAgent on representative fixtures with a stubbed story LLM."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analysis_v2.agents.base import AgentCtx
from src.analysis_v2.agents.profiling.tools import build_profile_summary
from src.analysis_v2.agents.targeted_eda import TargetedEDAAgent
from src.analysis_v2.core import AnalysisRunState, GateStatus
from src.analysis_v2.evals.fixtures import generators
from src.analysis_v2.spec import (
    CausalSpec,
    Confidence,
    DesignCandidate,
    EDACheckStatus,
    MethodLane,
    QuestionType,
    VariableRef,
)

DATA = Path(__file__).resolve().parents[3] / "evals" / "fixtures" / "data"


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path))
    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    yield tmp_path
    settings_mod.get_settings.cache_clear()


class StoryLLM:
    def __init__(self, text="The outcome is right-skewed and groups differ in size."):
        self.text = text

    async def generate(self, prompt, system_instruction=None, tools=None):
        class R:  # mimic a provider response carrying .text
            pass

        r = R()
        r.text = self.text
        return r


def _stub_llm(monkeypatch, llm) -> None:
    monkeypatch.setattr(
        "src.analysis_v2.agents.targeted_eda.agent.get_llm_client", lambda: llm
    )


def _candidate(lane: MethodLane) -> DesignCandidate:
    return DesignCandidate(
        lane=lane, design_label=lane.value, confidence=Confidence.MEDIUM, rationale="r"
    )


async def _run(spec, frame, lane, job_id="job-eda"):
    run = AnalysisRunState(job_id=job_id, causal_question="q?")
    run.causal_spec = spec
    run.dataset_profile = build_profile_summary(frame)
    run.design_candidates = [_candidate(lane)]
    ctx = AgentCtx(job_id=job_id, run=run, frame=frame)
    agent = TargetedEDAAgent()
    result = await agent.execute(ctx)
    if result.output is not None:
        agent.commit(ctx.run, result.output)
    return result, ctx.run


async def test_lalonde_observational_recipe_flags_imbalance(data_dir, monkeypatch):
    _stub_llm(monkeypatch, StoryLLM())
    frame = pd.read_csv(DATA / "lalonde.csv").drop(columns=["Unnamed: 0"])
    spec = CausalSpec(
        question_type=QuestionType.BINARY_TREATMENT,
        outcome=VariableRef(column="re78"),
        treatment=VariableRef(column="treat"),
        candidate_confounders=["age", "educ", "re74", "re75"],
    )
    result, run = await _run(spec, frame, MethodLane.OBSERVATIONAL)

    assert result.gate.status == GateStatus.ADVANCE
    summary = run.eda_summary
    balance = summary.check("covariate_balance")
    assert balance.status == EDACheckStatus.WARNING  # observational composite imbalance
    assert balance.metrics["max_abs_smd"] > 0.1
    raw = summary.check("group_outcome_comparison")
    assert raw.metrics["raw_mean_difference"] < 0  # the famous misleading naive diff
    assert "descriptive" in raw.detail
    # artifacts persisted and registered
    assert run.artifact_registry.get("eda/covariate_balance_plot") is not None
    plot_path = data_dir / "job-eda/analysis/eda/plots/covariate_balance_plot.png"
    assert plot_path.exists()
    assert run.eda_plan.target_lane == MethodLane.OBSERVATIONAL


async def test_did_panel_recipe_sees_parallel_pre_trends(data_dir, monkeypatch):
    _stub_llm(monkeypatch, StoryLLM())
    frame = generators.did_panel()
    spec = CausalSpec(
        question_type=QuestionType.DID,
        outcome=VariableRef(column="outcome"),
        treatment=VariableRef(derived=True, clue="treated group after period 4"),
        time_column=VariableRef(column="period"),
        group_column=VariableRef(column="group"),
    )
    result, run = await _run(spec, frame, MethodLane.DID)

    trends = run.eda_summary.check("did_trends")
    assert trends.status == EDACheckStatus.OK
    assert trends.metrics["n_periods"] == 8
    pre = run.eda_summary.check("did_pre_trends")
    assert pre.status == EDACheckStatus.OK  # parallel by construction


async def test_rdd_fixture_detects_the_sharp_pattern_at_the_cutoff(data_dir, monkeypatch):
    _stub_llm(monkeypatch, StoryLLM())
    frame = generators.scholarship_rdd()
    spec = CausalSpec(
        question_type=QuestionType.RDD,
        outcome=VariableRef(column="outcome_sharp"),
        treatment=VariableRef(column="scholarship_sharp"),
        running_variable=VariableRef(column="score"),
        cutoff_value=50.0,
    )
    result, run = await _run(spec, frame, MethodLane.RDD)

    side = run.eda_summary.check("rdd_treatment_by_side")
    assert side.metrics["take_below"] <= 0.02
    assert side.metrics["take_above"] >= 0.98
    assert "sharp" in side.detail
    near = run.eda_summary.check("rdd_around_cutoff")
    assert near.metrics["n_near_cutoff"] > 100


async def test_iv_fixture_shows_a_strong_first_stage(data_dir, monkeypatch):
    _stub_llm(monkeypatch, StoryLLM())
    frame = generators.synthetic_iv()
    spec = CausalSpec(
        question_type=QuestionType.IV,
        outcome=VariableRef(column="y"),
        treatment=VariableRef(column="x"),
        instrument=VariableRef(column="z"),
    )
    result, run = await _run(spec, frame, MethodLane.IV)

    first = run.eda_summary.check("iv_first_stage")
    assert first.status == EDACheckStatus.OK
    assert first.metrics["corr_z_x"] > 0.3


async def test_heart_failure_survival_overview(data_dir, monkeypatch):
    _stub_llm(monkeypatch, StoryLLM())
    frame = pd.read_csv(DATA / "heart_failure.csv")
    spec = CausalSpec(
        question_type=QuestionType.SURVIVAL,
        outcome=VariableRef(column="time"),
        treatment=VariableRef(column="high_blood_pressure"),
        duration_column=VariableRef(column="time"),
        event_column=VariableRef(column="DEATH_EVENT"),
    )
    result, run = await _run(spec, frame, MethodLane.SURVIVAL)

    overview = run.eda_summary.check("survival_overview")
    assert overview.status == EDACheckStatus.OK
    assert 0.25 < overview.metrics["event_rate"] < 0.4  # 96 deaths / 299


async def test_causal_language_from_the_llm_is_replaced_by_the_fallback(
    data_dir, monkeypatch
):
    _stub_llm(monkeypatch, StoryLLM("The training causes higher earnings."))
    frame = pd.read_csv(DATA / "lalonde.csv").drop(columns=["Unnamed: 0"])
    spec = CausalSpec(
        question_type=QuestionType.BINARY_TREATMENT,
        outcome=VariableRef(column="re78"),
        treatment=VariableRef(column="treat"),
    )
    result, _ = await _run(spec, frame, MethodLane.OBSERVATIONAL, "job-eda-guard")

    assert "causes" not in result.public_summary
    assert "checks ran" in result.public_summary  # the deterministic fallback


async def test_llm_failure_falls_back_without_failing_the_stage(data_dir, monkeypatch):
    class BoomLLM:
        async def generate(self, prompt, system_instruction=None, tools=None):
            raise RuntimeError("provider down")

    _stub_llm(monkeypatch, BoomLLM())
    frame = generators.mediation()
    spec = CausalSpec(
        question_type=QuestionType.MEDIATION,
        outcome=VariableRef(column="disease_risk"),
        treatment=VariableRef(column="exercise_program"),
        mediator=VariableRef(column="weight_change"),
    )
    result, run = await _run(spec, frame, MethodLane.MEDIATION, "job-eda-boom")

    assert result.gate.status == GateStatus.ADVANCE
    pathway = run.eda_summary.check("mediation_pathways")
    assert pathway.status == EDACheckStatus.WARNING  # timing assumption warning
    assert "timing is assumed" in pathway.detail
    assert "checks ran" in result.public_summary
