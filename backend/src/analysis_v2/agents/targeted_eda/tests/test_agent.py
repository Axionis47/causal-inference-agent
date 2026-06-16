"""TargetedEDAAgent on representative fixtures with a stubbed story LLM."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis_v2.agents.base import AgentCtx
from src.analysis_v2.agents.profiling.tools import build_profile_summary
from src.analysis_v2.agents.targeted_eda import TargetedEDAAgent
from src.analysis_v2.core import AnalysisRunState, GateStatus
from src.analysis_v2.evals.fixtures import generators
from src.analysis_v2.spec import (
    CausalDAG,
    CausalEdge,
    CausalNode,
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


class ToolLLM:
    """Scripts chat_with_tools turns; answers the story call via generate."""

    def __init__(self, turns, story="The groups differ in baseline covariates."):
        self.turns = list(turns)
        self.story = story

    def user_message(self, text):
        return {"role": "user", "content": text}

    def tool_results_message(self, results):
        return {"role": "tool_result",
                "content": [{"name": c["name"], "output": o} for c, o in results]}

    async def chat_with_tools(self, messages, system_instruction, specs):
        turn = self.turns.pop(0)
        return {
            "text": turn.get("text"),
            "tool_calls": turn.get("tool_calls", []),
            "assistant_message": {"role": "assistant", "content": turn.get("text") or ""},
            "usage": turn.get("usage", {"input_tokens": 10, "output_tokens": 5}),
        }

    async def generate(self, prompt, system_instruction=None, tools=None):
        class R:
            pass

        r = R()
        r.text = self.story
        return r


def _stub_llm(monkeypatch, llm) -> None:
    monkeypatch.setattr(
        "src.analysis_v2.agents.targeted_eda.agent.get_llm_client", lambda: llm
    )


def _candidate(lane: MethodLane) -> DesignCandidate:
    return DesignCandidate(
        lane=lane, design_label=lane.value, confidence=Confidence.MEDIUM, rationale="r"
    )


async def _run(spec, frame, lane, job_id="job-eda", dag=None):
    run = AnalysisRunState(job_id=job_id, causal_question="q?")
    run.causal_spec = spec
    run.dataset_profile = build_profile_summary(frame)
    run.design_candidates = [_candidate(lane)]
    run.causal_dag = dag
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


def _collider_dag() -> CausalDAG:
    """a -> c <- b: the only testable implication is a _||_ b (marginal)."""
    return CausalDAG(
        nodes=[CausalNode(name="a"), CausalNode(name="b"), CausalNode(name="c")],
        edges=[
            CausalEdge(source="a", target="c", mechanism="a drives c"),
            CausalEdge(source="b", target="c", mechanism="b drives c"),
        ],
        treatment="a",
        outcome="c",
    )


def _abc_frame(corr: bool) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    a = rng.normal(size=n)
    # corr=True wires b to a, contradicting the graph's a _||_ b implication
    b = a + rng.normal(scale=0.3, size=n) if corr else rng.normal(size=n)
    c = a + b + rng.normal(size=n)
    return pd.DataFrame({"a": a, "b": b, "c": c})


def _abc_spec() -> CausalSpec:
    return CausalSpec(
        question_type=QuestionType.SIMPLE_EFFECT,
        outcome=VariableRef(column="c"),
        treatment=VariableRef(column="a"),
        treatment_is_continuous=True,
    )


async def test_dag_consistency_flags_an_independence_the_data_violates(
    data_dir, monkeypatch
):
    _stub_llm(monkeypatch, StoryLLM())
    result, run = await _run(
        _abc_spec(), _abc_frame(corr=True), MethodLane.OBSERVATIONAL,
        "job-eda-dag-tension", dag=_collider_dag(),
    )

    chk = run.eda_summary.check("dag_consistency")
    assert chk.status == EDACheckStatus.WARNING
    assert chk.metrics["n_tested"] == 1
    assert chk.metrics["n_tension"] == 1
    assert chk.metrics["max_partial_corr"] > 0.1
    assert run.artifact_registry.get("eda/dag_implications") is not None
    table_path = data_dir / "job-eda-dag-tension/analysis/eda/tables/dag_implications.csv"
    assert table_path.exists()


async def test_dag_consistency_passes_when_the_data_backs_the_graph(
    data_dir, monkeypatch
):
    _stub_llm(monkeypatch, StoryLLM())
    result, run = await _run(
        _abc_spec(), _abc_frame(corr=False), MethodLane.OBSERVATIONAL,
        "job-eda-dag-ok", dag=_collider_dag(),
    )

    chk = run.eda_summary.check("dag_consistency")
    assert chk.status == EDACheckStatus.OK
    assert chk.metrics["n_tension"] == 0
    assert chk.metrics["max_partial_corr"] <= 0.1


async def test_dag_consistency_is_absent_without_a_graph(data_dir, monkeypatch):
    _stub_llm(monkeypatch, StoryLLM())
    result, run = await _run(
        _abc_spec(), _abc_frame(corr=True), MethodLane.OBSERVATIONAL, "job-eda-dag-none",
    )

    assert run.eda_summary.check("dag_consistency") is None


async def test_commit_records_a_replayable_tool_trace(data_dir, monkeypatch):
    _stub_llm(monkeypatch, StoryLLM())
    frame = pd.read_csv(DATA / "lalonde.csv").drop(columns=["Unnamed: 0"])
    spec = CausalSpec(
        question_type=QuestionType.BINARY_TREATMENT,
        outcome=VariableRef(column="re78"),
        treatment=VariableRef(column="treat"),
        candidate_confounders=["age", "educ", "re74", "re75"],
    )
    result, run = await _run(spec, frame, MethodLane.OBSERVATIONAL, "job-eda-trace")

    trace = run.eda_tool_trace
    assert trace is not None
    # one trace entry per check that ran, names and statuses preserved, in order
    assert [c.name for c in trace.calls] == [c.name for c in run.eda_summary.checks]
    assert [c.status for c in trace.calls] == [
        c.status.value for c in run.eda_summary.checks
    ]
    assert not trace.exhausted

    # the trace survives a full run-state serialize/load round-trip
    reloaded = AnalysisRunState.load(run.model_dump(mode="json"))
    assert reloaded.eda_tool_trace == trace


def _lalonde_observational_spec() -> CausalSpec:
    return CausalSpec(
        question_type=QuestionType.BINARY_TREATMENT,
        outcome=VariableRef(column="re78"),
        treatment=VariableRef(column="treat"),
        candidate_confounders=["age", "educ", "re74", "re75"],
    )


async def test_agentic_loop_runs_model_chosen_checks_over_the_base_floor(
    data_dir, monkeypatch
):
    story = "The treated and control groups differ in baseline covariates."
    _stub_llm(monkeypatch, ToolLLM(
        turns=[
            {"text": "let me check balance",
             "tool_calls": [{"id": "1", "name": "run_covariate_balance", "args": {}}]},
            {"text": story},
        ],
        story=story,
    ))
    frame = pd.read_csv(DATA / "lalonde.csv").drop(columns=["Unnamed: 0"])
    result, run = await _run(
        _lalonde_observational_spec(), frame, MethodLane.OBSERVATIONAL, "job-eda-agentic",
    )

    assert result.gate.status == GateStatus.ADVANCE
    # the base floor ran even though the model only chose one targeted check
    assert run.eda_summary.check("shape_and_missingness") is not None
    assert run.eda_summary.check("usable_sample_size") is not None
    # the model's chosen check ran too
    assert run.eda_summary.check("covariate_balance") is not None
    # the loop's tool call is recorded on the result and the trace
    assert [tc.name for tc in result.tool_calls] == ["run_covariate_balance"]
    assert [c.name for c in run.eda_tool_trace.calls] == ["covariate_balance"]
    # the model's final text is used as the story (no second generation needed)
    assert result.public_summary == story
    # the agentic path emits the replayable tool-call artifact
    assert run.artifact_registry.get("eda/tool_calls") is not None


async def test_agentic_abstention_falls_back_to_the_full_recipe(data_dir, monkeypatch):
    note = "Baseline looks clean; nothing further to check."
    _stub_llm(monkeypatch, ToolLLM(turns=[{"text": note}], story=note))
    frame = pd.read_csv(DATA / "lalonde.csv").drop(columns=["Unnamed: 0"])
    result, run = await _run(
        _lalonde_observational_spec(), frame, MethodLane.OBSERVATIONAL, "job-eda-abstain",
    )

    assert result.gate.status == GateStatus.ADVANCE
    # a model that calls no tools must not ship a thin floor-only EDA: the
    # deterministic recipe runs instead, so the design checks still appear
    assert run.eda_summary.check("shape_and_missingness") is not None  # floor
    assert run.eda_summary.check("covariate_balance") is not None  # recipe ran
    assert result.tool_calls == []  # no tools were used
    assert "covariate_balance" in {c.name for c in run.eda_tool_trace.calls}


async def test_dag_constraint_clamps_an_off_adjustment_set_covariate(data_dir, monkeypatch):
    # The model asks to balance on `educ`, which the DAG does not sanction (only
    # `age` lies on a backdoor path). The clamp must drop it from what is run and
    # recorded; the off-set column must never enter the analysis.
    story = "Balance was assessed on the covariates the causal model sanctions."
    _stub_llm(monkeypatch, ToolLLM(
        turns=[
            {"text": "checking balance",
             "tool_calls": [{"id": "1", "name": "run_covariate_balance",
                             "args": {"covariates": ["age", "educ"]}}]},
            {"text": story},
        ],
        story=story,
    ))
    frame = pd.read_csv(DATA / "lalonde.csv").drop(columns=["Unnamed: 0"])
    spec = CausalSpec(
        question_type=QuestionType.BINARY_TREATMENT,
        outcome=VariableRef(column="re78"),
        treatment=VariableRef(column="treat"),
        candidate_confounders=["age"],  # the DAG-sanctioned adjustment set
    )
    dag = CausalDAG(
        nodes=[CausalNode(name="treat"), CausalNode(name="re78"),
               CausalNode(name="age"), CausalNode(name="educ")],
        edges=[
            CausalEdge(source="age", target="treat", mechanism="age shapes selection"),
            CausalEdge(source="age", target="re78", mechanism="age shapes earnings"),
            CausalEdge(source="treat", target="re78", mechanism="training effect"),
        ],
        treatment="treat",
        outcome="re78",
    )
    result, run = await _run(spec, frame, MethodLane.OBSERVATIONAL, "job-eda-clamp", dag=dag)

    assert result.gate.status == GateStatus.ADVANCE
    balance = run.eda_summary.check("covariate_balance")
    assert balance is not None
    # the off-set covariate is dropped from the recorded (clamped) call args
    call = next(c for c in run.eda_tool_trace.calls if c.name == "covariate_balance")
    assert call.args["covariates"] == ["age"]
    # and it never entered the balance computation (which sources the adjustment set)
    assert "educ" not in balance.detail
    # EDA never mutates the adjustment set that flow_audit reads downstream
    assert run.causal_spec.candidate_confounders == ["age"]
