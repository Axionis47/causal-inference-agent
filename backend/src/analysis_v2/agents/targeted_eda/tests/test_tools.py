"""EDA tool registry, constraints, and mission, tested in isolation (no loop)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analysis_v2.agents.base import AgentCtx
from src.analysis_v2.agents.profiling.tools import build_profile_summary
from src.analysis_v2.agents.targeted_eda.common import ArtifactSink, EDAInputs
from src.analysis_v2.agents.targeted_eda.constraints import (
    clamp_covariates,
    reject_moderator_in_adjustment_set,
)
from src.analysis_v2.agents.targeted_eda.prompt import build_mission
from src.analysis_v2.agents.targeted_eda.tools import (
    build_eda_tools,
    collect_checks,
    collect_trace,
)
from src.analysis_v2.core import AnalysisRunState, AnalysisStage
from src.analysis_v2.spec import (
    CausalDAG,
    CausalEdge,
    CausalNode,
    CausalSpec,
    Confidence,
    DesignCandidate,
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


def _lalonde_spec() -> CausalSpec:
    return CausalSpec(
        question_type=QuestionType.BINARY_TREATMENT,
        outcome=VariableRef(column="re78"),
        treatment=VariableRef(column="treat"),
        candidate_confounders=["age", "educ", "re74", "re75"],
    )


def _setup(frame, spec, dag=None, lane=MethodLane.OBSERVATIONAL, job_id="job-tools"):
    run = AnalysisRunState(job_id=job_id, causal_question="does training raise earnings?")
    run.causal_spec = spec
    run.dataset_profile = build_profile_summary(frame)
    run.causal_dag = dag
    run.design_candidates = [
        DesignCandidate(lane=lane, design_label=lane.value,
                        confidence=Confidence.MEDIUM, rationale="r")
    ]
    ctx = AgentCtx(job_id=job_id, run=run, frame=frame)
    inputs = EDAInputs(frame=frame, spec=spec, profile=run.dataset_profile, dag=dag)
    sink = ArtifactSink(ctx=ctx, agent="targeted_eda",
                        stage=AnalysisStage.S4_TARGETED_EDA_COMPLETE)
    return inputs, sink, run


def test_offered_tools_exclude_the_data_health_floor_but_offer_distributions(data_dir):
    frame = pd.read_csv(DATA / "lalonde.csv").drop(columns=["Unnamed: 0"])
    inputs, sink, _ = _setup(frame, _lalonde_spec())
    _, tools = build_eda_tools(inputs, sink, MethodLane.OBSERVATIONAL, has_dag=False)
    names = {t.name for t in tools}
    assert "run_covariate_balance" in names
    assert "run_propensity_overlap" in names
    # distributions are now the model's to curate, not part of the floor
    assert "run_outcome_distribution" in names
    # the data-health floor runs separately and is never offered as a tool
    assert "run_shape_and_missingness" not in names
    assert "run_usable_sample_size" not in names
    assert "run_rdd_around_cutoff" not in names  # wrong lane is never offered


def test_dag_consistency_tool_appears_only_with_a_graph(data_dir):
    frame = pd.read_csv(DATA / "lalonde.csv").drop(columns=["Unnamed: 0"])
    inputs, sink, _ = _setup(frame, _lalonde_spec())
    _, without = build_eda_tools(inputs, sink, MethodLane.OBSERVATIONAL, has_dag=False)
    assert "run_dag_consistency" not in {t.name for t in without}
    _, with_dag = build_eda_tools(inputs, sink, MethodLane.OBSERVATIONAL, has_dag=True)
    assert "run_dag_consistency" in {t.name for t in with_dag}


def test_handler_runs_the_check_and_records_the_ledger(data_dir):
    frame = pd.read_csv(DATA / "lalonde.csv").drop(columns=["Unnamed: 0"])
    inputs, sink, _ = _setup(frame, _lalonde_spec())
    ledger, tools = build_eda_tools(inputs, sink, MethodLane.OBSERVATIONAL, has_dag=False)
    balance = next(t for t in tools if t.name == "run_covariate_balance")

    observation = balance.handler()

    assert "covariate_balance" in observation
    assert len(ledger) == 1
    assert ledger[0].call.name == "covariate_balance"
    assert ledger[0].check is not None
    assert collect_checks(ledger)[0].name == "covariate_balance"
    assert collect_trace(ledger).calls[0].name == "covariate_balance"


def test_clamp_covariates_keeps_only_adjustment_set_columns():
    frame = pd.DataFrame(
        {"treat": [0, 1, 0, 1], "age": [1, 2, 3, 4], "re78": [1.0, 2, 3, 4],
         "leak": [9, 9, 9, 9]}
    )
    spec = CausalSpec(
        question_type=QuestionType.BINARY_TREATMENT,
        treatment=VariableRef(column="treat"),
        outcome=VariableRef(column="re78"),
        candidate_confounders=["age"],
    )
    inputs = EDAInputs(frame=frame, spec=spec, profile=build_profile_summary(frame))

    verdict = clamp_covariates({"covariates": ["age", "leak"]}, inputs)

    assert verdict.rejected is False
    assert verdict.clamped_args["covariates"] == ["age"]
    assert "leak" in verdict.reason


def test_reject_moderator_that_is_an_adjustment_covariate():
    frame = pd.DataFrame({"treat": [0, 1], "age": [1, 2], "re78": [1.0, 2]})
    spec = CausalSpec(
        question_type=QuestionType.INTERACTION,
        treatment=VariableRef(column="treat"),
        outcome=VariableRef(column="re78"),
        candidate_confounders=["age"],
    )
    inputs = EDAInputs(frame=frame, spec=spec, profile=build_profile_summary(frame))

    assert reject_moderator_in_adjustment_set({"moderator": "age"}, inputs).rejected is True
    assert reject_moderator_in_adjustment_set({"moderator": "re78"}, inputs).rejected is False


def test_mission_summarizes_the_dag_and_offered_tools(data_dir):
    frame = pd.read_csv(DATA / "lalonde.csv").drop(columns=["Unnamed: 0"])
    dag = CausalDAG(
        nodes=[CausalNode(name="treat"), CausalNode(name="re78"), CausalNode(name="age")],
        edges=[
            CausalEdge(source="age", target="treat", mechanism="age shapes selection"),
            CausalEdge(source="age", target="re78", mechanism="age shapes earnings"),
            CausalEdge(source="treat", target="re78", mechanism="training effect"),
        ],
        treatment="treat",
        outcome="re78",
    )
    inputs, sink, run = _setup(frame, _lalonde_spec(), dag=dag)
    _, tools = build_eda_tools(inputs, sink, MethodLane.OBSERVATIONAL, has_dag=True)

    mission = build_mission(run, tools)

    assert "adjustment set" in mission
    assert "age" in mission  # the backdoor adjustment covariate
    assert "run_covariate_balance" in mission
    assert "identified by adjustment:" in mission
    # is_identifiable() is unpacked as (bool, str), not rendered as a raw tuple
    assert "(True," not in mission and "(False," not in mission
