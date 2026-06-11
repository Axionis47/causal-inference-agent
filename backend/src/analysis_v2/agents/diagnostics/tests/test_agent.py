"""Diagnostics and sensitivity per lane on the fixtures."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.analysis_v2.agents.base import AgentCtx
from src.analysis_v2.agents.diagnostics import DiagnosticsSensitivityAgent
from src.analysis_v2.agents.method_lane.lanes import LANES
from src.analysis_v2.agents.profiling.tools import build_profile_summary
from src.analysis_v2.core import AnalysisRunState, GateStatus
from src.analysis_v2.evals.fixtures import generators
from src.analysis_v2.spec import (
    CausalSpec,
    CheckStatus,
    MethodLane,
    MethodPlan,
    QuestionType,
    RobustnessStatus,
)

DATA = Path(__file__).resolve().parents[3] / "evals" / "fixtures" / "data"


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path))
    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    yield tmp_path
    settings_mod.get_settings.cache_clear()


async def _run_stage(frame, plan, question_type=QuestionType.BINARY_TREATMENT):
    spec = CausalSpec(question_type=question_type)
    lane_outcome = LANES[plan.lane](frame, plan, spec)
    run = AnalysisRunState(job_id="job-diag", causal_question="q?")
    run.causal_spec = spec
    run.dataset_profile = build_profile_summary(frame)
    run.method_plan = plan
    run.estimate_result = lane_outcome.result
    ctx = AgentCtx(job_id="job-diag", run=run, frame=frame)
    agent = DiagnosticsSensitivityAgent()
    result = await agent.execute(ctx)
    if result.output is not None and result.gate.status == GateStatus.ADVANCE:
        agent.commit(run, result.output)
    return result, run


async def test_lalonde_observational_gets_evalue_trimming_and_agreement(data_dir):
    frame = pd.read_csv(DATA / "lalonde.csv")
    plan = MethodPlan(
        lane=MethodLane.OBSERVATIONAL, estimator="regression_adjustment",
        estimand="ate", outcome="re78", treatment="treat",
        covariates=["age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75"],
        settings={"include_ipw": True},
    )
    result, run = await _run_stage(frame, plan)

    assert result.gate.status == GateStatus.ADVANCE
    names = {c.name for c in run.sensitivity_result.checks}
    assert {"e_value", "outlier_trimming"} <= names
    assert run.diagnostics_result.check("estimator_agreement") is not None
    assert run.sensitivity_result.robustness in (
        RobustnessStatus.ROBUST, RobustnessStatus.FRAGILE,
    )
    assert run.sensitivity_result.rerun_required is False


async def test_leakage_stops_the_run_with_an_actionable_message(data_dir):
    frame = pd.read_csv(DATA / "lalonde.csv")
    frame["re78_copy"] = frame["re78"] * 1.0
    plan = MethodPlan(
        lane=MethodLane.OBSERVATIONAL, estimator="regression_adjustment",
        estimand="ate", outcome="re78", treatment="treat",
        covariates=["age", "re78_copy"], settings={},
    )
    result, _ = await _run_stage(frame, plan)

    assert result.gate.status == GateStatus.FAIL
    assert "bad control" in result.gate.hard_failures[0]
    assert "re78_copy" in result.gate.hard_failures[0]
    assert "ignored" in result.gate.hard_failures[0]


async def test_synthetic_rdd_is_robust_under_bandwidth_and_placebo_checks(data_dir):
    frame = generators.scholarship_rdd()
    plan = MethodPlan(
        lane=MethodLane.RDD, estimator="local_linear_rdd", estimand="late",
        outcome="outcome_sharp", treatment="scholarship_sharp",
        settings={"running_variable": "score", "cutoff": 50.0, "bandwidth": 15.0},
    )
    result, run = await _run_stage(frame, plan, QuestionType.RDD)

    names = {c.name for c in run.sensitivity_result.checks}
    assert {"bandwidth_x0.5", "bandwidth_x2", "placebo_cutoff"} <= names
    placebo = next(c for c in run.sensitivity_result.checks if c.name == "placebo_cutoff")
    assert placebo.status in (CheckStatus.PASS, CheckStatus.NOT_APPLICABLE)
    assert run.sensitivity_result.robustness in (
        RobustnessStatus.ROBUST, RobustnessStatus.FRAGILE,
    )


async def test_strong_synthetic_iv_passes_first_stage_and_weak_iv_fails(data_dir):
    frame = generators.synthetic_iv()
    plan = MethodPlan(
        lane=MethodLane.IV, estimator="two_stage_least_squares", estimand="late",
        outcome="y", treatment="x", covariates=["c1", "c2"],
        settings={"instrument": "z"},
    )
    result, run = await _run_stage(frame, plan, QuestionType.IV)
    first = run.diagnostics_result.check("first_stage_strength")
    assert first.status == CheckStatus.PASS
    assert first.metrics["f_stat"] > 10

    weak = frame.copy()
    rng = __import__("numpy").random.default_rng(3)
    weak["z"] = rng.binomial(1, 0.5, len(weak))  # noise instrument
    plan_weak = plan.model_copy(deep=True)
    result_weak, run_weak = await _run_stage(weak, plan_weak, QuestionType.IV)
    first_weak = run_weak.diagnostics_result.check("first_stage_strength")
    assert first_weak.status == CheckStatus.FAIL
    assert run_weak.sensitivity_result.robustness == RobustnessStatus.NOT_SUPPORTED


async def test_did_panel_pre_trends_pass_with_enough_pre_periods(data_dir):
    frame = generators.did_panel()
    plan = MethodPlan(
        lane=MethodLane.DID, estimator="difference_in_differences", estimand="att",
        outcome="outcome", treatment=None,
        settings={"time_column": "period", "group_column": "group",
                  "post_column": "post", "needs_reshape": False,
                  "treated_group": "treated"},
    )
    result, run = await _run_stage(frame, plan, QuestionType.DID)
    pre = run.diagnostics_result.check("pre_trend")
    assert pre.status == CheckStatus.PASS  # parallel by construction


async def test_survival_km_crossing_check_runs_on_heart_failure(data_dir):
    frame = pd.read_csv(DATA / "heart_failure.csv")
    plan = MethodPlan(
        lane=MethodLane.SURVIVAL, estimator="cox_proportional_hazards",
        estimand="hazard_ratio", outcome="time", treatment="high_blood_pressure",
        covariates=["age"],
        settings={"duration_column": "time", "event_column": "DEATH_EVENT"},
    )
    result, run = await _run_stage(frame, plan, QuestionType.SURVIVAL)
    km = run.diagnostics_result.check("km_crossing")
    assert km is not None
    assert km.status in (CheckStatus.PASS, CheckStatus.WARNING)


async def test_a_fragile_estimate_advances_instead_of_rerunning(data_dir):
    """The core policy: weak results report honestly; no rerun, no failure."""
    frame = generators.mediation()
    plan = MethodPlan(
        lane=MethodLane.MEDIATION, estimator="product_of_coefficients",
        estimand="indirect_effect", outcome="disease_risk",
        treatment="exercise_program", covariates=["baseline_health"],
        settings={"mediator": "weight_change"},
    )
    result, run = await _run_stage(frame, plan, QuestionType.MEDIATION)

    assert result.gate.status == GateStatus.ADVANCE
    assert run.sensitivity_result.robustness == RobustnessStatus.FRAGILE
    assert "timing" in run.sensitivity_result.confidence_reason or any(
        "timing" in w for w in result.warnings
    )
    assert run.sensitivity_result.rerun_required is False
