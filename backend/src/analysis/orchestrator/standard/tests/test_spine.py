"""Tests for the deterministic spine (orchestrator/standard/spine.py).

These pin the order, the single parallel group, the resume-skip, and every
optional-agent skip rule. The skip rules are the determinism boundary: an
optional agent runs only when the data calls for it, and never just to refuse.
"""
from __future__ import annotations

from src.analysis.agents.base import DataProfile
from src.analysis.agents.base.state import AnalysisState, DatasetInfo
from src.analysis.agents.effect_estimator.output import TreatmentEffectResult
from src.analysis.orchestrator.standard.spine import (
    SPINE,
    Stage,
    _skip_confounder_discovery,
    _skip_data_repair,
    _skip_domain_knowledge,
    _skip_ps_diagnostics,
)
from src.domain.briefs import AgentBrief, Flag


def _state(**kw) -> AnalysisState:
    return AnalysisState(job_id="job-1", dataset_info=DatasetInfo(url="kaggle.com/x"), **kw)


def _profile() -> DataProfile:
    return DataProfile(
        n_samples=100,
        n_features=3,
        feature_names=["t", "y", "x"],
        feature_types={"t": "binary"},
        missing_values={"t": 0},
        numeric_stats={},
        categorical_stats={},
        treatment_candidates=["t"],
        outcome_candidates=["y"],
    )


def _brief(agent: str, flags: list[Flag]) -> AgentBrief:
    return AgentBrief(agent=agent, status="done", headline=f"{agent} done", flags=flags)


def test_spine_order_is_fixed_and_dependency_respecting():
    assert [s.name for s in SPINE] == [
        "domain_knowledge",
        "profile",
        "repair",
        "explore",
        "refine_dag",
        "confounders",
        "estimate",
        "ps_diagnostics",
        "sensitivity",
    ]


def test_only_the_explore_stage_runs_agents_in_parallel():
    by_name = {s.name: s for s in SPINE}
    assert by_name["explore"].agents == ("eda_agent", "causal_discovery")
    assert by_name["explore"].parallel is True
    assert all(s.parallel is False for s in SPINE if s.name != "explore")


def test_already_done_skips_a_completed_stage_on_resume():
    profile_stage = next(s for s in SPINE if s.name == "profile")
    assert profile_stage.already_done(_state()) is False
    assert profile_stage.already_done(_state(data_profile=_profile())) is True


def test_domain_knowledge_is_skipped_without_metadata_and_run_with_it():
    # No metadata, no designated pair -> domain_knowledge would refuse -> skip.
    assert _skip_domain_knowledge(_state()) is True
    # A user-designated treatment/outcome pair is enough signal -> run it.
    with_pair = _state(treatment_variable="t", outcome_variable="y")
    assert _skip_domain_knowledge(with_pair) is False


def test_data_repair_runs_only_on_high_missingness():
    assert _skip_data_repair(_state()) is True  # nothing flagged -> skip
    flagged = _state()
    flagged.agent_briefs["data_profiler"] = _brief("data_profiler", [Flag.HIGH_MISSINGNESS])
    assert _skip_data_repair(flagged) is False


def test_confounder_discovery_runs_only_when_no_adjustment_set():
    assert _skip_confounder_discovery(_state()) is True
    flagged = _state()
    flagged.agent_briefs["dag_expert"] = _brief("dag_expert", [Flag.NO_ADJUSTMENT_SET])
    assert _skip_confounder_discovery(flagged) is False


def test_ps_diagnostics_skipped_when_preconditions_or_method_absent():
    # Missing treatment/outcome -> ps_diagnostics would refuse -> skip.
    bare = _state(data_profile=_profile())
    bare.dataframe_path = "/tmp/df.parquet"
    assert _skip_ps_diagnostics(bare) is True

    ready = _state(treatment_variable="t", outcome_variable="y", data_profile=_profile())
    ready.dataframe_path = "/tmp/df.parquet"
    # A PS-based method was used -> overlap diagnostics apply -> run.
    ready.treatment_effects = [
        TreatmentEffectResult(method="PSM", estimand="ATE", estimate=1.0,
                              std_error=0.1, ci_lower=0.8, ci_upper=1.2)
    ]
    assert _skip_ps_diagnostics(ready) is False
    # Only OLS was used -> overlap diagnostics do not apply -> skip.
    ready.treatment_effects = [
        TreatmentEffectResult(method="OLS", estimand="ATE", estimate=1.0,
                              std_error=0.1, ci_lower=0.8, ci_upper=1.2)
    ]
    assert _skip_ps_diagnostics(ready) is True
