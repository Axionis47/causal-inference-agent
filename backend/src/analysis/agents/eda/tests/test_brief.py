"""Contract tests for the eda_agent brief.

Pin the capability shape, the preflight refusal cases, and build_brief
flag derivation. The ReAct loop and tool behaviour are exercised by
the other test modules in this folder.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.analysis.agents.base.state import AnalysisState, DatasetInfo
from src.analysis.agents.data_profiler.output import DataProfile
from src.analysis.agents.eda.brief import CAPABILITY, build_brief, preflight
from src.analysis.agents.eda.output import EDAResult
from src.domain.briefs import Flag


# --- fixtures ---------------------------------------------------------------


def _profile() -> DataProfile:
    return DataProfile(
        n_samples=100,
        n_features=3,
        feature_names=["t", "y", "x"],
        feature_types={"t": "binary", "y": "numeric", "x": "numeric"},
        missing_values={"t": 0, "y": 0, "x": 0},
        numeric_stats={"y": {"mean": 0.0, "std": 1.0, "min": -2.0, "max": 2.0}},
        categorical_stats={},
    )


def _make_state(
    *,
    dataframe_path: str | None = "/tmp/df.parquet",
    data_profile: DataProfile | None = None,
    treatment_variable: str | None = "t",
    outcome_variable: str | None = "y",
    eda_result: EDAResult | None = None,
) -> AnalysisState:
    return AnalysisState(
        job_id="test-job",
        dataset_info=DatasetInfo(url="https://example.com/data"),
        dataframe_path=dataframe_path,
        data_profile=data_profile if data_profile is not None else _profile(),
        treatment_variable=treatment_variable,
        outcome_variable=outcome_variable,
        eda_result=eda_result,
    )


# --- CAPABILITY -------------------------------------------------------------


class TestCapability:
    def test_name_is_eda_agent(self):
        assert CAPABILITY.name == "eda_agent"

    def test_needs_dataframe_path_and_data_profile(self):
        assert set(CAPABILITY.needs) == {"dataframe_path", "data_profile"}

    def test_does_not_need_treatment_or_outcome(self):
        # EDA can still report distributional and correlation findings
        # without a primary pair declared; only the balance flag depends
        # on having a treatment variable.
        assert "treatment_variable" not in CAPABILITY.needs
        assert "outcome_variable" not in CAPABILITY.needs

    def test_delivers_eda_result(self):
        assert "eda_result" in CAPABILITY.delivers

    def test_is_frozen(self):
        with pytest.raises(ValidationError):
            CAPABILITY.name = "other"

    def test_success_criteria_cover_each_flag_this_agent_raises(self):
        flags_with_criteria = {
            c.raises_flag for c in CAPABILITY.success_criteria if c.raises_flag
        }
        assert Flag.NEEDS_NOT_MET in flags_with_criteria
        assert Flag.OUTCOME_SKEWED in flags_with_criteria
        assert Flag.TC_IMBALANCE in flags_with_criteria
        assert Flag.SUSPECT_CORRELATION in flags_with_criteria


# --- preflight --------------------------------------------------------------


class TestPreflight:
    def test_refuses_when_dataframe_path_missing(self):
        state = _make_state(dataframe_path=None)
        brief = preflight(state)
        assert brief is not None
        assert brief.flags == [Flag.NEEDS_NOT_MET]
        assert "dataframe_path" in brief.headline

    def test_refuses_when_data_profile_missing(self):
        state = _make_state(data_profile=None)
        state.data_profile = None
        brief = preflight(state)
        assert brief is not None
        assert brief.flags == [Flag.NEEDS_NOT_MET]
        assert "data_profile" in brief.headline

    def test_passes_when_needs_satisfied(self):
        state = _make_state()
        assert preflight(state) is None

    def test_passes_when_treatment_outcome_missing(self):
        # EDA does not refuse on missing primary pair.
        state = _make_state(treatment_variable=None, outcome_variable=None)
        assert preflight(state) is None


# --- build_brief: status ----------------------------------------------------


class TestBuildBriefStatus:
    def test_failed_when_eda_result_is_none(self):
        state = _make_state(eda_result=None)
        brief = build_brief(state)
        assert brief.status == "failed"
        assert brief.artifact_keys == []
        assert "no result" in brief.headline.lower()

    def test_done_when_eda_result_is_populated(self):
        state = _make_state(eda_result=EDAResult(data_quality_score=0.85))
        brief = build_brief(state)
        assert brief.status == "done"
        assert brief.artifact_keys == ["eda_result"]


# --- build_brief: flag derivation ------------------------------------------


class TestBuildBriefOutcomeSkewed:
    def test_raises_when_outcome_skewness_exceeds_one(self):
        eda = EDAResult(
            distribution_stats={"y": {"skewness": 2.3}},
        )
        state = _make_state(eda_result=eda)
        brief = build_brief(state)
        assert Flag.OUTCOME_SKEWED in brief.flags

    def test_does_not_raise_at_exactly_one(self):
        eda = EDAResult(distribution_stats={"y": {"skewness": 1.0}})
        state = _make_state(eda_result=eda)
        brief = build_brief(state)
        assert Flag.OUTCOME_SKEWED not in brief.flags

    def test_uses_absolute_value(self):
        eda = EDAResult(distribution_stats={"y": {"skewness": -1.8}})
        state = _make_state(eda_result=eda)
        brief = build_brief(state)
        assert Flag.OUTCOME_SKEWED in brief.flags

    def test_skipped_when_outcome_variable_unset(self):
        eda = EDAResult(distribution_stats={"y": {"skewness": 2.3}})
        state = _make_state(outcome_variable=None, eda_result=eda)
        brief = build_brief(state)
        assert Flag.OUTCOME_SKEWED not in brief.flags

    def test_skipped_when_outcome_not_in_distribution_stats(self):
        eda = EDAResult(distribution_stats={"other": {"skewness": 2.3}})
        state = _make_state(eda_result=eda)
        brief = build_brief(state)
        assert Flag.OUTCOME_SKEWED not in brief.flags


class TestBuildBriefTcImbalance:
    def test_raises_when_max_smd_exceeds_threshold(self):
        eda = EDAResult(
            covariate_balance={
                "age": {"smd": 0.3, "is_balanced": False},
                "income": {"smd": 0.05, "is_balanced": True},
            }
        )
        state = _make_state(eda_result=eda)
        brief = build_brief(state)
        assert Flag.TC_IMBALANCE in brief.flags

    def test_does_not_raise_at_exactly_threshold(self):
        eda = EDAResult(covariate_balance={"age": {"smd": 0.25}})
        state = _make_state(eda_result=eda)
        brief = build_brief(state)
        assert Flag.TC_IMBALANCE not in brief.flags

    def test_uses_absolute_value(self):
        eda = EDAResult(covariate_balance={"age": {"smd": -0.4}})
        state = _make_state(eda_result=eda)
        brief = build_brief(state)
        assert Flag.TC_IMBALANCE in brief.flags

    def test_skipped_when_balance_table_empty(self):
        # No treatment set, no balance computed; flag never fires.
        eda = EDAResult(covariate_balance={})
        state = _make_state(eda_result=eda)
        brief = build_brief(state)
        assert Flag.TC_IMBALANCE not in brief.flags


class TestBuildBriefSuspectCorrelation:
    def test_raises_when_high_correlations_non_empty(self):
        eda = EDAResult(
            high_correlations=[{"var1": "x", "var2": "z", "r": 0.85}],
        )
        state = _make_state(eda_result=eda)
        brief = build_brief(state)
        assert Flag.SUSPECT_CORRELATION in brief.flags

    def test_does_not_raise_when_empty(self):
        eda = EDAResult(high_correlations=[])
        state = _make_state(eda_result=eda)
        brief = build_brief(state)
        assert Flag.SUSPECT_CORRELATION not in brief.flags


class TestBuildBriefAllFlags:
    def test_raises_all_three_when_everything_breached(self):
        eda = EDAResult(
            distribution_stats={"y": {"skewness": 2.0}},
            covariate_balance={"age": {"smd": 0.35}},
            high_correlations=[{"var1": "x", "var2": "z", "r": 0.9}],
            data_quality_score=0.5,
            data_quality_issues=["missing", "outliers"],
        )
        state = _make_state(eda_result=eda)
        brief = build_brief(state)
        assert Flag.OUTCOME_SKEWED in brief.flags
        assert Flag.TC_IMBALANCE in brief.flags
        assert Flag.SUSPECT_CORRELATION in brief.flags
        assert len(brief.flags) == 3
        assert len(brief.raised_issues) == 3
