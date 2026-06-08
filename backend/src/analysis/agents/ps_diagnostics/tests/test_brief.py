"""Contract tests for the ps_diagnostics brief.

These tests pin the contract layer: the frozen `CAPABILITY` declaration,
the `preflight` refusal function, and the `build_brief` flag derivation.
The ReAct loop is not exercised here — agent-level integration is the
job of `test_agent.py`.

Every assertion ties back to a `Criterion.id` declared on the
capability, so the audit checklist in CLAUDE.md §4 has a 1:1 mapping
between declared criteria and tests.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.analysis.agents.base.state import AnalysisState, DatasetInfo
from src.analysis.agents.data_profiler.output import DataProfile
from src.analysis.agents.ps_diagnostics.brief import (
    CAPABILITY,
    Metrics,
    build_brief,
    preflight,
)
from src.domain.briefs import Flag


# --- fixtures ---------------------------------------------------------------


def _make_data_profile() -> DataProfile:
    """A minimal but valid DataProfile so preflight passes."""
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
    ps_diagnostics: dict | None = None,
) -> AnalysisState:
    return AnalysisState(
        job_id="test-job",
        dataset_info=DatasetInfo(url="https://example.com/data"),
        dataframe_path=dataframe_path,
        data_profile=data_profile if data_profile is not None else _make_data_profile(),
        treatment_variable=treatment_variable,
        outcome_variable=outcome_variable,
        ps_diagnostics=ps_diagnostics,
    )


# --- CAPABILITY -------------------------------------------------------------


class TestCapability:
    def test_name_is_ps_diagnostics(self):
        assert CAPABILITY.name == "ps_diagnostics"

    def test_needs_declares_the_four_required_state_keys(self):
        assert set(CAPABILITY.needs) == {
            "dataframe_path",
            "data_profile",
            "treatment_variable",
            "outcome_variable",
        }

    def test_delivers_lists_ps_diagnostics(self):
        assert "ps_diagnostics" in CAPABILITY.delivers

    def test_is_frozen(self):
        with pytest.raises(ValidationError):
            CAPABILITY.name = "something_else"

    def test_one_line_format(self):
        line = CAPABILITY.one_line()
        assert line.startswith("ps_diagnostics: ")
        assert "overlap" in line.lower() or "balanced" in line.lower()

    def test_success_criteria_cover_each_flag_this_agent_raises(self):
        flags_with_criteria = {
            c.raises_flag for c in CAPABILITY.success_criteria if c.raises_flag
        }
        assert Flag.NEEDS_NOT_MET in flags_with_criteria
        assert Flag.PRECONDITION_FAILED in flags_with_criteria
        assert Flag.POOR_OVERLAP in flags_with_criteria
        assert Flag.SMD_HIGH in flags_with_criteria
        assert Flag.CALIBRATION_OFF in flags_with_criteria


# --- preflight --------------------------------------------------------------


class TestPreflight:
    def test_refuses_when_dataframe_path_is_missing(self):
        state = _make_state(dataframe_path=None)
        brief = preflight(state)
        assert brief is not None
        assert brief.status == "refused"
        assert brief.flags == [Flag.NEEDS_NOT_MET]
        assert "dataframe_path" in brief.headline

    def test_refuses_when_data_profile_is_missing(self):
        state = _make_state(data_profile=None)
        # data_profile is None here -> need to bypass the _make_data_profile default
        state.data_profile = None
        brief = preflight(state)
        assert brief is not None
        assert brief.status == "refused"
        assert brief.flags == [Flag.NEEDS_NOT_MET]
        assert "data_profile" in brief.headline

    def test_refuses_when_treatment_variable_is_missing(self):
        state = _make_state(treatment_variable=None)
        brief = preflight(state)
        assert brief is not None
        assert brief.status == "refused"
        assert brief.flags == [Flag.PRECONDITION_FAILED]
        assert "treatment_variable" in brief.headline

    def test_refuses_when_outcome_variable_is_missing(self):
        state = _make_state(outcome_variable=None)
        brief = preflight(state)
        assert brief is not None
        assert brief.status == "refused"
        assert brief.flags == [Flag.PRECONDITION_FAILED]
        assert "outcome_variable" in brief.headline

    def test_returns_none_when_all_needs_are_satisfied(self):
        state = _make_state()
        assert preflight(state) is None

    def test_dataframe_path_missing_takes_priority_over_profile(self):
        # If both fail, NEEDS_NOT_MET fires first via dataframe_path so the
        # orchestrator gets one clear reroute hint instead of compound failure.
        state = _make_state(dataframe_path=None)
        state.data_profile = None
        brief = preflight(state)
        assert brief is not None
        assert "dataframe_path" in brief.headline


# --- build_brief: flag derivation ------------------------------------------


class TestBuildBriefFlagDerivation:
    def test_raises_poor_overlap_when_overlap_below_90(self):
        state = _make_state()
        brief = build_brief(state, Metrics(overlap_pct=85.0))
        assert Flag.POOR_OVERLAP in brief.flags

    def test_does_not_raise_poor_overlap_at_exactly_90(self):
        # Threshold is strict-less-than; at 90 we are on the line and pass.
        state = _make_state()
        brief = build_brief(state, Metrics(overlap_pct=90.0))
        assert Flag.POOR_OVERLAP not in brief.flags

    def test_does_not_raise_poor_overlap_when_metric_absent(self):
        state = _make_state()
        brief = build_brief(state, Metrics(overlap_pct=None))
        assert Flag.POOR_OVERLAP not in brief.flags

    def test_raises_smd_high_when_max_weighted_smd_above_0_1(self):
        state = _make_state()
        brief = build_brief(state, Metrics(max_smd_weighted=0.25))
        assert Flag.SMD_HIGH in brief.flags

    def test_does_not_raise_smd_high_at_exactly_0_1(self):
        state = _make_state()
        brief = build_brief(state, Metrics(max_smd_weighted=0.1))
        assert Flag.SMD_HIGH not in brief.flags

    def test_raises_calibration_off_when_mae_above_0_1(self):
        state = _make_state()
        brief = build_brief(state, Metrics(calibration_mae=0.2))
        assert Flag.CALIBRATION_OFF in brief.flags

    def test_raises_all_three_flags_when_all_thresholds_are_breached(self):
        state = _make_state()
        metrics = Metrics(
            overlap_pct=70.0,
            max_smd_weighted=0.3,
            calibration_mae=0.2,
        )
        brief = build_brief(state, metrics)
        assert Flag.POOR_OVERLAP in brief.flags
        assert Flag.SMD_HIGH in brief.flags
        assert Flag.CALIBRATION_OFF in brief.flags
        assert len(brief.flags) == 3


# --- build_brief: status, headline, artifacts ------------------------------


class TestBuildBriefStatus:
    def test_status_done_when_at_least_one_metric_captured(self):
        state = _make_state()
        brief = build_brief(state, Metrics(overlap_pct=95.0))
        assert brief.status == "done"

    def test_no_metrics_is_soft_not_a_hard_failure(self):
        # ps_diagnostics is an optional diagnostic: when it cannot compute it must
        # not halt the run, so it reports status="done" with a soft
        # PRECONDITION_FAILED flag rather than status="failed".
        state = _make_state()
        brief = build_brief(state, Metrics())
        assert brief.status == "done"
        assert Flag.PRECONDITION_FAILED in brief.flags

    def test_headline_summarises_each_captured_metric(self):
        state = _make_state()
        metrics = Metrics(
            overlap_pct=95.0,
            max_smd_weighted=0.05,
            calibration_mae=0.04,
        )
        headline = build_brief(state, metrics).headline
        assert "overlap" in headline
        assert "SMD" in headline
        assert "calibration" in headline.lower()

    def test_uncomputed_headline_says_not_computed(self):
        state = _make_state()
        brief = build_brief(state, Metrics())
        assert "not computed" in brief.headline.lower()

    def test_raised_issues_quote_the_breaching_value(self):
        state = _make_state()
        brief = build_brief(state, Metrics(overlap_pct=72.5))
        assert any("72.5" in s for s in brief.raised_issues)


class TestBuildBriefArtifactKeys:
    def test_lists_ps_diagnostics_when_state_field_populated(self):
        state = _make_state(ps_diagnostics={"model_quality": "good"})
        brief = build_brief(state, Metrics(overlap_pct=95.0))
        assert brief.artifact_keys == ["ps_diagnostics"]

    def test_empty_when_state_field_not_populated(self):
        state = _make_state(ps_diagnostics=None)
        brief = build_brief(state, Metrics(overlap_pct=95.0))
        assert brief.artifact_keys == []
