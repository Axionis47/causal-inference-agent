"""Contract tests for the confounder_discovery brief."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.analysis.agents.base.state import AnalysisState, DatasetInfo
from src.analysis.agents.confounder_discovery.brief import (
    CAPABILITY,
    build_brief,
    preflight,
)
from src.analysis.agents.data_profiler.output import DataProfile
from src.domain.briefs import Flag


def _profile() -> DataProfile:
    return DataProfile(
        n_samples=100,
        n_features=3,
        feature_names=["t", "y", "x"],
        feature_types={"t": "binary", "y": "numeric", "x": "numeric"},
        missing_values={"t": 0, "y": 0, "x": 0},
        numeric_stats={},
        categorical_stats={},
    )


def _make_state(
    *,
    dataframe_path: str | None = "/tmp/df.parquet",
    data_profile: DataProfile | None = None,
    confounder_discovery: dict | None = None,
) -> AnalysisState:
    state = AnalysisState(
        job_id="test-job",
        dataset_info=DatasetInfo(url="https://example.com/data"),
        dataframe_path=dataframe_path,
        data_profile=data_profile if data_profile is not None else _profile(),
        treatment_variable="t",
        outcome_variable="y",
    )
    if confounder_discovery is not None:
        state.confounder_discovery = confounder_discovery
    return state


class TestCapability:
    def test_name_is_confounder_discovery(self):
        assert CAPABILITY.name == "confounder_discovery"

    def test_needs_profile_and_dataframe(self):
        assert set(CAPABILITY.needs) == {"data_profile", "dataframe_path"}

    def test_is_frozen(self):
        with pytest.raises(ValidationError):
            CAPABILITY.name = "other"

    def test_success_criteria_cover_each_flag(self):
        flags_with_criteria = {
            c.raises_flag for c in CAPABILITY.success_criteria if c.raises_flag
        }
        assert Flag.NEEDS_NOT_MET in flags_with_criteria
        assert Flag.WEAK_CONFOUNDER_EVIDENCE in flags_with_criteria


class TestPreflight:
    def test_refuses_when_data_profile_missing(self):
        state = _make_state()
        state.data_profile = None
        brief = preflight(state)
        assert brief is not None
        assert brief.flags == [Flag.NEEDS_NOT_MET]

    def test_refuses_when_dataframe_path_missing(self):
        state = _make_state(dataframe_path=None)
        brief = preflight(state)
        assert brief is not None
        assert brief.flags == [Flag.NEEDS_NOT_MET]

    def test_passes_when_needs_satisfied(self):
        assert preflight(_make_state()) is None


class TestBuildBriefStatus:
    def test_failed_when_no_result(self):
        state = _make_state(confounder_discovery=None)
        brief = build_brief(state)
        assert brief.status == "failed"
        assert brief.artifact_keys == []

    def test_done_when_result_present(self):
        state = _make_state(confounder_discovery={
            "ranked_confounders": ["age", "income"],
            "excluded_variables": ["zip"],
        })
        brief = build_brief(state)
        assert brief.status == "done"
        assert brief.artifact_keys == ["confounder_discovery"]


class TestBuildBriefWeakConfounder:
    def test_raises_when_ranked_empty(self):
        state = _make_state(confounder_discovery={
            "ranked_confounders": [],
            "excluded_variables": [],
        })
        brief = build_brief(state)
        assert Flag.WEAK_CONFOUNDER_EVIDENCE in brief.flags

    def test_does_not_raise_when_ranked_present(self):
        state = _make_state(confounder_discovery={
            "ranked_confounders": ["age"],
            "excluded_variables": [],
        })
        brief = build_brief(state)
        assert Flag.WEAK_CONFOUNDER_EVIDENCE not in brief.flags
