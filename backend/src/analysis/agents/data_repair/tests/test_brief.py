"""Contract tests for the data_repair brief."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.analysis.agents.base.state import AnalysisState, DatasetInfo
from src.analysis.agents.data_profiler.output import DataProfile
from src.analysis.agents.data_repair.brief import CAPABILITY, build_brief, preflight
from src.domain.briefs import Flag


def _profile(
    *,
    n_samples: int = 100,
    missing_values: dict[str, int] | None = None,
) -> DataProfile:
    return DataProfile(
        n_samples=n_samples,
        n_features=3,
        feature_names=["t", "y", "x"],
        feature_types={"t": "binary", "y": "numeric", "x": "numeric"},
        missing_values=missing_values or {"t": 5, "y": 3, "x": 10},
        numeric_stats={},
        categorical_stats={},
    )


def _make_state(
    *,
    dataframe_path: str | None = "/tmp/df.parquet",
    data_profile: DataProfile | None = None,
    data_repairs: list[dict] | None = None,
) -> AnalysisState:
    state = AnalysisState(
        job_id="test-job",
        dataset_info=DatasetInfo(url="https://example.com/data"),
        dataframe_path=dataframe_path,
        data_profile=data_profile if data_profile is not None else _profile(),
    )
    if data_repairs is not None:
        state.data_repairs = data_repairs
    return state


class TestCapability:
    def test_name(self):
        assert CAPABILITY.name == "data_repair"

    def test_is_frozen(self):
        with pytest.raises(ValidationError):
            CAPABILITY.name = "other"

    def test_success_criteria_cover_each_flag(self):
        flags_with_criteria = {
            c.raises_flag for c in CAPABILITY.success_criteria if c.raises_flag
        }
        assert Flag.NEEDS_NOT_MET in flags_with_criteria
        assert Flag.REPAIR_FAILED in flags_with_criteria
        assert Flag.DATA_LOST in flags_with_criteria


class TestPreflight:
    def test_refuses_when_profile_missing(self):
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
    def test_failed_when_no_record(self):
        state = _make_state()
        state.data_repairs = None
        brief = build_brief(state)
        assert brief.status == "failed"

    def test_done_when_record_present(self):
        state = _make_state(data_repairs=[])
        brief = build_brief(state)
        assert brief.status == "done"
        assert brief.artifact_keys == ["data_repairs"]


class TestBuildBriefRepairFailed:
    def test_raises_when_profile_had_issues_but_no_repairs(self):
        # profile has missing values, repairs list is empty
        state = _make_state(data_repairs=[])
        brief = build_brief(state)
        assert Flag.REPAIR_FAILED in brief.flags

    def test_does_not_raise_when_repairs_recorded(self):
        state = _make_state(data_repairs=[
            {"type": "missing", "strategy": "median_impute", "columns": ["x"]}
        ])
        brief = build_brief(state)
        assert Flag.REPAIR_FAILED not in brief.flags

    def test_does_not_raise_when_profile_clean(self):
        clean = _profile(missing_values={"t": 0, "y": 0, "x": 0})
        state = _make_state(data_profile=clean, data_repairs=[])
        brief = build_brief(state)
        assert Flag.REPAIR_FAILED not in brief.flags


class TestBuildBriefDataLost:
    def test_raises_when_rows_dropped_exceeds_5pct(self):
        state = _make_state(data_repairs=[
            {"type": "outliers", "strategy": "drop", "rows_dropped": 10}  # 10/100 = 10%
        ])
        brief = build_brief(state)
        assert Flag.DATA_LOST in brief.flags

    def test_does_not_raise_when_rows_dropped_within_threshold(self):
        state = _make_state(data_repairs=[
            {"type": "outliers", "strategy": "drop", "rows_dropped": 3}  # 3/100 = 3%
        ])
        brief = build_brief(state)
        assert Flag.DATA_LOST not in brief.flags

    def test_aggregates_across_repairs(self):
        # 3 + 3 = 6 / 100 = 6% > 5%
        state = _make_state(data_repairs=[
            {"type": "missing", "rows_dropped": 3},
            {"type": "outliers", "rows_dropped": 3},
        ])
        brief = build_brief(state)
        assert Flag.DATA_LOST in brief.flags
