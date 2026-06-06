"""Contract tests for the data_profiler brief."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.analysis.agents.base.state import AnalysisState, DatasetInfo
from src.analysis.agents.data_profiler.brief import (
    CAPABILITY,
    build_brief,
    preflight,
)
from src.analysis.agents.data_profiler.output import DataProfile, TreatmentEncoding
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
        missing_values=missing_values or {"t": 0, "y": 0, "x": 0},
        numeric_stats={},
        categorical_stats={},
        treatment_candidates=["t"],
        outcome_candidates=["y"],
    )


def _make_state(
    *,
    data_profile: DataProfile | None = None,
    dataframe_path: str | None = "/tmp/df.parquet",
    treatment_encoding: TreatmentEncoding | None = None,
) -> AnalysisState:
    state = AnalysisState(
        job_id="test-job",
        dataset_info=DatasetInfo(url="https://example.com/data"),
        dataframe_path=dataframe_path,
        data_profile=data_profile,
        treatment_encoding=treatment_encoding,
    )
    return state


class TestCapability:
    def test_name(self):
        assert CAPABILITY.name == "data_profiler"

    def test_needs_dataset_info(self):
        assert CAPABILITY.needs == ("dataset_info",)

    def test_is_frozen(self):
        with pytest.raises(ValidationError):
            CAPABILITY.name = "other"

    def test_success_criteria_cover_each_flag(self):
        flags_with_criteria = {
            c.raises_flag for c in CAPABILITY.success_criteria if c.raises_flag
        }
        assert Flag.HIGH_MISSINGNESS in flags_with_criteria
        assert Flag.ENCODING_REQUIRED in flags_with_criteria


class TestPreflight:
    def test_never_refuses(self):
        # Profiler is the entry point; preflight always passes.
        assert preflight(_make_state()) is None
        assert preflight(_make_state(data_profile=_profile())) is None


class TestBuildBriefStatus:
    def test_failed_when_no_profile(self):
        state = _make_state(data_profile=None)
        brief = build_brief(state)
        assert brief.status == "failed"
        assert brief.artifact_keys == []

    def test_done_when_profile_present(self):
        state = _make_state(data_profile=_profile())
        brief = build_brief(state)
        assert brief.status == "done"
        assert "data_profile" in brief.artifact_keys
        assert "dataframe_path" in brief.artifact_keys


class TestBuildBriefHighMissingness:
    def test_raises_when_column_over_50_pct(self):
        profile = _profile(
            n_samples=100,
            missing_values={"t": 0, "y": 0, "x": 80},  # 80% missing
        )
        state = _make_state(data_profile=profile)
        brief = build_brief(state)
        assert Flag.HIGH_MISSINGNESS in brief.flags

    def test_does_not_raise_at_50_pct_exactly(self):
        profile = _profile(
            n_samples=100,
            missing_values={"t": 0, "y": 0, "x": 50},  # exactly 50%
        )
        state = _make_state(data_profile=profile)
        brief = build_brief(state)
        assert Flag.HIGH_MISSINGNESS not in brief.flags

    def test_does_not_raise_when_all_clean(self):
        state = _make_state(data_profile=_profile())
        brief = build_brief(state)
        assert Flag.HIGH_MISSINGNESS not in brief.flags


class TestBuildBriefEncodingRequired:
    def test_raises_when_encoding_strategy_set(self):
        encoding = TreatmentEncoding(
            original_type="multi_categorical",
            strategy="collapse_to_binary",
            control_value="No",
            value_mapping={"No": 0, "Yes": 1},
        )
        state = _make_state(
            data_profile=_profile(),
            treatment_encoding=encoding,
        )
        brief = build_brief(state)
        assert Flag.ENCODING_REQUIRED in brief.flags

    def test_does_not_raise_when_strategy_is_none(self):
        encoding = TreatmentEncoding(original_type="binary", strategy="none")
        state = _make_state(
            data_profile=_profile(),
            treatment_encoding=encoding,
        )
        brief = build_brief(state)
        assert Flag.ENCODING_REQUIRED not in brief.flags

    def test_does_not_raise_when_no_encoding(self):
        state = _make_state(data_profile=_profile())
        brief = build_brief(state)
        assert Flag.ENCODING_REQUIRED not in brief.flags
