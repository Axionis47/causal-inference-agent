"""Contract tests for the notebook_generator brief."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.analysis.agents.base.state import AnalysisState, DatasetInfo
from src.analysis.agents.data_profiler.output import DataProfile
from src.analysis.agents.notebook.brief import CAPABILITY, build_brief, preflight
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
    data_profile: DataProfile | None = None,
    notebook_path: str | None = None,
) -> AnalysisState:
    state = AnalysisState(
        job_id="test-job",
        dataset_info=DatasetInfo(url="https://example.com/data"),
        data_profile=data_profile if data_profile is not None else _profile(),
    )
    if notebook_path is not None:
        state.notebook_path = notebook_path
    return state


class TestCapability:
    def test_name(self):
        assert CAPABILITY.name == "notebook_generator"

    def test_needs_data_profile(self):
        assert CAPABILITY.needs == ("data_profile",)

    def test_is_frozen(self):
        with pytest.raises(ValidationError):
            CAPABILITY.name = "other"

    def test_capability_does_not_advertise_issue_flags(self):
        # Notebook is a terminal renderer; only NEEDS_NOT_MET appears.
        flags_with_criteria = {
            c.raises_flag for c in CAPABILITY.success_criteria if c.raises_flag
        }
        assert flags_with_criteria == {Flag.NEEDS_NOT_MET}


class TestPreflight:
    def test_refuses_when_profile_missing(self):
        state = _make_state()
        state.data_profile = None
        brief = preflight(state)
        assert brief is not None
        assert brief.flags == [Flag.NEEDS_NOT_MET]

    def test_passes_when_profile_present(self):
        assert preflight(_make_state()) is None


class TestBuildBriefStatus:
    def test_failed_when_no_notebook_path(self):
        state = _make_state(notebook_path=None)
        brief = build_brief(state)
        assert brief.status == "failed"
        assert brief.artifact_keys == []

    def test_done_when_notebook_path_written(self):
        state = _make_state(notebook_path="/tmp/notebook.ipynb")
        brief = build_brief(state)
        assert brief.status == "done"
        assert brief.artifact_keys == ["notebook_path"]
        assert "/tmp/notebook.ipynb" in brief.headline

    def test_emits_no_closed_enum_flags_on_success(self):
        state = _make_state(notebook_path="/tmp/notebook.ipynb")
        brief = build_brief(state)
        assert brief.flags == []
