"""AnalysisState.file_profiles: per-candidate-file DataProfile map.

Populated by dataset_inspector before the rest of the pipeline runs;
exposes the multi-file reality the UI renders. data_profile (singular)
stays as the chosen winner the downstream agents read.
"""
from __future__ import annotations

from src.analysis.agents.base.state import AnalysisState, DatasetInfo
from src.analysis.agents.data_profiler import DataProfile


def _base_state() -> AnalysisState:
    return AnalysisState(
        job_id="job-1",
        dataset_info=DatasetInfo(url="kaggle.com/x"),
    )


def _profile(n: int) -> DataProfile:
    return DataProfile(
        n_samples=n,
        n_features=3,
        feature_names=["a", "b", "c"],
        feature_types={"a": "numeric", "b": "binary", "c": "categorical"},
        missing_values={"a": 0, "b": 0, "c": 0},
        numeric_stats={"a": {"mean": 0.0, "std": 1.0, "min": -1.0, "max": 1.0}},
        categorical_stats={"c": {"x": 5, "y": 5}},
    )


def test_file_profiles_defaults_to_empty_dict():
    state = _base_state()
    assert state.file_profiles == {}


def test_file_profiles_accepts_one_entry_per_candidate_file():
    state = _base_state()
    state.file_profiles["train.csv"] = _profile(1000)
    state.file_profiles["test.csv"] = _profile(250)
    assert set(state.file_profiles) == {"train.csv", "test.csv"}
    assert state.file_profiles["train.csv"].n_samples == 1000


def test_file_profiles_round_trips_through_json_mode():
    state = _base_state()
    state.file_profiles["train.csv"] = _profile(1000)
    dumped = state.model_dump(mode="json")
    restored = AnalysisState.model_validate(dumped)
    assert restored.file_profiles.keys() == {"train.csv"}
    assert restored.file_profiles["train.csv"].n_samples == 1000


def test_file_profiles_and_data_profile_are_independent():
    """data_profile is the chosen winner; file_profiles holds all candidates.
    Updating one does not implicitly update the other."""
    state = _base_state()
    state.file_profiles["train.csv"] = _profile(1000)
    state.file_profiles["test.csv"] = _profile(250)
    assert state.data_profile is None
    state.data_profile = state.file_profiles["train.csv"]
    # Mutating the singular slot does not retroactively change the dict
    state.data_profile.n_samples = 999
    assert state.file_profiles["test.csv"].n_samples == 250
