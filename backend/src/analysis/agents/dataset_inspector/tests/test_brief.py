"""CAPABILITY + preflight + build_brief for dataset_inspector."""
from __future__ import annotations

from src.analysis.agents.base.state import (
    AnalysisState,
    DatasetInfo,
    FileEntry,
)
from src.analysis.agents.data_profiler import DataProfile
from src.analysis.agents.dataset_inspector import CAPABILITY
from src.analysis.agents.dataset_inspector.brief import (
    AGENT_NAME,
    build_brief,
    preflight,
)
from src.domain.briefs import Flag


def _profile(n: int = 1000, treatment: int = 1, outcome: int = 1) -> DataProfile:
    return DataProfile(
        n_samples=n,
        n_features=3,
        feature_names=["a", "b", "c"],
        feature_types={"a": "numeric", "b": "binary", "c": "categorical"},
        missing_values={"a": 0, "b": 0, "c": 0},
        numeric_stats={"a": {"mean": 0.0, "std": 1.0, "min": -1.0, "max": 1.0}},
        categorical_stats={"c": {"x": 5, "y": 5}},
        treatment_candidates=[f"t{i}" for i in range(treatment)],
        outcome_candidates=[f"y{i}" for i in range(outcome)],
    )


def _state(files: list[FileEntry]) -> AnalysisState:
    return AnalysisState(
        job_id="job-1",
        dataset_info=DatasetInfo(url="kaggle.com/x", files=files),
    )


# --- Capability shape -------------------------------------------------------


def test_capability_is_frozen_and_named_dataset_inspector():
    assert CAPABILITY.name == AGENT_NAME == "dataset_inspector"
    assert CAPABILITY.model_config.get("frozen") is True


def test_capability_declares_dataset_info_as_only_need():
    assert CAPABILITY.needs == ("dataset_info",)


def test_capability_delivers_advertise_the_winner_slots():
    delivers = set(CAPABILITY.delivers)
    assert {"file_profiles", "data_profile", "dataframe_path"}.issubset(delivers)


# --- preflight: refusal when no data files ---------------------------------


def test_preflight_refuses_when_files_list_is_empty():
    brief = preflight(_state([]))
    assert brief is not None
    assert brief.status == "refused"
    assert Flag.NEEDS_NOT_MET in brief.flags


def test_preflight_refuses_when_only_metadata_files_present():
    """README.md and dataset-metadata.json are not analyseable on their own."""
    files = [
        FileEntry(name="README.md", size_bytes=100, format="md"),
        FileEntry(name="metadata.json", size_bytes=200, format="json"),
    ]
    brief = preflight(_state(files))
    assert brief is not None
    assert brief.status == "refused"


def test_preflight_allows_through_when_csv_present():
    files = [FileEntry(name="train.csv", size_bytes=1000, format="csv")]
    assert preflight(_state(files)) is None


def test_preflight_allows_through_when_parquet_present():
    files = [FileEntry(name="data.parquet", size_bytes=1000, format="parquet")]
    assert preflight(_state(files)) is None


# --- build_brief: winner-pointing shape ------------------------------------


def test_build_brief_done_when_winner_is_flagged_used_and_profiled():
    files = [
        FileEntry(name="train.csv", size_bytes=10_000, format="csv", used=True),
        FileEntry(name="test.csv", size_bytes=2_000, format="csv", used=False),
    ]
    state = _state(files)
    state.file_profiles["train.csv"] = _profile(n=950, treatment=1, outcome=1)
    state.file_profiles["test.csv"] = _profile(n=200, treatment=0, outcome=0)

    brief = build_brief(state)
    assert brief.status == "done"
    assert "train.csv" in brief.headline
    assert "950 rows" in brief.headline
    assert set(brief.artifact_keys) == {"file_profiles", "data_profile", "dataframe_path"}


def test_build_brief_failed_when_no_file_marked_used():
    """Defence in depth: if execute() returned without flipping the
    used flag (bug), the brief must surface failure rather than silently
    claim success."""
    files = [FileEntry(name="train.csv", size_bytes=10_000, format="csv", used=False)]
    state = _state(files)
    state.file_profiles["train.csv"] = _profile()
    brief = build_brief(state)
    assert brief.status == "failed"
    assert "no winner" in brief.headline


def test_build_brief_failed_when_winner_has_no_profile():
    """Another defensive case: used=True but no profile recorded."""
    files = [FileEntry(name="train.csv", size_bytes=10_000, format="csv", used=True)]
    state = _state(files)
    # file_profiles intentionally empty
    brief = build_brief(state)
    assert brief.status == "failed"
