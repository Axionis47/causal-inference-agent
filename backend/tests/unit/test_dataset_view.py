"""Tests for the dataset-view assembly helpers.

The helpers translate AnalysisState (or persisted results) into the
DatasetViewResponse the frontend Data panel consumes. They must
distinguish three kinds of "missing" so the UI can render each
correctly:

    pending  — we haven't reached this step yet
    unavailable — the source genuinely does not provide it
    error    — fetch ran and failed

The mappings live in build_from_state and build_from_persisted; these
tests pin the contract.
"""

from __future__ import annotations

from src.analysis.agents.base import (
    AnalysisState,
    DataProfile,
    DatasetInfo,
    FileEntry,
    FileSample,
    JobStatus,
)
from src.api.utils import (
    build_dataset_view_from_persisted,
    build_dataset_view_from_state,
)


def _make_state(**overrides) -> AnalysisState:
    info = DatasetInfo(url="https://www.kaggle.com/datasets/owner/name", name="name")
    state = AnalysisState(job_id="t", dataset_info=info)
    for k, v in overrides.items():
        setattr(state, k, v)
    return state


# ─── build_from_state ─────────────────────────────────────────────────────


def test_state_pending_for_fresh_job():
    state = _make_state()
    view = build_dataset_view_from_state(state)
    assert view.download.status == "pending"
    assert view.download.url == state.dataset_info.url
    assert view.kaggle_meta.status == "pending"
    assert view.profile.status == "pending"


def test_state_downloading_when_status_progressed_but_no_files():
    state = _make_state(status=JobStatus.PROFILING)
    view = build_dataset_view_from_state(state)
    assert view.download.status == "downloading"


def test_state_downloaded_when_files_present():
    state = _make_state()
    state.dataset_info.files = [
        FileEntry(name="a.csv", size_bytes=100, format="csv", used=True)
    ]
    view = build_dataset_view_from_state(state)
    assert view.download.status == "downloaded"
    assert view.download.files[0].name == "a.csv"
    assert view.download.files[0].used is True


def test_state_failed_when_profiler_reported_error():
    state = _make_state()
    state.status = JobStatus.FAILED
    state.error_agent = "data_profiler"
    state.error_message = "Dataset 'owner/name' is not accessible."
    view = build_dataset_view_from_state(state)
    assert view.download.status == "failed"
    assert "not accessible" in view.download.error
    # Profile inherits the failure.
    assert view.profile.status == "error"


def test_kaggle_meta_loaded_when_fields_present():
    state = _make_state()
    state.dataset_info.kaggle_description = "Job-training RCT."
    state.dataset_info.kaggle_tags = ["economics"]
    state.dataset_info.metadata_quality = "high"
    view = build_dataset_view_from_state(state)
    assert view.kaggle_meta.status == "loaded"
    assert view.kaggle_meta.data is not None
    assert view.kaggle_meta.data.description == "Job-training RCT."
    assert view.kaggle_meta.data.metadata_quality == "high"


def test_kaggle_meta_unavailable_when_quality_set_but_fields_empty():
    """Fetch ran (quality is no longer 'unknown') but Kaggle didn't
    provide anything useful — distinct from 'pending' (haven't fetched
    yet)."""
    state = _make_state()
    state.dataset_info.metadata_quality = "low"
    view = build_dataset_view_from_state(state)
    assert view.kaggle_meta.status == "unavailable"
    assert view.kaggle_meta.data is None


def test_profile_loaded_when_data_profile_set():
    state = _make_state()
    state.data_profile = DataProfile(
        n_samples=100,
        n_features=5,
        feature_names=["a", "b", "c", "d", "e"],
        feature_types={"a": "numeric"},
        missing_values={"a": 0},
        numeric_stats={},
        categorical_stats={},
        treatment_candidates=["t"],
        outcome_candidates=["y"],
        potential_confounders=["c1"],
    )
    view = build_dataset_view_from_state(state)
    assert view.profile.status == "loaded"
    assert view.profile.data["n_samples"] == 100
    assert view.profile.data["treatment_candidates"] == ["t"]


# ─── build_from_persisted ─────────────────────────────────────────────────


def test_persisted_downloaded_with_files():
    job = {"kaggle_url": "https://www.kaggle.com/datasets/owner/name", "status": "completed"}
    results = {
        "dataset_files": [
            {"name": "a.csv", "size_bytes": 100, "format": "csv", "used": True}
        ],
        "data_profile": {"n_samples": 100, "n_features": 5},
    }
    view = build_dataset_view_from_persisted(job, results)
    assert view.download.status == "downloaded"
    assert view.download.files[0].name == "a.csv"
    # No persisted kaggle metadata — completed job, treat as unavailable
    # rather than spinning forever on pending.
    assert view.kaggle_meta.status == "unavailable"
    assert view.profile.status == "loaded"
    assert view.profile.data["n_samples"] == 100


def test_persisted_loaded_kaggle_meta():
    job = {"kaggle_url": "https://www.kaggle.com/datasets/owner/name", "status": "completed"}
    results = {
        "dataset_files": [{"name": "a.csv", "size_bytes": 1, "format": "csv", "used": True}],
        "kaggle_meta": {
            "description": "X.",
            "column_descriptions": {},
            "tags": ["t"],
            "domain": "econ",
            "metadata_quality": "high",
        },
    }
    view = build_dataset_view_from_persisted(job, results)
    assert view.kaggle_meta.status == "loaded"
    assert view.kaggle_meta.data.description == "X."


def test_persisted_failed_job_with_no_files():
    job = {
        "kaggle_url": "https://www.kaggle.com/datasets/owner/name",
        "status": "failed",
        "error_message": "Dataset not accessible.",
    }
    view = build_dataset_view_from_persisted(job, {})
    assert view.download.status == "failed"
    assert view.download.error == "Dataset not accessible."
    assert view.profile.status == "error"


def test_persisted_returns_pending_when_results_missing():
    job = {"kaggle_url": "https://www.kaggle.com/datasets/owner/name", "status": "pending"}
    view = build_dataset_view_from_persisted(job, None)
    assert view.download.status == "pending"
    assert view.kaggle_meta.status == "pending"
    assert view.profile.status == "pending"


# ─── sample rows (the raw-data preview) ───────────────────────────────────


def test_sample_pending_before_anything_downloaded():
    """No samples and no parquet yet: the panel shows progress, not error."""
    state = _make_state()
    view = build_dataset_view_from_state(state)
    assert view.sample.status == "pending"
    assert view.sample.files == []


def test_sample_multi_file_returns_one_preview_per_file():
    """Per-file samples captured at download surface as one entry each,
    with `used` marking the file we loaded."""
    state = _make_state()
    state.dataset_info.files = [
        FileEntry(name="train.csv", size_bytes=10, format="csv", used=True),
        FileEntry(name="meta.csv", size_bytes=5, format="csv", used=False),
    ]
    state.dataset_info.file_samples = [
        FileSample(
            name="train.csv",
            columns=["treat", "re78"],
            rows=[{"treat": 0, "re78": 0.0}, {"treat": 1, "re78": 9930.05}],
            total_rows=445,
        ),
        FileSample(name="meta.csv", columns=["k", "v"], rows=[{"k": "a", "v": 1}], total_rows=12),
    ]
    view = build_dataset_view_from_state(state)

    assert view.sample.status == "loaded"
    assert [f.name for f in view.sample.files] == ["train.csv", "meta.csv"]
    train = view.sample.files[0]
    assert train.used is True
    assert train.total_rows == 445
    assert train.rows[1]["re78"] == 9930.05
    assert view.sample.files[1].used is False


def test_sample_falls_back_to_parquet_when_no_file_samples(tmp_path):
    """Older jobs without per-file capture still preview the loaded parquet."""
    import pandas as pd

    df = pd.DataFrame({"x": list(range(50))})
    parquet = tmp_path / "big.parquet"
    df.to_parquet(parquet)

    state = _make_state()
    state.dataframe_path = str(parquet)
    view = build_dataset_view_from_state(state)

    assert view.sample.status == "loaded"
    assert len(view.sample.files) == 1
    f = view.sample.files[0]
    assert f.used is True
    assert len(f.rows) == 10  # head() cap
    assert f.total_rows == 50  # true count still reported


def test_sample_pending_when_fallback_path_points_nowhere():
    """A stale path (file evicted) reads as pending, not a hard error."""
    state = _make_state()
    state.dataframe_path = "/tmp/does-not-exist-12345.parquet"
    view = build_dataset_view_from_state(state)
    assert view.sample.status == "pending"
