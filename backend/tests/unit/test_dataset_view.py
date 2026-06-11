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

import types

import pytest

from src.analysis_v2.state import (
    AnalysisState,
    DataProfile,
    DatasetInfo,
    FileEntry,
    JobStatus,
)
from src.api.utils import (
    build_dataset_view_from_persisted,
    build_dataset_view_from_state,
)
from src.domain import DatasetManifest, ManifestFile
from src.storage.job_data import write_manifest


@pytest.fixture(autouse=True)
def isolate_job_storage(tmp_path, monkeypatch):
    """Point the per-job storage root at a temp dir so the live view reads
    only manifests this test wrote (the download file list comes from the
    manifest now)."""
    fake = types.SimpleNamespace(local_storage_path=str(tmp_path / "store"))
    monkeypatch.setattr("src.storage.job_data.get_settings", lambda: fake)


def _make_state(**overrides) -> AnalysisState:
    info = DatasetInfo(url="https://www.kaggle.com/datasets/owner/name", name="name")
    state = AnalysisState(job_id="t", dataset_info=info)
    for k, v in overrides.items():
        setattr(state, k, v)
    return state


def _manifest_file(name: str, used: bool, columns: list[str], n_rows: int) -> ManifestFile:
    return ManifestFile(
        name=name,
        relative_path=f"raw/{name}",
        size_bytes=10,
        format="csv",
        sha256="x",
        n_rows=n_rows,
        n_columns=len(columns),
        columns=columns,
        used=used,
    )


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


def test_state_downloaded_files_carry_columns_and_rows_from_manifest():
    """Status flips to downloaded once files are captured; the file list,
    with columns + row counts, comes from the manifest (single source)."""
    state = _make_state()
    state.dataset_info.files = [
        FileEntry(name="a.csv", size_bytes=100, format="csv", used=True)
    ]
    write_manifest(
        state.job_id,
        DatasetManifest(
            job_id=state.job_id,
            kaggle_url=state.dataset_info.url,
            raw_dir="/tmp/raw",
            files=[_manifest_file("a.csv", used=True, columns=["x", "y"], n_rows=42)],
            winner="a.csv",
        ),
    )
    view = build_dataset_view_from_state(state)
    assert view.download.status == "downloaded"
    f = view.download.files[0]
    assert f.name == "a.csv"
    assert f.used is True
    assert f.columns == ["x", "y"]
    assert f.n_rows == 42


def test_state_file_entries_carry_tabular_flag_from_manifest():
    """The view surfaces each file's tabular flag so the UI can offer a row
    preview for tabular files and mark non-tabular ones as not previewable."""
    state = _make_state()
    state.dataset_info.files = [
        FileEntry(name="a.csv", size_bytes=100, format="csv", used=True),
        FileEntry(name="logo.png", size_bytes=50, format="other", used=False),
    ]
    write_manifest(
        state.job_id,
        DatasetManifest(
            job_id=state.job_id,
            kaggle_url=state.dataset_info.url,
            raw_dir="/tmp/raw",
            files=[
                _manifest_file("a.csv", used=True, columns=["x"], n_rows=1),
                ManifestFile(
                    name="logo.png",
                    relative_path="raw/logo.png",
                    size_bytes=50,
                    format="other",
                    sha256="y",
                    tabular=False,
                ),
            ],
            winner="a.csv",
        ),
    )
    files = {f.name: f for f in build_dataset_view_from_state(state).download.files}
    assert files["a.csv"].tabular is True
    assert files["logo.png"].tabular is False


def test_state_downloaded_but_no_manifest_yet_has_empty_file_list():
    """Brief window where files are captured but the manifest is not on disk:
    status is downloaded, file list is empty rather than wrong."""
    state = _make_state()
    state.dataset_info.files = [
        FileEntry(name="a.csv", size_bytes=100, format="csv", used=True)
    ]
    view = build_dataset_view_from_state(state)
    assert view.download.status == "downloaded"
    assert view.download.files == []


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


def test_kaggle_meta_carries_full_raw_metadata_from_state():
    """At the data-review gate the full raw Kaggle dict is on
    state.raw_metadata; the view must surface every fact (title, license,
    counts, ...) and leave the derived domain unset."""
    state = _make_state()
    state.raw_metadata = {
        "extraction_success": True,
        "title": "Lalonde A/B Testing",
        "subtitle": "A/B test training program",
        "description": "### Context\nFamous job-training dataset.",
        "source": "https://www.kaggle.com/datasets/o/lalonde",
        "license": "CC0-1.0",
        "tags": ["business", "education"],
        "keywords": ["business", "education"],
        "column_descriptions": {},
        "total_size": 9734,
        "download_count": 552,
        "vote_count": 6,
        "usability_rating": 0.647,
        "metadata_quality": "medium",
    }
    view = build_dataset_view_from_state(state)
    d = view.kaggle_meta.data
    assert view.kaggle_meta.status == "loaded"
    assert d is not None
    assert d.title == "Lalonde A/B Testing"
    assert d.license == "CC0-1.0"
    assert d.tags == ["business", "education"]
    assert d.download_count == 552
    assert d.usability_rating == 0.647
    # Derived field is not inferred at the gate.
    assert d.domain is None


def test_kaggle_meta_zero_counts_become_none_for_placeholder():
    """A source that reports 0 downloads/votes should render as a
    placeholder, not a misleading zero."""
    state = _make_state()
    state.raw_metadata = {
        "extraction_success": True,
        "title": "Sparse set",
        "total_size": 0,
        "download_count": 0,
        "vote_count": 0,
        "usability_rating": 0,
    }
    d = build_dataset_view_from_state(state).kaggle_meta.data
    assert d.download_count is None
    assert d.vote_count is None
    assert d.usability_rating is None


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


def test_profile_surfaces_time_tag():
    """The profile block carries the deterministic time tag so the preview can
    show whether the dataset has a time dimension."""
    state = _make_state()
    state.data_profile = DataProfile(
        n_samples=12,
        n_features=2,
        feature_names=["date", "sales"],
        feature_types={"date": "datetime", "sales": "numeric"},
        missing_values={"date": 0, "sales": 0},
        numeric_stats={},
        categorical_stats={},
        has_time_dimension=True,
        time_column="date",
    )
    view = build_dataset_view_from_state(state)
    assert view.profile.status == "loaded"
    assert view.profile.data["has_time_dimension"] is True
    assert view.profile.data["time_column"] == "date"


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


# ─── file list sourced from the manifest ──────────────────────────────────


def test_state_no_files_before_download():
    """Fresh job: empty file list, no rows embedded anywhere."""
    state = _make_state()
    view = build_dataset_view_from_state(state)
    assert view.download.files == []


def test_persisted_files_come_from_manifest_when_present():
    """A completed job's file list (with columns + rows) is read from the
    persisted manifest, marking the winner."""
    job = {"kaggle_url": "https://www.kaggle.com/datasets/owner/name", "status": "completed"}
    manifest = DatasetManifest(
        job_id="t",
        kaggle_url=job["kaggle_url"],
        raw_dir="/tmp/raw",
        files=[
            _manifest_file("train.csv", used=True, columns=["treat", "re78"], n_rows=445),
            _manifest_file("meta.csv", used=False, columns=["k", "v"], n_rows=12),
        ],
        winner="train.csv",
    )
    results = {"dataset_manifest": manifest.model_dump()}
    view = build_dataset_view_from_persisted(job, results)

    assert view.download.status == "downloaded"
    assert [f.name for f in view.download.files] == ["train.csv", "meta.csv"]
    train = view.download.files[0]
    assert train.used is True
    assert train.columns == ["treat", "re78"]
    assert train.n_rows == 445
    assert view.download.files[1].used is False


def test_persisted_falls_back_to_dataset_files_without_manifest():
    """Pre-manifest jobs still list files from dataset_files, without
    columns/row counts."""
    job = {"kaggle_url": "https://www.kaggle.com/datasets/owner/name", "status": "completed"}
    results = {
        "dataset_files": [
            {"name": "a.csv", "size_bytes": 100, "format": "csv", "used": True}
        ]
    }
    view = build_dataset_view_from_persisted(job, results)
    assert view.download.status == "downloaded"
    assert view.download.files[0].name == "a.csv"
    assert view.download.files[0].columns == []
    assert view.download.files[0].n_rows is None
