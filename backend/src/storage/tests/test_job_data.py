"""Tests for durable per-job dataset storage (raw bundle + manifest)."""

from __future__ import annotations

import types

import pytest

from src.analysis.agents.base import FileEntry
from src.storage.job_data import (
    build_manifest,
    job_data_dir,
    job_raw_dir,
    read_file_page,
    read_manifest,
    reset_job_raw_dir,
    write_manifest,
)


@pytest.fixture(autouse=True)
def storage_root(tmp_path, monkeypatch):
    """Point the storage root at a temp dir for every test here."""
    root = tmp_path / "store"
    fake = types.SimpleNamespace(local_storage_path=str(root))
    monkeypatch.setattr("src.storage.job_data.get_settings", lambda: fake)
    return root


def test_job_dirs_resolve_under_storage_root(storage_root):
    assert job_data_dir("job-1") == storage_root / "job-1"
    raw = job_raw_dir("job-1")
    assert raw == storage_root / "job-1" / "raw"
    assert raw.is_dir()  # created on access


def test_reset_job_raw_dir_drops_prior_contents():
    raw = job_raw_dir("job-2")
    (raw / "stale.csv").write_text("old")
    assert (raw / "stale.csv").exists()

    fresh = reset_job_raw_dir("job-2")
    assert fresh.is_dir()
    assert not (fresh / "stale.csv").exists()


def _state_with_bundle(job_id: str) -> types.SimpleNamespace:
    """Write a two-file bundle to disk and return a state-like object that
    mirrors what data_profiler populates before build_manifest runs.

    build_manifest reads columns and row counts straight from these files, so
    no sample data is supplied on the state."""
    raw = reset_job_raw_dir(job_id)
    (raw / "big.csv").write_text("a,b\n1,2\n3,4\n")
    (raw / "small.csv").write_text("a,b\n9,9\n")

    return types.SimpleNamespace(
        job_id=job_id,
        raw_metadata={"description": "a study", "downloadCount": 5},
        dataset_info=types.SimpleNamespace(
            url="https://www.kaggle.com/datasets/owner/name",
            files=[
                FileEntry(name="big.csv", size_bytes=12, format="csv", used=True),
                FileEntry(name="small.csv", size_bytes=4, format="csv", used=False),
            ],
        ),
    )


def test_build_manifest_describes_bundle_with_winner_and_hashes():
    state = _state_with_bundle("job-3")

    manifest = build_manifest(state)

    assert manifest.job_id == "job-3"
    assert manifest.dataset_ref == "owner/name"
    assert manifest.winner == "big.csv"
    assert {f.name for f in manifest.files} == {"big.csv", "small.csv"}

    big = next(f for f in manifest.files if f.name == "big.csv")
    assert big.used is True
    assert big.n_rows == 2
    assert big.n_columns == 2
    assert big.columns == ["a", "b"]
    assert big.relative_path == "raw/big.csv"
    assert len(big.sha256) == 64  # real content hash of the file on disk

    # Lossless metadata, including fields outside the typed Kaggle schema.
    assert manifest.kaggle_metadata["downloadCount"] == 5


def test_write_then_read_manifest_round_trips():
    state = _state_with_bundle("job-4")
    manifest = build_manifest(state)

    path = write_manifest("job-4", manifest)
    assert path == job_data_dir("job-4") / "manifest.json"
    assert path.is_file()

    loaded = read_manifest("job-4")
    assert loaded == manifest


def test_build_manifest_counts_small_file_rows_from_disk():
    """Row counts come from the files themselves, not any supplied sample."""
    manifest = build_manifest(_state_with_bundle("job-5"))
    small = next(f for f in manifest.files if f.name == "small.csv")
    assert small.n_rows == 1
    assert small.columns == ["a", "b"]
    assert small.used is False


def test_build_manifest_records_normalized_path_and_tabular_flag():
    """Each tabular file gets a normalised parquet path and tabular=True; a
    non-tabular file is listed but flagged tabular=False with no parquet."""
    import types

    job_id = "job-norm"
    raw = reset_job_raw_dir(job_id)
    (raw / "data.csv").write_text("a,b\n1,2\n")
    (raw / "logo.png").write_bytes(b"\x89PNG\r\n")
    state = types.SimpleNamespace(
        job_id=job_id,
        raw_metadata={},
        dataset_info=types.SimpleNamespace(
            url="https://www.kaggle.com/datasets/owner/name",
            files=[
                FileEntry(name="data.csv", size_bytes=8, format="csv", used=True),
                FileEntry(name="logo.png", size_bytes=8, format="other", used=False),
            ],
        ),
    )

    manifest = build_manifest(state)
    csv = next(f for f in manifest.files if f.name == "data.csv")
    png = next(f for f in manifest.files if f.name == "logo.png")

    assert csv.tabular is True
    assert csv.normalized_path == "normalized/data.csv.parquet"
    assert csv.columns == ["a", "b"]
    assert png.tabular is False
    assert png.normalized_path is None


def test_read_file_page_returns_slice_and_total():
    state = _state_with_bundle("job-page")
    write_manifest("job-page", build_manifest(state))

    page = read_file_page("job-page", "big.csv", offset=1, limit=1)
    assert page is not None
    assert page["columns"] == ["a", "b"]
    assert page["total_rows"] == 2  # from the manifest, not a re-count
    assert page["offset"] == 1 and page["limit"] == 1
    assert page["rows"] == [{"a": 3, "b": 4}]  # the second data row only


def test_read_file_page_first_page():
    state = _state_with_bundle("job-page2")
    write_manifest("job-page2", build_manifest(state))
    page = read_file_page("job-page2", "big.csv", offset=0, limit=10)
    assert [r["a"] for r in page["rows"]] == [1, 3]  # both data rows, in order


def test_read_file_page_rejects_unknown_file():
    state = _state_with_bundle("job-x2")
    write_manifest("job-x2", build_manifest(state))
    assert read_file_page("job-x2", "nope.csv", 0, 10) is None


def test_read_file_page_rejects_path_traversal():
    state = _state_with_bundle("job-x3")
    write_manifest("job-x3", build_manifest(state))
    # A name not listed in the manifest is refused before any path work.
    assert read_file_page("job-x3", "../manifest.json", 0, 10) is None


def test_read_file_page_returns_none_without_manifest():
    assert read_file_page("never-existed", "big.csv", 0, 10) is None
