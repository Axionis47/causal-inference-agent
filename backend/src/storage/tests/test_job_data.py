"""Tests for durable per-job dataset storage (raw bundle + manifest)."""

from __future__ import annotations

import types

import pytest

from src.analysis.agents.base import FileEntry, FileSample
from src.storage.job_data import (
    build_manifest,
    job_data_dir,
    job_raw_dir,
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
    mirrors what data_profiler populates before build_manifest runs."""
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
            file_samples=[
                FileSample(
                    name="big.csv", columns=["a", "b"], rows=[], total_rows=2
                ),
                FileSample(
                    name="small.csv", columns=["a", "b"], rows=[], total_rows=1
                ),
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


def test_read_manifest_returns_none_when_absent():
    assert read_manifest("never-existed") is None
