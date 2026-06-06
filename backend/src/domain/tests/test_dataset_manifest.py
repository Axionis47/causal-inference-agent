"""Tests for the job-scoped dataset manifest contract."""

from __future__ import annotations

from src.domain import DatasetManifest, ManifestFile


def test_manifest_file_defaults_are_safe():
    """A file record needs only the required identity fields; the rest
    default to empty so a non-tabular file (no rows/cols) is still valid."""
    rec = ManifestFile(
        name="notes.txt",
        relative_path="raw/notes.txt",
        size_bytes=12,
        format="txt",
        sha256="abc",
    )
    assert rec.n_rows is None
    assert rec.n_columns is None
    assert rec.columns == []
    assert rec.used is False


def test_manifest_round_trips_through_json():
    """The manifest is persisted as JSON and re-read by the storage layer;
    a round-trip must preserve every field, including the lossless metadata."""
    manifest = DatasetManifest(
        job_id="job-1",
        kaggle_url="https://www.kaggle.com/datasets/owner/name",
        dataset_ref="owner/name",
        raw_dir="/data/job-1/raw",
        files=[
            ManifestFile(
                name="data.csv",
                relative_path="raw/data.csv",
                size_bytes=100,
                format="csv",
                sha256="deadbeef",
                n_rows=42,
                n_columns=3,
                columns=["a", "b", "c"],
                used=True,
            )
        ],
        winner="data.csv",
        kaggle_metadata={"description": "study", "downloadCount": 7},
    )

    restored = DatasetManifest.model_validate_json(manifest.model_dump_json())

    assert restored == manifest
    assert restored.winner == "data.csv"
    assert restored.files[0].columns == ["a", "b", "c"]
    # Stats outside the typed metadata schema survive because the blob is a dict.
    assert restored.kaggle_metadata["downloadCount"] == 7


def test_manifest_defaults_to_empty_bundle():
    """A manifest built before any files land is still valid and empty."""
    manifest = DatasetManifest(
        job_id="job-2",
        kaggle_url="https://www.kaggle.com/datasets/owner/name",
        raw_dir="/data/job-2/raw",
    )
    assert manifest.files == []
    assert manifest.winner is None
    assert manifest.kaggle_metadata == {}
