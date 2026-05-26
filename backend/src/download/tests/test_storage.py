"""On-disk dataset layout and manifest."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from src.domain.download import (
    DownloadRecord,
    DownloadStatus,
    KaggleMetadata,
)
from src.download import storage


def test_dataset_dir_includes_profile_owner_slug_version(tmp_path):
    got = storage.dataset_dir(tmp_path, "default", "uciml", "iris", "3")
    assert got == tmp_path / "datasets" / "default" / "uciml__iris" / "3"


def test_dataset_dir_falls_back_to_latest_when_version_none(tmp_path):
    got = storage.dataset_dir(tmp_path, "default", "uciml", "iris", None)
    assert got.name == "latest"


def test_index_directory_returns_empty_for_missing_dir(tmp_path):
    assert storage.index_directory(tmp_path / "nope") == []


def test_index_directory_walks_files_with_sha256_and_format(tmp_path):
    (tmp_path / "Iris.csv").write_bytes(b"a,b,c\n1,2,3\n")
    (tmp_path / "notes.md").write_text("# Notes")
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01")

    entries = storage.index_directory(tmp_path)
    by_name = {e.name: e for e in entries}

    assert set(by_name) == {"Iris.csv", "notes.md", "blob.bin"}
    assert by_name["Iris.csv"].format == "csv"
    assert by_name["notes.md"].format == "md"
    assert by_name["blob.bin"].format == "other"

    expected_sha = hashlib.sha256(b"a,b,c\n1,2,3\n").hexdigest()
    assert by_name["Iris.csv"].sha256 == expected_sha
    assert by_name["Iris.csv"].size_bytes == len(b"a,b,c\n1,2,3\n")


def test_index_directory_excludes_manifest(tmp_path):
    (tmp_path / "data.csv").write_text("x")
    (tmp_path / "_manifest.json").write_text("{}")
    names = [e.name for e in storage.index_directory(tmp_path)]
    assert names == ["data.csv"]


def test_index_directory_is_sorted_by_relative_path(tmp_path):
    (tmp_path / "z.csv").write_text("z")
    (tmp_path / "a.csv").write_text("a")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "b.csv").write_text("b")

    entries = storage.index_directory(tmp_path)
    assert [e.relative_path for e in entries] == ["a.csv", "sub/b.csv", "z.csv"]


def test_write_and_read_manifest_round_trip(tmp_path):
    record = DownloadRecord(
        download_id="dl-1",
        profile_id="default",
        kaggle_url="https://www.kaggle.com/datasets/uciml/iris",
        dataset_ref="uciml/iris",
        status=DownloadStatus.COMPLETE,
        storage_path=str(tmp_path),
        files=[],
        metadata=KaggleMetadata(
            owner="uciml",
            slug="iris",
            url="https://www.kaggle.com/datasets/uciml/iris",
        ),
        created_at=datetime(2026, 5, 26, tzinfo=timezone.utc),
    )
    storage.write_manifest(tmp_path, record)
    restored = storage.read_manifest(tmp_path)
    assert restored == record


def test_read_manifest_returns_none_when_absent(tmp_path):
    assert storage.read_manifest(tmp_path) is None
