"""DownloadRecord persistence."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.domain.download import (
    DownloadRecord,
    DownloadStatus,
    KaggleMetadata,
)
from src.download import download_store


def _make_record(
    download_id: str,
    profile_id: str = "default",
    minutes_ago: int = 0,
    status: DownloadStatus = DownloadStatus.COMPLETE,
) -> DownloadRecord:
    return DownloadRecord(
        download_id=download_id,
        profile_id=profile_id,
        kaggle_url=f"https://www.kaggle.com/datasets/o/{download_id}",
        dataset_ref=f"o/{download_id}",
        status=status,
        created_at=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
        metadata=KaggleMetadata(
            owner="o",
            slug=download_id,
            url=f"https://www.kaggle.com/datasets/o/{download_id}",
        ),
    )


def test_save_and_load_round_trip(tmp_path):
    record = _make_record("dl-1")
    download_store.save(record, root=tmp_path)
    got = download_store.load("dl-1", root=tmp_path)
    assert got == record


def test_load_returns_none_when_absent(tmp_path):
    assert download_store.load("nope", root=tmp_path) is None


def test_list_records_newest_first(tmp_path):
    download_store.save(_make_record("old", minutes_ago=60), root=tmp_path)
    download_store.save(_make_record("newer", minutes_ago=5), root=tmp_path)
    download_store.save(_make_record("newest", minutes_ago=0), root=tmp_path)

    ids = [r.download_id for r in download_store.list_records(root=tmp_path)]
    assert ids == ["newest", "newer", "old"]


def test_list_records_filters_by_profile_id(tmp_path):
    download_store.save(_make_record("a", profile_id="alice"), root=tmp_path)
    download_store.save(_make_record("b", profile_id="bob"), root=tmp_path)
    download_store.save(_make_record("c", profile_id="alice"), root=tmp_path)

    alice = {r.download_id for r in download_store.list_records(root=tmp_path, profile_id="alice")}
    assert alice == {"a", "c"}


def test_list_records_returns_empty_when_dir_missing(tmp_path):
    assert download_store.list_records(root=tmp_path) == []


def test_list_records_skips_malformed_files(tmp_path):
    download_store.save(_make_record("good"), root=tmp_path)
    (tmp_path / "downloads" / "junk.json").write_text("not valid json")

    ids = [r.download_id for r in download_store.list_records(root=tmp_path)]
    assert ids == ["good"]


def test_delete_removes_record(tmp_path):
    download_store.save(_make_record("dl-1"), root=tmp_path)
    download_store.delete("dl-1", root=tmp_path)
    assert download_store.load("dl-1", root=tmp_path) is None


def test_delete_is_idempotent(tmp_path):
    download_store.delete("never-existed", root=tmp_path)  # no error


def test_save_overwrites_existing_record(tmp_path):
    download_store.save(_make_record("dl-1", status=DownloadStatus.DOWNLOADING_FILES), root=tmp_path)
    download_store.save(_make_record("dl-1", status=DownloadStatus.COMPLETE), root=tmp_path)
    got = download_store.load("dl-1", root=tmp_path)
    assert got.status == DownloadStatus.COMPLETE
