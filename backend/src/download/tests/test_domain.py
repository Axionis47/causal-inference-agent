"""Round-trip tests for download domain types.

The point is the contract: every downstream consumer (analysis stages,
the API, the SSE transport) reads these models. If serialization breaks
silently the whole subsystem breaks silently.
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.domain.download import (
    DownloadEvent,
    DownloadRecord,
    DownloadRequest,
    DownloadStatus,
    FileEntry,
    KaggleMetadata,
    KaggleProfile,
)


def test_download_status_string_values():
    # the enum values are persisted to disk so they must stay stable
    assert DownloadStatus.QUEUED.value == "queued"
    assert DownloadStatus.COMPLETE.value == "complete"
    assert DownloadStatus.FAILED.value == "failed"


def test_file_entry_round_trip():
    entry = FileEntry(
        name="data.csv",
        relative_path="data.csv",
        size_bytes=12345,
        format="csv",
        sha256="a" * 64,
    )
    payload = entry.model_dump()
    restored = FileEntry.model_validate(payload)
    assert restored == entry


def test_kaggle_metadata_minimal_fields():
    meta = KaggleMetadata(owner="uciml", slug="iris", url="https://www.kaggle.com/datasets/uciml/iris")
    assert meta.owner == "uciml"
    assert meta.tags == []
    assert meta.column_descriptions == {}
    assert meta.license_name is None


def test_kaggle_profile_never_carries_raw_key():
    # The model intentionally has no field for the key itself; only "has_key".
    profile = KaggleProfile(
        profile_id="default",
        kaggle_username="alice",
        has_key=True,
        validated_at=datetime(2026, 5, 26, tzinfo=timezone.utc),
    )
    assert "key" not in profile.model_dump()
    assert "kaggle_key" not in profile.model_dump()


def test_download_request_default_profile():
    req = DownloadRequest(kaggle_url="https://www.kaggle.com/datasets/uciml/iris")
    assert req.profile_id == "default"


def test_download_record_round_trip_with_files_and_metadata():
    record = DownloadRecord(
        download_id="dl-1",
        profile_id="default",
        kaggle_url="https://www.kaggle.com/datasets/uciml/iris",
        dataset_ref="uciml/iris",
        status=DownloadStatus.COMPLETE,
        storage_path="data/datasets/default/uciml__iris/1/",
        files=[
            FileEntry(
                name="Iris.csv",
                relative_path="Iris.csv",
                size_bytes=5000,
                format="csv",
                sha256="b" * 64,
            )
        ],
        metadata=KaggleMetadata(
            owner="uciml",
            slug="iris",
            url="https://www.kaggle.com/datasets/uciml/iris",
            tags=["classification"],
        ),
        created_at=datetime(2026, 5, 26, tzinfo=timezone.utc),
        completed_at=datetime(2026, 5, 26, tzinfo=timezone.utc),
    )
    payload = record.model_dump(mode="json")
    restored = DownloadRecord.model_validate(payload)
    assert restored == record


def test_download_event_round_trip():
    event = DownloadEvent(
        download_id="dl-1",
        event_type="metadata.ready",
        timestamp=datetime(2026, 5, 26, tzinfo=timezone.utc),
        data={"tags": ["x"], "owner": "uciml"},
    )
    payload = event.model_dump(mode="json")
    restored = DownloadEvent.model_validate(payload)
    assert restored == event
