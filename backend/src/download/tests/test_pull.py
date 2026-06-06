"""Pull orchestrator: end-to-end flow with the Kaggle client mocked.

These tests pin the event sequence and the final record shape, since
those are the contracts the frontend and analysis stages will consume.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.fernet import Fernet

from src.domain.download import DownloadRequest, DownloadStatus, KaggleMetadata
from src.download import (
    download_store,
    events,
    profile_store,
    pull,
    storage,
)


@pytest.fixture
def enc_key() -> str:
    return Fernet.generate_key().decode()


@pytest.fixture
def bus() -> events.DownloadEventBus:
    return events.DownloadEventBus()


@pytest.fixture
def with_profile(tmp_path, enc_key):
    profile_store.set_kaggle_key(
        "default",
        username="alice",
        key="real-key",
        root=tmp_path,
        encryption_key=enc_key,
    )
    return tmp_path


def _mock_client_class(meta: KaggleMetadata | None = None, write_files: list[tuple[str, bytes]] | None = None):
    """Build a patch target for KaggleClient that simulates a successful download."""
    instance = MagicMock()
    instance.fetch_metadata = AsyncMock(return_value=meta or KaggleMetadata(
        owner="uciml", slug="iris", url="https://www.kaggle.com/datasets/uciml/iris",
        title="Iris", tags=["classification"], version="1",
    ))

    async def _download_files(owner, slug, dest, *, version=None):
        Path(dest).mkdir(parents=True, exist_ok=True)
        for name, content in (write_files or [("Iris.csv", b"a,b\n1,2\n")]):
            (Path(dest) / name).write_bytes(content)

    instance.download_files = AsyncMock(side_effect=_download_files)
    instance.list_files = AsyncMock(return_value=[])
    return instance


def _collected_events(bus: events.DownloadEventBus, download_id: str) -> list[str]:
    return [e.event_type for e in bus.history(download_id)]


@pytest.mark.asyncio
async def test_happy_path_writes_files_indexes_them_and_completes(with_profile, enc_key, bus):
    root = with_profile

    req = DownloadRequest(kaggle_url="https://www.kaggle.com/datasets/uciml/iris")
    fake_instance = _mock_client_class(
        write_files=[("Iris.csv", b"a,b\n1,2\n"), ("README.md", b"# notes")],
    )

    with patch("src.download.pull.client_mod.KaggleClient", return_value=fake_instance):
        download_id = await pull.start_download(req, root=root, encryption_key=enc_key, bus=bus)
        # Drain background task by awaiting one execute_pull directly:
        record = download_store.load(download_id, root=root)
        record = await pull.execute_pull(record, root=root, encryption_key=enc_key, bus=bus)

    assert record.status == DownloadStatus.COMPLETE
    assert record.error is None
    assert record.storage_path is not None

    stored_dir = Path(record.storage_path)
    assert (stored_dir / "Iris.csv").exists()
    assert (stored_dir / "README.md").exists()
    assert (stored_dir / "_manifest.json").exists()

    names = {f.name for f in record.files}
    assert names == {"Iris.csv", "README.md"}


@pytest.mark.asyncio
async def test_event_sequence_for_happy_path(with_profile, enc_key, bus):
    root = with_profile
    req = DownloadRequest(kaggle_url="https://www.kaggle.com/datasets/uciml/iris")
    fake_instance = _mock_client_class()

    with patch("src.download.pull.client_mod.KaggleClient", return_value=fake_instance):
        download_id = await pull.start_download(req, root=root, encryption_key=enc_key, bus=bus)
        record = download_store.load(download_id, root=root)
        await pull.execute_pull(record, root=root, encryption_key=enc_key, bus=bus)

    seq = _collected_events(bus, download_id)
    assert seq == [
        events.EVENT_QUEUED,
        events.EVENT_AUTH_STARTED,
        events.EVENT_AUTH_OK,
        events.EVENT_METADATA_STARTED,
        events.EVENT_METADATA_READY,
        events.EVENT_FILES_STARTED,
        events.EVENT_FILES_COMPLETE,
        events.EVENT_INDEXED,
        events.EVENT_COMPLETE,
    ]


@pytest.mark.asyncio
async def test_no_credentials_fails_with_clear_message(tmp_path, enc_key, bus):
    # profile NOT set up
    req = DownloadRequest(kaggle_url="https://www.kaggle.com/datasets/uciml/iris")
    download_id = await pull.start_download(req, root=tmp_path, encryption_key=enc_key, bus=bus)
    record = download_store.load(download_id, root=tmp_path)
    final = await pull.execute_pull(record, root=tmp_path, encryption_key=enc_key, bus=bus)

    assert final.status == DownloadStatus.FAILED
    assert "No Kaggle credentials" in final.error
    seq = _collected_events(bus, download_id)
    assert events.EVENT_AUTH_FAILED in seq
    assert seq[-1] == events.EVENT_FAILED


@pytest.mark.asyncio
async def test_metadata_failure_marks_record_failed(with_profile, enc_key, bus):
    root = with_profile
    req = DownloadRequest(kaggle_url="https://www.kaggle.com/datasets/uciml/iris")
    fake_instance = MagicMock()
    fake_instance.fetch_metadata = AsyncMock(side_effect=RuntimeError("kaggle api down"))

    with patch("src.download.pull.client_mod.KaggleClient", return_value=fake_instance):
        download_id = await pull.start_download(req, root=root, encryption_key=enc_key, bus=bus)
        record = download_store.load(download_id, root=root)
        final = await pull.execute_pull(record, root=root, encryption_key=enc_key, bus=bus)

    assert final.status == DownloadStatus.FAILED
    assert "metadata" in final.error.lower()
    seq = _collected_events(bus, download_id)
    assert events.EVENT_METADATA_FAILED in seq
    assert seq[-1] == events.EVENT_FAILED


@pytest.mark.asyncio
async def test_download_403_translated_to_user_message(with_profile, enc_key, bus):
    root = with_profile
    req = DownloadRequest(kaggle_url="https://www.kaggle.com/datasets/private/owner")
    fake_instance = _mock_client_class()
    fake_instance.download_files = AsyncMock(
        side_effect=RuntimeError("403 Client Error: Forbidden")
    )

    with patch("src.download.pull.client_mod.KaggleClient", return_value=fake_instance):
        download_id = await pull.start_download(req, root=root, encryption_key=enc_key, bus=bus)
        record = download_store.load(download_id, root=root)
        final = await pull.execute_pull(record, root=root, encryption_key=enc_key, bus=bus)

    assert final.status == DownloadStatus.FAILED
    assert "not accessible" in final.error
    assert "403" in final.error


@pytest.mark.asyncio
async def test_completed_record_has_manifest_matching_record(with_profile, enc_key, bus):
    root = with_profile
    req = DownloadRequest(kaggle_url="https://www.kaggle.com/datasets/uciml/iris")
    fake_instance = _mock_client_class()

    with patch("src.download.pull.client_mod.KaggleClient", return_value=fake_instance):
        download_id = await pull.start_download(req, root=root, encryption_key=enc_key, bus=bus)
        record = download_store.load(download_id, root=root)
        final = await pull.execute_pull(record, root=root, encryption_key=enc_key, bus=bus)

    manifest = storage.read_manifest(Path(final.storage_path))
    assert manifest == final


@pytest.mark.asyncio
async def test_profile_last_used_is_marked_on_success(with_profile, enc_key, bus):
    root = with_profile
    req = DownloadRequest(kaggle_url="https://www.kaggle.com/datasets/uciml/iris")
    fake_instance = _mock_client_class()

    with patch("src.download.pull.client_mod.KaggleClient", return_value=fake_instance):
        download_id = await pull.start_download(req, root=root, encryption_key=enc_key, bus=bus)
        record = download_store.load(download_id, root=root)
        await pull.execute_pull(record, root=root, encryption_key=enc_key, bus=bus)

    profile = profile_store.get_profile("default", root=root)
    assert profile.last_used_at is not None


@pytest.mark.asyncio
async def test_versioned_url_lands_under_versioned_dir(with_profile, enc_key, bus):
    root = with_profile
    req = DownloadRequest(kaggle_url="https://www.kaggle.com/datasets/uciml/iris/versions/3")
    fake_instance = _mock_client_class()

    with patch("src.download.pull.client_mod.KaggleClient", return_value=fake_instance):
        download_id = await pull.start_download(req, root=root, encryption_key=enc_key, bus=bus)
        record = download_store.load(download_id, root=root)
        final = await pull.execute_pull(record, root=root, encryption_key=enc_key, bus=bus)

    assert final.storage_path.endswith("/uciml__iris/3") or final.storage_path.endswith("uciml__iris/3")
