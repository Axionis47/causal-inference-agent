"""Live integration test against real Kaggle.

Skipped unless ~/.kaggle/kaggle.json exists or KAGGLE_USERNAME +
KAGGLE_KEY are set in the environment. Pulls two small public
datasets and asserts the full flow produces complete records with
files on disk and a valid manifest.

Run explicitly:
    pytest -m integration backend/src/download/tests/test_live_kaggle.py -s
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from src.domain.download import DownloadRequest, DownloadStatus
from src.download import (
    events,
    profile_store,
)
from src.download.download_store import load as load_record
from src.download.pull import execute_pull, start_download

pytestmark = pytest.mark.integration


def _kaggle_creds() -> tuple[str, str] | None:
    env_user = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")
    if env_user and env_key:
        return env_user, env_key
    creds_path = Path.home() / ".kaggle" / "kaggle.json"
    if creds_path.exists():
        data = json.loads(creds_path.read_text())
        return data["username"], data["key"]
    return None


@pytest.fixture
def real_creds():
    creds = _kaggle_creds()
    if creds is None:
        pytest.skip("No Kaggle credentials available")
    return creds


@pytest.fixture
def live_root(tmp_path):
    return tmp_path


@pytest.fixture
def enc_key():
    return Fernet.generate_key().decode()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url, expected_owner",
    [
        ("https://www.kaggle.com/datasets/uciml/iris", "uciml"),
        ("https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009", "uciml"),
    ],
)
async def test_real_kaggle_download_completes(
    url, expected_owner, real_creds, live_root, enc_key
):
    username, key = real_creds
    profile_store.set_kaggle_key(
        "default",
        username=username,
        key=key,
        root=live_root,
        encryption_key=enc_key,
    )

    bus = events.DownloadEventBus()
    req = DownloadRequest(kaggle_url=url)
    download_id = await start_download(req, root=live_root, encryption_key=enc_key, bus=bus)
    record = load_record(download_id, root=live_root)
    final = await execute_pull(record, root=live_root, encryption_key=enc_key, bus=bus)

    assert final.status == DownloadStatus.COMPLETE, f"Download failed: {final.error}"
    assert final.metadata is not None
    assert final.metadata.owner == expected_owner
    assert final.storage_path is not None
    assert len(final.files) > 0

    storage = Path(final.storage_path)
    assert storage.exists()
    assert (storage / "_manifest.json").exists()

    # Every reported file must exist on disk with the reported size.
    for entry in final.files:
        f = storage / entry.relative_path
        assert f.exists(), f"Missing file: {entry.relative_path}"
        assert f.stat().st_size == entry.size_bytes
