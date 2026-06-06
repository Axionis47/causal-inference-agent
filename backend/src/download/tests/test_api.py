"""HTTP boundary for the download module.

Underlying logic is already covered by the other test files in this
directory; these tests verify only that the routes are wired correctly
and surface the right status codes.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.download import api as download_api
from src.download import events, profile_store
from src.download.auth import ValidationResult


@pytest.fixture
def enc_key() -> str:
    return Fernet.generate_key().decode()


@pytest.fixture
def app(tmp_path: Path, enc_key: str) -> FastAPI:
    bus = events.DownloadEventBus()
    app = FastAPI()
    app.include_router(download_api.router)
    app.dependency_overrides[download_api.get_root] = lambda: tmp_path
    app.dependency_overrides[download_api.get_encryption_key] = lambda: enc_key
    app.dependency_overrides[download_api.get_event_bus] = lambda: bus
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def test_set_kaggle_key_without_validation_persists_profile(client, tmp_path):
    resp = client.post(
        "/api/v1/download/profile/kaggle",
        json={"username": "alice", "key": "real-key", "validate_key": False},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["kaggle_username"] == "alice"
    assert body["has_key"] is True
    assert "key" not in body and "kaggle_key" not in body


def test_set_kaggle_key_with_validation_calls_validator(client):
    with patch(
        "src.download.api.auth.validate_kaggle_key",
        new=AsyncMock(return_value=ValidationResult(valid=True)),
    ) as mock_validate:
        resp = client.post(
            "/api/v1/download/profile/kaggle",
            json={"username": "alice", "key": "real-key"},
        )
    assert resp.status_code == 200
    mock_validate.assert_awaited_once_with("alice", "real-key")


def test_set_kaggle_key_with_invalid_key_returns_400(client):
    with patch(
        "src.download.api.auth.validate_kaggle_key",
        new=AsyncMock(return_value=ValidationResult(valid=False, error="401 Unauthorized")),
    ):
        resp = client.post(
            "/api/v1/download/profile/kaggle",
            json={"username": "alice", "key": "bad"},
        )
    assert resp.status_code == 400
    assert "401" in resp.json()["detail"]


def test_get_kaggle_profile_returns_null_when_unset(client):
    resp = client.get("/api/v1/download/profile/kaggle")
    assert resp.status_code == 200
    assert resp.json() is None


def test_get_kaggle_profile_returns_profile_when_set(client, tmp_path, enc_key):
    profile_store.set_kaggle_key(
        "default",
        username="alice",
        key="real-key",
        root=tmp_path,
        encryption_key=enc_key,
    )
    resp = client.get("/api/v1/download/profile/kaggle")
    assert resp.status_code == 200
    assert resp.json()["kaggle_username"] == "alice"


def test_delete_kaggle_profile_removes_it(client, tmp_path, enc_key):
    profile_store.set_kaggle_key(
        "default",
        username="alice",
        key="real-key",
        root=tmp_path,
        encryption_key=enc_key,
    )
    resp = client.delete("/api/v1/download/profile/kaggle")
    assert resp.status_code == 204
    assert profile_store.get_profile("default", root=tmp_path) is None


def test_post_download_returns_record_with_queued_status(client, tmp_path, enc_key):
    profile_store.set_kaggle_key(
        "default",
        username="alice",
        key="real-key",
        root=tmp_path,
        encryption_key=enc_key,
    )
    # Patch the orchestrator so we don't actually hit Kaggle.
    with patch(
        "src.download.api.pull.start_download",
        new=AsyncMock(return_value="dl-fake-123"),
    ):
        with patch(
            "src.download.api.download_store.load",
            return_value=_minimal_record("dl-fake-123"),
        ):
            resp = client.post(
                "/api/v1/download/downloads",
                json={"kaggle_url": "https://www.kaggle.com/datasets/uciml/iris"},
            )
    assert resp.status_code == 202
    body = resp.json()
    assert body["download_id"] == "dl-fake-123"
    assert body["dataset_ref"] == "uciml/iris"


def test_get_download_404_when_missing(client):
    resp = client.get("/api/v1/download/downloads/nope")
    assert resp.status_code == 404


def test_list_downloads_returns_empty_list_initially(client):
    resp = client.get("/api/v1/download/downloads")
    assert resp.status_code == 200
    assert resp.json() == []


def test_delete_download_is_idempotent(client):
    resp = client.delete("/api/v1/download/downloads/never-existed")
    assert resp.status_code == 204


def _minimal_record(download_id: str):
    from datetime import datetime, timezone

    from src.domain.download import DownloadRecord, DownloadStatus

    return DownloadRecord(
        download_id=download_id,
        profile_id="default",
        kaggle_url="https://www.kaggle.com/datasets/uciml/iris",
        dataset_ref="uciml/iris",
        status=DownloadStatus.QUEUED,
        created_at=datetime.now(timezone.utc),
    )
