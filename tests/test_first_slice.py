import hashlib
from datetime import timedelta
from uuid import UUID, uuid4

import pytest
from conftest import PASSWORD
from fastapi.testclient import TestClient
from sqlalchemy import func, select, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from backend.auth import COOKIE, digest
from backend.db import engine
from backend.models import AuthSession, Dataset, DatasetVersion, now
from backend.storage import LocalFiles

CSV = b'unit,group,value,note\r\n001,control,10,"hello, world"\r\n002,treated,,"two\nlines"\r\n003,control,15,ok\r\n'


def upload(
    client: TestClient, content: bytes = CSV, *, title: str = "Study data", submission: str | None = None
):
    return client.post(
        "/api/datasets",
        data={"title": title, "submission_id": submission or str(uuid4())},
        files={"file": ("study.csv", content, "text/csv")},
    )


def test_login_session_expiry_logout_and_csrf(client: TestClient):
    assert client.get("/api/datasets").status_code == 401
    assert (
        client.post("/api/auth/login", json={"email": "owner@example.com", "password": "wrong"}).status_code
        == 401
    )
    login = client.post("/api/auth/login", json={"email": "owner@example.com", "password": PASSWORD})
    assert login.status_code == 200
    assert "HttpOnly" in login.headers["set-cookie"] and "SameSite=strict" in login.headers["set-cookie"]
    token = client.cookies.get(COOKIE)
    with Session(engine()) as db:
        record = db.get(AuthSession, digest(token))
        assert record and record.token_hash != token
        db.execute(update(AuthSession).values(expires_at=now() - timedelta(seconds=1)))
        db.commit()
    assert client.get("/api/auth/me").status_code == 401
    assert (
        client.post(
            "/api/auth/login",
            json={"email": "owner@example.com", "password": PASSWORD},
            headers={"Origin": "https://other.example"},
        ).status_code
        == 403
    )
    client.post("/api/auth/login", json={"email": "owner@example.com", "password": PASSWORD})
    old = client.cookies.get(COOKIE)
    assert client.post("/api/auth/logout").status_code == 204
    assert client.get("/api/auth/me", headers={"Cookie": f"{COOKIE}={old}"}).status_code == 401


def test_upload_identity_source_bytes_preview_and_new_connection(owner: TestClient):
    saved = upload(owner)
    assert saved.status_code == 201, saved.text
    data = saved.json()
    version = data["version"]
    assert version["sha256"] == hashlib.sha256(CSV).hexdigest()
    assert version["row_count"] == 3 and version["number"] == 1
    assert version["columns"] == ["unit", "group", "value", "note"]
    path = f"/api/datasets/{data['id']}"
    preview = owner.get(path + "/preview?page=1&page_size=2").json()
    assert preview["version_id"] == version["id"]
    assert preview["rows"] == [["001", "control", "10", "hello, world"], ["002", "treated", "", "two\nlines"]]
    assert owner.get(path + "/preview?page=2&page_size=2").json()["rows"] == [["003", "control", "15", "ok"]]
    engine().dispose()
    assert owner.get(path).json() == data
    assert owner.get(path + "/file").content == CSV
    assert owner.get("/api/datasets?search=Study").json()["total"] == 1
    assert owner.get("/api/datasets?search=missing").json()["items"] == []


def test_retry_and_conflicting_submission(owner: TestClient):
    submission = str(uuid4())
    first = upload(owner, submission=submission)
    assert first.status_code == 201
    assert upload(owner, submission=submission).json() == first.json()
    assert upload(owner, title="Changed title", submission=submission).status_code == 409
    with Session(engine()) as db:
        assert db.scalar(select(func.count()).select_from(Dataset)) == 1
        assert db.scalar(select(func.count()).select_from(DatasetVersion)) == 1


def test_owner_isolation_covers_list_preview_and_download(owner: TestClient):
    dataset_id = upload(owner).json()["id"]
    owner.post("/api/auth/login", json={"email": "other@example.com", "password": PASSWORD})
    assert owner.get("/api/datasets").json()["items"] == []
    for suffix in ("", "/preview", "/file"):
        assert owner.get(f"/api/datasets/{dataset_id}{suffix}").status_code == 404


@pytest.mark.parametrize(
    "bad", [b"", b"a,b\n", b"a,a\n1,2\n", b"a,b\n1\n", b"a,b\n\x00,2\n", b"a,b\n\xff,2\n", b",b\n1,2\n"]
)
def test_invalid_csv_never_publishes_dataset(owner: TestClient, bad: bytes):
    assert upload(owner, bad).status_code == 422
    assert owner.get("/api/datasets").json()["total"] == 0


def test_storage_failure_can_be_retried_without_partial_records(owner: TestClient, monkeypatch):
    original = LocalFiles.put

    def broken(*args):
        raise OSError("disk full")

    monkeypatch.setattr(LocalFiles, "put", broken)
    submission = str(uuid4())
    assert upload(owner, submission=submission).status_code == 503
    assert owner.get("/api/datasets").json()["total"] == 0
    monkeypatch.setattr(LocalFiles, "put", original)
    assert upload(owner, submission=submission).status_code == 201


def test_detects_changed_source_bytes(owner: TestClient):
    data = upload(owner).json()
    with Session(engine()) as db:
        version = db.get(DatasetVersion, UUID(data["version"]["id"]))
        from backend.main import app
        from backend.storage import files

        storage = app.dependency_overrides[files]()
        storage.path(version.storage_key).write_bytes(b"changed")
    assert owner.get(f"/api/datasets/{data['id']}/preview").status_code == 503


def test_database_rejects_in_place_version_mutation(owner: TestClient):
    upload(owner)
    with Session(engine()) as db, pytest.raises(DBAPIError, match="immutable"):
        db.execute(update(DatasetVersion).values(row_count=999))
        db.commit()


def test_upload_request_limit_rejects_before_publication(owner: TestClient):
    assert (
        owner.post(
            "/api/datasets", content=b"", headers={"Content-Length": str(12 * 1024 * 1024)}
        ).status_code
        == 413
    )
