"""Phase 2: real owned metadata reads and pagination."""

import io
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import select, update

from causal_agent.server.ingestion import library
from causal_agent.server.ingestion import schema as S

pytestmark = pytest.mark.ingestion


def save(storage, owner, context=""):
    return storage.ingest(owner, str(uuid4()), "Same title", context, "same.csv", io.BytesIO(b"a\n1\n"))


def test_owner_filter(storage):
    a, b = str(uuid4()), str(uuid4())
    mine = [save(storage, a), save(storage, a, "original")]
    save(storage, b)
    page = library.list_owned(storage, a, 20)
    assert {r["dataset_id"] for r in page["items"]} == {r["dataset_id"] for r in mine}
    assert sorted(r["has_initial_context"] for r in page["items"]) == [False, True]
    assert all(set(r) == {"dataset_id", "version_id", "title", "filename", "size_bytes", "created_at", "status", "has_initial_context"} for r in page["items"])
    assert library.list_owned(storage, str(uuid4()), 20) == {"items": [], "next_cursor": None}


def test_pagination(storage):
    owner = str(uuid4())
    saved = [save(storage, owner) for _ in range(5)]
    stamp = datetime(2026, 1, 1, tzinfo=UTC)
    with storage.engine.begin() as db:
        db.execute(update(S.datasets).values(created_at=stamp))
    ids, cursor = [], None
    for _ in range(3):
        page = library.list_owned(storage, owner, 2, cursor)
        ids.extend(row["dataset_id"] for row in page["items"])
        cursor = page["next_cursor"]
    assert cursor is None
    assert ids == sorted((r["dataset_id"] for r in saved), reverse=True)
    new = save(storage, owner)
    assert library.list_owned(storage, owner, 2)["items"][0]["dataset_id"] == new["dataset_id"]


def test_only_ready(storage):
    owner = str(uuid4())
    records = [save(storage, owner) for _ in range(3)]
    with storage.engine.begin() as db:
        for record, state in zip(records, ["pending", "failed"]):
            op = db.scalar(select(S.datasets.c.operation_id).where(S.datasets.c.id == record["dataset_id"]))
            db.execute(update(S.operations).where(S.operations.c.id == op).values(status=state))
    assert [r["dataset_id"] for r in library.list_owned(storage, owner, 20)["items"]] == [records[-1]["dataset_id"]]


def test_http_read_contract(ingestion_client, storage, monkeypatch):
    import os

    owner = os.environ["DEV_USER_ID"]
    record = save(storage, owner, " original\naccount ")

    def forbidden(*args, **kwargs):
        raise AssertionError("Metadata read fetched a CSV")

    monkeypatch.setattr(storage.objects, "open", forbidden)
    with storage.engine.connect() as db:
        before = {t.name: [dict(r) for r in db.execute(select(t)).mappings()] for t in S.metadata.sorted_tables}
    response = ingestion_client.get("/api/datasets")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"
    assert response.json()["items"][0]["dataset_id"] == record["dataset_id"]
    detail = ingestion_client.get(f"/api/datasets/{record['dataset_id']}")
    assert detail.status_code == 200
    assert detail.json()["initial_context"] == record["initial_context"]
    assert detail.json()["dataset_id"] == record["dataset_id"]
    assert detail.headers["cache-control"] == "private, no-store"
    with storage.engine.connect() as db:
        after = {t.name: [dict(r) for r in db.execute(select(t)).mappings()] for t in S.metadata.sorted_tables}
    assert before == after
    monkeypatch.setenv("DEV_USER_ID", str(uuid4()))
    assert ingestion_client.get("/api/datasets").json()["items"] == []
    assert ingestion_client.get(f"/api/datasets/{record['dataset_id']}").status_code == 404
    monkeypatch.setenv("APP_ENV", "production")
    assert ingestion_client.get("/api/datasets").status_code == 503


@pytest.mark.parametrize("query", ["limit=0", "limit=101", "limit=abc", "cursor=bad", "cursor=e30="])
def test_invalid_query(ingestion_client, query):
    assert ingestion_client.get("/api/datasets?" + query).status_code == 422


def test_other_owner_cursor(ingestion_client, storage, monkeypatch):
    owner = str(uuid4())
    save(storage, owner)
    save(storage, owner)
    cursor = library.list_owned(storage, owner, 1)["next_cursor"]
    assert ingestion_client.get("/api/datasets", params={"cursor": cursor}).status_code == 422


def test_unavailable_is_not_empty(ingestion_client, storage, monkeypatch):
    def failed():
        raise OSError("database unavailable")

    monkeypatch.setattr(storage.engine, "connect", failed)
    for path in ["/api/datasets", f"/api/datasets/{uuid4()}"]:
        response = ingestion_client.get(path)
        assert response.status_code == 503
        assert "items" not in response.json()


@pytest.mark.parametrize("field,value", [("id", 1), ("created_at", []), ("created_at", "2026-01-01")])
def test_cursor_field_types(ingestion_client, field, value):
    import base64
    import json
    import os

    payload = {"owner": os.environ["DEV_USER_ID"], "id": str(uuid4()), "created_at": "2026-01-01T00:00:00+00:00"}
    payload[field] = value
    cursor = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    assert ingestion_client.get("/api/datasets", params={"cursor": cursor}).status_code == 422
