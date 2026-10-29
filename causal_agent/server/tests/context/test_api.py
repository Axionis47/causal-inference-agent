"""Phase 4 owned reads, publication errors and original-source preservation."""

import io
import os
from uuid import uuid4

import pytest
from sqlalchemy import select, update

from causal_agent.common.llm import set_llm
from causal_agent.memory.context import extract as E
from causal_agent.memory.context.tests.builders import SOURCE, Scripted
from causal_agent.server.context import schema as S
from causal_agent.server.context import service
from causal_agent.server.ingestion import schema as I

pytestmark = pytest.mark.ingestion


def test_roundtrip(isolated_client, storage, prepared, monkeypatch):
    owner = os.environ["DEV_USER_ID"]
    record = prepared(owner)
    path = f"/api/datasets/{record['dataset_id']}"
    assert isolated_client.get(path + "/context").json() == {"status": "not_prepared", "pack": None}
    response = isolated_client.post(path + "/context")
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "private, no-store"
    pack = response.json()["pack"]
    assert [c["name"] for c in pack["columns"]] == ["code", "amount"]
    assert pack["columns"][0]["statement_count"] == 0 and pack["columns"][1]["statement_count"] == 1
    assert [c["relationship_count"] for c in pack["columns"]] == [1, 1]
    assert pack["csv_checksum"] == record["sha256"]
    base = path + "/contexts/" + pack["revision_id"]
    detail = isolated_client.get(base).json()
    assert detail["source_text"] == SOURCE
    values = [isolated_client.get(base + "/columns/" + c["column_id"]).json() for c in pack["columns"]]
    assert values[1]["statements"][0]["value"] == "dollars"
    assert values[0]["relationships"] == values[1]["relationships"]
    assert isolated_client.get(path + "/file").content == b"code,amount\r\n001,10\r\n002,20\r\n"
    assert storage.get(owner, record["dataset_id"])["initial_context"] == SOURCE

    def forbidden(*args, **kwargs):
        raise AssertionError("Saved context must not read CSV or run extraction")

    monkeypatch.setattr(storage.objects, "open", forbidden)
    monkeypatch.setattr(E, "extract", forbidden)
    set_llm(Scripted([]))
    with storage.engine.connect() as db:
        before = [dict(r) for r in db.execute(select(S.packs)).mappings()]
    assert isolated_client.get(path + "/context").json() == response.json()
    assert isolated_client.post(path + "/context").json() == response.json()
    assert isolated_client.get(base).json() == detail
    assert isolated_client.get(base + "/columns/" + pack["columns"][1]["column_id"]).json() == values[1]
    with storage.engine.connect() as db:
        assert [dict(r) for r in db.execute(select(S.packs)).mappings()] == before


def test_owner_isolation(isolated_client, prepared, monkeypatch):
    record = prepared(os.environ["DEV_USER_ID"])
    base = f"/api/datasets/{record['dataset_id']}"
    pack = isolated_client.post(base + "/context").json()["pack"]
    monkeypatch.setenv("DEV_USER_ID", str(uuid4()))
    paths = [
        base + "/context",
        base + "/contexts/" + pack["revision_id"],
        base + "/contexts/" + pack["revision_id"] + "/columns/" + pack["columns"][0]["column_id"],
    ]
    for path in paths:
        r = isolated_client.get(path)
        assert r.status_code == 404, "Other owners must not retrieve context"
        assert r.json()["detail"] == "Dataset not found."
    assert isolated_client.post(base + "/context").status_code == 404
    monkeypatch.setenv("APP_ENV", "production")
    assert isolated_client.get(paths[0]).status_code == 503


def test_column_version_join(isolated_client, prepared):
    owner = os.environ["DEV_USER_ID"]
    a = prepared(owner)
    pa = isolated_client.post(f"/api/datasets/{a['dataset_id']}/context").json()["pack"]
    b = prepared(owner)
    pb = isolated_client.post(f"/api/datasets/{b['dataset_id']}/context").json()["pack"]
    base = f"/api/datasets/{a['dataset_id']}/contexts/"
    assert isolated_client.get(base + pb["revision_id"]).status_code == 404, "Pack must belong to routed dataset"
    assert isolated_client.get(base + pb["revision_id"] + "/columns/" + pb["columns"][0]["column_id"]).status_code == 404, (
        "Column must belong to routed version"
    )
    assert isolated_client.get(base + pa["revision_id"] + "/columns/" + pb["columns"][0]["column_id"]).status_code == 404


def test_missing_profile_and_absent_context(isolated_client, storage, prepared):
    owner = os.environ["DEV_USER_ID"]
    record = storage.ingest(owner, str(uuid4()), "No profile", "", "one.csv", io.BytesIO(b"x\n1\n"))
    r = isolated_client.post(f"/api/datasets/{record['dataset_id']}/context")
    assert r.status_code == 409 and r.json()["detail"]["code"] == "profile_required"
    record = prepared(owner, " \n ")
    r = isolated_client.post(f"/api/datasets/{record['dataset_id']}/context")
    assert r.status_code == 200 and not r.json()["pack"]["source_present"]
    assert all(c["statement_count"] == 0 for c in r.json()["pack"]["columns"])


@pytest.mark.parametrize("state", ["pending", "failed"])
def test_unpublished_is_hidden(isolated_client, storage, prepared, state):
    record = prepared(os.environ["DEV_USER_ID"])
    with storage.engine.begin() as db:
        op = db.scalar(select(I.datasets.c.operation_id).where(I.datasets.c.id == record["dataset_id"]))
        db.execute(update(I.operations).where(I.operations.c.id == op).values(status=state))
    path = f"/api/datasets/{record['dataset_id']}/context"
    assert isolated_client.get(path).status_code == isolated_client.post(path).status_code == 404


def test_storage_and_model_failures_are_visible(isolated_client, storage, prepared, monkeypatch):
    record = prepared(os.environ["DEV_USER_ID"])
    path = f"/api/datasets/{record['dataset_id']}/context"

    def fail(*a):
        raise E.InvalidExtraction("invalid citation")

    monkeypatch.setattr(E, "extract", fail)
    assert isolated_client.post(path).status_code == 502
    assert isolated_client.get(path).json()["status"] == "not_prepared"

    def outage(*a):
        raise OSError("database unavailable")

    monkeypatch.setattr(storage.engine, "connect", outage)
    assert isolated_client.get(path).status_code == 503


def test_extraction_provenance_cannot_be_edited(storage, prepared):
    from sqlalchemy.exc import DBAPIError

    owner = str(uuid4())
    record = prepared(owner)
    pack = service.generate(storage, owner, record["dataset_id"]).pack
    with pytest.raises(DBAPIError, match="immutable"):
        with storage.engine.begin() as db:
            db.execute(update(S.column_contexts).where(S.column_contexts.c.column_id == pack.columns[0].column_id).values(statements=[{"broken": True}]))
    result = service.column(storage, owner, record["dataset_id"], pack.revision_id, pack.columns[1].column_id)
    assert result.statements[0].value == "dollars"


def test_invalid_citation_rejected(isolated_client, prepared, monkeypatch):
    record = prepared(os.environ["DEV_USER_ID"])
    original = E.extract

    def corrupt(source, columns):
        result = original(source, columns)
        result.content.columns[1].statements[0].citations[0].quote = "invented quotation"
        return result

    monkeypatch.setattr(E, "extract", corrupt)
    response = isolated_client.post(f"/api/datasets/{record['dataset_id']}/context")
    assert response.status_code == 502, "Unverifiable citations must not be published"
