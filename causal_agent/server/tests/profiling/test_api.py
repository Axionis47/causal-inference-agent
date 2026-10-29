"""Phase 3: owned API, independent column retrieval and read-only originals."""

import os
from uuid import uuid4

import pytest
from sqlalchemy import select, update

from causal_agent.profile import inspection
from causal_agent.server.ingestion import schema as I
from causal_agent.server.profiling import schema as S
from causal_agent.server.profiling import service

pytestmark = pytest.mark.ingestion


def test_roundtrip(ingestion_client, storage, saved, monkeypatch):
    owner = os.environ["DEV_USER_ID"]
    original = b"id,value\r\n1,10\r\n2,20\r\n"
    record = saved(owner, original)
    path = f"/api/datasets/{record['dataset_id']}/profile"
    assert ingestion_client.get(path).json() == {"status": "not_profiled", "profile": None}
    response = ingestion_client.post(path)
    assert response.status_code == 200 and response.headers["cache-control"] == "private, no-store"
    profile = response.json()["profile"]
    assert profile["status"] == "complete" and profile["dataset"]["rows"] == 2
    assert [c["name"] for c in profile["columns"]] == ["id", "value"]
    assert profile["source_checksum"] == record["sha256"] and profile["version_id"] == record["version_id"]
    column_path = f"/api/datasets/{record['dataset_id']}/profiles/{profile['profile_id']}/columns/{profile['columns'][1]['column_id']}"
    assert ingestion_client.get(column_path).json()["result"]["numeric"]["mean"] == 15
    assert ingestion_client.get(f"/api/datasets/{record['dataset_id']}/file").content == original
    assert storage.get(owner, record["dataset_id"])["initial_context"] == record["initial_context"]

    def forbidden(*args):
        raise AssertionError("Saved report fetched CSV or recomputed")

    monkeypatch.setattr(storage.objects, "open", forbidden)
    monkeypatch.setattr(inspection, "inspect", forbidden)
    with storage.engine.connect() as db:
        before = [dict(r) for r in db.execute(select(S.profiles)).mappings()]
    assert ingestion_client.get(path).json()["profile"] == profile
    assert ingestion_client.post(path).json()["profile"] == profile
    assert ingestion_client.get(column_path).status_code == 200
    with storage.engine.connect() as db:
        assert [dict(r) for r in db.execute(select(S.profiles)).mappings()] == before


def test_owner_isolation(ingestion_client, saved, monkeypatch):
    owner = os.environ["DEV_USER_ID"]
    mine, theirs = saved(owner), saved(str(uuid4()))
    path = f"/api/datasets/{mine['dataset_id']}/profile"
    profile = ingestion_client.post(path).json()["profile"]
    for method in (ingestion_client.get, ingestion_client.post):
        assert method(f"/api/datasets/{theirs['dataset_id']}/profile").status_code == 404
    monkeypatch.setenv("DEV_USER_ID", str(uuid4()))
    assert ingestion_client.get(path).status_code == 404
    assert ingestion_client.post(path).status_code == 404
    assert (
        ingestion_client.get(f"/api/datasets/{mine['dataset_id']}/profiles/{profile['profile_id']}/columns/{profile['columns'][0]['column_id']}").status_code
        == 404
    )
    monkeypatch.setenv("APP_ENV", "production")
    assert ingestion_client.get(path).status_code == 503


def test_column_version_join(ingestion_client, saved):
    owner = os.environ["DEV_USER_ID"]
    a, b = saved(owner), saved(owner, b"code,amount\nother,999\n")
    pa = ingestion_client.post(f"/api/datasets/{a['dataset_id']}/profile").json()["profile"]
    pb = ingestion_client.post(f"/api/datasets/{b['dataset_id']}/profile").json()["profile"]
    # Valid profile/column IDs from another owned version must not leak through a dataset route.
    assert ingestion_client.get(f"/api/datasets/{a['dataset_id']}/profiles/{pb['profile_id']}/columns/{pb['columns'][0]['column_id']}").status_code == 404
    assert ingestion_client.get(f"/api/datasets/{a['dataset_id']}/profiles/{pa['profile_id']}/columns/{pb['columns'][0]['column_id']}").status_code == 404


@pytest.mark.parametrize("state", ["pending", "failed"])
def test_unpublished_is_hidden(ingestion_client, storage, saved, state):
    record = saved(os.environ["DEV_USER_ID"])
    with storage.engine.begin() as db:
        op = db.scalar(select(I.datasets.c.operation_id).where(I.datasets.c.id == record["dataset_id"]))
        db.execute(update(I.operations).where(I.operations.c.id == op).values(status=state))
    path = f"/api/datasets/{record['dataset_id']}/profile"
    assert ingestion_client.get(path).status_code == ingestion_client.post(path).status_code == 404


def test_partial_is_explicit(ingestion_client, saved):
    record = saved(os.environ["DEV_USER_ID"], b"x\n1\noops\n")
    profile = ingestion_client.post(f"/api/datasets/{record['dataset_id']}/profile").json()["profile"]
    col = ingestion_client.get(f"/api/datasets/{record['dataset_id']}/profiles/{profile['profile_id']}/columns/{profile['columns'][0]['column_id']}").json()[
        "result"
    ]
    assert profile["status"] == "partial" and col["numeric"] is None
    assert col["numeric_exception_count"] == 1 and col["issues"][0]["examples"] == ["oops"]


def test_storage_failure_visible(ingestion_client, storage, saved, monkeypatch):
    record = saved(os.environ["DEV_USER_ID"])

    def failed(*args):
        raise OSError("object service unavailable")

    monkeypatch.setattr(storage.objects, "open", failed)
    path = f"/api/datasets/{record['dataset_id']}/profile"
    assert ingestion_client.post(path).status_code == 503
    assert ingestion_client.get(path).json()["status"] == "not_profiled"
    monkeypatch.setattr(storage.engine, "connect", failed)
    assert ingestion_client.get(path).status_code == 503


def test_column_reads_only_requested_row(storage, saved, monkeypatch):
    owner = str(uuid4())
    record = saved(owner)
    profile = service.generate(storage, owner, record["dataset_id"]).profile
    with storage.engine.begin() as db:
        # An unrelated corrupted row must not be loaded or validated by the requested-column query.
        db.execute(update(S.column_results).where(S.column_results.c.column_id == profile.columns[1].column_id).values(result={}))
    assert service.column(storage, owner, record["dataset_id"], profile.profile_id, profile.columns[0].column_id).result.name == "code"


def test_numeric_exception_not_discarded(ingestion_client, saved):
    record = saved(os.environ["DEV_USER_ID"], ("amount\n" + "\n".join(str(i) for i in range(99)) + "\noops\n").encode())
    profile = ingestion_client.post(f"/api/datasets/{record['dataset_id']}/profile").json()["profile"]
    col = ingestion_client.get(f"/api/datasets/{record['dataset_id']}/profiles/{profile['profile_id']}/columns/{profile['columns'][0]['column_id']}").json()[
        "result"
    ]
    assert col["numeric"] is None, "Invalid values must not disappear from numerical summaries"
    assert col["numeric_exception_count"] == 1 and profile["status"] == "partial"


def test_check_failure_visible(ingestion_client, saved, monkeypatch):
    record = saved(os.environ["DEV_USER_ID"], b"amount\n1\n2\n")

    def fail(values):
        raise RuntimeError("deliberate numeric failure")

    monkeypatch.setattr(inspection, "numeric_summary", fail)
    response = ingestion_client.post(f"/api/datasets/{record['dataset_id']}/profile")
    assert response.status_code == 200
    assert response.json()["status"] == "partial", "Failed checks must not be published as a complete profile"
