"""Behavioral ingestion tests. Real storage is created by the harness, never supplied by a deployment."""

import hashlib
import io
import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from threading import Event
from uuid import uuid4

import pytest
from botocore.exceptions import ClientError
from sqlalchemy import Connection, func, select, update
from sqlalchemy.exc import DBAPIError, IntegrityError

from causal_agent.common.config import ingestion
from causal_agent.server.ingestion import schema as S
from causal_agent.server.ingestion.service import Conflict, Missing, Store
from causal_agent.server.ingestion.validation import InvalidUpload, validate

CSV = b'\xef\xbb\xbfid,note\r\n1,"hello, world"\r\n2,"two\nlines"\r\n'
LIMITS = ingestion({})


@pytest.mark.parametrize(
    "data,reason",
    [
        pytest.param(b"", "nonempty header", id="empty"),
        pytest.param(b"a,b\n", "no data records", id="header-only"),
        pytest.param(b"a,a\n1,2\n", "unique", id="duplicate-header"),
        pytest.param(b"exam score,exam-score\n1,2\n", "normalization", id="normalized-collision"),
        pytest.param(b"a, \n1,2\n", "nonempty header", id="blank-header"),
        pytest.param(b"a,b\n1\n", "record 2 has 1 fields", id="short-row"),
        pytest.param(b"a,b\n1,2\n3,4,5\n", "record 3 has 3 fields", id="late-wide-row"),
        pytest.param(b'a,b\n1,"unfinished', "Invalid UTF-8 CSV", id="unclosed-quote"),
        pytest.param(b"a\n\xff\n", "Invalid UTF-8 CSV", id="invalid-encoding"),
        pytest.param(b"a\n\x00\n", "NUL", id="binary-nul"),
    ],
)
def test_invalid_csv(data, reason):
    with pytest.raises(InvalidUpload, match=reason):
        validate(io.BytesIO(data), "x.csv", LIMITS)


@pytest.mark.parametrize("data", [CSV, 'name,note\nÉva,"said ""hello"""\n'.encode(), b"a\n" + b"x" * 150000 + b"\n"])
def test_valid_bytes_checksum_and_bounded_reads(data):
    class CountingStream(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= 64 * 1024, "Unbounded application byte read"
            return super().read(size)

    stream = CountingStream(data)
    assert validate(stream, "x.CSV", LIMITS) == (hashlib.sha256(data).hexdigest(), len(data))
    assert stream.tell() == 0 and stream.getvalue() == data


@pytest.mark.parametrize("field,exact", [("max_bytes", 12), ("max_rows", 2), ("max_columns", 2), ("max_cells", 4)])
def test_limit_boundary(field, exact):
    data = b"a,b\n1,2\n3,4\n"
    assert len(data) == 12
    validate(io.BytesIO(data), "x.csv", replace(LIMITS, **{field: exact}))
    with pytest.raises(InvalidUpload, match="exceeds"):
        validate(io.BytesIO(data), "x.csv", replace(LIMITS, **{field: exact - 1}))


def test_extension():
    with pytest.raises(InvalidUpload, match="Choose a .csv"):
        validate(io.BytesIO(CSV), "x.txt", LIMITS)


def save(storage, owner, key=None, context="  original\naccount  ", data=CSV, title="Study", filename="same.csv"):
    return storage.ingest(owner, key or str(uuid4()), title, context, filename, io.BytesIO(data))


def op_for(storage, owner):
    with storage.engine.connect() as db:
        return db.execute(select(S.operations).where(S.operations.c.owner_id == owner)).mappings().one()


def age(storage, owner):
    with storage.engine.begin() as db:
        db.execute(update(S.operations).where(S.operations.c.owner_id == owner).values(created_at=datetime.now(UTC) - timedelta(days=2)))


def absent(storage, key):
    with pytest.raises(ClientError) as exc:
        storage.objects.client.head_object(Bucket=storage.objects.bucket, Key=key)
    assert exc.value.response["ResponseMetadata"]["HTTPStatusCode"] == 404


def bytes_for(storage, owner, dataset_id):
    _, body = storage.download(owner, dataset_id)
    try:
        return body.read()
    finally:
        body.close()


def counts(storage):
    with storage.engine.connect() as db:
        return [db.scalar(select(func.count()).select_from(t)) for t in (S.datasets, S.versions, S.files)]


def orphan(storage, monkeypatch, owner):
    original = storage.objects.put

    def fail(*args):
        original(*args)
        raise OSError("lost response after object write")

    with monkeypatch.context() as patch:
        patch.setattr(storage.objects, "put", fail)
        with pytest.raises(OSError, match="lost response"):
            save(storage, owner, key=owner)
    assert counts(storage) == [0, 0, 0]
    operation = op_for(storage, owner)
    assert operation["status"] == "failed"
    body = storage.objects.open(operation["object_key"])
    try:
        assert body.read() == CSV
    finally:
        body.close()
    return operation


@pytest.mark.ingestion
def test_roundtrip_owner_isolation_and_replay(storage):
    owner, other, key = str(uuid4()), str(uuid4()), str(uuid4())
    result = save(storage, owner, key)
    assert result == save(storage, owner, key)
    assert result["initial_context"] == "  original\naccount  "
    assert result["sha256"] == hashlib.sha256(CSV).hexdigest()
    assert bytes_for(storage, owner, result["dataset_id"]) == CSV
    assert counts(storage) == [1, 1, 1]
    assert storage.objects.client.list_objects_v2(Bucket=storage.objects.bucket)["KeyCount"] == 1
    with pytest.raises(Missing):
        storage.get(other, result["dataset_id"])
    with pytest.raises(Missing):
        storage.download(other, result["dataset_id"])
    second = save(storage, other, key, context="")
    assert second["dataset_id"] != result["dataset_id"] and second["initial_context"] is None
    assert counts(storage) == [2, 2, 2]


@pytest.mark.ingestion
@pytest.mark.parametrize("change", [{"context": "changed"}, {"data": b"a\n1\n"}, {"title": "Changed"}, {"filename": "another.csv"}])
def test_changed_payload_conflicts(storage, change):
    owner, key = str(uuid4()), str(uuid4())
    original = save(storage, owner, key)
    with pytest.raises(Conflict, match="different content"):
        save(storage, owner, key, **change)
    assert storage.get(owner, original["dataset_id"]) == original
    assert counts(storage) == [1, 1, 1]


@pytest.mark.ingestion
def test_cleanup_removes_orphan_and_retry_publishes_once(storage, monkeypatch):
    owner = str(uuid4())
    operation = orphan(storage, monkeypatch, owner)
    age(storage, owner)
    storage.cleanup()
    absent(storage, operation["object_key"])
    assert op_for(storage, owner)["status"] == "failed"
    storage.cleanup()
    absent(storage, operation["object_key"])
    result = save(storage, owner, key=owner)
    storage.cleanup()
    assert bytes_for(storage, owner, result["dataset_id"]) == CSV
    assert counts(storage) == [1, 1, 1]


@pytest.mark.ingestion
def test_cleanup_preserves_recent_locked_and_ready_objects(storage, monkeypatch):
    owner = str(uuid4())
    operation = orphan(storage, monkeypatch, owner)
    storage.cleanup()
    storage.objects.client.head_object(Bucket=storage.objects.bucket, Key=operation["object_key"])
    age(storage, owner)
    with storage.locked(owner, owner):
        storage.cleanup()
        storage.objects.client.head_object(Bucket=storage.objects.bucket, Key=operation["object_key"])
    result = save(storage, owner, key=owner)
    storage.cleanup()
    assert op_for(storage, owner)["status"] == "ready"
    assert bytes_for(storage, owner, result["dataset_id"]) == CSV


@pytest.mark.ingestion
def test_cleanup_delete_failure_is_retryable(storage, monkeypatch):
    owner = str(uuid4())
    operation = orphan(storage, monkeypatch, owner)
    age(storage, owner)
    with monkeypatch.context() as patch:

        def broken(key):
            raise OSError("storage unavailable")

        patch.setattr(storage.objects, "delete", broken)
        with pytest.raises(OSError, match="storage unavailable"):
            storage.cleanup()
    storage.objects.client.head_object(Bucket=storage.objects.bucket, Key=operation["object_key"])
    storage.cleanup()
    absent(storage, operation["object_key"])


@pytest.mark.ingestion
def test_two_submissions_compete_through_publication(storage, monkeypatch):
    owner, key = str(uuid4()), str(uuid4())
    entered, release = Event(), Event()
    original = storage.objects.put
    calls = []

    def blocked(*args):
        calls.append(args[0])
        if len(calls) == 1:
            entered.set()
            assert release.wait(15), "writer release timed out"
        return original(*args)

    monkeypatch.setattr(storage.objects, "put", blocked)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(save, storage, owner, key)
        try:
            assert entered.wait(15), "first writer never reached object storage"
            second = pool.submit(save, storage, owner, key)
            with pytest.raises(Conflict, match="still being saved"):
                second.result(timeout=10)
        finally:
            release.set()
        result = first.result(timeout=15)
    assert save(storage, owner, key) == result
    assert counts(storage) == [1, 1, 1] and len(calls) == 1
    assert bytes_for(storage, owner, result["dataset_id"]) == CSV


@pytest.mark.ingestion
def test_context_ownership_fk_rejects_wrong_owner(storage):
    owner, other = str(uuid4()), str(uuid4())
    result = save(storage, owner, context="")  # No context exists: UNIQUE cannot mask a missing FK.
    with storage.engine.begin() as db:
        db.execute(S.users.insert().values(id=other))
    with pytest.raises(IntegrityError) as exc, storage.engine.begin() as db:
        db.execute(S.contexts.insert().values(id=str(uuid4()), owner_id=other, version_id=result["version_id"], text="wrong owner"))
    assert exc.value.orig.sqlstate == "23503"
    assert exc.value.orig.diag.constraint_name == "contexts_version_id_owner_id_fkey"
    with storage.engine.connect() as db:
        assert db.scalar(select(func.count()).select_from(S.contexts)) == 0


@pytest.mark.ingestion
@pytest.mark.parametrize(
    "table,field,value", [(S.contexts, "text", "changed"), (S.files, "sha256", "0" * 64), (S.versions, "number", 0)], ids=["context", "file", "version"]
)
def test_original_records_are_immutable(storage, table, field, value):
    owner = str(uuid4())
    result = save(storage, owner)
    with storage.engine.connect() as db:
        before = db.execute(select(table)).mappings().one()
    with pytest.raises(DBAPIError) as exc, storage.engine.begin() as db:
        db.execute(update(table).where(table.c.owner_id == owner).values(**{field: value}))
    assert exc.value.orig.sqlstate == "P0001"
    assert "Original ingestion records are immutable" in str(exc.value.orig)
    with storage.engine.connect() as db:
        assert dict(db.execute(select(table)).mappings().one()) == dict(before)
    assert storage.get(owner, result["dataset_id"]) == result


@pytest.mark.ingestion
def test_corrupted_readback_cannot_publish(storage, monkeypatch):
    original = storage.objects.client.put_object

    def corrupt(**kwargs):
        return original(**{**kwargs, "Body": b"x" * len(CSV)})

    monkeypatch.setattr(storage.objects.client, "put_object", corrupt)
    owner = str(uuid4())
    with pytest.raises(OSError, match="verification failed"):
        save(storage, owner)
    assert counts(storage) == [0, 0, 0]
    assert op_for(storage, owner)["status"] == "failed"


@pytest.mark.ingestion
def test_lost_commit_acknowledgement_replays_ready_dataset(storage, monkeypatch):
    original, raised = Connection.commit, []

    def commit(db):
        original(db)
        if db.engine is storage.engine and not raised:
            with storage.engine.connect() as observer:
                published = observer.scalar(select(func.count()).select_from(S.datasets))
            if published:
                raised.append(True)
                raise OSError("commit acknowledgement lost")

    owner, key = str(uuid4()), str(uuid4())
    with monkeypatch.context() as patch:
        patch.setattr(Connection, "commit", commit)
        with pytest.raises(OSError, match="acknowledgement lost"):
            save(storage, owner, key)
    assert raised and op_for(storage, owner)["status"] == "ready"
    result = save(storage, owner, key)
    assert counts(storage) == [1, 1, 1]
    assert bytes_for(storage, owner, result["dataset_id"]) == CSV


def _writer_process(settings, owner, pipe):
    service = Store(settings)
    original = service.objects.put

    def stop_after_object(*args):
        original(*args)
        pipe.send("object_written")
        pipe.recv()  # Parent kills this process with a pending operation and held DB lock.

    service.objects.put = stop_after_object
    save(service, owner, owner)


@pytest.mark.ingestion
def test_process_death_releases_lock_and_retry_recovers(storage):
    owner = str(uuid4())
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=_writer_process, args=(storage.settings, owner, child))
    process.start()
    try:
        assert parent.poll(30), "child did not reach object write"
        assert parent.recv() == "object_written"
    finally:
        process.terminate()
        process.join(10)
        parent.close()
        child.close()
    assert not process.is_alive() and process.exitcode != 0
    assert op_for(storage, owner)["status"] == "pending"
    assert counts(storage) == [0, 0, 0]
    result = save(storage, owner, owner)
    assert counts(storage) == [1, 1, 1]
    assert bytes_for(storage, owner, result["dataset_id"]) == CSV


@pytest.mark.ingestion
def test_full_app_roundtrip_identity_and_no_runtime_calls(ingestion_client, monkeypatch):
    client = ingestion_client
    payload = {"title": "My study", "context": json.dumps(" Exact account\n"), "idempotency_key": str(uuid4()), "owner_id": str(uuid4())}
    response = client.post("/api/datasets", data=payload, files={"file": ("test.csv", CSV)})
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["initial_context"] == json.loads(payload["context"])
    path = f"/api/datasets/{result['dataset_id']}"
    assert client.get(path).json() == result
    assert client.get(path + "/file").content == CSV
    monkeypatch.setenv("DEV_USER_ID", payload["owner_id"])
    assert client.get(path).status_code == 404
    assert client.get(path + "/file").status_code == 404
    monkeypatch.setenv("APP_ENV", "production")
    assert client.get(path).status_code == 503
    assert client.get(path + "/file").status_code == 503
    assert client.post("/api/datasets", data=payload, files={"file": ("x.csv", CSV)}).status_code == 503


@pytest.mark.ingestion
@pytest.mark.parametrize("data,status", [(b"a,a\n1,2\n", 422), (b"a\n" + b"x" * (26 * 1024 * 1024), 413)])
def test_http_invalid_input_never_publishes(ingestion_client, storage, data, status):
    response = ingestion_client.post("/api/datasets", data={"title": "Study", "idempotency_key": str(uuid4())}, files={"file": ("x.csv", data)})
    assert response.status_code == status
    assert counts(storage) == [0, 0, 0]
    assert storage.objects.client.list_objects_v2(Bucket=storage.objects.bucket).get("KeyCount", 0) == 0


@pytest.mark.ingestion
def test_private_objects_refuse_anonymous_reads(storage):
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    owner = str(uuid4())
    result = save(storage, owner)
    assert bytes_for(storage, owner, result["dataset_id"]) == CSV
    key = op_for(storage, owner)["object_key"]
    anonymous = boto3.client("s3", endpoint_url=storage.settings.endpoint, config=Config(signature_version=UNSIGNED))
    try:
        with pytest.raises(ClientError) as exc:
            anonymous.get_object(Bucket=storage.objects.bucket, Key=key)
        assert exc.value.response["ResponseMetadata"]["HTTPStatusCode"] == 403
    finally:
        anonymous.close()


@pytest.mark.ingestion
@pytest.mark.parametrize("edge", ["files.operation_id", "datasets.operation_id", "versions.dataset_id", "versions.file_id"])
def test_other_ownership_edges_fail_on_their_foreign_key(storage, edge):
    owner, other, op_a, op_b, dataset_id, file_id, child_id = [str(uuid4()) for _ in range(7)]
    table_name, field = edge.split(".")
    table = getattr(S, table_name)
    with storage.engine.begin() as db:
        db.execute(S.users.insert(), [{"id": owner}, {"id": other}])
        for op_id, user in [(op_a, owner), (op_b, other)]:
            db.execute(S.operations.insert().values(id=op_id, owner_id=user, idempotency_key=op_id, fingerprint="0" * 64, object_key=op_id, status="pending"))
        if table_name == "files":
            values = dict(id=child_id, owner_id=other, operation_id=op_a, object_key=child_id, filename="x.csv", sha256="0" * 64, size_bytes=1)
        elif table_name == "datasets":
            values = dict(id=child_id, owner_id=other, operation_id=op_a, title="Study")
        else:
            dataset_owner, dataset_op = (owner, op_a) if field == "dataset_id" else (other, op_b)
            file_owner, file_op = (other, op_b) if field == "dataset_id" else (owner, op_a)
            db.execute(S.datasets.insert().values(id=dataset_id, owner_id=dataset_owner, operation_id=dataset_op, title="Study"))
            db.execute(
                S.files.insert().values(
                    id=file_id, owner_id=file_owner, operation_id=file_op, object_key=file_id, filename="x.csv", sha256="0" * 64, size_bytes=1
                )
            )
            values = dict(id=child_id, owner_id=other, dataset_id=dataset_id, file_id=file_id, number=0)
    with pytest.raises(IntegrityError) as exc, storage.engine.begin() as db:
        db.execute(table.insert().values(**values))
    assert exc.value.orig.sqlstate == "23503"
    assert exc.value.orig.diag.constraint_name == f"{table_name}_{field}_owner_id_fkey"
    with storage.engine.connect() as db:
        assert db.scalar(select(table.c.id).where(table.c.id == child_id)) is None


@pytest.mark.ingestion
@pytest.mark.parametrize("context", ["plain text", "null", "{}", '"\\u0000"', json.dumps("x" * 100001)])
def test_http_invalid_context_never_publishes(ingestion_client, storage, context):
    response = ingestion_client.post(
        "/api/datasets",
        data={"title": "Study", "idempotency_key": str(uuid4()), "context": context},
        files={"file": ("x.csv", CSV)},
    )
    assert response.status_code == 422
    assert counts(storage) == [0, 0, 0]
    assert storage.objects.client.list_objects_v2(Bucket=storage.objects.bucket).get("KeyCount", 0) == 0


@pytest.mark.ingestion
def test_http_omitted_context_is_optional(ingestion_client, storage):
    response = ingestion_client.post("/api/datasets", data={"title": "Study", "idempotency_key": str(uuid4())}, files={"file": ("x.csv", CSV)})
    assert response.status_code == 201, response.text
    assert response.json()["initial_context"] is None
    assert counts(storage) == [1, 1, 1]
    with storage.engine.connect() as db:
        assert db.scalar(select(func.count()).select_from(S.contexts)) == 0
