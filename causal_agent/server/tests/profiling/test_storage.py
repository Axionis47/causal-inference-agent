"""Phase 3: real relational integrity, atomic publication and lock recovery."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from uuid import uuid4

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.exc import IntegrityError

from causal_agent.profile import inspection
from causal_agent.server.ingestion.service import Conflict
from causal_agent.server.profiling import schema as S
from causal_agent.server.profiling import service

pytestmark = pytest.mark.ingestion


def test_column_identity_survives_retry_and_profiler_change(storage, saved, monkeypatch):
    owner = str(uuid4())
    record = saved(owner)
    first = service.generate(storage, owner, record["dataset_id"]).profile
    retry = service.generate(storage, owner, record["dataset_id"]).profile
    assert first.profile_id == retry.profile_id
    assert [c.column_id for c in first.columns] == [c.column_id for c in retry.columns]
    monkeypatch.setattr(inspection, "VERSION", "another-version")
    later = service.generate(storage, owner, record["dataset_id"]).profile
    assert later.profile_id != first.profile_id
    assert [c.column_id for c in later.columns] == [c.column_id for c in first.columns]


@pytest.mark.parametrize("other_owner", [False, True])
def test_column_foreign_key_rejects_wrong_version(storage, saved, other_owner):
    owner = str(uuid4())
    a, b = saved(owner), saved(str(uuid4()) if other_owner else owner)
    pa = service.generate(storage, owner, a["dataset_id"]).profile
    with storage.engine.connect() as db:
        b_owner = db.scalar(select(S.metadata.tables["ingestion.versions"].c.owner_id).where(S.metadata.tables["ingestion.versions"].c.id == b["version_id"]))
    pb = service.generate(storage, b_owner, b["dataset_id"]).profile
    with pytest.raises(IntegrityError) as exc:
        with storage.engine.begin() as db:
            # New pair avoids a primary-key failure masking the ownership/version FK.
            db.execute(
                S.column_results.insert().values(
                    profile_id=pa.profile_id, column_id=pb.columns[0].column_id, owner_id=owner, version_id=a["version_id"], status="complete", result={}
                )
            )
    assert exc.value.orig.sqlstate == "23503"


def test_profile_foreign_key_rejects_wrong_version(storage, saved):
    owner = str(uuid4())
    a, b = saved(owner), saved(owner)
    pa = service.generate(storage, owner, a["dataset_id"]).profile
    pb = service.generate(storage, owner, b["dataset_id"]).profile
    with pytest.raises(IntegrityError) as exc:
        with storage.engine.begin() as db:
            db.execute(
                S.column_results.insert().values(
                    profile_id=pa.profile_id, column_id=pb.columns[0].column_id, owner_id=owner, version_id=b["version_id"], status="complete", result={}
                )
            )
    assert exc.value.orig.sqlstate == "23503"


def test_atomic_publication_and_retry(storage, saved):
    owner = str(uuid4())
    record = saved(owner)
    calls = 0

    def interrupt(conn, cursor, statement, parameters, context, executemany):
        nonlocal calls
        if statement.startswith("INSERT INTO profiling.column_results"):
            calls += 1
            if calls == 2:
                raise RuntimeError("interrupted between column writes")

    event.listen(storage.engine, "before_cursor_execute", interrupt)
    try:
        with pytest.raises(RuntimeError, match="between column"):
            service.generate(storage, owner, record["dataset_id"])
    finally:
        event.remove(storage.engine, "before_cursor_execute", interrupt)
    with storage.engine.connect() as db:
        assert all(db.scalar(select(func.count()).select_from(t)) == 0 for t in [S.columns, S.profiles, S.column_results])
    assert service.generate(storage, owner, record["dataset_id"]).profile.dataset.rows == 2


def test_concurrent_generation(storage, saved, monkeypatch):
    owner = str(uuid4())
    record = saved(owner)
    entered, release = Event(), Event()
    original = inspection.inspect

    def slow(stream):
        entered.set()
        assert release.wait(10)
        return original(stream)

    monkeypatch.setattr(inspection, "inspect", slow)
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(service.generate, storage, owner, record["dataset_id"])
        try:
            assert entered.wait(10)
            with pytest.raises(Conflict):
                service.generate(storage, owner, record["dataset_id"])
        finally:
            release.set()
        assert task.result().profile is not None
    with storage.engine.connect() as db:
        assert db.scalar(select(func.count()).select_from(S.profiles)) == 1


def test_failed_computation_and_recovery(storage, saved, monkeypatch):
    owner = str(uuid4())
    record = saved(owner)
    original = inspection.inspect

    def fail(stream):
        raise RuntimeError("calculation crashed")

    monkeypatch.setattr(inspection, "inspect", fail)
    first = service.generate(storage, owner, record["dataset_id"])
    assert first.status == "failed" and first.profile.columns == []
    monkeypatch.setattr(inspection, "inspect", original)
    assert service.generate(storage, owner, record["dataset_id"]).profile.dataset.rows == 2


def test_failed_retry_preserves_report(storage, saved, monkeypatch):
    owner = str(uuid4())
    record = saved(owner)
    before = service.generate(storage, owner, record["dataset_id"])

    def fail(stream):
        raise RuntimeError("retry crashed")

    monkeypatch.setattr(inspection, "inspect", fail)
    with pytest.raises(OSError, match="previous report"):
        service.generate(storage, owner, record["dataset_id"])
    assert service.view(storage, owner, record["dataset_id"]) == before


def test_corrupted_original_is_rejected(storage, saved):
    owner = str(uuid4())
    record = saved(owner)
    info = service.owned(storage, owner, record["dataset_id"])
    storage.objects.client.put_object(Bucket=storage.objects.bucket, Key=info["object_key"], Body=b"code,amount\n003,3\n004,4\n")
    with pytest.raises(OSError, match="checksum"):
        service.generate(storage, owner, record["dataset_id"])
    assert service.view(storage, owner, record["dataset_id"]).status == "not_profiled"


def test_dataset_and_columns_read_one_snapshot(storage, saved):
    owner = str(uuid4())
    record = saved(owner)
    before = service.generate(storage, owner, record["dataset_id"])
    changed = False

    def change_between_reads(conn, cursor, statement, parameters, context, executemany):
        nonlocal changed
        if not changed and statement.startswith("SELECT profiling.columns"):
            changed = True
            from sqlalchemy import update

            with storage.engine.begin() as writer:
                writer.execute(
                    update(S.profiles)
                    .where(S.profiles.c.id == before.profile.profile_id)
                    .values(status="complete", dataset={**before.profile.dataset.model_dump(), "rows": 999})
                )
                row = writer.execute(select(S.column_results).where(S.column_results.c.profile_id == before.profile.profile_id)).mappings().first()
                writer.execute(
                    update(S.column_results)
                    .where(S.column_results.c.profile_id == before.profile.profile_id)
                    .values(status="complete", result={**row["result"], "suggested_dtype": "unknown"})
                )

    event.listen(storage.engine, "before_cursor_execute", change_between_reads)
    try:
        during = service.view(storage, owner, record["dataset_id"])
    finally:
        event.remove(storage.engine, "before_cursor_execute", change_between_reads)
    assert changed and during == before, "One response must not mix dataset and column publications"
    assert service.view(storage, owner, record["dataset_id"]).profile.dataset.rows == 999


def test_migration_preserves_ingested_records_and_objects(storage, saved):
    from alembic import command
    from alembic.config import Config

    command.downgrade(Config("alembic.ini"), "0002_library_index")
    owner = str(uuid4())
    record = saved(owner)
    command.upgrade(Config("alembic.ini"), "head")
    assert storage.get(owner, record["dataset_id"]) == record
    assert service.generate(storage, owner, record["dataset_id"]).profile.source_checksum == record["sha256"]


def test_stream_failure_closes_body_and_preserves_existing(storage, saved, monkeypatch):
    owner = str(uuid4())
    record = saved(owner)
    before = service.generate(storage, owner, record["dataset_id"])
    closed = []

    class BrokenBody:
        def iter_chunks(self, size):
            yield b"code,"
            raise OSError("interrupted object stream")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(storage.objects, "open", lambda key: BrokenBody())
    with pytest.raises(OSError, match="interrupted object"):
        service.generate(storage, owner, record["dataset_id"])
    assert closed == [True]
    assert service.view(storage, owner, record["dataset_id"]) == before
