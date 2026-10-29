"""Real foreign keys, atomic revisions, lock contention and interrupted publication."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from uuid import uuid4

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.exc import IntegrityError

from causal_agent.common.llm import set_llm
from causal_agent.memory.context import extract as E
from causal_agent.memory.context.tests.builders import SOURCE, happy
from causal_agent.server.context import schema as S
from causal_agent.server.context import service
from causal_agent.server.ingestion.service import Conflict

pytestmark = pytest.mark.ingestion


def reset_model(storage, owner, version):
    with storage.engine.connect() as db:
        columns = service.column_index(db, owner, version)
    set_llm(happy(columns, SOURCE))


@pytest.mark.parametrize("other_owner", [False, True])
def test_column_fk(storage, prepared, other_owner):
    owner = str(uuid4())
    a = prepared(owner)
    pa = service.generate(storage, owner, a["dataset_id"]).pack
    with storage.engine.connect() as db:
        provenance_id = db.scalar(select(S.revisions.c.source_pack_id).where(S.revisions.c.id == pa.revision_id))
    other = str(uuid4()) if other_owner else owner
    b = prepared(other)
    pb = service.generate(storage, other, b["dataset_id"]).pack
    with pytest.raises(IntegrityError) as exc:
        with storage.engine.begin() as db:
            db.execute(
                S.column_contexts.insert().values(
                    pack_id=provenance_id, column_id=pb.columns[0].column_id, owner_id=owner, version_id=a["version_id"], statements=[]
                )
            )
    assert exc.value.orig.sqlstate == "23503"
    with pytest.raises(IntegrityError) as exc:
        with storage.engine.begin() as db:
            db.execute(S.relationships.insert().values(pack_id=provenance_id, id="new-relation", owner_id=other, version_id=b["version_id"], result={}))
    assert exc.value.orig.sqlstate == "23503"


def test_source_fk(storage, prepared):
    owner = str(uuid4())
    a = prepared(owner)
    pa = service.generate(storage, owner, a["dataset_id"]).pack
    b = prepared(owner)
    with storage.engine.connect() as db:
        sid, _ = service.source(db, owner, b["version_id"])
        values = dict(
            db.execute(select(S.packs).where(S.packs.c.id == db.scalar(select(S.revisions.c.source_pack_id).where(S.revisions.c.id == pa.revision_id))))
            .mappings()
            .one()
        )
    values.update(id=str(uuid4()), revision=2, identity="a" * 64, source_id=sid)
    with pytest.raises(IntegrityError) as exc:
        with storage.engine.begin() as db:
            db.execute(S.packs.insert().values(**values))
    assert exc.value.orig.sqlstate == "23503"


def test_atomic_interruption_and_retry(storage, prepared):
    owner = str(uuid4())
    record = prepared(owner)

    def interrupt(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("INSERT INTO context.relationships"):
            raise RuntimeError("interrupted after columns")

    event.listen(storage.engine, "before_cursor_execute", interrupt)
    try:
        with pytest.raises(RuntimeError, match="after columns"):
            service.generate(storage, owner, record["dataset_id"])
    finally:
        event.remove(storage.engine, "before_cursor_execute", interrupt)
    with storage.engine.connect() as db:
        assert all(db.scalar(select(func.count()).select_from(t)) == 0 for t in [S.packs, S.column_contexts, S.relationships, S.revisions])
    reset_model(storage, owner, record["version_id"])
    assert service.generate(storage, owner, record["dataset_id"]).pack.revision == 1


def test_concurrency_returns_conflict(storage, prepared, monkeypatch):
    owner = str(uuid4())
    record = prepared(owner)
    entered, release = Event(), Event()
    extract = E.extract

    def slow(source, columns):
        entered.set()
        assert release.wait(15)
        return extract(source, columns)

    monkeypatch.setattr(E, "extract", slow)
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(service.generate, storage, owner, record["dataset_id"])
        try:
            assert entered.wait(15)
            with pytest.raises(Conflict):
                service.generate(storage, owner, record["dataset_id"])
        finally:
            release.set()
        assert task.result().pack.revision == 1
    assert service.generate(storage, owner, record["dataset_id"]).pack.revision == 1


def test_preparation_never_forks_existing_context(storage, prepared, monkeypatch):
    owner = str(uuid4())
    record = prepared(owner)
    first = service.generate(storage, owner, record["dataset_id"])
    monkeypatch.setattr(E, "EXTRACTOR_VERSION", "next")

    def forbidden(*a):
        raise AssertionError("Existing context must be updated through its revision gate")

    monkeypatch.setattr(E, "extract", forbidden)
    assert service.generate(storage, owner, record["dataset_id"]) == first
    assert service.dataset(storage, owner, record["dataset_id"], first.pack.revision_id).pack == first.pack


def test_invalid_participant_cannot_be_published(storage, prepared, monkeypatch):
    owner = str(uuid4())
    record = prepared(owner)
    original = E.extract

    def corrupt(source, columns):
        result = original(source, columns)
        result.content.relationships[0].participants[0].id = "foreign-column"
        return result

    monkeypatch.setattr(E, "extract", corrupt)
    with pytest.raises(E.InvalidExtraction, match="participant"):
        service.generate(storage, owner, record["dataset_id"])
    assert service.view(storage, owner, record["dataset_id"]).status == "not_prepared"


def test_migration_preserves_originals_and_profiles(storage, prepared):
    from alembic import command
    from alembic.config import Config

    owner = str(uuid4())
    record = prepared(owner)
    command.downgrade(Config("alembic.ini"), "0003_profiling")
    assert storage.get(owner, record["dataset_id"]) == record
    command.upgrade(Config("alembic.ini"), "head")
    assert service.generate(storage, owner, record["dataset_id"]).pack.csv_checksum == record["sha256"]


def test_profile_version_changes_do_not_reextract_context(storage, prepared, monkeypatch):
    from causal_agent.profile import inspection
    from causal_agent.server.profiling import service as profiling

    owner = str(uuid4())
    record = prepared(owner)
    first = service.generate(storage, owner, record["dataset_id"])
    monkeypatch.setattr(inspection, "VERSION", "new-profiler")
    profiling.generate(storage, owner, record["dataset_id"])

    def forbidden(*args):
        raise AssertionError("Profile statistics must not invalidate semantic context")

    monkeypatch.setattr(E, "extract", forbidden)
    assert service.generate(storage, owner, record["dataset_id"]) == first


def test_unresolved_assessment_cannot_be_published(storage, prepared, monkeypatch):
    from causal_agent.memory.context.contracts import Correction

    owner = str(uuid4())
    record = prepared(owner)
    original = E.extract

    def unresolved(source, columns):
        result = original(source, columns)
        result.assessment.corrections.append(Correction(statement_ids=["s1"], source_lines=[2], reason="Units mapped incorrectly"))
        return result

    monkeypatch.setattr(E, "extract", unresolved)
    with pytest.raises(E.InvalidExtraction, match="Units mapped incorrectly"):
        service.generate(storage, owner, record["dataset_id"])
    assert service.view(storage, owner, record["dataset_id"]).status == "not_prepared"
