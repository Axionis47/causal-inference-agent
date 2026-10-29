"""Real PostgreSQL revisions, isolated by the existing private storage harness."""

from uuid import uuid4

import pytest

from causal_agent.memory.context.review import ColumnDisposition, ContextEdit, RevisionConflict, SourceReference
from causal_agent.server.context import service
from causal_agent.server.context.repository import Repository
from causal_agent.server.ingestion.service import Missing

pytestmark = pytest.mark.ingestion


def test_extraction_review_and_dataset_reads_share_one_context(prepared, storage, monkeypatch):
    from sqlalchemy import func, select

    from causal_agent.memory.context import extract
    from causal_agent.server.context import schema as S

    owner = str(uuid4())
    dataset = prepared(owner)
    initial = service.generate(storage, owner, dataset["dataset_id"]).pack
    repo = Repository(storage, owner, dataset["dataset_id"])
    review = repo.start_review("What do these amounts mean?")
    assert review["starting_revision_id"] == initial.revision_id
    with storage.engine.connect() as db:
        assert db.scalar(select(func.count()).select_from(S.revisions)) == 1
    first = repo.read(initial.revision_id)
    statement = next(s for s in first.knowledge.statements if s.id == "s1")
    updated = repo.commit(
        first.id,
        str(uuid4()),
        [
            ContextEdit(
                action="replace",
                target_id=statement.id,
                statement=statement.model_copy(update={"id": "corrected-units", "value": "US dollars"}),
                reason="Clarified the currency.",
            )
        ],
        SourceReference(kind="message", id="human-correction", text="The amount is in US dollars."),
    )
    latest = service.view(storage, owner, dataset["dataset_id"]).pack
    assert latest.revision_id == updated.id and updated.parent_id == initial.revision_id
    column_id = statement.column_ids[0]
    current = service.column(storage, owner, dataset["dataset_id"], latest.revision_id, column_id)
    assert [(s.id, s.value) for s in current.statements] == [("corrected-units", "US dollars")]
    assert service.column(storage, owner, dataset["dataset_id"], initial.revision_id, column_id).statements[0].value == "dollars"
    assert repo.read(updated.id).knowledge.for_column(column_id)[0].source.kind in {"extraction", "message"}

    def forbidden(*args):
        raise AssertionError("Preparation must not reset edited context")

    monkeypatch.setattr(extract, "extract", forbidden)
    monkeypatch.setattr(extract, "EXTRACTOR_VERSION", "changed-model-version")
    assert service.generate(storage, owner, dataset["dataset_id"]).pack == latest
    with storage.engine.connect() as db:
        assert db.scalar(select(func.count()).select_from(S.packs)) == 1
        assert db.scalar(select(func.count()).select_from(S.revisions)) == 2


def setup(prepared, storage, owner):
    dataset = prepared(owner)
    service.generate(storage, owner, dataset["dataset_id"])
    repo = Repository(storage, owner, dataset["dataset_id"])
    review = repo.start_review("How do these amounts vary?")
    return dataset, repo, review


def finish(repo, review):
    saved = repo.save_review(review["id"], review["generation"], [], [ColumnDisposition(column_id=c.column_id, status="unresolved") for c in review["columns"]])
    return repo.finish_review(review["id"], saved["generation"])


def test_review_round_trip_and_duplicate_completion(prepared, storage):
    _, repo, review = setup(prepared, storage, str(uuid4()))
    revision = finish(repo, review)
    assert repo.finish_review(review["id"], 2) == revision
    assert repo.review(review["id"])["question"] == "How do these amounts vary?"
    assert all(s.status == "proposed" for s in revision.knowledge.statements)
    assert all(d.status == "unresolved" for d in revision.knowledge.dispositions)


def test_duplicate_commit_and_payload_reuse(prepared, storage):
    _, repo, review = setup(prepared, storage, str(uuid4()))
    revision = finish(repo, review)
    target = revision.knowledge.statements[0].id
    edits = [ContextEdit(action="review", target_id=target)]
    op = str(uuid4())
    source = SourceReference(kind="message", id="m", text="I confirm this statement.")
    next_revision = repo.commit(revision.id, op, edits, source)
    assert repo.commit(revision.id, op, edits, source) == next_revision
    with pytest.raises(ValueError, match="different submission"):
        repo.commit(revision.id, op, [], source)
    assert repo.read(revision.id).knowledge.statements[0].status == "proposed"


def test_stale_edit_reports_changes_without_overwrite(prepared, storage):
    _, repo, review = setup(prepared, storage, str(uuid4()))
    revision = finish(repo, review)
    target = revision.knowledge.statements[0].id
    newer = repo.commit(revision.id, str(uuid4()), [ContextEdit(action="review", target_id=target)], SourceReference(kind="review", id="r"))
    with pytest.raises(RevisionConflict) as error:
        repo.commit(revision.id, str(uuid4()), [ContextEdit(action="unresolved", target_id=target)], SourceReference(kind="message", id="m"))
    assert error.value.current_id == newer.id
    assert target in error.value.changed_ids
    assert repo.read(newer.id).knowledge.statements[0].status == "reviewed"


def test_wrong_owner_and_dataset_cannot_read_revision(prepared, storage):
    owner = str(uuid4())
    dataset, repo, review = setup(prepared, storage, owner)
    revision = finish(repo, review)
    with pytest.raises(Missing):
        Repository(storage, str(uuid4()), dataset["dataset_id"])
    _, other, _ = setup(prepared, storage, owner)
    with pytest.raises(Missing):
        other.read(revision.id)


def test_incomplete_review_cannot_commit(prepared, storage):
    _, repo, review = setup(prepared, storage, str(uuid4()))
    with pytest.raises(ValueError, match="every column"):
        repo.finish_review(review["id"], 1)
    assert repo.review(review["id"])["completed_revision_id"] is None


def test_revision_rows_are_immutable_even_outside_repository(prepared, storage):
    from sqlalchemy.exc import DBAPIError

    from causal_agent.server.context import schema

    _, repo, review = setup(prepared, storage, str(uuid4()))
    revision = finish(repo, review)
    with pytest.raises(DBAPIError, match="immutable"):
        with storage.engine.begin() as db:
            db.execute(schema.revisions.update().where(schema.revisions.c.id == revision.id).values(number=99))
    assert repo.read(revision.id) == revision
