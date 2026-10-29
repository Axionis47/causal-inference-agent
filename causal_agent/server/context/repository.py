"""PostgreSQL implementation of the context repository, including review drafts."""

import hashlib
import json

from sqlalchemy import select

from causal_agent.memory.context.projection import validate_bindings
from causal_agent.memory.context.review import (
    ColumnDisposition,
    ContextEdit,
    Knowledge,
    Revision,
    RevisionConflict,
    SourceReference,
    apply_edits,
    complete_review,
    validate_sources,
)
from causal_agent.server.context import schema as S
from causal_agent.server.context.service import Prerequisite
from causal_agent.server.context.service import source as original_source
from causal_agent.server.ingestion import schema as I
from causal_agent.server.ingestion.lookup import owned
from causal_agent.server.ingestion.service import Missing, uid


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class Repository:
    def __init__(self, storage, owner, dataset_id):
        self.storage, self.owner, self.dataset_id = storage, owner, dataset_id
        self.info = owned(storage, owner, dataset_id)
        self.version_id = self.info["version_id"]

    def lock(self, db):
        row = db.execute(select(I.versions.c.id).where(I.versions.c.id == self.version_id, I.versions.c.owner_id == self.owner).with_for_update()).first()
        if row is None:
            raise Missing("Dataset not found.")

    def head(self, db):
        return (
            db.execute(
                select(S.revisions)
                .where(S.revisions.c.owner_id == self.owner, S.revisions.c.version_id == self.version_id)
                .order_by(S.revisions.c.number.desc())
                .limit(1)
            )
            .mappings()
            .first()
        )

    def row(self, db, revision_id):
        row = (
            db.execute(
                select(S.revisions).where(S.revisions.c.id == revision_id, S.revisions.c.owner_id == self.owner, S.revisions.c.version_id == self.version_id)
            )
            .mappings()
            .first()
        )
        if row is None:
            raise Missing("Context revision not found.")
        return row

    def read(self, revision_id):
        with self.storage.engine.connect() as db:
            return self.decode(self.row(db, revision_id))

    @staticmethod
    def decode(row):
        return Revision(**{key: row[key] for key in Revision.model_fields})

    def insert(self, db, knowledge, pack_id, previous=None):
        rev = Revision(
            id=uid(),
            version_id=self.version_id,
            source_pack_id=pack_id,
            parent_id=previous["id"] if previous else None,
            number=previous["number"] + 1 if previous else 1,
            knowledge=knowledge,
        )
        db.execute(S.revisions.insert().values(owner_id=self.owner, **rev.model_dump(mode="json")))
        return rev

    def selected_head(self, db):
        current = self.head(db)
        if current is None:
            raise Prerequisite("Prepare the profile and context before reviewing it.")
        return self.decode(current)

    def check_revision(self, db, expected_id):
        expected = self.row(db, expected_id)
        current = self.head(db)
        if current["id"] != expected_id:
            before = {s["id"]: s for s in expected["knowledge"]["statements"]}
            after = {s["id"]: s for s in current["knowledge"]["statements"]}
            raise RevisionConflict(current["id"], sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k)))
        return current

    def commit_in(self, db, expected_id, operation_id, edits, source, dispositions=None):
        digest = fingerprint(
            {
                "expected": expected_id,
                "edits": [e.model_dump() for e in edits],
                "source": source.model_dump(),
                "dispositions": None if dispositions is None else [d.model_dump() for d in dispositions],
            }
        )
        receipt = db.execute(select(S.receipts).where(S.receipts.c.id == operation_id)).mappings().first()
        if receipt:
            if receipt["owner_id"] != self.owner or receipt["version_id"] != self.version_id:
                raise Missing("Operation not found.")
            if receipt["request_hash"] != digest:
                raise ValueError("This operation ID already belongs to a different submission.")
            return self.decode(self.row(db, receipt["revision_id"]))
        current = self.check_revision(db, expected_id)
        knowledge = apply_edits(Knowledge.model_validate(current["knowledge"]), edits, source)
        if dispositions is not None:
            knowledge = complete_review(knowledge, dispositions)
        validate_bindings(knowledge)
        validate_sources(knowledge, original_source(db, self.owner, self.version_id)[1])
        rev = (
            self.decode(current)
            if knowledge.model_dump(mode="json") == current["knowledge"]
            else self.insert(db, knowledge, current["source_pack_id"], current)
        )
        db.execute(
            S.receipts.insert().values(
                id=operation_id,
                owner_id=self.owner,
                version_id=self.version_id,
                request_hash=digest,
                revision_id=rev.id,
                source=source.model_dump(),
                edits=[e.model_dump() for e in edits],
            )
        )
        return rev

    def commit(self, expected_id, operation_id, edits, source):
        with self.storage.engine.begin() as db:
            self.lock(db)
            return self.commit_in(db, expected_id, operation_id, edits, source)

    def start_review(self, question):
        question = question.strip()
        if not question:
            raise ValueError("Enter the question you want to explore first.")
        with self.storage.engine.begin() as db:
            self.lock(db)
            revision = self.selected_head(db)
            review_id = uid()
            db.execute(
                S.reviews.insert().values(id=review_id, owner_id=self.owner, version_id=self.version_id, question=question, starting_revision_id=revision.id)
            )
        return self.review(review_id)

    def review_row(self, db, review_id):
        row = (
            db.execute(select(S.reviews).where(S.reviews.c.id == review_id, S.reviews.c.owner_id == self.owner, S.reviews.c.version_id == self.version_id))
            .mappings()
            .first()
        )
        if row is None:
            raise Missing("Review not found.")
        return row

    def review(self, review_id):
        with self.storage.engine.connect() as db:
            row = self.review_row(db, review_id)
            revision = self.decode(self.row(db, row["completed_revision_id"] or row["starting_revision_id"]))
            return {**dict(row), "dataset_id": self.dataset_id, "columns": revision.knowledge.columns}

    def review_knowledge(self, review_id):
        with self.storage.engine.connect() as db:
            row = self.review_row(db, review_id)
            if row["completed_revision_id"]:
                return self.decode(self.row(db, row["completed_revision_id"])).knowledge
            knowledge = Knowledge.model_validate(self.row(db, row["starting_revision_id"])["knowledge"])
            return apply_edits(knowledge, [ContextEdit.model_validate(e) for e in row["edits"]], SourceReference(kind="review", id=review_id))

    def save_review(self, review_id, generation, edits, dispositions):
        with self.storage.engine.begin() as db:
            self.lock(db)
            row = self.review_row(db, review_id)
            if row["completed_revision_id"] or row["generation"] != generation:
                raise ValueError("This review changed. Reload it before editing.")
            self.check_revision(db, row["starting_revision_id"])
            knowledge = Knowledge.model_validate(self.row(db, row["starting_revision_id"])["knowledge"])
            proposed = apply_edits(knowledge, edits, SourceReference(kind="review", id=review_id))
            validate_bindings(proposed)
            validate_sources(proposed, original_source(db, self.owner, self.version_id)[1])
            # Partial dispositions are allowed while reviewing, but must name real columns once.
            Knowledge(columns=knowledge.columns, dispositions=dispositions)
            db.execute(
                S.reviews.update()
                .where(S.reviews.c.id == review_id)
                .values(generation=generation + 1, edits=[e.model_dump() for e in edits], dispositions=[d.model_dump() for d in dispositions])
            )
        return self.review(review_id)

    def finish_review(self, review_id, generation):
        with self.storage.engine.begin() as db:
            self.lock(db)
            row = self.review_row(db, review_id)
            if row["generation"] != generation:
                raise ValueError("This review changed. Reload it before confirming.")
            if row["completed_revision_id"]:
                return self.decode(self.row(db, row["completed_revision_id"]))
            revision = self.commit_in(
                db,
                row["starting_revision_id"],
                review_id,
                [ContextEdit.model_validate(e) for e in row["edits"]],
                SourceReference(kind="review", id=review_id, text=row["question"]),
                [ColumnDisposition.model_validate(d) for d in row["dispositions"]],
            )
            db.execute(S.reviews.update().where(S.reviews.c.id == review_id).values(completed_revision_id=revision.id))
            return revision
