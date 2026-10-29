"""Publication and recovery. A PostgreSQL advisory lock fences retries and cleanup."""

import hashlib
import json
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import create_engine, select, text, update
from sqlalchemy.dialects.postgresql import insert

from causal_agent.server.ingestion import schema as S
from causal_agent.server.ingestion.objects import Objects
from causal_agent.server.ingestion.validation import validate


class Conflict(ValueError):
    pass


class Missing(LookupError):
    pass


def uid():
    return str(uuid4())


class Store:
    def __init__(self, settings):
        self.settings = settings
        self.engine = create_engine(settings.database_url, pool_pre_ping=True, connect_args={"connect_timeout": 5})
        self.objects = Objects(settings)

    @contextmanager
    def locked(self, owner, key):
        lock = int.from_bytes(hashlib.sha256(f"{owner}:{key}".encode()).digest()[:8], "big", signed=True)
        with self.engine.connect() as db:
            acquired = db.scalar(text("SELECT pg_try_advisory_lock(:key)"), {"key": lock})
            db.commit()
            if not acquired:
                raise Conflict("This submission is still being saved. Retry with the same form shortly.")
            try:
                yield db
            finally:
                db.rollback()
                db.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": lock})
                db.commit()

    def ingest(self, owner, idem, title, context, filename, stream):
        digest, size = validate(stream, filename, self.settings)
        fingerprint = hashlib.sha256(json.dumps([title, context, filename, digest], ensure_ascii=False).encode()).hexdigest()
        with self.locked(owner, idem) as db:
            op = db.execute(select(S.operations).where(S.operations.c.owner_id == owner, S.operations.c.idempotency_key == idem)).mappings().first()
            if op:
                if op["fingerprint"] != fingerprint:
                    raise Conflict("This submission key was already used for different content.")
                if op["status"] == "ready":
                    dataset_id = db.scalar(select(S.datasets.c.id).where(S.datasets.c.operation_id == op["id"]))
                    return self.get(owner, dataset_id, db=db)
                operation_id, object_key = op["id"], op["object_key"]
                db.execute(update(S.operations).where(S.operations.c.id == operation_id).values(status="pending"))
            else:
                operation_id = uid()
                object_key = f"users/{owner}/uploads/{operation_id}/original.csv"
                db.execute(insert(S.users).values(id=owner).on_conflict_do_nothing())
                db.execute(
                    S.operations.insert().values(
                        id=operation_id, owner_id=owner, idempotency_key=idem, fingerprint=fingerprint, object_key=object_key, status="pending"
                    )
                )
            db.commit()
            try:
                self.objects.put(object_key, stream, size, digest)
                dataset_id, version_id, file_id = uid(), uid(), uid()
                db.execute(
                    S.files.insert().values(
                        id=file_id, owner_id=owner, operation_id=operation_id, object_key=object_key, filename=filename, sha256=digest, size_bytes=size
                    )
                )
                db.execute(S.datasets.insert().values(id=dataset_id, owner_id=owner, operation_id=operation_id, title=title))
                db.execute(S.versions.insert().values(id=version_id, owner_id=owner, dataset_id=dataset_id, file_id=file_id, number=0))
                if context != "":
                    db.execute(S.contexts.insert().values(id=uid(), owner_id=owner, version_id=version_id, text=context))
                db.execute(update(S.operations).where(S.operations.c.id == operation_id).values(status="ready"))
                db.commit()
            except Exception:
                db.rollback()
                # A lost commit acknowledgement may have published successfully. Never delete here.
                db.execute(update(S.operations).where(S.operations.c.id == operation_id, S.operations.c.status != "ready").values(status="failed"))
                db.commit()
                raise
        return self.get(owner, dataset_id)

    def get(self, owner, dataset_id, db=None):
        query = select(
            S.datasets.c.id.label("dataset_id"),
            S.datasets.c.title,
            S.datasets.c.created_at,
            S.versions.c.id.label("version_id"),
            S.files.c.filename,
            S.files.c.size_bytes,
            S.files.c.sha256,
            S.contexts.c.text.label("initial_context"),
        ).select_from(
            S.datasets.join(S.versions, S.versions.c.dataset_id == S.datasets.c.id)
            .join(S.files, S.files.c.id == S.versions.c.file_id)
            .outerjoin(S.contexts, S.contexts.c.version_id == S.versions.c.id)
        )
        with nullcontext(db) if db is not None else self.engine.connect() as connection:
            row = connection.execute(query.where(S.datasets.c.id == dataset_id, S.datasets.c.owner_id == owner)).mappings().first()
            if not row:
                raise Missing("Dataset not found.")
            return {**dict(row), "status": "saved"}

    def download(self, owner, dataset_id):
        metadata = self.get(owner, dataset_id)
        with self.engine.connect() as db:
            key = db.scalar(
                select(S.files.c.object_key)
                .join(S.versions, S.versions.c.file_id == S.files.c.id)
                .where(S.versions.c.dataset_id == dataset_id, S.files.c.owner_id == owner)
            )
        return metadata, self.objects.open(key)

    def cleanup(self):
        cutoff = datetime.now(UTC) - timedelta(hours=24)
        with self.engine.connect() as db:
            pending = db.execute(select(S.operations).where(S.operations.c.status != "ready", S.operations.c.created_at < cutoff)).mappings().all()
        removed = 0
        for op in pending:
            try:
                with self.locked(op["owner_id"], op["idempotency_key"]) as db:
                    status = db.scalar(select(S.operations.c.status).where(S.operations.c.id == op["id"]))
                    if status == "ready":
                        continue
                    self.objects.delete(op["object_key"])
                    # Retain the idempotency record. A later retry can use the original request again.
                    db.execute(update(S.operations).where(S.operations.c.id == op["id"]).values(status="failed"))
                    db.commit()
                    removed += 1
            except Conflict:
                continue
        return removed
