"""Owned, read-only metadata listing with keyset pagination."""

import base64
import binascii
import json
from datetime import datetime
from uuid import UUID

from sqlalchemy import select, tuple_

from causal_agent.server.ingestion import schema as S


def boundary(owner, cursor):
    if cursor is None:
        return None
    try:
        value = json.loads(base64.b64decode(cursor, altchars=b"-_", validate=True))
        if not isinstance(value, dict) or set(value) != {"owner", "created_at", "id"} or value["owner"] != owner:
            raise ValueError
        if not all(isinstance(v, str) for v in value.values()):
            raise ValueError
        stamp = datetime.fromisoformat(value["created_at"])
        if stamp.tzinfo is None:
            raise ValueError
        return stamp, str(UUID(value["id"]))
    except (ValueError, TypeError, KeyError, binascii.Error) as exc:
        raise ValueError("Invalid dataset cursor. Refresh the library.") from exc


def list_owned(storage, owner, limit, cursor=None):
    after = boundary(owner, cursor)
    query = (
        select(
            S.datasets.c.id.label("dataset_id"),
            S.datasets.c.title,
            S.datasets.c.created_at,
            S.versions.c.id.label("version_id"),
            S.files.c.filename,
            S.files.c.size_bytes,
            S.contexts.c.id.is_not(None).label("has_initial_context"),
        )
        .select_from(
            S.datasets.join(S.operations, S.operations.c.id == S.datasets.c.operation_id)
            .join(S.versions, S.versions.c.dataset_id == S.datasets.c.id)
            .join(S.files, S.files.c.id == S.versions.c.file_id)
            .outerjoin(S.contexts, S.contexts.c.version_id == S.versions.c.id)
        )
        .where(S.datasets.c.owner_id == owner, S.operations.c.status == "ready")
    )
    if after:
        query = query.where(tuple_(S.datasets.c.created_at, S.datasets.c.id) < after)
    query = query.order_by(S.datasets.c.created_at.desc(), S.datasets.c.id.desc()).limit(limit + 1)
    with storage.engine.connect() as db:
        rows = db.execute(query).mappings().all()
    items = [{**dict(row), "status": "saved"} for row in rows[:limit]]
    next_cursor = None
    if len(rows) > limit:
        last = items[-1]
        next_cursor = base64.urlsafe_b64encode(
            json.dumps({"owner": owner, "created_at": last["created_at"].isoformat(), "id": last["dataset_id"]}).encode()
        ).decode()
    return {"items": items, "next_cursor": next_cursor}
