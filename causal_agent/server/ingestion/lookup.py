"""One ready, owned dataset lookup shared by profile and context services."""

from sqlalchemy import select

from causal_agent.server.ingestion import schema as I
from causal_agent.server.ingestion.service import Missing


def owned(storage, owner, dataset_id):
    with storage.engine.connect() as db:
        row = (
            db.execute(
                select(I.versions.c.id.label("version_id"), I.files.c.sha256, I.files.c.size_bytes, I.files.c.filename, I.files.c.object_key)
                .select_from(
                    I.datasets.join(I.operations, I.operations.c.id == I.datasets.c.operation_id)
                    .join(I.versions, I.versions.c.dataset_id == I.datasets.c.id)
                    .join(I.files, I.files.c.id == I.versions.c.file_id)
                )
                .where(I.datasets.c.id == dataset_id, I.datasets.c.owner_id == owner, I.operations.c.status == "ready")
            )
            .mappings()
            .first()
        )
    if row is None:
        raise Missing("Dataset not found.")
    return dict(row)
