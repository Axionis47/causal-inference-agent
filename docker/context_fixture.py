"""Seed deterministic context through publication ONLY in the disposable smoke container.

Executed explicitly by docker/smoke.py, never copied into the application image or
used at startup. This checks storage/browser behavior, not semantic model quality.
"""

import hashlib
import sys

from causal_agent.common.config import ingestion
from causal_agent.memory.context import extract as E
from causal_agent.memory.context.contracts import (
    Assessment,
    Citation,
    ColumnContext,
    ColumnStatement,
    DraftContext,
    ExtractedContext,
    Participant,
    Relationship,
)
from causal_agent.server.context import service
from causal_agent.server.ingestion.service import Store

settings = ingestion()
storage = Store(settings)
owner = settings.user_id
for dataset_id in sys.argv[1:]:
    info = service.owned(storage, owner, dataset_id)
    with storage.locked(owner, "context:" + info["version_id"]) as db:
        source_id, text = service.source(db, owner, info["version_id"])
        columns = service.column_index(db, owner, info["version_id"])
        content = DraftContext(columns=[ColumnContext(column_id=c.column_id) for c in columns])
        for c, col in zip(columns, content.columns, strict=True):
            quote = {"id": "id identifies a participant.", "note": "note contains a supplied note."}.get(c.name)
            if quote and quote in text:
                start = text.index(quote)
                col.statements.append(
                    ColumnStatement(
                        id="statement-" + c.column_id,
                        field="definition",
                        value=quote,
                        basis="source_statement",
                        citations=[Citation(source_line=text[:start].count("\n") + 1, start=start, end=start + len(quote), quote=quote)],
                    )
                )
        quote = "id and note describe the same participant."
        if quote in text:
            start = text.index(quote)
            content.relationships.append(
                Relationship(
                    id="relationship-1",
                    kind="membership",
                    value=quote,
                    basis="source_statement",
                    citations=[Citation(source_line=text[:start].count("\n") + 1, start=start, end=start + len(quote), quote=quote)],
                    participants=[Participant(kind="column", id=c.column_id) for c in columns],
                )
            )
        provenance = dict(
            source_checksum=hashlib.sha256(text.encode()).hexdigest(),
            csv_checksum=info["sha256"],
            column_index_fingerprint=E.fingerprint([c.model_dump() for c in columns]),
            schema_version=E.SCHEMA_VERSION,
            extractor_version=E.EXTRACTOR_VERSION,
            model="smoke-fixture",
            identity=E.fingerprint({"fixture": dataset_id}),
        )
        service.publish(
            db,
            owner,
            info,
            source_id,
            text,
            columns,
            provenance,
            ExtractedContext(content=content, assessment=Assessment(notes=["Deterministic smoke fixture, not a model acceptance run."]), model_calls=0),
        )
        db.commit()
print("Published deterministic context fixtures for", len(sys.argv) - 1, "datasets")
