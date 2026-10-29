"""Reuse a successful real Phase 4 extraction for chat-only acceptance.

This evaluation setup copies a validated same-schema product onto an isolated copy
of the exact dataset bytes. Only owner-local column identities change. Source text,
statements, relationship semantics and the original model trace remain intact.
It is not a runtime compatibility loader or a synthetic context generator.
"""

import hashlib
import json

from causal_agent.common import trace
from causal_agent.memory.context import extract as E
from causal_agent.memory.context.contracts import ColumnIdentity, ExtractedContext
from causal_agent.server.context import service


def reuse(storage, owner, data, evidence):
    recorded = json.loads(evidence.with_name("conversation.json").read_text())
    if recorded["dataset"]["sha256"] != data["sha256"]:
        raise ValueError("Retained preparation belongs to different dataset bytes")
    runs = json.loads(evidence.read_text())
    extraction = next(r for r in runs if r["name"] == "context:extract" and not r.get("error") and (r.get("outputs") or {}).get("content"))
    result = ExtractedContext.model_validate(extraction["outputs"])
    original = [ColumnIdentity.model_validate(c) for c in extraction["inputs"]["columns"]]
    with storage.locked(owner, "context:" + data["version_id"]) as db:
        source_id, text = service.source(db, owner, data["version_id"])
        columns = service.column_index(db, owner, data["version_id"])
        if extraction["inputs"]["source"] != text or [(c.name, c.key, c.position) for c in original] != [(c.name, c.key, c.position) for c in columns]:
            raise ValueError("Retained preparation differs from the immutable account or column index")
        identities = {old.column_id: new.column_id for old, new in zip(original, columns, strict=True)}
        for column in result.content.columns:
            column.column_id = identities[column.column_id]
        for statement in result.content.dataset:
            statement.column_ids = [identities[c] for c in statement.column_ids]
        for relationship in result.content.relationships:
            for participant in relationship.participants:
                if participant.kind == "column":
                    participant.id = identities[participant.id]
        errors = E.validate(result.content, text, columns) + E.validate_assessment(result.assessment, result.content, text)
        if errors or result.assessment.corrections:
            raise ValueError("Retained preparation failed validation: " + "; ".join(errors))
        provenance = dict(
            source_checksum=hashlib.sha256(text.encode()).hexdigest(),
            csv_checksum=data["sha256"],
            column_index_fingerprint=E.fingerprint([c.model_dump() for c in columns]),
            schema_version=E.SCHEMA_VERSION,
            extractor_version=E.EXTRACTOR_VERSION,
            model="retained Gemini extraction: " + str(extraction["id"]),
            identity=E.fingerprint({"trace_run": extraction["id"], "identities": identities}),
        )
        with trace.span(
            "context:reuse-evidence",
            inputs={
                "trace_file": str(evidence),
                "trace_run": extraction["id"],
                "column_identities": identities,
                "source_checksum": provenance["source_checksum"],
                "csv_checksum": data["sha256"],
            },
        ) as span:
            revision_id = service.publish(db, owner, data, source_id, text, columns, provenance, result)
            db.commit()
            span.end(outputs={"revision_id": revision_id, "validation": "passed", "new_model_calls": 0})
    return {"trace_file": str(evidence), "trace_run": extraction["id"]}
