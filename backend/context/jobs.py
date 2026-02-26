"""Durable delivery, frozen inputs, and bounded resumable context preparation."""

import asyncio
import logging
from typing import Any, cast
from uuid import UUID, uuid5

from langsmith import trace, tracing_context
from pydantic import ValidationError
from sqlalchemy import text, update
from sqlalchemy.orm import Session

from backend import model_access
from backend.config import settings
from backend.context.contracts import COLUMN_FIELDS, DATASET_FIELDS, ContextContent, empty_content
from backend.context.extraction import (
    ASSESS,
    EXTRACT,
    Assessment,
    Extraction,
    materialize,
    schema_digest,
    validate_assessment,
)
from backend.context.service import append_revision, check_parent, locked_pack, revision_record
from backend.db import engine
from backend.jobs import app, dsn  # noqa: F401
from backend.models import ContextPreparation, now


def enqueue(db: Session, operation_id: UUID) -> int:
    # SQLAlchemy owns commit/rollback. The supported psycopg connector executes the
    # queue INSERT using that same DBAPI connection and does not commit it.
    connection = db.connection().connection.driver_connection
    return int(
        prepare_context.configure(connection=connection, lock=f"context:{operation_id}").defer(
            operation_id=str(operation_id)
        )
    )


def process(operation_id: UUID) -> None:
    # Session-level lock protects publication and persisted call budgets across
    # duplicate delivery. It does not hold a transaction during network calls.
    lock_id = int.from_bytes(operation_id.bytes[:8], "big", signed=True)
    with engine().connect().execution_options(isolation_level="AUTOCOMMIT") as guard:
        if not guard.scalar(text("SELECT pg_try_advisory_lock(:key)"), {"key": lock_id}):
            return
        try:
            _process(operation_id)
        finally:
            guard.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": lock_id})


def _process(identity: UUID) -> None:
    with Session(engine()) as db:
        loaded = db.get(ContextPreparation, identity)
        if loaded is None or loaded.status in {"succeeded", "failed"}:
            return
        operation = loaded
        operation.status, operation.error, operation.updated_at = "running", None, now()
        db.commit()
        try:
            inputs = operation.inputs
            if inputs.get("schema_digest", schema_digest()) != schema_digest():
                raise ValueError("This preparation uses an earlier model contract. Start a new preparation.")
            extract_prompt = str(inputs.get("extract_prompt", EXTRACT))
            assess_prompt = str(inputs.get("assess_prompt", ASSESS))
            version = UUID(str(inputs["version_id"]))
            parent = (
                revision_record(db, version, operation.base_revision_id)
                if operation.base_revision_id
                else None
            )
            base = (
                ContextContent.model_validate(parent.content)
                if parent
                else empty_content(version, cast(list[str], inputs["columns"]))
            )
            source_records: dict[str, dict[str, Any]] = inputs["sources"]  # type: ignore[assignment]
            sources = {sid: entry["text"] for sid, entry in source_records.items()}
            from typing import get_args

            from backend.context.contracts import RelationshipKind

            payload: dict[str, object] = {
                "columns": [column.identity.model_dump() for column in base.columns.values()],
                "existing_concepts": {
                    key: value.model_dump(mode="json") for key, value in base.concepts.items()
                },
                "dataset_fields": list(DATASET_FIELDS),
                "column_fields": list(COLUMN_FIELDS),
                "relationship_kinds": list(get_args(RelationshipKind)),
                "sources": [
                    {
                        "source_id": sid,
                        "title": source_records[sid]["title"],
                        "lines": [
                            {"source_line": n, "text": line}
                            for n, line in enumerate(source.splitlines(keepends=True), 1)
                        ],
                    }
                    for sid, source in sources.items()
                ],
            }
            with tracing_context(project_name=settings().langsmith_project):
                with trace(
                    "context-preparation",
                    run_id=operation.trace_id,
                    inputs={
                        "preparation_id": str(identity),
                        "version_id": str(version),
                        "sources": inputs["sources"],
                        "model": inputs["model"],
                    },
                    metadata={"extractor_version": inputs["extractor_version"]},
                ) as span:

                    def ask(stage: str, schema: type[Any], system: str, data: dict[str, object]) -> Any:
                        steps = operation.steps
                        if stage in steps:
                            return schema.model_validate(steps[stage])
                        if operation.calls >= 4:
                            raise ValueError(
                                "Preparation reached its four-call limit. No context was replaced."
                            )
                        operation.stage = stage
                        operation.calls += (
                            1  # Reserve before contacting the provider, including interrupted calls.
                        )
                        operation.updated_at = now()
                        attempts = cast(list[dict[str, object]], operation.steps.get("attempts", []))
                        operation.steps = {
                            **operation.steps,
                            "attempts": [
                                *attempts,
                                {"call": operation.calls, "stage": stage, "started_at": now().isoformat()},
                            ],
                        }
                        db.commit()
                        attempts = cast(list[dict[str, object]], operation.steps["attempts"])
                        try:
                            result = model_access.structured(model_access.prepare(schema, system, data, str(inputs["model"])))
                        except Exception as error:
                            operation.steps = {
                                **operation.steps,
                                "attempts": [
                                    *attempts[:-1],
                                    {
                                        **attempts[-1],
                                        "error": str(error)[:3500],
                                        "ended_at": now().isoformat(),
                                    },
                                ],
                            }
                            db.commit()
                            raise
                        operation.steps = {
                            **operation.steps,
                            stage: result.model_dump(mode="json"),
                            "attempts": [*attempts[:-1], {**attempts[-1], "completed_at": now().isoformat()}],
                        }
                        operation.updated_at = now()
                        db.commit()
                        return result

                    correction = operation.steps.get("validation_correction")
                    if correction is None:
                        try:
                            extracted = ask("extract", Extraction, extract_prompt, payload)
                            candidate = materialize(extracted, base, sources, identity, operation.author_id)
                            assessment = ask(
                                "assess",
                                Assessment,
                                assess_prompt,
                                {
                                    **payload,
                                    "extraction_contract": Extraction.model_json_schema(),
                                    "extraction": extracted.model_dump(mode="json"),
                                },
                            )
                            validate_assessment(assessment, extracted, sources)
                            if assessment.corrections:
                                correction = {
                                    "previous": extracted.model_dump(mode="json"),
                                    "corrections": assessment.model_dump(mode="json"),
                                }
                        except ValueError as error:
                            correction = {
                                "previous": getattr(error, "raw", operation.steps.get("extract")),
                                "validation_errors": str(error),
                            }
                        if correction is not None:
                            operation.steps = {**operation.steps, "validation_correction": correction}
                            db.commit()
                    if correction is not None:
                        extracted = ask(
                            "correct", Extraction, extract_prompt, {**payload, "correction": correction}
                        )
                        candidate = materialize(extracted, base, sources, identity, operation.author_id)
                        assessment = ask(
                            "reassess",
                            Assessment,
                            assess_prompt,
                            {
                                **payload,
                                "extraction_contract": Extraction.model_json_schema(),
                                "extraction": extracted.model_dump(mode="json"),
                            },
                        )
                        validate_assessment(assessment, extracted, sources)
                        if assessment.corrections:
                            raise ValueError(
                                "Source assessment still found unsupported or omitted context. Review the source material and start a new preparation."
                            )
                    operation.steps = {**operation.steps, "candidate": candidate.model_dump(mode="json")}
                    db.commit()
                    pack = locked_pack(db, version)
                    check_parent(pack, operation.base_revision_id)
                    record = append_revision(
                        db,
                        pack,
                        parent,
                        uuid5(identity, "revision"),
                        operation.submission_id,
                        operation.request_hash,
                        operation.author_id,
                        candidate,
                        [UUID(sid) for sid in cast(list[str], inputs["source_ids"])],
                        "draft",
                    )
                    operation.status, operation.stage, operation.result_revision_id = (
                        "succeeded",
                        "published",
                        record.id,
                    )
                    operation.updated_at = now()
                    db.commit()
                    span.end(
                        outputs={
                            "revision_id": str(record.id),
                            "calls": operation.calls,
                            "content": candidate.model_dump(mode="json"),
                        }
                    )
        except Exception as exc:
            db.rollback()
            failed = db.get(ContextPreparation, identity)
            if failed is None:
                raise
            operation = failed
            from fastapi import HTTPException

            if isinstance(exc, HTTPException):
                message = str(exc.detail)
            elif isinstance(exc, (model_access.InvalidModelOutput, ValidationError)):
                message = "The prepared statements did not pass validation after correction. No context was replaced. Start a new preparation."
                operation.stage = "rejected"
            elif isinstance(exc, ValueError):
                message = str(exc)[:3500]
                operation.stage = "rejected"
            else:
                message = f"Preparation failed ({type(exc).__name__}). Saved context is retained. Check worker logs and retry."
            logging.getLogger(__name__).exception("Context preparation %s failed", identity)
            operation.status, operation.error, operation.updated_at = "failed", message, now()
            attempts = cast(list[dict[str, object]], operation.steps.get("attempts", []))
            if attempts and "completed_at" not in attempts[-1]:
                operation.steps = {
                    **operation.steps,
                    "attempts": [
                        *attempts[:-1],
                        {**attempts[-1], "error": message, "ended_at": now().isoformat()},
                    ],
                }
            db.commit()


@app.task(name="context.prepare", queue="context")
async def prepare_context(operation_id: str) -> None:
    await asyncio.to_thread(process, UUID(operation_id))


@app.periodic(cron="* * * * *")
@app.task(name="context.recover", queue="context")
async def recover_stalled(timestamp: int) -> None:
    for job in await app.job_manager.get_stalled_jobs(
        task_name="context.prepare", seconds_since_heartbeat=60
    ):
        await app.job_manager.retry_job(job, retry_at=now())
    failed = await app.job_manager.list_jobs_async(task="context.prepare", status="failed")
    await asyncio.to_thread(mark_failed_deliveries, [job.id for job in failed if job.id is not None])


def mark_failed_deliveries(job_ids: list[int]) -> None:
    """Expose failures before the domain handler, such as a lost DB connection."""
    if not job_ids:
        return
    with Session(engine()) as db:
        # Predicate includes the current delivery identity. An older failed job
        # cannot change an operation that has since been retried on a new job.
        db.execute(
            update(ContextPreparation)
            .where(
                ContextPreparation.job_id.in_(job_ids), ContextPreparation.status.in_(["queued", "running"])
            )
            .values(
                status="failed",
                error="The worker delivery failed before reporting its result. Saved context is retained. Retry this preparation.",
                updated_at=now(),
            )
        )
        db.commit()
