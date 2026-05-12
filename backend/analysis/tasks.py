"""Read-only task, attempt and budget views for preserved historical runs."""

from contextlib import nullcontext
from copy import deepcopy
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session, defer

from backend.db import engine
from backend.models import AnalysisRun, AnalysisTask, AnalysisTaskAttempt

ATTEMPTS_LIMIT = 2

COLUMN_LIMIT = 16

SHARED_LIMIT = 27

def _counts(db: Session, run: AnalysisRun) -> dict[str, Any]:
    columns: dict[str, int] = {}
    obligations: dict[str, int] = {}
    query = (
        select(
            AnalysisTask.column_ids,
            AnalysisTask.obligation_keys,
            AnalysisTask.budget_kind,
            func.count(AnalysisTaskAttempt.id),
        )
        .join(AnalysisTaskAttempt, AnalysisTaskAttempt.task_id == AnalysisTask.id)
        .where(AnalysisTask.run_id == run.id)
        .group_by(AnalysisTask.id)
    )
    for owned_columns, owned_obligations, budget_kind, count in db.execute(query):
        for key in owned_obligations:
            obligations[key] = obligations.get(key, 0) + count
        if budget_kind == "column":
            for column in owned_columns:
                columns[column] = columns.get(column, 0) + count
    return {
        "columns": columns,
        "obligations": obligations,
        "shared_calls": run.shared_calls,
        "model_calls": run.calls,
    }

def _shared_key(key: str, metrics: dict[str, Any], field: str) -> str:
    return str(metrics.get(field, key.removeprefix("shared:")))

def shared_exhaustion(
    db: Session, run: AnalysisRun, *, attempts: list[dict[str, Any]] | None = None,
) -> str | None:
    """Explain an unrecoverable provider handoff without changing saved attempts.

    Only the latest failed/uncertain provider attempt is relevant. An accepted
    proposal can still be replayed even when its obligation has no calls left.
    The current model-start event also fences off later, unrelated task errors.
    """
    values = shared_attempts(db, run.id) if attempts is None else attempts
    if not values:
        return None
    latest = values[-1]
    if latest["status"] not in {"failed", "uncertain"} or latest["has_proposal"]:
        return None
    logical = _shared_key(latest["key"], latest["metrics"], "logical_key")
    if "model:" + logical in run.receipts:
        return None
    from backend.models import AnalysisEvent

    current = db.scalar(select(AnalysisEvent).where(
        AnalysisEvent.run_id == run.id,
        AnalysisEvent.key == f"model_start:{run.calls}",
        AnalysisEvent.stage == run.stage,
    ))
    if current is None or current.detail.get("provider_attempt_key") != latest["key"]:
        return None
    group = _shared_key(latest["key"], latest["metrics"], "obligation_key")
    grouped = [value for value in values if _shared_key(value["key"], value["metrics"], "obligation_key") == group]
    limit = latest["metrics"].get("obligation_limit", ATTEMPTS_LIMIT)
    if type(limit) is not int or not 1 <= limit <= SHARED_LIMIT:
        return None
    if any(value["metrics"].get("obligation_limit", limit) != limit for value in grouped):
        return None
    prior = [value for value in values if _shared_key(value["key"], value["metrics"], "logical_key") == logical]
    if len(grouped) >= limit:
        spent = f"{len(grouped)} of {limit} reserved attempts"
    elif len(prior) >= ATTEMPTS_LIMIT:
        spent = f"{len(prior)} of {ATTEMPTS_LIMIT} transport attempts for '{logical}'"
    elif run.shared_calls >= SHARED_LIMIT:
        spent = f"{run.shared_calls} of {SHARED_LIMIT} shared calls"
    else:
        return None
    return (
        f"Shared scientific obligation '{group}' is exhausted: {spent} spent, including failed or uncertain calls. "
        f"The latest attempt is {latest['status']}. No further provider call or retry is available for this obligation."
    )

def _view(
    db: Session,
    task: AnalysisTask,
    attempts: list[AnalysisTaskAttempt] | None = None,
    *,
    include_packets: bool = True,
) -> dict[str, Any]:
    if attempts is None:
        attempts = list(
            db.scalars(
                select(AnalysisTaskAttempt)
                .options(defer(AnalysisTaskAttempt.request))
                .where(AnalysisTaskAttempt.task_id == task.id)
                .order_by(AnalysisTaskAttempt.number)
            )
        )
    return {
        **{
            name: deepcopy(getattr(task, name))
            for name in (
                "key",
                "checkpoint",
                "kind",
                "reason",
                "status",
                "budget_kind",
                "column_ids",
                "obligation_keys",
                "scope",
                "dependency_receipt_ids",
                "input_hash",
            )
        },
        "id": str(task.id),
        "packet": deepcopy(task.packet) if include_packets else {},
        "allowance": {
            "attempts_limit": task.max_attempts,
            "column_limit": COLUMN_LIMIT,
            "shared_limit": SHARED_LIMIT,
        },
        "attempts": [deepcopy(item.receipt) for item in attempts if item.receipt],
        "attempt_details": [
            {
                "id": str(item.id),
                "receipt_id": item.receipt.get("id") if item.receipt else None,
                "status": item.status,
                "generation": item.generation,
                "request_hash": item.request_hash,
                "raw_output": item.raw_output,
                "metrics": deepcopy(item.metrics),
            }
            for item in attempts
        ],
    }

def view(task_id: UUID) -> dict[str, Any]:
    with Session(engine()) as db:
        task = db.get(AnalysisTask, task_id)
        if task is None:
            raise ValueError("Task not found.")
        return _view(db, task)

def list_tasks(
    run_id: UUID,
    *,
    checkpoint: str | None = None,
    include_packets: bool = True,
    session: Session | None = None,
) -> list[dict[str, Any]]:
    with nullcontext(session) if session is not None else Session(engine()) as db:
        query = (
            select(AnalysisTask, AnalysisTaskAttempt)
            .options(defer(AnalysisTaskAttempt.request))
            .outerjoin(AnalysisTaskAttempt, AnalysisTaskAttempt.task_id == AnalysisTask.id)
            .where(AnalysisTask.run_id == run_id)
            .order_by(AnalysisTask.key, AnalysisTaskAttempt.number)
        )
        if checkpoint is not None:
            query = query.where(AnalysisTask.checkpoint == checkpoint)
        if not include_packets:
            query = query.options(defer(AnalysisTask.packet))
        records: dict[UUID, AnalysisTask] = {}
        attempts: dict[UUID, list[AnalysisTaskAttempt]] = {}
        # One database snapshot prevents accepted status being observed without
        # its atomically published receipt while another worker is finishing.
        for task, attempt in db.execute(query):
            records[task.id] = task
            if attempt is not None:
                attempts.setdefault(task.id, []).append(attempt)
        return [
            _view(db, task, attempts.get(task.id, []), include_packets=include_packets)
            for task in records.values()
        ]

def task_statuses(
    run_id: UUID, *, kind: str | None = None, status: str | None = None
) -> list[dict[str, Any]]:
    """Thin frontier inventory, without evidence, responses or provider requests."""
    with Session(engine()) as db:
        query = (
            select(
                AnalysisTask.id, AnalysisTask.key, AnalysisTask.kind, AnalysisTask.status, AnalysisTask.scope
            )
            .where(AnalysisTask.run_id == run_id)
            .order_by(AnalysisTask.key)
        )
        if kind is not None:
            query = query.where(AnalysisTask.kind == kind)
        if status is not None:
            query = query.where(AnalysisTask.status == status)
        return [{**dict(row), "id": str(row["id"])} for row in db.execute(query).mappings()]

def coverage(run_id: UUID, obligation_keys: list[str]) -> dict[str, Any]:
    records = list_tasks(run_id, include_packets=False)
    accepted = {key for task in records if task["status"] == "accepted" for key in task["obligation_keys"]}
    expected = set(obligation_keys)
    return {
        "complete": expected <= accepted,
        "accepted": sorted(expected & accepted),
        "pending": sorted(expected - accepted),
    }

def budgets(run_id: UUID) -> dict[str, Any]:
    with Session(engine()) as db:
        run = db.get(AnalysisRun, run_id)
        if run is None:
            raise ValueError("Analysis not found.")
        return {
            **_counts(db, run),
            "column_calls_limit": COLUMN_LIMIT,
            "shared_calls_limit": SHARED_LIMIT,
            "obligation_attempts_limit": ATTEMPTS_LIMIT,
        }

def input_for_receipt(db: Session, run_id: UUID, receipt_id: str) -> dict[str, Any] | None:
    attempt = db.scalar(
        select(AnalysisTaskAttempt).where(
            AnalysisTaskAttempt.run_id == run_id,
            AnalysisTaskAttempt.task_id.is_not(None),
            AnalysisTaskAttempt.receipt["id"].astext == receipt_id,
        )
    )
    if attempt is None:
        return None
    if attempt.request is not None:
        return deepcopy(attempt.request)
    task = db.get(AnalysisTask, attempt.task_id)
    return deepcopy(task.packet) if task else None

_SHARED_FIELDS = ("id", "key", "number", "status", "input_hash", "request_hash", "started_at", "completed_at", "metrics")

def _shared_view(value: dict[str, Any]) -> dict[str, Any]:
    return {
        **value,
        "logical_key": value["metrics"].get("logical_key", value["key"].removeprefix("shared:")),
        "obligation_key": value["metrics"].get("obligation_key"),
    }

def shared_attempts(db: Session, run_id: UUID) -> list[dict[str, Any]]:
    """Thin same-session view; full requests, proposals and raw bytes are on demand."""
    query = select(
        *(getattr(AnalysisTaskAttempt, name) for name in _SHARED_FIELDS),
        func.coalesce(func.jsonb_typeof(AnalysisTaskAttempt.request) == "object", False).label("has_request"),
        AnalysisTaskAttempt.raw_output.is_not(None).label("has_raw_output"),
        func.coalesce(func.jsonb_typeof(AnalysisTaskAttempt.receipt) == "object", False).label("has_proposal"),
    ).where(AnalysisTaskAttempt.run_id == run_id, AnalysisTaskAttempt.task_id.is_(None)).order_by(AnalysisTaskAttempt.number)
    return [_shared_view(dict(row)) for row in db.execute(query).mappings()]

def shared_attempt(db: Session, run_id: UUID, attempt_id: UUID) -> dict[str, Any] | None:
    attempt = db.scalar(select(AnalysisTaskAttempt).where(
        AnalysisTaskAttempt.id == attempt_id, AnalysisTaskAttempt.run_id == run_id,
        AnalysisTaskAttempt.task_id.is_(None),
    ))
    if attempt is None:
        return None
    return _shared_view({
        **{name: getattr(attempt, name) for name in _SHARED_FIELDS},
        "has_request": attempt.request is not None,
        "has_raw_output": attempt.raw_output is not None,
        "has_proposal": attempt.receipt is not None,
        "request": deepcopy(attempt.request),
        "proposal": deepcopy(attempt.receipt),
        "raw_output": attempt.raw_output,
    })

