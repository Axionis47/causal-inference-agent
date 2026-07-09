"""Test-only historical reservation fixtures for saved-contract regressions.

All mutations lock the run before its tasks. Workers use their own short sessions;
provider calls happen between claim and finish, outside database transactions.
"""

from collections.abc import Iterable
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from uuid import UUID, uuid4, uuid5

from sqlalchemy import func, select, text
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session, defer, load_only

from backend.context.service import fingerprint
from backend.db import engine
from backend.models import AnalysisRun, AnalysisTask, AnalysisTaskAttempt, now

ATTEMPTS_LIMIT = 2
COLUMN_LIMIT = 16
SHARED_LIMIT = 27
TERMINAL = {"accepted", "blocked", "exhausted", "expanded"}


class TaskConflict(ValueError):
    pass


class AllowanceExhausted(ValueError):
    pass


@dataclass(frozen=True)
class SharedOwner:
    run_id: UUID
    connection: Connection
    invocation_id: UUID

    def verify(self, run_id: UUID) -> None:
        unsigned = int.from_bytes(run_id.bytes[:8], "big")
        held = not self.connection.closed and self.connection.info.get("analysis_shared_owner") is self and self.connection.scalar(
            text("SELECT EXISTS (SELECT 1 FROM pg_locks WHERE locktype='advisory' AND pid=pg_backend_pid() AND granted AND mode='ExclusiveLock' AND classid::bigint=:high AND objid::bigint=:low AND objsubid=1)"),
            {"high": unsigned >> 32, "low": unsigned & 0xFFFFFFFF},
        )
        if self.run_id != run_id or not held:
            raise TaskConflict("Shared recovery requires the current run's advisory-lock owner.")


def shared_owner(run_id: UUID, connection: Connection) -> SharedOwner:
    """Issue one capability per owned invocation, including pooled-connection reuse."""
    owner = connection.info.get("analysis_shared_owner")
    if not isinstance(owner, SharedOwner) or owner.connection is not connection or owner.run_id != run_id:
        owner = SharedOwner(run_id, connection, uuid4())
    connection.info["analysis_shared_owner"] = owner
    owner.verify(run_id)
    return owner


@dataclass(frozen=True)
class Claim:
    task_id: UUID
    run_id: UUID
    attempt_id: UUID
    number: int
    generation: int
    input_hash: str
    packet: dict[str, Any]


def _run(db: Session, identity: UUID) -> AnalysisRun:
    run = db.scalar(
        select(AnalysisRun)
        .options(
            load_only(
                AnalysisRun.id,
                AnalysisRun.calls,
                AnalysisRun.shared_calls,
                AnalysisRun.status,
            )
        )
        .where(AnalysisRun.id == identity)
        .with_for_update()
    )
    if run is None:
        raise ValueError("Analysis not found.")
    return run


def _event(
    db: Session, run: AnalysisRun, key: str, task: AnalysisTask, kind: str, detail: dict[str, Any]
) -> None:
    from backend.analysis.service import emit

    emit(db, run, key, task.checkpoint, kind, task.reason, {"task_id": str(task.id), **detail})


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


def ensure(
    run_id: UUID,
    *,
    key: str,
    checkpoint: str,
    kind: str,
    column_ids: list[str],
    scope: dict[str, Any],
    reason: str,
    dependency_receipt_ids: list[str],
    packet: dict[str, Any],
    budget_kind: str = "column",
    max_attempts: int = ATTEMPTS_LIMIT,
    obligation_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Create once. Reuse requires the exact same task definition and evidence."""
    if not key or len(key) > 200 or max_attempts not in {1, 2}:
        raise ValueError("A task requires a bounded key and one or two attempts.")
    if budget_kind not in {"column", "relationship", "shared"}:
        raise ValueError("Unknown task allowance.")
    columns = sorted(set(column_ids))
    if budget_kind == "column" and not columns:
        raise ValueError("Column work must name the columns it owns.")
    obligations = sorted(
        set(
            obligation_keys
            or [f"{checkpoint}:{kind}:{cid}:{fingerprint(scope)}" for cid in columns]
            or [f"{checkpoint}:{kind}:{fingerprint(scope)}"]
        )
    )
    definition: dict[str, Any] = {
        "checkpoint": checkpoint,
        "kind": kind,
        "budget_kind": budget_kind,
        "column_ids": columns,
        "obligation_keys": obligations,
        "scope": deepcopy(scope),
        "reason": reason,
        "dependency_receipt_ids": sorted(set(dependency_receipt_ids)),
        "input_hash": fingerprint(packet),
        "packet": deepcopy(packet),
        "max_attempts": max_attempts,
    }
    with Session(engine()) as db:
        run = _run(db, run_id)
        task = db.scalar(select(AnalysisTask).where(AnalysisTask.run_id == run_id, AnalysisTask.key == key))
        if task is not None:
            if any(getattr(task, name) != value for name, value in definition.items()):
                raise TaskConflict("A resumed task differs from its immutable definition or evidence packet.")
        else:
            task = AnalysisTask(id=uuid5(run_id, "task:" + key), run_id=run_id, key=key, **definition)
            db.add(task)
            db.flush()
            _event(db, run, f"task_created:{task.id}", task, "task_created", {"input_hash": task.input_hash})
            db.commit()
        return _view(db, task)


def _accepted_dependencies(db: Session, run: AnalysisRun, required: set[str]) -> set[str]:
    if not required:
        return set()
    accepted = set(
        db.scalars(
            text(
                "SELECT entry.value->>'id' FROM analysis_runs run CROSS JOIN LATERAL jsonb_each(run.receipts) entry WHERE run.id=:identity AND entry.value->>'status'='accepted' AND entry.value->>'complete'='true' AND entry.value->>'id' IS NOT NULL"
            ),
            {"identity": run.id},
        )
    )
    if required <= accepted:
        return accepted
    accepted.update(
        db.scalars(
            select(AnalysisTaskAttempt.receipt["id"].astext)
            .join(AnalysisTask, AnalysisTask.id == AnalysisTaskAttempt.task_id)
            .where(
                AnalysisTaskAttempt.run_id == run.id,
                AnalysisTaskAttempt.task_id.is_not(None),
                AnalysisTaskAttempt.status == "accepted",
                AnalysisTask.run_id == run.id,
                AnalysisTask.status == "accepted",
                AnalysisTaskAttempt.receipt["status"].astext == "accepted",
                AnalysisTaskAttempt.receipt["id"].astext.is_not(None),
                AnalysisTaskAttempt.receipt["id"].astext.in_(required - accepted),
            )
        )
    )
    return accepted


def require_dependencies(db: Session, run: AnalysisRun, required_ids: Iterable[str]) -> None:
    """Require accepted science owned by this run, not a merely existing receipt.

    Coarse checkpoints must be complete. Worker receipts have complete=False
    because they do not complete the coarse checkpoint; an accepted task and
    its accepted immutable attempt certify that worker's narrower obligation.
    """
    required = set(required_ids)
    if any(not isinstance(identity, str) or not identity for identity in required):
        raise TaskConflict("Scientific prerequisites must name nonempty receipt identities.")
    missing = required - _accepted_dependencies(db, run, required)
    if missing:
        raise TaskConflict(
            "Scientific work requires accepted prerequisites from this run: " + ", ".join(sorted(missing))
        )


def claim(
    task_id: UUID,
    *,
    lease_seconds: int = 300,
    request: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> Claim | None:
    """Reserve a paid attempt before dispatch. An expired attempt stays charged."""
    if lease_seconds <= 0:
        raise ValueError("Task leases must have a positive duration.")
    with Session(engine()) as db:
        original = db.get(AnalysisTask, task_id)
        if original is None:
            raise ValueError("Task not found.")
        run = _run(db, original.run_id)
        db.refresh(original, with_for_update=True)
        task = original
        if task.status in TERMINAL:
            return None
        if task.status == "running" and task.lease_until and task.lease_until > now():
            return None
        required = set(task.dependency_receipt_ids)
        if not required <= _accepted_dependencies(db, run, required):
            return None
        previous = list(
            db.scalars(
                select(AnalysisTaskAttempt)
                .options(
                    load_only(
                        AnalysisTaskAttempt.id,
                        AnalysisTaskAttempt.number,
                        AnalysisTaskAttempt.status,
                        AnalysisTaskAttempt.generation,
                        AnalysisTaskAttempt.completed_at,
                    )
                )
                .where(AnalysisTaskAttempt.task_id == task.id)
            )
        )
        for attempt in previous:
            if attempt.status == "reserved":
                attempt.status, attempt.completed_at = "uncertain", now()
                _event(
                    db,
                    run,
                    f"task_uncertain:{attempt.id}",
                    task,
                    "task_interrupted",
                    {"attempt_id": str(attempt.id)},
                )
        counts = _counts(db, run)
        exhausted = len(previous) >= task.max_attempts or any(
            counts["obligations"].get(key, 0) >= ATTEMPTS_LIMIT for key in task.obligation_keys
        )
        exhausted = exhausted or (
            task.budget_kind == "column"
            and any(counts["columns"].get(cid, 0) >= COLUMN_LIMIT for cid in task.column_ids)
        )
        exhausted = exhausted or (task.budget_kind == "shared" and run.shared_calls >= SHARED_LIMIT)
        if exhausted:
            task.status, task.lease_until, task.updated_at = "exhausted", None, now()
            _event(db, run, f"task_exhausted:{task.id}", task, "task_exhausted", {"allowance": counts})
            db.commit()
            return None
        number = len(previous) + 1
        task.generation += 1
        task.status, task.lease_until, task.updated_at = (
            "running",
            now() + timedelta(seconds=lease_seconds),
            now(),
        )
        identity = uuid5(task.id, f"attempt:{number}")
        db.add(
            AnalysisTaskAttempt(
                id=identity,
                run_id=run.id,
                task_id=task.id,
                key=f"task:{task.id}:{number}",
                number=number,
                generation=task.generation,
                input_hash=task.input_hash,
                request=deepcopy(request),
                request_hash=fingerprint(request) if request is not None else None,
                metrics=deepcopy(metrics or {}),
            )
        )
        run.calls += 1
        if task.budget_kind == "shared":
            run.shared_calls += 1
        _event(
            db,
            run,
            f"task_reserved:{identity}",
            task,
            "model_started",
            {
                "attempt_id": str(identity),
                "input_hash": fingerprint(request) if request is not None else task.input_hash,
                "task_input_hash": task.input_hash,
                "call": run.calls,
                "number": number,
            },
        )
        value = Claim(
            task.id, run.id, identity, number, task.generation, task.input_hash, deepcopy(task.packet)
        )
        db.commit()
        return value


def finish(
    work: Claim,
    output: dict[str, Any],
    checks: list[dict[str, Any]],
    *,
    status: str = "accepted",
    raw_output: str | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish one immutable attempt receipt and its event in the same transaction."""
    if status not in {"accepted", "rejected", "failed", "blocked", "exhausted", "expanded"}:
        raise ValueError("Invalid task result state.")
    if status == "accepted" and (not checks or not all(check.get("passed") is True for check in checks)):
        raise ValueError("Accepted task results require nonempty passing checks.")
    with Session(engine()) as db:
        run = _run(db, work.run_id)
        task = db.get(AnalysisTask, work.task_id, with_for_update=True)
        attempt = db.get(AnalysisTaskAttempt, work.attempt_id, with_for_update=True)
        if (
            task is None or attempt is None or attempt.task_id != task.id
            or task.run_id != work.run_id or attempt.run_id != work.run_id
            or attempt.number != work.number
        ):
            raise TaskConflict("Task claim does not exist.")
        if attempt.receipt is not None:
            if (
                attempt.status != status
                or attempt.receipt.get("output") != output
                or attempt.receipt.get("checks") != checks
                or attempt.raw_output != raw_output
            ):
                raise TaskConflict("An immutable task receipt cannot be replaced.")
            return deepcopy(attempt.receipt)
        if (
            task.generation != work.generation
            or task.status != "running"
            or attempt.status != "reserved"
            or attempt.generation != work.generation
        ):
            raise TaskConflict("This task claim was superseded. Its attempt remains charged.")
        if task.input_hash != work.input_hash:
            raise TaskConflict("The task input differs from its reserved input.")
        from backend.analysis import domain

        evidence = task.packet.get("evidence_records", [])
        offered = (
            [str(item["key"]) for item in evidence if isinstance(item, dict) and "key" in item]
            if isinstance(evidence, list)
            else []
        )
        counts = _counts(db, run)
        receipt: dict[str, Any] = {
            "id": fingerprint(
                {
                    "task": str(task.id),
                    "attempt": work.number,
                    "output": output,
                    "checks": checks,
                    "status": status,
                }
            ),
            "checkpoint": task.checkpoint,
            "attempt": work.number - 1,
            "status": "rejected" if status == "expanded" else status,
            "complete": False,
            "input_hash": attempt.request_hash or task.input_hash,
            "output_hash": fingerprint(output),
            "input_refs": ["question", *offered, *task.dependency_receipt_ids],
            "prerequisite_ids": task.dependency_receipt_ids,
            "offered": offered,
            "retrieved": offered,
            "cited": sorted(domain.references(output)),
            "checks": deepcopy(checks),
            "output": deepcopy(output),
            "assumptions": output.get("assumptions", []),
            "started_at": attempt.started_at.isoformat(),
            "completed_at": now().isoformat(),
            "budgets": {
                "model_calls": run.calls,
                "shared_calls": run.shared_calls,
                "shared_calls_limit": SHARED_LIMIT,
                "attempts": work.number,
                "attempts_limit": task.max_attempts,
                "column_calls_limit": COLUMN_LIMIT,
            },
            "evidence_scope": task.packet.get("evidence_scope"),
            "trace_id": (metrics or {}).get("trace_id"),
            "trace_url": (metrics or {}).get("trace_url"),
        }
        attempt.receipt, attempt.raw_output = receipt, raw_output
        attempt.metrics = {**attempt.metrics, **deepcopy(metrics or {})}
        attempt.status, attempt.completed_at = status, now()
        task.status, task.lease_until, task.updated_at = status, None, now()
        from backend.analysis.authority_view import summary

        _event(
            db,
            run,
            f"task_finished:{attempt.id}",
            task,
            "task_completed",
            {"receipt_id": receipt["id"], "status": status, "allowance": counts,
             **({"claim_authority": summary(output)} if run.inputs.get("frozen_context") else {})},
        )
        db.commit()
        return deepcopy(receipt)


def reject(
    work: Claim, output: dict[str, Any], checks: list[dict[str, Any]], **kwargs: Any
) -> dict[str, Any]:
    return finish(work, output, checks, status="rejected", **kwargs)


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


def reserve_shared(
    db: Session, run: AnalysisRun, key: str, input_hash: str, metadata: dict[str, Any] | None = None,
    *, obligation_key: str | None = None, obligation_limit: int = 2,
    recover: bool = False, owner: SharedOwner | None = None,
) -> str:
    """Reserve coordinator work in its existing transaction, before a provider call.

    A replacement has its own charged key. Only a new advisory-lock owner may
    settle an interrupted reservation; a parallel call cannot invalidate it.
    """
    if not 1 <= obligation_limit <= SHARED_LIMIT:
        raise ValueError("Shared work requires a bounded obligation allowance.")
    if owner is not None:
        owner.verify(run.id)
    if recover and owner is None:
        raise TaskConflict("Shared recovery requires the current run's advisory-lock owner.")
    db.flush()
    locked = _run(db, run.id)
    db.refresh(locked, attribute_names=["calls", "shared_calls"])
    attempts = list(db.scalars(
        select(AnalysisTaskAttempt).options(defer(AnalysisTaskAttempt.request)).where(
            AnalysisTaskAttempt.run_id == run.id, AnalysisTaskAttempt.task_id.is_(None)
        ).order_by(AnalysisTaskAttempt.number)
    ))
    prior = [item for item in attempts if _shared_key(item.key, item.metrics, "logical_key") == key]
    group = obligation_key or key
    grouped = [item for item in attempts if _shared_key(item.key, item.metrics, "obligation_key") == group]
    if any(item.metrics.get("obligation_limit", obligation_limit) != obligation_limit for item in grouped):
        raise TaskConflict("A shared obligation's allowance cannot change during recovery.")
    spent = len(grouped)
    if prior and any(item.input_hash != input_hash for item in prior):
        raise TaskConflict("A reserved shared request has different inputs.")
    if any(item.metrics.get("obligation_key", group) != group for item in prior):
        raise TaskConflict("A reserved shared request cannot change its scientific obligation.")
    if prior and (not recover or any(item.status not in {"reserved", "failed", "uncertain"} for item in prior)):
        raise TaskConflict("This shared request was already reserved. Resume its saved result or record an explicit new attempt.")
    for previous in prior:
        if previous.status != "reserved":
            continue
        if owner is None or previous.metrics.get("owner_invocation") == str(owner.invocation_id):
            raise TaskConflict("A live shared reservation cannot be invalidated by the same invocation.")
        previous.status, previous.completed_at = "uncertain", now()
        from backend.analysis.service import emit

        emit(db, locked, f"shared_uncertain:{previous.id}", locked.stage, "model_interrupted",
             "An interrupted provider attempt remains charged; its response is unknown.",
             {"attempt_id": str(previous.id), "logical_key": key})
    if len(prior) >= ATTEMPTS_LIMIT or spent >= obligation_limit:
        raise AllowanceExhausted("This shared scientific obligation has spent its allowance, including interrupted calls.")
    if locked.shared_calls >= SHARED_LIMIT:
        raise AllowanceExhausted(
            "The shared checkpoint allowance is exhausted; column work cannot spend its reserve."
        )
    reserved_key = key if not prior else f"{key}:transport:{len(prior)}"
    locked.calls += 1
    locked.shared_calls += 1
    details = deepcopy(metadata or {})
    request = details.pop("request", None)
    details.update(
        logical_key=key, obligation_key=group, obligation_limit=obligation_limit,
        owner_invocation=str(owner.invocation_id) if owner is not None else None,
    )
    db.add(
        AnalysisTaskAttempt(
            id=uuid5(run.id, "shared:" + reserved_key),
            run_id=run.id,
            task_id=None,
            key="shared:" + reserved_key,
            number=locked.shared_calls,
            generation=0,
            input_hash=input_hash,
            request=request,
            request_hash=fingerprint(request) if request is not None else None,
            metrics=details,
        )
    )
    db.flush()
    return reserved_key


def settle_shared(
    db: Session,
    run_id: UUID,
    key: str,
    *,
    status: str,
    receipt: dict[str, Any] | None = None,
    raw_output: str | None = None,
    metrics: dict[str, Any] | None = None,
) -> None:
    _run(db, run_id)
    attempt = db.scalar(
        select(AnalysisTaskAttempt)
        .where(AnalysisTaskAttempt.run_id == run_id, AnalysisTaskAttempt.key == "shared:" + key)
        .with_for_update()
    )
    if attempt is None:
        raise TaskConflict("Shared work must be reserved before publishing a result.")
    if attempt.status != "reserved":
        if attempt.status != status or attempt.receipt != receipt:
            raise TaskConflict("A settled shared attempt cannot be replaced.")
        return
    attempt.status, attempt.receipt, attempt.raw_output = status, deepcopy(receipt), raw_output
    attempt.metrics, attempt.completed_at = {**attempt.metrics, **deepcopy(metrics or {})}, now()


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
