"""Analysis ownership, exact approval and durable event reads."""

from typing import Any, cast
from uuid import UUID, uuid4, uuid5

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.analysis import prompts
from backend.analysis.contracts import (
    AnalysisView,
    ApproveAnalysis,
    EventView,
    PrepareAnalysis,
    RejectAnalysis,
)
from backend.analysis.readiness import AnalysisReadiness, assess
from backend.config import settings
from backend.handoffs import ConversationScope
from backend.models import (
    AnalysisEvent,
    AnalysisRun,
    Dataset,
    DatasetVersion,
    now,
)


def owned(db: Session, owner: UUID, identity: UUID, *, lock: bool = False) -> AnalysisRun:
    query = (
        select(AnalysisRun)
        .join(DatasetVersion)
        .join(Dataset)
        .where(AnalysisRun.id == identity, Dataset.owner_id == owner)
    )
    if lock:
        query = query.with_for_update(of=AnalysisRun)
    row = db.scalar(query)
    if row is None:
        raise HTTPException(404, "Analysis not found.")
    return row


def events(db: Session, identity: UUID, after: int = 0) -> list[EventView]:
    return [
        EventView.model_validate(e)
        for e in db.scalars(
            select(AnalysisEvent)
            .where(AnalysisEvent.run_id == identity, AnalysisEvent.sequence > after)
            .order_by(AnalysisEvent.sequence)
        )
    ]


def emit(
    db: Session,
    run: AnalysisRun,
    key: str,
    stage: str,
    kind: str,
    summary: str,
    detail: dict[str, Any] | None = None,
) -> None:
    # Task workers have independent sessions. Serialize sequence allocation and
    # the receipt/event commit on the run, including duplicate event delivery.
    db.execute(select(AnalysisRun.id).where(AnalysisRun.id == run.id).with_for_update())
    if (
        db.scalar(
            select(AnalysisEvent.sequence).where(AnalysisEvent.run_id == run.id, AnalysisEvent.key == key)
        )
        is not None
    ):
        return
    sequence = (
        db.scalar(select(func.max(AnalysisEvent.sequence)).where(AnalysisEvent.run_id == run.id)) or 0
    ) + 1
    db.add(
        AnalysisEvent(
            run_id=run.id,
            sequence=sequence,
            key=key,
            stage=stage,
            kind=kind,
            summary=summary[:4000],
            detail=detail or {},
        )
    )
    db.flush()


def view(db: Session, run: AnalysisRun) -> AnalysisView:
    from backend.analysis import agent_views, tasks
    from backend.analysis.agent_store import CONTRACT, Store
    from backend.analysis.authority_view import inspect
    from backend.analysis.checkpoints import budgets, views
    from backend.analysis.records import programs
    from backend.context.contracts import empty_content
    from backend.context.service import fingerprint, revision_record

    current = run.inputs.get("contract") in prompts.READABLE_VERSIONS
    bounded = run.inputs.get("contract") == CONTRACT
    agents = agent_views.progress(db, run) if bounded else []
    design = Store(db, run.id).accepted("design") if bounded else None
    retryable = bounded and run.status == "failed" and run.result is None
    version = db.get(DatasetVersion, run.version_id)
    assert version
    context = (
        revision_record(db, run.version_id, run.context_revision_id).content
        if run.context_revision_id
        else empty_content(run.version_id, version.columns).model_dump(mode="json")
    )
    saved_programs = []
    saved_tasks = tasks.list_tasks(run.id, session=db) if run.inputs.get("contract") in prompts.DYNAMIC_VERSIONS else []
    checkpoint_views = views(run) if current and not bounded else []
    shared_attempts = tasks.shared_attempts(db, run.id)
    exhaustion = (
        (Store(db, run.id).exhaustion(run) if bounded else tasks.shared_exhaustion(db, run, attempts=shared_attempts))
        if run.status == "failed" and run.result is None
        else None
    )
    retryable = retryable and exhaustion is None
    for program in programs(run) if current else []:
        index = program["revision"]
        approval = run.receipts.get(f"approval:{index}")
        rejection = run.receipts.get(f"review_rejection:{index}")
        saved_programs.append(
            {
                **program,
                "approved_at": approval.get("approved_at") if isinstance(approval, dict) else None,
                "rejected_reason": rejection.get("reason") if isinstance(rejection, dict) else None,
                "execution": run.receipts.get(f"execution:{index}"),
            }
        )
    return AnalysisView.model_validate(
        {
            **{
                name: getattr(run, name)
                for name in (
                    "id",
                    "conversation_id",
                    "version_id",
                    "context_revision_id",
                    "question",
                    "status",
                    "stage",
                    "calls",
                    "snapshot_hash",
                    "result",
                    "error",
                    "trace_id",
                    "created_at",
                )
            },
            "frame": design["records"]["snapshot"]["frame"] if design else run.receipts.get("frame") if current else None,
            "snapshot": run.snapshot if current else None,
            "programs": saved_programs,
            "budgets": budgets(run),
            "checkpoints": checkpoint_views,
            "tasks": saved_tasks,
            "agents": agents,
            "outcome": next((cast(dict[str, Any], task.draft["blocker"]).get("outcome") for task in agent_views.sessions(db, run) if isinstance(task.draft.get("blocker"), dict)), None) if bounded else None,
            "recovery_actions": ["retry"] if retryable else (["approve"] + (["revise_program"] if saved_programs and saved_programs[-1]["revision"] < 2 and agents[2]["model_calls"] < agents[2]["model_limit"] else [])) if run.status == "awaiting_approval" and bounded else [],
            "authority": inspect([
                *[attempt.model_dump(mode="json") for checkpoint in checkpoint_views for attempt in checkpoint.attempts],
                *[attempt for task in saved_tasks for attempt in task["attempts"]],
            ]) if run.inputs.get("frozen_context") else None,
            "shared_attempts": shared_attempts,
            "error": exhaustion or run.error,
            "contract": run.inputs.get("contract"),
            "frozen_context": run.inputs.get("frozen_context"),
            "read_only": run.inputs.get("contract") != prompts.VERSION,
            "context": {
                "revision_id": str(run.context_revision_id) if run.context_revision_id else None,
                "hash": fingerprint(context),
                "content": context,
            },
            "diagnostics": [
                value for key, value in sorted(run.receipts.items()) if key.startswith("diagnostic:")
            ]
            if current
            else [],
            "legacy_record": None
            if current
            else {
                "contract": run.inputs.get("contract"),
                "snapshot": run.snapshot,
                "receipts": run.receipts,
                "message": "Historical draft. Inspect its saved records; prepare a new analysis to use scientific checkpoints.",
            },
            "events": events(db, run.id),
            "retryable": retryable,
        }
    )


def readiness(db: Session, conversation: ConversationScope) -> AnalysisReadiness:
    from backend.context.contracts import ContextContent, empty_content
    from backend.context.service import revision_record

    version = db.get(DatasetVersion, conversation.version_id)
    assert version
    revision = conversation.context_revision_id
    content = (
        ContextContent.model_validate(revision_record(db, version.id, revision).content)
        if revision else empty_content(version.id, version.columns)
    )
    return assess(content, version.id, revision, source_sha256=version.sha256)


def prepare(db: Session, conversation: ConversationScope, owner: UUID, body: PrepareAnalysis) -> AnalysisView:
    from backend.analysis.agent_prompts import SYSTEMS
    from backend.analysis.agent_store import LIMITS
    from backend.analysis.delivery import enqueue

    identity = uuid5(conversation.id, f"analysis:{body.submission_id}")
    existing = db.get(AnalysisRun, identity)
    if existing:
        frozen = existing.inputs.get("frozen_context")
        if (
            (body.question is not None and existing.question != body.question)
            or existing.context_revision_id != body.expected_context_revision_id
            or not isinstance(frozen, dict)
            or frozen.get("readiness_fingerprint") != body.expected_readiness_fingerprint
        ):
            raise HTTPException(409, "This analysis submission has different inputs.")
        return view(db, existing)
    if body.expected_context_revision_id != conversation.context_revision_id:
        raise HTTPException(409, "The chat context changed. Refresh before preparing an analysis.")
    ready = readiness(db, conversation)
    if ready.fingerprint != body.expected_readiness_fingerprint:
        raise HTTPException(409, "The readiness receipt changed. Refresh the applied analysis brief before running.")
    if ready.status == "needs_context":
        raise HTTPException(409, "Apply the missing essential context and analysis brief before running analysis.")
    if body.question is not None and body.question != ready.question:
        raise HTTPException(409, "The question differs from the applied analysis brief. Apply a revised brief first.")
    if db.scalar(
        select(AnalysisRun.id).where(
            AnalysisRun.conversation_id == conversation.id,
            AnalysisRun.status.in_(["queued", "running", "awaiting_approval"]),
            AnalysisRun.inputs["contract"].astext == prompts.VERSION,
        )
    ):
        raise HTTPException(409, "This chat already has an active analysis. Inspect its saved plan first.")
    count = (
        db.scalar(
            select(func.count())
            .select_from(AnalysisRun)
            .where(AnalysisRun.conversation_id == conversation.id)
        )
        or 0
    )
    if count >= 20:
        raise HTTPException(
            409, "This chat has reached its 20-analysis limit. Its saved work remains available."
        )
    exposure = conversation.exploration_count
    version = db.get(DatasetVersion, conversation.version_id)
    assert version
    from backend.context.contracts import ContextContent
    from backend.context.service import revision_record

    assert body.expected_context_revision_id is not None
    pinned = ContextContent.model_validate(
        revision_record(db, version.id, body.expected_context_revision_id).content
    )
    assert pinned.analysis_brief is not None
    relationship_count = sum(item.status != "withdrawn" for item in pinned.relationships.values())
    run = AnalysisRun(
        id=identity,
        conversation_id=conversation.id,
        version_id=conversation.version_id,
        context_revision_id=body.expected_context_revision_id,
        author_id=owner,
        submission_id=body.submission_id,
        question=ready.question,
        trace_id=uuid4(),
        model_limit=sum(LIMITS.values()),
        inputs={
            "model": settings().gemini_model,
            "contract": prompts.VERSION,
            "agent_prompts": SYSTEMS,
            "agent_limits": LIMITS,
            "frozen_context": {
                "dataset_version_id": str(version.id),
                "source_sha256": version.sha256,
                "context_revision_id": str(body.expected_context_revision_id),
                "context_hash": ready.context_hash,
                "question": ready.question,
                "brief": pinned.analysis_brief.model_dump(mode="json"),
                "readiness_fingerprint": ready.fingerprint,
                "readiness": ready.model_dump(mode="json"),
                "policy_version": ready.policy_version,
            },
            "column_count": len(version.columns),
            "relationship_count": relationship_count,
            "prior_exposure": f"This chat already has {exposure} visualization artifacts and {count} analysis preparations. Their numerical results are withheld from planning. Prior user exposure cannot be undone.",
        },
    )
    db.add(run)
    db.flush()
    emit(
        db,
        run,
        "queued",
        "bind",
        "status",
        "Applied context, dataset and question frozen; analysis preparation queued.",
        {"readiness_fingerprint": ready.fingerprint, "context_hash": ready.context_hash,
         "policy_version": ready.policy_version},
    )
    run.job_id = enqueue(db, identity)
    db.commit()
    return view(db, run)


def approve(db: Session, run: AnalysisRun, body: ApproveAnalysis) -> AnalysisView:
    from backend.analysis.delivery import enqueue
    from backend.analysis.records import programs
    from backend.context.service import fingerprint

    if run.inputs.get("contract") != prompts.VERSION:
        raise HTTPException(409, "Prepare a new analysis for this historical draft.")
    candidates = programs(run)
    if (
        not candidates
        or run.snapshot is None
        or body.snapshot_hash != run.snapshot_hash
        or fingerprint(run.snapshot) != body.snapshot_hash
    ):
        raise HTTPException(409, "Approval must match the exact saved scientific snapshot.")
    program = candidates[-1]
    if (
        body.program_hash != program["hash"]
        or fingerprint({key: value for key, value in program.items() if key != "hash"}) != body.program_hash
    ):
        raise HTTPException(409, "Approval must match the current program revision and runtime.")
    key = f"approval:{program['revision']}"
    if key in run.receipts:
        return view(db, run)
    if run.status != "awaiting_approval":
        raise HTTPException(409, "This program is not waiting for approval.")
    run.receipts = {
        **run.receipts,
        key: {
            "snapshot_hash": body.snapshot_hash,
            "program_hash": body.program_hash,
            "approved_at": now().isoformat(),
            "author_id": str(run.author_id),
        },
    }
    emit(
        db,
        run,
        key,
        "freeze",
        "approval_recorded",
        "User approved the exact science, code and runtime.",
        {"revision": program["revision"], "program_hash": body.program_hash},
    )
    run.status, run.updated_at = "queued", now()
    run.job_id = enqueue(db, run.id)
    db.commit()
    return view(db, run)


def reject(db: Session, run: AnalysisRun, body: RejectAnalysis, *, request_revision: bool = False) -> AnalysisView:
    """Retain a human review failure without altering or executing frozen science."""
    from backend.analysis.records import programs
    from backend.context.service import fingerprint

    if run.inputs.get("contract") != prompts.VERSION:
        raise HTTPException(409, "This historical analysis is read-only.")
    reason = body.reason.strip()
    if not reason:
        raise HTTPException(422, "Explain why this program should not execute.")
    candidates = programs(run)
    if not candidates or run.snapshot is None or body.snapshot_hash != run.snapshot_hash or fingerprint(run.snapshot) != body.snapshot_hash:
        raise HTTPException(409, "Rejection must match the exact saved scientific snapshot.")
    program = candidates[-1]
    if body.program_hash != program["hash"] or fingerprint({key: value for key, value in program.items() if key != "hash"}) != body.program_hash:
        raise HTTPException(409, "Rejection must match the current program revision and runtime.")
    key = f"review_rejection:{program['revision']}"
    revision_key = f"revision_request:{program['revision']}"
    if key in run.receipts:
        saved = run.receipts[key]
        if isinstance(saved, dict) and saved.get("reason") == reason and saved.get("program_hash") == body.program_hash:
            if not request_revision or revision_key in run.receipts:
                return view(db, run)
        else:
            raise HTTPException(409, "This program already has a different saved rejection.")
    if run.status != "awaiting_approval" or f"approval:{program['revision']}" in run.receipts:
        raise HTTPException(409, "Only an unapproved program waiting for review can be rejected.")
    if request_revision:
        from backend.analysis.agent_store import LIMITS, Store
        if program["revision"] >= 2 or Store(db, run.id).counts("computation") >= LIMITS["computation"]:
            raise HTTPException(409, "No further program revision is available within this analysis's saved allowance.")
    record = {**body.model_dump(), "reason": reason, "revision": program["revision"],
              "author_id": str(run.author_id), "recorded_at": now().isoformat()}
    record = dict(cast(dict[str, Any], run.receipts.get(key, record)))
    run.receipts = {**run.receipts, key: record}
    rejection_summary = "Program rejected during human review: " + reason
    run.error = None if request_revision else rejection_summary
    run.status, run.updated_at = ("queued" if request_revision else "blocked"), now()
    emit(db, run, key, "computation", "program_rejected", rejection_summary, record)
    if request_revision:
        from backend.analysis.delivery import enqueue
        run.receipts = {**run.receipts, revision_key: record}
        run.status, run.error = "queued", None
        run.job_id = enqueue(db, run.id)
        emit(db, run, revision_key, "computation", "program_revision_requested",
             "Review feedback returned to computation. The frozen study and design remain accepted.", record)
    db.commit()
    return view(db, run)


def retry(db: Session, run: AnalysisRun) -> AnalysisView:
    from backend.analysis.delivery import enqueue

    current = view(db, run)
    if not current.retryable:
        raise HTTPException(
            409, current.error or "This analysis cannot be retried. The saved primary outcome and spent budgets remain intact."
        )
    run.status, run.error, run.updated_at = "queued", None, now()
    run.job_id = enqueue(db, run.id)
    db.commit()
    return view(db, run)
