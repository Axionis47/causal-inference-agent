"""Read-only projections of durable agent progress and optional execution inspection."""

from typing import Any, cast
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.analysis.agent_store import LIMITS, TITLES
from backend.models import AnalysisRun, AnalysisTask, AnalysisTaskAttempt


def sessions(db: Session, run: AnalysisRun) -> list[AnalysisTask]:
    return list(db.scalars(select(AnalysisTask).where(
        AnalysisTask.run_id == run.id, AnalysisTask.agent.is_not(None)
    ).order_by(AnalysisTask.parent_id.is_not(None), AnalysisTask.created_at, AnalysisTask.id)))


def progress(db: Session, run: AnalysisRun) -> list[dict[str, Any]]:
    saved = sessions(db, run)
    turns = list(db.scalars(select(AnalysisTaskAttempt).where(AnalysisTaskAttempt.run_id == run.id)))
    result = []
    for agent, limit in LIMITS.items():
        root = next((task for task in saved if task.agent == agent and task.parent_id is None), None)
        ids = {task.id for task in saved if task.agent == agent}
        attempts = [turn for turn in turns if turn.task_id in ids]
        metrics = [cast(dict[str, Any], turn.metrics) for turn in attempts]
        handoff = root.handoff if root else None
        blocker = cast(dict[str, Any], root.draft.get("blocker", {})) if root else {}
        status = root.status if root else "pending"
        if root and status == "running" and run.stage == agent and run.status in {"awaiting_approval", "failed", "blocked"}:
            status = run.status
        result.append({"id": root.id if root else None, "agent": agent, "title": TITLES[agent],
            "status": status,
            "summary": handoff["summary"] if handoff else str(blocker.get("reason", "")),
            "model_calls": len(attempts), "model_limit": limit, "submissions": root.submissions if root else 0,
            "handoff_id": handoff["id"] if handoff else None, "handoff_hash": root.handoff_hash if root else None,
            "dependencies": [item["producer"] for item in cast(list[dict[str, Any]], handoff["dependencies"])] if handoff else root.dependency_receipt_ids if root else [],
            "input_tokens": sum(int(item.get("input_tokens", 0) or 0) for item in metrics),
            "output_tokens": sum(int(item.get("output_tokens", 0) or 0) for item in metrics),
            "elapsed_seconds": sum((turn.completed_at - turn.started_at).total_seconds() for turn in attempts if turn.completed_at)})
    return result


def detail(db: Session, run: AnalysisRun, identity: UUID) -> dict[str, Any]:
    saved = sessions(db, run)
    root = next((task for task in saved if task.id == identity and task.parent_id is None), None)
    if root is None:
        raise HTTPException(404, "Analysis agent not found.")
    selected = [task for task in saved if task.agent == root.agent]
    attempts = list(db.scalars(select(AnalysisTaskAttempt).where(AnalysisTaskAttempt.task_id.in_([task.id for task in selected])).order_by(AnalysisTaskAttempt.started_at)))
    return {"id": root.id, "agent": root.agent, "packet": root.packet, "draft": root.draft,
        "handoff": root.handoff, "handoff_hash": root.handoff_hash,
        "sessions": [{"id": str(task.id), "parent_id": str(task.parent_id) if task.parent_id else None,
            "status": task.status, "handoff": task.handoff, "handoff_hash": task.handoff_hash,
            "attempts": [{"id": str(turn.id), "number": turn.number, "status": turn.status,
                "request_hash": turn.request_hash, "request": turn.request, "provider_content": turn.provider_content,
                "tool_receipts": turn.tool_receipts, "metrics": turn.metrics, "started_at": turn.started_at.isoformat(),
                "completed_at": turn.completed_at.isoformat() if turn.completed_at else None} for turn in attempts if turn.task_id == task.id]} for task in selected]}
