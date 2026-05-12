"""Durable preparation, approval resume and recovery without re-fitting."""

import logging
from typing import Any
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command
from langsmith import trace, tracing_context
from sqlalchemy import text, update
from sqlalchemy.orm import Session

from backend.analysis import packets, prompts
from backend.analysis.agent_store import AgentLimit, HandoffConflict, Store
from backend.analysis.agents import Runtime, build
from backend.analysis.records import approval_hash, requested_revision
from backend.analysis.service import emit
from backend.config import settings
from backend.db import engine
from backend.jobs import dsn
from backend.models import AnalysisRun, now


def process(identity: UUID) -> None:
    key = int.from_bytes(identity.bytes[:8], "big", signed=True)
    with engine().connect().execution_options(isolation_level="AUTOCOMMIT") as guard:
        guard.execute(text("SELECT pg_advisory_lock(:key)"), {"key": key})
        try:
            with Session(engine()) as db:
                run = db.get(AnalysisRun, identity)
                if (
                    run is None
                    or run.inputs.get("contract") != prompts.VERSION
                    or run.status in {"succeeded", "blocked", "failed"}
                    or (run.status == "awaiting_approval" and not approval_hash(run))
                ):
                    return
                run.status, run.error, run.updated_at = "running", None, now()
                db.commit()
                try:
                    with tracing_context(project_name=settings().langsmith_project):
                        with trace(
                            "causal-analysis",
                            inputs={"run_id": str(identity), "question": run.question},
                            metadata={
                                "analysis_trace_id": str(run.trace_id),
                                "snapshot_hash": run.snapshot_hash,
                            },
                        ) as span:
                            with PostgresSaver.from_conn_string(dsn()) as saver:
                                Store(db, identity).recover()
                                graph = build(Runtime(db, run), saver)
                                config: RunnableConfig = {
                                    "configurable": {"thread_id": f"analysis:{identity}"},
                                    "recursion_limit": 128,
                                }
                                previous = graph.get_state(config)
                                command: Any = (
                                    Command(resume={"revise_program": requested_revision(run)})
                                    if previous.interrupts and requested_revision(run)
                                    else Command(resume={"program_hash": approval_hash(run)})
                                    if previous.interrupts and approval_hash(run)
                                    else None
                                    if previous.next
                                    else {"run_id": str(identity)}
                                )
                                graph.invoke(command, config)
                                # Recovery can reach an interrupt after approval was recorded but before
                                # its checkpoint existed. Resume the same saved snapshot, never restart.
                                if graph.get_state(config).interrupts:
                                    if requested_revision(run):
                                        graph.invoke(Command(resume={"revise_program": requested_revision(run)}), config)
                                    elif approval_hash(run):
                                        graph.invoke(Command(resume={"program_hash": approval_hash(run)}), config)
                            span.end(
                                outputs={
                                    "status": run.status,
                                    "snapshot_hash": run.snapshot_hash,
                                    "calls": run.calls,
                                }
                            )
                except Exception as error:
                    db.rollback()
                    db.refresh(run)
                    if run.result is None and run.status not in {"succeeded", "blocked"}:
                        terminal = isinstance(error, (AgentLimit, HandoffConflict, packets.PacketOverflow))
                        run.status, run.updated_at = ("blocked" if terminal else "failed"), now()
                        run.error = (
                            str(error)[:3500]
                            if isinstance(error, (ValueError, RuntimeError))
                            else f"Analysis interrupted ({type(error).__name__}). Retry to resume the saved trajectory."
                        )
                        emit(db, run, f"failure:{run.calls}:{run.stage}", run.stage, "failure", run.error)
                        db.commit()
                    logging.getLogger(__name__).exception("Analysis %s failed", identity)
        finally:
            guard.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": key})




def mark_failed(ids: list[int]) -> None:
    with Session(engine()) as db:
        db.execute(
            update(AnalysisRun)
            .where(AnalysisRun.job_id.in_(ids), AnalysisRun.status.in_(["queued", "running"]))
            .values(
                status="failed", error="Worker delivery failed. Retry to resume saved work.", updated_at=now()
            )
        )
        db.commit()
