"""Durable chat delivery, bounded retry and PostgreSQL graph recovery."""

import logging
from uuid import UUID

from langgraph.checkpoint.postgres import PostgresSaver
from langsmith import trace, tracing_context
from sqlalchemy import text, update
from sqlalchemy.orm import Session

from backend.chat.graph import Runtime, build
from backend.config import settings
from backend.db import engine
from backend.handoffs import AnalysisDiscussionReader
from backend.jobs import dsn
from backend.model_access import InvalidModelOutput
from backend.models import ChatTurn, now


def process(identity: UUID, analysis_reader: AnalysisDiscussionReader) -> None:
    key = int.from_bytes(identity.bytes[:8], "big", signed=True)
    with engine().connect().execution_options(isolation_level="AUTOCOMMIT") as guard:
        if not guard.scalar(text("SELECT pg_try_advisory_lock(:key)"), {"key": key}):
            return
        try:
            with Session(engine()) as db:
                turn = db.get(ChatTurn, identity)
                if turn is None or turn.status in {"succeeded", "failed"}:
                    return
                turn.status, turn.error, turn.updated_at = "running", None, now()
                db.commit()
                try:
                    with tracing_context(project_name=settings().langsmith_project):
                        with trace(
                            "chat-turn",
                            run_id=turn.trace_id,
                            inputs={"turn_id": str(turn.id), "message": turn.text},
                            metadata={"conversation_id": str(turn.conversation_id)},
                        ) as span:
                            with PostgresSaver.from_conn_string(dsn()) as saver:
                                graph = build(Runtime(db, turn, analysis_reader), saver)
                                config = {"configurable": {"thread_id": f"chat:{identity}"}}
                                previous = graph.get_state(config)
                                graph.invoke(None if previous.next else {"turn_id": str(identity)}, config)
                            span.end(outputs={"reply": turn.reply, "calls": turn.calls})
                except Exception as error:
                    db.rollback()
                    turn = db.get(ChatTurn, identity)
                    assert turn is not None
                    if turn.status != "succeeded":
                        turn.status, turn.updated_at = "failed", now()
                        if isinstance(error, ValueError) and not isinstance(error, InvalidModelOutput):
                            turn.stage = "rejected"
                        turn.error = (
                            str(error)[:3500]
                            if isinstance(error, (ValueError, RuntimeError))
                            else f"The reply failed ({type(error).__name__}). Your message and earlier work are saved. Retry this turn."
                        )
                        db.commit()
                    logging.getLogger(__name__).exception("Chat turn %s failed", identity)
        finally:
            guard.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": key})




def mark_failed(ids: list[int]) -> None:
    if ids:
        with Session(engine()) as db:
            db.execute(
                update(ChatTurn)
                .where(ChatTurn.job_id.in_(ids), ChatTurn.status.in_(["queued", "running"]))
                .values(
                    status="failed",
                    error="The worker delivery failed. Retry this turn to continue its saved work.",
                    updated_at=now(),
                )
            )
            db.commit()
