"""Atomic job submission on the caller's database transaction."""

from uuid import UUID

from sqlalchemy.orm import Session

from backend.jobs import app


def enqueue(db: Session, identity: UUID) -> int:
    connection = db.connection().connection.driver_connection
    return int(app.configure_task(name="chat.respond", allow_unknown=True, queue="chat", connection=connection, lock=f"chat:{identity}").defer(turn_id=str(identity)))

