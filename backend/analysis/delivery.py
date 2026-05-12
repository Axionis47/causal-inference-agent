"""Atomic job submission on the caller's database transaction."""

from uuid import UUID

from sqlalchemy.orm import Session

from backend.jobs import app


def enqueue(db: Session, identity: UUID) -> int:
    return int(
        app.configure_task(name="analysis.run", allow_unknown=True, queue="analysis", 
            connection=db.connection().connection.driver_connection, lock=f"analysis:{identity}"
        ).defer(run_id=str(identity))
    )

