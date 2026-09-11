"""Design schema migration 0004 against a live Postgres (T-009 §5, §6; PRD-002 §21)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest

from tests.infrastructure import requires_docker

pytestmark = requires_docker

HASH = "a" * 64
NOW = datetime(2026, 8, 24, 12, 0, 0, 0, tzinfo=UTC)
DESIGN_TABLES = {
    "design_runs", "causal_graph_views", "design_tasks", "context_requirements",
    "accepted_facts",
}


def seed_artifact(conn: psycopg.Connection[Any], artifact_id: str = "art-1") -> str:
    """Insert the causal.stage_runs and causal.artifacts rows the design FKs point at."""
    conn.execute(
        "INSERT INTO causal.stage_runs (stage_run_id, analysis_id, stage, run_state,"
        " created_at_utc, updated_at_utc) VALUES ('run-1', 'an-1', 'design', 'running', %s, %s)"
        " ON CONFLICT DO NOTHING",
        (NOW, NOW),
    )
    conn.execute(
        "INSERT INTO causal.artifacts (artifact_id, artifact_type, schema_version, content_hash,"
        " analysis_id, stage_run_id, producer_component, producer_version, sensitivity_class,"
        " created_at_utc, payload_locator) VALUES (%s, 'DesignOutcome', 'design-outcome.v2', %s,"
        " 'an-1', 'run-1', 'design-harness', '0.1.0', 'internal', %s, %s)",
        (artifact_id, HASH, NOW, f"objects/{HASH}"),
    )
    return artifact_id


def insert_run(
    conn: psycopg.Connection[Any],
    *,
    stage_run_id: str = "run-1",
    design_revision: int = 1,
    state: str = "running",
) -> None:
    conn.execute(
        "INSERT INTO design.design_runs (stage_run_id, analysis_id, graph_thread_id,"
        " design_revision, state, created_at, updated_at)"
        " VALUES (%s, 'an-1', 'thread-1', %s, %s, %s, %s)",
        (stage_run_id, design_revision, state, NOW, NOW),
    )


def test_design_schema_tables_exist(conn: psycopg.Connection[Any]) -> None:
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'design'"
    ).fetchall()
    assert {row[0] for row in rows} == DESIGN_TABLES


def test_run_insert(conn: psycopg.Connection[Any]) -> None:
    seed_artifact(conn)
    insert_run(conn)
    row = conn.execute("SELECT state, selected_table FROM design.design_runs").fetchone()
    assert row == ("running", None)


def test_duplicate_analysis_revision_rejected(conn: psycopg.Connection[Any]) -> None:
    insert_run(conn)
    with pytest.raises(psycopg.errors.UniqueViolation):
        insert_run(conn, stage_run_id="run-2")
    conn.rollback()
    insert_run(conn, stage_run_id="run-2", design_revision=2)  # a new revision is fine
    assert conn.execute("SELECT count(*) FROM design.design_runs").fetchone() == (2,)


def test_invalid_run_state_rejected(conn: psycopg.Connection[Any]) -> None:
    with pytest.raises(psycopg.errors.CheckViolation):
        insert_run(conn, state="daydreaming")
    conn.rollback()


def test_a_resolved_requirement_requires_an_accepted_fact(conn: psycopg.Connection[Any]) -> None:
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute(
            "INSERT INTO design.context_requirements (analysis_id, design_revision,"
            " requirement_id, scope_kind, scope_id, criticality, missing_action, state,"
            " attempted_evidence) VALUES ('an-1', 1, 'design.table_grain', 'design', 'design',"
            " 'blocking', 'ask_user', 'resolved', '[]')")
    conn.rollback()
