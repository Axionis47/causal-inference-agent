"""Preparation schema migration 0006 against a live Postgres (T-015 §1.5; PRD-003 §19)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
import pytest

from tests.infrastructure import MIGRATIONS, requires_docker

pytestmark = requires_docker

HASH = "a" * 64
NOW = datetime(2026, 8, 26, 12, 0, 0, 0, tzinfo=UTC)
MIGRATION = Path(MIGRATIONS) / "0006_preparation.sql"
PREPARATION_TABLES = {
    "preparation_runs", "preparation_artifact_refs", "preparation_tasks", "plan_items",
}


def seed_artifact(conn: psycopg.Connection[Any], artifact_id: str = "art-1") -> str:
    """Insert the causal.stage_runs and causal.artifacts rows the preparation FKs point at."""
    conn.execute(
        "INSERT INTO causal.stage_runs (stage_run_id, analysis_id, stage, run_state,"
        " created_at_utc, updated_at_utc) VALUES ('run-1', 'an-1', 'preparation', 'running',"
        " %s, %s) ON CONFLICT DO NOTHING",
        (NOW, NOW),
    )
    conn.execute(
        "INSERT INTO causal.artifacts (artifact_id, artifact_type, schema_version, content_hash,"
        " analysis_id, stage_run_id, producer_component, producer_version, sensitivity_class,"
        " created_at_utc, payload_locator) VALUES (%s, 'PreparationPlan', 'preparation-plan.v1',"
        " %s, 'an-1', 'run-1', 'preparation-harness', '0.1.0', 'internal', %s, %s)",
        (artifact_id, HASH, NOW, f"objects/{HASH}"),
    )
    return artifact_id


def insert_run(
    conn: psycopg.Connection[Any],
    *,
    stage_run_id: str = "run-1",
    preparation_revision: int = 1,
    state: str = "running",
) -> None:
    conn.execute(
        "INSERT INTO preparation.preparation_runs (stage_run_id, analysis_id, graph_thread_id,"
        " state, design_stage_run_id, preparation_revision, created_at, updated_at)"
        " VALUES (%s, 'an-1', 'prep-thread-1', %s, 'design-run-1', %s, %s, %s)",
        (stage_run_id, state, preparation_revision, NOW, NOW),
    )


def test_preparation_schema_tables_exist(conn: psycopg.Connection[Any]) -> None:
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'preparation'"
    ).fetchall()
    assert {row[0] for row in rows} == PREPARATION_TABLES


def test_reapplying_0006_is_idempotent(conn: psycopg.Connection[Any]) -> None:
    conn.execute(MIGRATION.read_text(encoding="utf-8"))
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'preparation'"
    ).fetchall()
    assert {row[0] for row in rows} == PREPARATION_TABLES


def test_run_artifact_ref_and_plan_item_insert(conn: psycopg.Connection[Any]) -> None:
    artifact_id = seed_artifact(conn)
    insert_run(conn)
    conn.execute(
        "INSERT INTO preparation.preparation_artifact_refs (stage_run_id, kind, artifact_id,"
        " content_hash, schema_version)"
        " VALUES ('run-1', 'PreparationPlan', %s, %s, 'preparation-plan.v1')",
        (artifact_id, HASH),
    )
    conn.execute(
        "INSERT INTO preparation.plan_items (plan_artifact_id, plan_item_id, stage_run_id, phase,"
        " state, created_at, updated_at)"
        " VALUES (%s, 'pi-1', 'run-1', 'imputation', 'committed', %s, %s)",
        (artifact_id, NOW, NOW),
    )
    row = conn.execute(
        "SELECT r.state, r.row_set_hash, i.state, i.receipt_artifact_id"
        " FROM preparation.preparation_runs r JOIN preparation.plan_items i"
        " ON i.stage_run_id = r.stage_run_id"
    ).fetchone()
    assert row == ("running", None, "committed", None)


def test_duplicate_analysis_revision_rejected(conn: psycopg.Connection[Any]) -> None:
    insert_run(conn)
    with pytest.raises(psycopg.errors.UniqueViolation):
        insert_run(conn, stage_run_id="run-2")
    conn.rollback()
    insert_run(conn, stage_run_id="run-2", preparation_revision=2)
    assert conn.execute("SELECT count(*) FROM preparation.preparation_runs").fetchone() == (2,)


def test_invalid_run_state_rejected(conn: psycopg.Connection[Any]) -> None:
    with pytest.raises(psycopg.errors.CheckViolation):
        insert_run(conn, state="daydreaming")
    conn.rollback()


def test_task_stop_states_and_attempt_ceiling(conn: psycopg.Connection[Any]) -> None:
    insert_run(conn)
    task = (
        "INSERT INTO preparation.preparation_tasks (task_id, stage_run_id, analysis_id, task_kind,"
        " phase, scope_ids, envelope_hash, status, attempt_count, created_at, updated_at)"
        " VALUES (%s, 'run-1', 'an-1', 'coupled_columns', 'preparation', '[]'::jsonb, %s, %s, %s,"
        " %s, %s)"
    )
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute(task, ("task-1", HASH, "improvising", 1, NOW, NOW))
    conn.rollback()
    with pytest.raises(psycopg.errors.CheckViolation):
        # §17.7: one initial response plus at most two corrections.
        conn.execute(task, ("task-1", HASH, "proposed", 4, NOW, NOW))
    conn.rollback()
    conn.execute(task, ("task-1", HASH, "design_conflict", 3, NOW, NOW))
    assert conn.execute(
        "SELECT status FROM preparation.preparation_tasks"
    ).fetchone() == ("design_conflict",)


def test_a_row_set_hash_must_be_a_sha256(conn: psycopg.Connection[Any]) -> None:
    insert_run(conn)
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute("UPDATE preparation.preparation_runs SET row_set_hash = 'nope'")
    conn.rollback()
    conn.execute("UPDATE preparation.preparation_runs SET row_set_hash = %s", (HASH,))
    assert conn.execute(
        "SELECT row_set_hash FROM preparation.preparation_runs"
    ).fetchone() == (HASH,)
