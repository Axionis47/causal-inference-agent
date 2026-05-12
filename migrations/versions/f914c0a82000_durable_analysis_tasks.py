"""Durable task attempts, fenced leases and column-aware analysis allowances."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "f914c0a82000"
down_revision = "d83190ab4100"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "analysis_runs", sa.Column("shared_calls", sa.Integer(), server_default="0", nullable=False)
    )
    op.drop_constraint("ck_analysis_budget", "analysis_runs", type_="check")
    op.create_check_constraint(
        "ck_analysis_budget", "analysis_runs", "calls >= 0 AND shared_calls BETWEEN 0 AND 27"
    )
    op.create_table(
        "analysis_tasks",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("run_id", sa.Uuid(), sa.ForeignKey("analysis_runs.id"), nullable=False),
        sa.Column("key", sa.String(200), nullable=False),
        sa.Column("checkpoint", sa.String(40), nullable=False),
        sa.Column("kind", sa.String(40), nullable=False),
        sa.Column("budget_kind", sa.String(20), nullable=False),
        sa.Column("column_ids", postgresql.JSONB(), nullable=False),
        sa.Column("obligation_keys", postgresql.JSONB(), nullable=False),
        sa.Column("scope", postgresql.JSONB(), nullable=False),
        sa.Column("reason", sa.String(4000), nullable=False),
        sa.Column("dependency_receipt_ids", postgresql.JSONB(), nullable=False),
        sa.Column("input_hash", sa.String(64), nullable=False),
        sa.Column("packet", postgresql.JSONB(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("lease_until", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("run_id", "key", name="uq_analysis_task_key"),
        sa.CheckConstraint(
            "max_attempts BETWEEN 1 AND 2 AND generation >= 0", name="ck_analysis_task_allowance"
        ),
        sa.CheckConstraint(
            "budget_kind IN ('column','relationship','shared')", name="ck_analysis_task_budget_kind"
        ),
        sa.CheckConstraint(
            "status IN ('pending','running','accepted','rejected','failed','blocked','exhausted','expanded')",
            name="ck_analysis_task_status",
        ),
    )
    op.create_index("ix_analysis_tasks_run_id", "analysis_tasks", ["run_id"])
    op.create_table(
        "analysis_task_attempts",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("run_id", sa.Uuid(), sa.ForeignKey("analysis_runs.id"), nullable=False),
        sa.Column("task_id", sa.Uuid(), sa.ForeignKey("analysis_tasks.id")),
        sa.Column("key", sa.String(240), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("input_hash", sa.String(64), nullable=False),
        sa.Column("request", postgresql.JSONB()),
        sa.Column("request_hash", sa.String(64)),
        sa.Column("receipt", postgresql.JSONB()),
        sa.Column("raw_output", sa.String()),
        sa.Column("metrics", postgresql.JSONB(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("run_id", "key", name="uq_analysis_attempt_key"),
        sa.UniqueConstraint("task_id", "number", name="uq_analysis_task_attempt"),
        sa.CheckConstraint("number > 0 AND generation >= 0", name="ck_analysis_attempt_number"),
        sa.CheckConstraint(
            "status IN ('reserved','accepted','rejected','failed','uncertain','blocked','exhausted','expanded')",
            name="ck_analysis_attempt_status",
        ),
    )
    op.create_index("ix_analysis_task_attempts_run_id", "analysis_task_attempts", ["run_id"])
    op.create_index("ix_analysis_task_attempts_task_id", "analysis_task_attempts", ["task_id"])
    op.execute("""
    CREATE FUNCTION protect_analysis_task() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF TG_OP = 'DELETE' THEN RAISE EXCEPTION 'Analysis tasks cannot be deleted'; END IF;
      IF (to_jsonb(NEW) - ARRAY['status','generation','lease_until','updated_at'])
           IS DISTINCT FROM (to_jsonb(OLD) - ARRAY['status','generation','lease_until','updated_at'])
      THEN RAISE EXCEPTION 'Analysis task inputs are immutable'; END IF;
      IF OLD.status IN ('accepted','blocked','exhausted','expanded') AND NEW IS DISTINCT FROM OLD
      THEN RAISE EXCEPTION 'A terminal analysis task is immutable'; END IF;
      IF NEW.generation < OLD.generation THEN RAISE EXCEPTION 'Task generations cannot decrease'; END IF;
      RETURN NEW;
    END $$;
    CREATE TRIGGER protect_analysis_task BEFORE UPDATE OR DELETE ON analysis_tasks
      FOR EACH ROW EXECUTE FUNCTION protect_analysis_task();
    CREATE FUNCTION protect_analysis_task_attempt() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF TG_OP = 'DELETE' THEN RAISE EXCEPTION 'Reserved analysis attempts cannot be deleted'; END IF;
      IF (to_jsonb(NEW) - ARRAY['status','receipt','raw_output','metrics','completed_at'])
           IS DISTINCT FROM (to_jsonb(OLD) - ARRAY['status','receipt','raw_output','metrics','completed_at'])
      THEN RAISE EXCEPTION 'Reserved analysis attempt inputs are immutable'; END IF;
      IF OLD.status <> 'reserved' AND NEW IS DISTINCT FROM OLD
      THEN RAISE EXCEPTION 'A settled analysis attempt is immutable'; END IF;
      RETURN NEW;
    END $$;
    CREATE TRIGGER protect_analysis_task_attempt BEFORE UPDATE OR DELETE ON analysis_task_attempts
      FOR EACH ROW EXECUTE FUNCTION protect_analysis_task_attempt();
    CREATE FUNCTION protect_analysis_shared_budget() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF NEW.shared_calls < OLD.shared_calls THEN RAISE EXCEPTION 'Spent shared allowance cannot decrease'; END IF;
      RETURN NEW;
    END $$;
    CREATE TRIGGER protect_analysis_shared_budget BEFORE UPDATE ON analysis_runs
      FOR EACH ROW EXECUTE FUNCTION protect_analysis_shared_budget();
    """)


def downgrade():
    op.execute(
        "DROP TRIGGER IF EXISTS protect_analysis_shared_budget ON analysis_runs; DROP FUNCTION IF EXISTS protect_analysis_shared_budget(); DROP TRIGGER protect_analysis_task_attempt ON analysis_task_attempts; DROP FUNCTION protect_analysis_task_attempt(); DROP TRIGGER protect_analysis_task ON analysis_tasks; DROP FUNCTION protect_analysis_task();"
    )
    op.drop_table("analysis_task_attempts")
    op.drop_table("analysis_tasks")
    op.drop_constraint("ck_analysis_budget", "analysis_runs", type_="check")
    # Downgrade fails honestly if any retained run has spent more than 32 calls.
    op.create_check_constraint("ck_analysis_budget", "analysis_runs", "calls BETWEEN 0 AND 32")
    op.drop_column("analysis_runs", "shared_calls")
