"""Durable bounded agent sessions, native tool turns and sealed handoffs.

Revision ID: ab2840d19000
Revises: f914c0a82000
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "ab2840d19000"
down_revision = "f914c0a82000"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("analysis_runs", sa.Column("model_limit", sa.Integer(), nullable=True))
    op.drop_constraint("ck_analysis_budget", "analysis_runs", type_="check")
    op.create_check_constraint("ck_analysis_budget", "analysis_runs", "calls >= 0 AND shared_calls >= 0 AND (model_limit IS NULL OR calls <= model_limit)")
    op.add_column("analysis_tasks", sa.Column("parent_id", sa.Uuid(), sa.ForeignKey("analysis_tasks.id"), nullable=True))
    op.add_column("analysis_tasks", sa.Column("agent", sa.String(40), nullable=True))
    op.add_column("analysis_tasks", sa.Column("model_limit", sa.Integer(), nullable=True))
    op.add_column("analysis_tasks", sa.Column("submissions", sa.Integer(), server_default="0", nullable=False))
    op.add_column("analysis_tasks", sa.Column("draft", postgresql.JSONB(), server_default="{}", nullable=False))
    op.add_column("analysis_tasks", sa.Column("handoff", postgresql.JSONB(), nullable=True))
    op.add_column("analysis_tasks", sa.Column("handoff_hash", sa.String(64), nullable=True))
    op.add_column("analysis_task_attempts", sa.Column("provider_content", postgresql.JSONB(), nullable=True))
    op.add_column("analysis_task_attempts", sa.Column("tool_receipts", postgresql.JSONB(), server_default="{}", nullable=False))
    op.execute("""
    CREATE OR REPLACE FUNCTION protect_analysis_task() RETURNS trigger LANGUAGE plpgsql AS $$
    DECLARE mutable text[];
    BEGIN
      IF TG_OP = 'DELETE' THEN RAISE EXCEPTION 'Analysis tasks cannot be deleted'; END IF;
      mutable := ARRAY['status','generation','lease_until','updated_at'];
      IF OLD.agent IS NOT NULL THEN mutable := mutable || ARRAY['draft','submissions','handoff','handoff_hash']; END IF;
      IF (to_jsonb(NEW) - mutable) IS DISTINCT FROM (to_jsonb(OLD) - mutable)
      THEN RAISE EXCEPTION 'Analysis task inputs are immutable'; END IF;
      IF OLD.status IN ('accepted','blocked','exhausted','expanded') AND NEW IS DISTINCT FROM OLD
      THEN RAISE EXCEPTION 'A terminal analysis task is immutable'; END IF;
      IF NEW.generation < OLD.generation OR NEW.submissions < OLD.submissions OR NEW.submissions > 3
      THEN RAISE EXCEPTION 'Agent allowances cannot be reset or exceeded'; END IF;
      IF OLD.handoff IS NOT NULL AND (NEW.handoff IS DISTINCT FROM OLD.handoff OR NEW.handoff_hash IS DISTINCT FROM OLD.handoff_hash)
      THEN RAISE EXCEPTION 'An accepted handoff is immutable'; END IF;
      IF NEW.agent IS NOT NULL AND NEW.status = 'accepted' AND (NEW.handoff IS NULL OR NEW.handoff_hash IS NULL)
      THEN RAISE EXCEPTION 'Accepted agents require a sealed handoff'; END IF;
      RETURN NEW;
    END $$;
    CREATE OR REPLACE FUNCTION protect_analysis_task_attempt() RETURNS trigger LANGUAGE plpgsql AS $$
    DECLARE native boolean; item record;
    BEGIN
      IF TG_OP = 'DELETE' THEN RAISE EXCEPTION 'Reserved analysis attempts cannot be deleted'; END IF;
      SELECT agent IS NOT NULL INTO native FROM analysis_tasks WHERE id = OLD.task_id;
      IF NOT coalesce(native, false) THEN
        IF (to_jsonb(NEW) - ARRAY['status','receipt','raw_output','metrics','completed_at']) IS DISTINCT FROM
           (to_jsonb(OLD) - ARRAY['status','receipt','raw_output','metrics','completed_at'])
        THEN RAISE EXCEPTION 'Reserved analysis attempt inputs are immutable'; END IF;
        IF OLD.status <> 'reserved' AND NEW IS DISTINCT FROM OLD
        THEN RAISE EXCEPTION 'A settled analysis attempt is immutable'; END IF;
        RETURN NEW;
      END IF;
      IF (to_jsonb(NEW) - ARRAY['status','receipt','raw_output','metrics','completed_at','provider_content','tool_receipts']) IS DISTINCT FROM
         (to_jsonb(OLD) - ARRAY['status','receipt','raw_output','metrics','completed_at','provider_content','tool_receipts'])
      THEN RAISE EXCEPTION 'Native provider inputs are immutable'; END IF;
      IF OLD.status <> 'reserved' AND (to_jsonb(NEW) - 'tool_receipts') IS DISTINCT FROM (to_jsonb(OLD) - 'tool_receipts')
      THEN RAISE EXCEPTION 'A settled provider response is immutable'; END IF;
      IF NEW.tool_receipts IS DISTINCT FROM OLD.tool_receipts AND (NEW.status <> 'accepted' OR NEW.provider_content IS NULL)
      THEN RAISE EXCEPTION 'Tools require a saved native provider response'; END IF;
      FOR item IN SELECT key, value FROM jsonb_each(OLD.tool_receipts) LOOP
        IF NOT NEW.tool_receipts ? item.key OR
          (item.value->>'complete' = 'true' AND NEW.tool_receipts->item.key IS DISTINCT FROM item.value) OR
          (((NEW.tool_receipts->item.key) - ARRAY['complete','result']) IS DISTINCT FROM (item.value - ARRAY['complete','result']))
        THEN RAISE EXCEPTION 'Saved tool arguments and completed results are immutable'; END IF;
      END LOOP;
      RETURN NEW;
    END $$;
    CREATE OR REPLACE FUNCTION protect_analysis_shared_budget() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF NEW.shared_calls < OLD.shared_calls OR NEW.model_limit IS DISTINCT FROM OLD.model_limit
      THEN RAISE EXCEPTION 'Saved model limits and spent allowance cannot change'; END IF;
      RETURN NEW;
    END $$;
    """)


def downgrade() -> None:
    connection = op.get_bind()
    if connection.scalar(sa.text("SELECT EXISTS (SELECT 1 FROM analysis_tasks WHERE agent IS NOT NULL)")):
        raise RuntimeError("Agent sessions exist. Preserve their audit records and use application rollback instead of a destructive downgrade.")
    if connection.scalar(sa.text("SELECT EXISTS (SELECT 1 FROM analysis_runs WHERE shared_calls > 27)")):
        raise RuntimeError("Spent model allowances exceed the historical limit; counters cannot be truncated.")
    op.execute("""
    CREATE OR REPLACE FUNCTION protect_analysis_task() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF TG_OP = 'DELETE' THEN RAISE EXCEPTION 'Analysis tasks cannot be deleted'; END IF;
      IF (to_jsonb(NEW) - ARRAY['status','generation','lease_until','updated_at']) IS DISTINCT FROM
         (to_jsonb(OLD) - ARRAY['status','generation','lease_until','updated_at'])
      THEN RAISE EXCEPTION 'Analysis task inputs are immutable'; END IF;
      IF OLD.status IN ('accepted','blocked','exhausted','expanded') AND NEW IS DISTINCT FROM OLD
      THEN RAISE EXCEPTION 'A terminal analysis task is immutable'; END IF;
      IF NEW.generation < OLD.generation THEN RAISE EXCEPTION 'Task generations cannot decrease'; END IF;
      RETURN NEW;
    END $$;
    CREATE OR REPLACE FUNCTION protect_analysis_task_attempt() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF TG_OP = 'DELETE' THEN RAISE EXCEPTION 'Reserved analysis attempts cannot be deleted'; END IF;
      IF (to_jsonb(NEW) - ARRAY['status','receipt','raw_output','metrics','completed_at']) IS DISTINCT FROM
         (to_jsonb(OLD) - ARRAY['status','receipt','raw_output','metrics','completed_at'])
      THEN RAISE EXCEPTION 'Reserved analysis attempt inputs are immutable'; END IF;
      IF OLD.status <> 'reserved' AND NEW IS DISTINCT FROM OLD THEN RAISE EXCEPTION 'A settled analysis attempt is immutable'; END IF;
      RETURN NEW;
    END $$;
    CREATE OR REPLACE FUNCTION protect_analysis_shared_budget() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF NEW.shared_calls < OLD.shared_calls THEN RAISE EXCEPTION 'Spent shared allowance cannot decrease'; END IF;
      RETURN NEW;
    END $$;
    """)
    for name in ("tool_receipts", "provider_content"):
        op.drop_column("analysis_task_attempts", name)
    for name in ("handoff_hash", "handoff", "draft", "submissions", "model_limit", "agent", "parent_id"):
        op.drop_column("analysis_tasks", name)
    op.drop_constraint("ck_analysis_budget", "analysis_runs", type_="check")
    op.create_check_constraint("ck_analysis_budget", "analysis_runs", "calls >= 0 AND shared_calls BETWEEN 0 AND 27")
    op.drop_column("analysis_runs", "model_limit")
