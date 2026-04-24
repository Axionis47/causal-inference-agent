"""Scientific checkpoint budgets and approval receipts, preserving historical records."""

from alembic import op

revision = "d83190ab4100"
down_revision = "c410678d5e00"
branch_labels = None
depends_on = None


def upgrade():
    op.drop_constraint("ck_analysis_budget", "analysis_runs", type_="check")
    op.create_check_constraint("ck_analysis_budget", "analysis_runs", "calls BETWEEN 0 AND 32")
    op.execute("""
    CREATE FUNCTION protect_analysis_harness() RETURNS trigger LANGUAGE plpgsql AS $$
    DECLARE entry RECORD; program jsonb;
    BEGIN
      IF NEW.inputs->>'contract' <> 'checkpoint-harness-2' THEN RETURN NEW; END IF;
      IF (SELECT count(*) FROM jsonb_object_keys(NEW.receipts) k WHERE k LIKE 'read:%') > 6
         OR (SELECT count(*) FROM jsonb_object_keys(NEW.receipts) k WHERE k LIKE 'diagnostic_request:%') > 4
         OR (SELECT count(*) FROM jsonb_object_keys(NEW.receipts) k WHERE k LIKE 'repair_request:%') > 2
         OR (SELECT count(*) FROM jsonb_object_keys(NEW.receipts) k WHERE k LIKE 'program:%') > 3
      THEN RAISE EXCEPTION 'The persisted harness allowance cannot be exceeded'; END IF;
      FOR entry IN SELECT key, value FROM jsonb_each(NEW.receipts) WHERE key LIKE 'approval:%' LOOP
        program := OLD.receipts->('program:' || split_part(entry.key, ':', 2));
        IF program IS NULL OR entry.value->>'program_hash' IS DISTINCT FROM program->>'hash'
           OR entry.value->>'snapshot_hash' IS DISTINCT FROM OLD.snapshot_hash
           OR entry.value->>'author_id' IS DISTINCT FROM OLD.author_id::text
           OR entry.value->>'approved_at' IS NULL
        THEN RAISE EXCEPTION 'Approval must bind an existing frozen program, science and owner'; END IF;
      END LOOP;
      RETURN NEW;
    END $$;
    CREATE TRIGGER protect_analysis_harness BEFORE UPDATE ON analysis_runs
      FOR EACH ROW EXECUTE FUNCTION protect_analysis_harness();
    """)


def downgrade():
    op.execute(
        "DROP TRIGGER protect_analysis_harness ON analysis_runs; DROP FUNCTION protect_analysis_harness();"
    )
    # Do not truncate real spent budgets to make an older application appear compatible.
    op.drop_constraint("ck_analysis_budget", "analysis_runs", type_="check")
    op.create_check_constraint("ck_analysis_budget", "analysis_runs", "calls BETWEEN 0 AND 16")
