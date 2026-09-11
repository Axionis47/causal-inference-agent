# The estimation harness: the §21 schema, the run-row lifecycle, flush-gated commits and
# their replay, the wall runner, the §16 conflict return, and the frozen frame read
# (T-024 §2; PRD-004 §19.1, §20.5, §21, §26.1).

from __future__ import annotations

import json
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import psycopg
import pytest

from causal.analysis.integration import RESOURCE_ROOT
from causal.analysis.integration import harness as eh
from causal.analysis.integration import walls as ew
from causal.analysis.integration.plancompile import DesignConflictDraftV1
from causal.analysis.tests.support.builders import PACKS, REGISTRIES, ROLES, make_plan
from causal.shared import frames, persistence
from causal.shared.canonical import canonical_bytes
from causal.shared.contracts import ArtifactEnvelopeV1
from causal.shared.events import EventEmitter
from causal.shared.registry import load_artifact_type_registry
from tests.infrastructure import MIGRATIONS, requires_docker

pytestmark = requires_docker

NOW = datetime(2026, 8, 26, 12, 0, 0, 0, tzinfo=UTC)
ANALYSIS, UPSTREAM_RUN = "an-1", "pr:an-1:1"
MIGRATION = Path(MIGRATIONS) / "0007_estimation.sql"
ESTIMATION_TABLES = {"estimation_runs", "estimation_artifact_refs", "contribution_mask_index"}
FRAME = pl.DataFrame({"y": [1.0, 2.0], "arm": ["a", "b"], "uid": [1, 2], "score": [0.5, 1.5],
                      "period": [1, 2], "spare": ["x", "y"]})
DRAFT = DesignConflictDraftV1(
    conflict_code="method_structure_changed", failed_rule_id="approved_unit_count_changed",
    affected_row_count=0, affected_unit_count=0, affected_dimension_counts={"contrasts": 9},
    evidence_artifact_ids=(), why_no_permitted_operation="the observed unit structure differs from the approved request",
    material_design_fields=("primary_contrasts",), recommended_action="revise_design")


class FlushCounter:
    # A tracer stand-in that counts the SC §8.2 flushes a commit must wait for.

    flushes = 0

    def flush(self) -> None:
        self.flushes += 1


@pytest.fixture()
def parts(conn: Any, object_store: Any) -> dict[str, Any]:
    registry = load_artifact_type_registry(REGISTRIES / "artifact-types.v1.json")
    sink, tracer = StringIO(), FlushCounter()
    emitter = EventEmitter(sink)
    products = persistence.ProductStore(conn)
    products.create_stage_run(UPSTREAM_RUN, ANALYSIS, "preparation")
    deps = eh.EstimationDeps(
        conn=conn, products=products, objects=object_store, registry=registry, emitter=emitter,
        committer=persistence.ArtifactCommitter(object_store, products, registry, emitter,
                                                tracer=tracer),
        clock=lambda: NOW, packs=PACKS,
        rules=ew.load_validation_rules(RESOURCE_ROOT / "estimation-validation-rules.v1.json"),
        frames=frames.FrameStore(objects=object_store), registries=REGISTRIES,
        repo_root=REGISTRIES.parent)
    return {"deps": deps, "sink": sink, "tracer": tracer, "conn": conn}


def seed(parts: dict[str, Any], artifact_type: str, payload: dict[str, Any]
         ) -> ArtifactEnvelopeV1:
    # One upstream artifact placed straight into the store; PRD-004 only ever reads it.
    deps = parts["deps"]
    envelope = persistence.build_envelope(
        deps.registry, artifact_type, payload, analysis_id=ANALYSIS,
        stage_run_id=UPSTREAM_RUN, producer_version="0.1.0", parents=(), created_at_utc=NOW)
    deps.objects.put_if_absent(envelope.content_hash, canonical_bytes(payload))
    deps.products.insert_artifact(envelope)
    return envelope


def opened(parts: dict[str, Any], revision: int = 1,
           **over: Any) -> tuple[eh.HarnessBase, eh.EstimationState]:
    state = eh.new_state(ANALYSIS, f"es:{ANALYSIS}:{revision}", revision,
                         {"prepared_bundle": "pfb-1"}) | over
    base = eh.HarnessBase(parts["deps"])
    base.open_run(state, UPSTREAM_RUN)
    return base, state


def bind(state: eh.EstimationState, kind: str, envelope: ArtifactEnvelopeV1) -> None:
    state["artifacts"][kind] = envelope.artifact_id
    state["hashes"][envelope.artifact_id] = envelope.content_hash


def events(parts: dict[str, Any]) -> list[dict[str, Any]]:
    return [json.loads(line) for line in parts["sink"].getvalue().splitlines()]


# -- the §21 schema -------------------------------------------------------


def test_the_estimation_schema_carries_exactly_its_three_tables(conn: Any) -> None:
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'estimation'"
    ).fetchall()
    assert {row[0] for row in rows} == ESTIMATION_TABLES


def test_reapplying_0007_is_idempotent(conn: Any) -> None:
    conn.execute(MIGRATION.read_text(encoding="utf-8"))
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'estimation'"
    ).fetchall()
    assert {row[0] for row in rows} == ESTIMATION_TABLES


# -- the run row (SC §4; PRD-004 §26.1) -----------------------------------


@pytest.mark.parametrize(("status", "row_state"), [
    ("complete", "completed"), ("design_conflict", "completed"),
    ("not_estimable", "completed"), ("invalidated", "completed"),
    ("failed_observability", "failed_observability"), ("failed", "failed")])
def test_the_run_row_is_terminal_on_every_exit_path(
    parts: dict[str, Any], status: str, row_state: str
) -> None:
    base, state = opened(parts)
    assert parts["conn"].execute(
        "SELECT state FROM estimation.estimation_runs").fetchone() == ("running",)
    result = base.close_run(dict(state) | {"status": status, "overall_ceiling": "reportable"})
    assert result.status == status and result.estimation_revision == 1
    assert result.thread_id == f"et:{ANALYSIS}:1"
    assert parts["conn"].execute(
        "SELECT state, overall_ceiling FROM estimation.estimation_runs"
    ).fetchone() == (row_state, "reportable")


def test_a_crashed_revision_re_enters_at_the_next_revision(parts: dict[str, Any]) -> None:
    # D-035: no checkpoint exists, so the next attempt replays the artifacts under a NEW
    # stage_run_id; the UNIQUE (analysis_id, estimation_revision) row forbids reusing one.
    opened(parts, 1)
    with pytest.raises(psycopg.errors.UniqueViolation):
        opened(parts, 1, stage_run_id=f"es:{ANALYSIS}:1b")
    parts["conn"].rollback()
    _, state = opened(parts, 2)
    rows = parts["conn"].execute(
        "SELECT estimation_revision, state FROM estimation.estimation_runs"
        " ORDER BY estimation_revision").fetchall()
    assert rows == [(1, "running"), (2, "running")]
    assert state["thread_id"] == f"et:{ANALYSIS}:2"


def test_reopening_a_live_run_row_keeps_one_row_and_one_stage_run(
    parts: dict[str, Any]
) -> None:
    base, state = opened(parts)
    base.open_run(state, UPSTREAM_RUN)
    assert parts["conn"].execute(
        "SELECT count(*) FROM estimation.estimation_runs").fetchone() == (1,)
    assert parts["deps"].products.get_stage_run_state(state["stage_run_id"]) == "running"


# -- commits, replay, and events (§8.2, §20.5) ----------------------------


def test_a_commit_is_flush_gated_indexed_and_replays_as_a_no_op(
    parts: dict[str, Any]
) -> None:
    base, state = opened(parts)
    plan = seed(parts, "EstimationPlan", {"schema_version": "estimation-plan.v1"})
    payload = base.environment(make_plan()).canonical_payload()
    first = base.commit(state, "NumericalEnvironmentManifest", payload, (plan,))
    second = base.commit(state, "NumericalEnvironmentManifest", payload, (plan,))
    assert first.artifact_id == second.artifact_id
    assert first.content_hash == second.content_hash
    assert parts["tracer"].flushes == 1
    assert parts["conn"].execute(
        "SELECT count(*) FROM causal.artifacts WHERE artifact_type ="
        " 'NumericalEnvironmentManifest'").fetchone() == (1,)
    assert parts["conn"].execute(
        "SELECT kind, artifact_id FROM estimation.estimation_artifact_refs"
    ).fetchall() == [("NumericalEnvironmentManifest", first.artifact_id)]


def test_every_event_names_the_estimation_stage_and_its_eval_ids(
    parts: dict[str, Any]
) -> None:
    base, state = opened(parts)
    plan = seed(parts, "EstimationPlan", {"schema_version": "estimation-plan.v1"})
    base.commit(state, "NumericalEnvironmentManifest",
                base.environment(make_plan()).canonical_payload(), (plan,))
    base.fail(state, "entry_validation_failed")
    committed, blocker = events(parts)
    assert committed["event_name"] == "artifact.committed"
    assert committed["stage"] == "estimation" and committed["component_id"] == eh.COMPONENT
    assert committed["graph_thread_id"] == f"et:{ANALYSIS}:1"
    assert tuple(committed["required_eval_ids"]) == eh.EVAL_STAGE
    assert blocker["event_name"] == "blocker.raised" and blocker["severity"] == "error"
    assert blocker["error_code"] == "entry_validation_failed"


def test_the_wall_runner_stops_at_the_first_failing_wall_and_reports_it(
    parts: dict[str, Any]
) -> None:
    base, state = opened(parts)
    report = base.wall(state, 3, ew.WallContext(handoff_accepted=False,
                                                entry_codes=("row_set_hash_mismatch",)))
    assert not report.passed and report.wall == 1 and state["wall"] == 1
    codes = {issue.code for issue in report.issues}
    assert "handoff_not_accepted" in codes
    failed = [row for row in events(parts) if row["event_name"] == "artifact.validation_failed"]
    assert len(failed) == 1 and failed[0]["safe_dimensions"]["wall"] == 1
    # A passing wall records its number and says nothing.
    assert base.wall(state, 1, ew.WallContext(handoff_accepted=True)).passed
    assert state["wall"] == 1 and len(events(parts)) == 1


# -- the §16 conflict return and the §21 mask index -----------------------


def test_a_design_conflict_is_committed_and_returned_to_prd_002(
    parts: dict[str, Any]
) -> None:
    base, state = opened(parts)
    manifest = seed(parts, "EstimationContextManifest",
                    {"schema_version": "estimation-context-manifest.v1"})
    bind(state, "EstimationContextManifest", manifest)
    out = base.conflict(state, DRAFT)
    assert out["status"] == "design_conflict" and out["conflict_code"] == "method_structure_changed"
    committed = parts["deps"].products.load_envelope(out["artifacts"]["DesignConflict"])
    assert committed.schema_version == "design-conflict.v1"
    assert committed.parent_artifacts[0].artifact_id == manifest.artifact_id
    body = json.loads(parts["deps"].objects.get(committed.payload_locator))
    assert body["conflict_code"] == "method_structure_changed"
    assert body["evidence_artifact_ids"] == [manifest.artifact_id]
    # §19: PRD-004 never interrupts the user; the conflict is the only route out.
    assert not [row for row in events(parts) if row["event_name"].startswith("user_interrupt")]
    assert base.close_run(out).conflict_code == "method_structure_changed"


def test_the_mask_index_records_identities_and_counts_only(parts: dict[str, Any]) -> None:
    from causal.analysis.common import legacy_engine as engine

    base, state = opened(parts)
    plan = seed(parts, "EstimationPlan", {"schema_version": "estimation-plan.v1"})
    bits = engine.mask_bits("arm_membership", FRAME, ROLES, {})
    locator = parts["deps"].frames.put_object(engine.mask_object_payload("arm_membership", bits))
    mask = engine.contribution_mask(
        make_plan(), "arm_membership", bits,
        eh.ec.ObjectRefV1(object_locator=locator, content_hash=locator.split("/")[-1]),
        frame_row_set_hash=make_plan().row_set_hash, calculation_id="primary",
        parents=(eh.ArtifactRef(artifact_id=plan.artifact_id, content_hash=plan.content_hash),))
    committed = base.commit(state, "AnalysisContributionMask", mask.canonical_payload(), (plan,))
    base.index_mask(state, committed, mask)
    row = parts["conn"].execute(
        "SELECT calculation_id, mask_rule_id, included_rows, noncontributing_rows"
        " FROM estimation.contribution_mask_index").fetchone()
    assert row == ("primary", "arm_membership", 2, 0)
    assert state["mask_ids"] == [committed.artifact_id]


# -- the frozen prepared frame (§19.1) ------------------------------------


def test_the_prepared_frame_is_reopened_under_its_recorded_dtypes(
    parts: dict[str, Any]
) -> None:
    deps = parts["deps"]
    columns = [{"column_name": name, "dtype": str(dtype), "prepared_from": []}
               for name, dtype in FRAME.schema.items()]
    locator, digest = deps.frames.write_frame(
        FRAME, [frames.FrameColumnV1(column_name=str(row["column_name"]),
                                     dtype=str(row["dtype"])) for row in columns])
    prepared = seed(parts, "PreparedFrame", {
        "schema_version": "prepared-frame.v1", "columns": columns,
        "frame_object": {"object_locator": locator, "content_hash": digest}})
    bundle = seed(parts, "PreparedFrameBundle", {
        "schema_version": "prepared-frame-bundle.v1",
        "prepared_frame": {"artifact_id": prepared.artifact_id,
                           "content_hash": prepared.content_hash}})
    base, state = opened(parts, upstream={"prepared_bundle": bundle.artifact_id})
    reopened = base.prepared_frame(state)
    assert reopened.schema == FRAME.schema and reopened.equals(FRAME)
    # §19.1: the adapter's view is built from the declared roles, never the whole frame.
    view = base.view(state, SimpleNamespace(role_columns=dict(ROLES)))
    assert set(view.columns) == set(ROLES.values()) and "spare" not in view.columns
