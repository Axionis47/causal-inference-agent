"""Scripted preparation coordinator runs over the four method packs."""

from __future__ import annotations

import io
import json
import uuid
from collections.abc import Iterator
from dataclasses import fields
from typing import Any

import psycopg
import pytest

from causal.preparation import nodes
from causal.preparation.harness import PreparationDeps
from causal.shared import handoff, persistence
from tests.infrastructure import MIGRATIONS, requires_docker
from tests.preparation.support import (
    METHODS,
    ROOT,
    VERSIONS,
    approved_design,
    csv_for,
    event,
    make_deps,
)

pytestmark = requires_docker

# The approved delivery-capacity cardinality vector every fixture design carries (SC §11).


Stack = tuple[Any, PreparationDeps, io.StringIO, dict[str, str]]


@pytest.fixture(scope="module")
def stack(postgres_dsn: str, minio_s3: dict[str, Any]) -> Iterator[Stack]:
    """One database, one object store, and the four approved designs, built once per session."""
    admin = psycopg.connect(postgres_dsn, autocommit=True)
    name = f"test_{uuid.uuid4().hex[:10]}"
    admin.execute(f'CREATE DATABASE "{name}"')
    admin.close()
    conn = psycopg.connect(postgres_dsn.rsplit("/", 1)[0] + f"/{name}", autocommit=True)
    persistence.apply_migrations(conn, MIGRATIONS)
    sink = io.StringIO()
    deps = make_deps(conn, persistence.ObjectStore(minio_s3["client"], minio_s3["bucket"]), sink)
    designs = {method: approved_design(deps, f"an-{method}", method, csv_for(method))
               for method in METHODS}
    yield conn, deps, sink, designs
    conn.close()


def run_one(deps: PreparationDeps, analysis_id: str, outcome: str, revision: int = 1) -> Any:
    return nodes.run_preparation(
        deps, analysis_id=analysis_id, design_outcome_artifact_id=outcome,
        stage_run_id=f"pr:{analysis_id}:{revision}", preparation_revision=revision)


def artifact_ids(conn: Any, analysis_id: str, *kinds: str) -> set[str]:
    return {str(row[0]) for row in conn.execute(
        "SELECT artifact_id FROM causal.artifacts WHERE analysis_id = %s"
        " AND artifact_type = ANY(%s)", (analysis_id, list(kinds))).fetchall()}


def test_the_preparation_scope_holds_no_model_plumbing() -> None:
    """Amendment 2: no gateway dependency, no task runner, no checkpointer in this scope."""
    named = {found.name for found in fields(PreparationDeps)}
    assert not named & {"gateway", "checkpointer", "task_table", "prompts_root"}
    sources = "".join(path.read_text(encoding="utf-8")
                      for path in (ROOT / "src" / "causal" / "preparation").glob("*.py"))
    assert not any(name in sources for name in (
        "causal.shared.gateway", "VertexGateway", "TaskRunner", "StateGraph", "PostgresSaver"))


@pytest.mark.parametrize("method", METHODS)
def test_a_scripted_run_reaches_prepared_and_opens_the_prd004_handoff(
    stack: Stack, method: str
) -> None:
    """The happy path per pack: no model call, a `prepared` outcome, a T-006-clean handoff."""
    conn, deps, _, designs = stack
    run = run_one(deps, f"an-{method}", designs[method])
    assert run.status == "prepared", run.error_code
    assert run.handoff_id is not None and run.row_set_hash is not None
    # D-090: the §20 bundle carries the prepared-stage post-repair reports the final wall gated
    # on, so PRD-004's §4 condition 7 can read their statuses instead of guessing.
    held = deps.products.load_envelope(next(iter(
        artifact_ids(conn, f"an-{method}", "PreparedFrameBundle"))))
    carried = json.loads(deps.objects.get(held.payload_locator))["postrepair_diagnostics"]
    pack = deps.packs.get(method, VERSIONS[method])
    assert {(row["diagnostic_id"], row["status"]) for row in carried} >= {
        (name, "not_computable") for name in pack.required_postrepair_diagnostic_ids}
    opened = nodes.open_preparation_handoff(
        deps, run.analysis_id, str(run.outcome_artifact_id), "sr:estimation")
    gate = handoff.HandoffGate(deps.objects, deps.products, handoff.HandoffStore(conn),
                               deps.registry, deps.emitter)
    result = gate.accept(opened, "estimation-harness", frozenset({"prepared"}),
                         lambda verdict, codes: event(run.analysis_id, f"handoff.{verdict}"))
    assert result.accepted, result.error_codes
    assert conn.execute("SELECT state FROM preparation.preparation_runs WHERE stage_run_id = %s",
                        (run.stage_run_id,)).fetchone() == ("completed",)


def test_a_permitted_imputation_gap_compiles_and_executes(stack: Stack) -> None:
    """§25.1: a permitted imputation target compiles to its registered operation, no model."""
    conn, deps, _, _ = stack
    outcome = approved_design(deps, "an-gap", "randomized_experiment",
                              csv_for("randomized_experiment", blank="age"), impute=("age",))
    run = run_one(deps, "an-gap", outcome)
    assert run.status == "prepared", run.error_code
    items = {str(row[0]) for row in conn.execute(
        "SELECT plan_item_id FROM preparation.plan_items WHERE stage_run_id = %s",
        (run.stage_run_id,)).fetchall()}
    assert items == {"pi:required_derivation_missing:outcome_observed",
                     "pi:imputation_target_missing:age"}


def test_a_satisfied_contract_prepares_with_no_plan_item(stack: Stack) -> None:
    """EV-P3-001, §25: no gap compiles no item, and the run still reaches `prepared`."""
    conn, deps, _, _ = stack
    outcome = approved_design(deps, "an-clean", "randomized_experiment",
                              csv_for("randomized_experiment"), indicators=())
    run = run_one(deps, "an-clean", outcome)
    assert (run.status, run.handoff_id is not None) == ("prepared", True), run.error_code
    assert not conn.execute("SELECT plan_item_id FROM preparation.plan_items"
                            " WHERE stage_run_id = %s", (run.stage_run_id,)).fetchall()
    held = deps.products.load_envelope(next(iter(
        artifact_ids(conn, "an-clean", "ExecutionReceiptBundle"))))
    assert json.loads(deps.objects.get(held.payload_locator))["receipts"] == []


def test_a_rerun_replays_the_artifacts_and_lands_the_same_terminal(stack: Stack) -> None:
    """D-035: a new stage run recommits the deterministic artifacts as no-ops (T-019 §1.3)."""
    conn, deps, _, _ = stack
    outcome = approved_design(deps, "an-rerun", "aipw", csv_for("aipw"))
    kinds = ("PreparationContextManifest", "StabilizationRecord", "StabilizedFrame")
    first = run_one(deps, "an-rerun", outcome)
    replayed = artifact_ids(conn, "an-rerun", *kinds)
    second = run_one(deps, "an-rerun", outcome, revision=2)
    assert (second.status, second.row_set_hash) == (first.status, first.row_set_hash) == (
        "prepared", first.row_set_hash)
    assert artifact_ids(conn, "an-rerun", *kinds) == replayed
    assert conn.execute("SELECT count(*) FROM preparation.preparation_runs"
                        " WHERE analysis_id = %s AND state = 'completed'",
                        ("an-rerun",)).fetchone() == (2,)


@pytest.mark.parametrize(("name", "method", "blank", "impute", "drop", "code"), [
    ("unresolved", "randomized_experiment", "treat", (), ("missing_treatment_assignment",),
     "unresolved_conflict"),
    ("uncompilable", "did", "age", ("age",), (), "no_registered_resolution")])
def test_a_refused_frame_returns_a_design_conflict(
    stack: Stack, name: str, method: str, blank: str, impute: tuple[str, ...],
    drop: tuple[str, ...], code: str
) -> None:
    """§16: PRD-003 never resolves a conflict; it commits one and hands design the revision."""
    _, deps, sink, _ = stack
    outcome = approved_design(deps, f"an-{name}", method, csv_for(method, blank=blank),
                              impute=impute, drop_rules=drop)
    run = run_one(deps, f"an-{name}", outcome)
    assert (run.status, run.conflict_code) == ("design_conflict", code)
    emitted = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert any(row["event_name"] == "blocker.raised" and row["error_code"] == code
               and row["analysis_id"] == f"an-{name}" for row in emitted)


def test_a_frame_without_method_structure_is_not_runnable(stack: Stack) -> None:
    """§12: the retained rows lose the pack's minimum support, so the frame cannot run."""
    _, deps, _, _ = stack
    outcome = approved_design(deps, "an-thin", "aipw", csv_for("aipw", limit=4))
    run = run_one(deps, "an-thin", outcome)
    assert (run.status, run.error_code) == ("not_runnable", "method_support_lost")
