"""Scripted preparation coordinator runs over the four method packs (T-019 §2; EV-P3-001..007).

Every fixture here is committed through the real stores and the real committer, so the nine
§4 entry conditions are satisfied by honest artifacts: the capacity check is bound from both
the design and the outcome, and the pre-repair report is a parent of the design.
"""

from __future__ import annotations

import hashlib
import io
import json
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import psycopg
import pytest

from causal.preparation import nodes
from causal.preparation.harness import PreparationDeps
from causal.runtime.failures import preparation_deps
from causal.shared import events, handoff, persistence
from causal.shared.canonical import content_hash
from causal.shared.contracts import ArtifactEnvelopeV1
from causal.shared.registry import load_artifact_type_registry
from tests.conftest import MIGRATIONS, requires_docker

pytestmark = requires_docker

ROOT = Path(__file__).resolve().parents[2]
REGISTRIES = ROOT / "registries"
REGISTRY = load_artifact_type_registry(REGISTRIES / "artifact-types.v1.json")
NOW = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)
METHODS = ("randomized_experiment", "aipw", "did", "sharp_rdd")
VERSIONS = {method: f"{method.replace('_', '-')}-pack.v1" for method in METHODS}
CATALOG, CAPACITY = "visualization-catalog.v1", "delivery-capacity.v1"
PREREPAIR = "frame_completeness"
COLUMNS: dict[str, tuple[str, ...]] = {
    "randomized_experiment": ("unit_id", "treat", "earn", "age"),
    "aipw": ("unit_id", "treat", "earn", "age"),
    "did": ("unit_id", "period", "grp", "treat", "earn", "age"),
    "sharp_rdd": ("unit_id", "score", "treat", "earn", "age")}
ROLES: dict[str, dict[str, str]] = {
    "randomized_experiment": {"unit_id": "unit_identifier", "treat": "treatment",
                              "earn": "outcome", "age": "precision_covariate"},
    "aipw": {"unit_id": "unit_identifier", "treat": "treatment", "earn": "outcome",
             "age": "precision_covariate"},
    "did": {"unit_id": "unit_identifier", "period": "time", "grp": "group",
            "treat": "treatment", "earn": "outcome", "age": "confounder_candidate"},
    "sharp_rdd": {"unit_id": "unit_identifier", "score": "running_variable",
                  "treat": "treatment", "earn": "outcome", "age": "precision_covariate"}}
KEYS: dict[str, tuple[str, ...]] = {"did": ("unit_id", "period")}
# The approved delivery-capacity cardinality vector every fixture design carries (SC §11).
CARDINALITIES: dict[str, int] = {"arms": 2, "contrasts": 1, "subgroups": 0, "cohorts": 0,
                                 "periods": 0, "event_times": 0, "cutoff_sides": 0, "series": 2,
                                 "evidence_items": 20}
STRUCTURE: dict[str, dict[str, str]] = {"did": {"adoption_time": "2"},
                                        "sharp_rdd": {"cutoff": "10"}}


def csv_for(method: str, blank: str = "", limit: int = 0) -> bytes:
    """One frozen source table per pack: the smallest frame its §12 structure gates accept."""
    if method == "did":
        rows = [[f"u{u}", str(p), str(u % 2), str(int(p > 1)), str(u * p), str(u + p)]
                for u in range(1, 5) for p in (1, 2)]
    elif method == "randomized_experiment":
        rows = [[f"u{u}", str(u % 2), str(u * 3), str(u + 1)] for u in range(1, 5)]
    elif method == "aipw":
        rows = [[f"u{u}", str(int(u > 10)), str(u * 3), str(u + 1)] for u in range(1, 21)]
    else:
        rows = [[f"u{u}", str(u), str(int(u >= 10)), str(u * 3), str(u + 1)]
                for u in range(1, 21)]
    columns = COLUMNS[method]
    if blank:
        rows[0][columns.index(blank)] = ""
    rows = rows[:limit] if limit else rows
    return ("\n".join([",".join(columns), *(",".join(row) for row in rows)]) + "\n").encode()


def event(analysis_id: str, name: str = "artifact.committed") -> events.OperationalEventV1:
    return events.build_event(
        occurred_at_utc=NOW, event_name=name, analysis_id=analysis_id,
        event_id=f"evt:{uuid.uuid4().hex}", stage=events.Stage.DESIGN,
        stage_run_id=f"dr:{analysis_id}:1", component_id="design-harness",
        component_version="0.1.0", required_eval_ids=("EV-P2-001",))


def make_deps(conn: Any, objects: Any, sink: io.StringIO) -> PreparationDeps:
    """The production wiring, over the docker stores and the frozen registries (T-019 §1.4)."""
    emitter = events.EventEmitter(sink)
    products = persistence.ProductStore(conn)
    held = SimpleNamespace(
        conn=conn, products=products, objects=objects, registry=REGISTRY, emitter=emitter,
        clock=lambda: NOW,
        committer=persistence.ArtifactCommitter(objects, products, REGISTRY, emitter))
    return preparation_deps(cast(Any, held), REGISTRIES, ROOT)


def ident(kind: str, analysis_id: str, payload: Mapping[str, Any]) -> dict[str, str]:
    """One artifact's deterministic identity, computed before its parents exist (D-031)."""
    digest = content_hash(dict(payload))
    return {"artifact_id": f"{kind.lower()}:{analysis_id}:{digest[:16]}", "content_hash": digest}


def ref(envelope: ArtifactEnvelopeV1) -> dict[str, str]:
    return {"artifact_id": envelope.artifact_id, "content_hash": envelope.content_hash}


def commit(deps: PreparationDeps, analysis_id: str, kind: str, payload: dict[str, Any],
           parents: tuple[ArtifactEnvelopeV1, ...] = ()) -> ArtifactEnvelopeV1:
    built = persistence.build_envelope(
        deps.registry, kind, payload, analysis_id=analysis_id,
        stage_run_id=f"dr:{analysis_id}:1", producer_version="0.1.0", parents=parents,
        created_at_utc=NOW)
    return deps.committer.commit(built, payload, event(analysis_id))


def approved_design(deps: PreparationDeps, analysis_id: str, method: str, data: bytes, *,
                    impute: Sequence[str] = (), drop_rules: Sequence[str] = (),
                    indicators: Sequence[str] = ("outcome_observed",),
                    roles: Mapping[str, str] | None = None,
                    contrasts: Sequence[str] = ("treated_vs_control",),
                    cardinalities: Mapping[str, int] | None = None) -> str:
    """Commit the approved PRD-002 chain the §4 gate reads, and return its DesignOutcome id.

    The role map, the approved primary contrasts, and the delivery-capacity cardinalities are
    overridable so PRD-004's own entry gate (T-026) can reuse this one honest chain.
    """
    pack = deps.packs.get(method, VERSIONS[method])
    deps.products.create_stage_run(f"dr:{analysis_id}:1", analysis_id, "design")
    locator = deps.objects.put_if_absent(hashlib.sha256(data).hexdigest(), data)
    roles, keys = roles or ROLES[method], KEYS.get(method, ("unit_id",))
    made = commit(deps, analysis_id, "QuestionRecord", {"question_text": "does it work?"})
    intake = commit(deps, analysis_id, "IntakeOutcome", {"status": "usable"}, (made,))
    selection = commit(deps, analysis_id, "TableSelection", {
        "resource_object_locator": locator, "logical_name": "frame.csv",
        "resource_sha256": hashlib.sha256(data).hexdigest()}, (intake, made))
    book = commit(deps, analysis_id, "DesignContextManifest", {"table": "frame.csv"}, (selection,))
    intent = commit(deps, analysis_id, "DesignIntent", {"question_kind": "causal"}, (book,))
    mapped = commit(deps, analysis_id, "MeasurementMap", {"links": [
        {"column_name": name, "concept_id": f"c:{name}"} for name in sorted(roles)]}, (intent,))
    context = commit(deps, analysis_id, "CausalContext", {"concept_ids": sorted(roles)}, (mapped,))
    ledger = commit(deps, analysis_id, "RoleLedger", {"claims": [
        {"role": role, "column_refs": [name]} for name, role in sorted(roles.items())]}, (context,))
    probe = commit(deps, analysis_id, "PreRepairFeasibilityReport", {
        "method_id": method, "results": [{"diagnostic_id": PREREPAIR}]}, (ledger,))
    capacity = {"status": "pass", "method_id": method, "cardinalities": dict(
        cardinalities or CARDINALITIES) | {"contrasts": len(contrasts)},
        "visualization_catalog_version": CATALOG, "capacity_registry_version": CAPACITY}
    design = commit(deps, analysis_id, "ExperimentDesign", {
        "selected_csv": ref(selection), "method_id": method, "unit": "participant",
        "method_pack_version": VERSIONS[method], "comparator": "the untreated", "estimand": "ate",
        "primary_contrasts": list(contrasts),
        "frame": {"treatment": "tr", "outcome": "out", "population": "pop", "timeframe": "tf"},
        "measurement_map": ref(mapped), "role_ledger": ref(ledger),
        "causal_context": ref(context), "capacity_check": ident(
            "DeliveryCapacityCheck", analysis_id, capacity),
        "visualization_catalog_version": CATALOG, "capacity_registry_version": CAPACITY,
        "registry_versions": {"schema": "experiment-design.v1"},
        "required_prerepair_diagnostics": [PREREPAIR]}, (ledger, probe))
    contract = commit(deps, analysis_id, "RunnableFrameContract", {
        "selected_csv": ref(selection), "experiment_design_hash": design.content_hash,
        "estimator_input_schema": pack.prepared_frame_schema_id,
        "output_grain": "one row per unit", "key_columns": list(keys), "eligibility_rules": [],
        "exclusion_reason_vocabulary": [rule for rule in pack.permitted_disposition_rule_ids
                                        if rule not in drop_rules],
        "imputation_permitted": list(impute), "imputation_forbidden": [],
        "required_missingness_indicators": list(indicators), "type_constraints": {},
        "required_final_diagnostics": list(pack.required_postrepair_diagnostic_ids),
        "deletion_impact_dimensions": list(pack.dimension_impact_dimensions),
        "method_structure": STRUCTURE.get(method, {})}, (design,))
    view = commit(deps, analysis_id, "CausalGraphView", {"nodes": sorted(roles)}, (contract,))
    checked = commit(deps, analysis_id, "DeliveryCapacityCheck", capacity, (view,))
    approval = commit(deps, analysis_id, "DesignApproval", {"decision": "approved"}, (design,))
    return commit(deps, analysis_id, "DesignOutcome", {
        "status": "approved", "approval": ref(approval), "causal_graph_view": ref(view),
        "experiment_design": ref(design), "runnable_frame_contract": ref(contract),
        "capacity_check": ref(checked)}, (intake,)).artifact_id


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
    _, deps, _, _ = stack
    outcome = approved_design(deps, f"an-{name}", method, csv_for(method, blank=blank),
                              impute=impute, drop_rules=drop)
    run = run_one(deps, f"an-{name}", outcome)
    assert (run.status, run.conflict_code) == ("design_conflict", code)


def test_a_frame_without_method_structure_is_not_runnable(stack: Stack) -> None:
    """§12: the retained rows lose the pack's minimum support, so the frame cannot run."""
    _, deps, _, _ = stack
    outcome = approved_design(deps, "an-thin", "aipw", csv_for("aipw", limit=4))
    run = run_one(deps, "an-thin", outcome)
    assert (run.status, run.error_code) == ("not_runnable", "method_support_lost")
