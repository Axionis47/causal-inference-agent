"""End-to-end design-harness runs over the T-008 intake fixture (T-013 §1.6; EV-P2-001..008)."""

from __future__ import annotations

import copy
import io
import json
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
import pytest

from causal.design import graph
from causal.design.capacity import load_capacity_registry
from causal.design.compile import load_task_table
from causal.design.contracts import REGISTRY_VERSION_KEYS
from causal.design.entry import PsycopgCatalogReader
from causal.design.packs import (
    METHOD_IDS,
    load_method_packs,
    load_requirement_templates,
    load_tool_registry,
)
from causal.design.semantics import COLUMN_CARD_SLOTS
from causal.design.validators import load_validation_rules
from causal.intake.catalog import CatalogStore
from causal.intake.contracts import IntakeSubmissionV1
from causal.intake.coordinator import IntakeCoordinator
from causal.intake.fields import load_field_classes
from causal.shared.contracts import ArtifactEnvelopeV1
from causal.shared.envelope import AgentTaskEnvelopeV1
from causal.shared.events import EventEmitter
from causal.shared.gateway import GatewayResultV1
from causal.shared.persistence import ArtifactCommitter, ObjectStore, ProductStore
from causal.shared.registry import load_artifact_type_registry
from tests.conftest import requires_docker
from tests.intake.conftest import CSV, FILES_RESPONSE, README, FrozenKaggleClient

pytestmark = requires_docker

ROOT = Path(__file__).resolve().parents[2]
REGISTRIES = ROOT / "registries"
REGISTRY = load_artifact_type_registry(REGISTRIES / "artifact-types.v1.json")
CLASSES = load_field_classes(REGISTRIES / "kaggle-field-classes.v1.json")
PACKS = load_method_packs(REGISTRIES / "method-packs.v1.json")
TEMPLATES = load_requirement_templates(REGISTRIES / "context-requirements.v1.json")
TOOLS = load_tool_registry(REGISTRIES / "design-tools.v1.json")
RULES = load_validation_rules(REGISTRIES / "design-validation-rules.v1.json")
TASKS = load_task_table(REGISTRIES / "design-tasks.v1.json")
CAPACITY = load_capacity_registry(REGISTRIES / "delivery-capacity.v1.json")
PACK = PACKS.get("randomized_experiment")
NOW = datetime(2026, 8, 25, 12, 0, 0, tzinfo=UTC)
TSV = CSV.replace(b",", b"\t")
FRAME = {"treatment": "c:treatment", "outcome": "c:outcome", "population": "applicants",
         "timeframe": "1975-1978"}
EDGE = {"edge_id": "e-1", "source_concept_id": "c:treatment", "target_concept_id": "c:outcome",
        "timeframe": "1975-1978", "mechanism_summary": "training raises skill and so earnings",
        "supporting_evidence_ids": [], "contrary_evidence_ids": [], "status": "hypothesis",
        "differing_alternative_ids": []}
TWO_FILES: dict[str, Any] = {"files": [
    copy.deepcopy(FILES_RESPONSE["files"][0]),  # type: ignore[index]
    copy.deepcopy(FILES_RESPONSE["files"][0]) | {"name": "psid.csv"}]}  # type: ignore[index]
needs_dot = pytest.mark.skipif(shutil.which("dot") is None, reason="graphviz `dot` is absent")


# --- canned model payloads, one builder per task kind ---------------------

def proposal(name: str, description: str, columns: list[str]) -> dict[str, Any]:
    return {"name": name, "description": description, "candidate_columns": columns}


def role(name: str, concept: str, columns: list[str], timing: str) -> dict[str, Any]:
    return {"role": name, "concept_id": concept, "column_refs": columns, "evidence_ids": [],
            "timing": timing, "graph_edge_ids": ["e-1"], "alternatives": [],
            "support_class": "direct_source_statement", "status": "evidenced",
            "methods": ["randomized_experiment"]}


def intent_payload(envelope: AgentTaskEnvelopeV1) -> dict[str, Any]:
    return {"schema_version": "design-intent.v1", "question_kind": "causal",
            "causal_claim": "the training programme raises earnings",
            "intended_decision": "whether to expand the programme",
            "treatment": proposal("treatment", "the training programme", ["group"]),
            "outcome": proposal("outcome", "earnings after the programme", ["earnings"]),
            "population": proposal("population", "enrolled applicants", ["unit_id"]),
            "comparator": proposal("comparator", "applicants not enrolled", ["group"]),
            "unit": proposal("unit", "one applicant", ["unit_id"]),
            "timeframe": proposal("timeframe", "the 1975-1978 window", []),
            "candidate_grain": "one row per applicant", "mandatory_concepts": [], "claims": []}


def card(column: str) -> dict[str, Any]:
    return {"schema_version": "column-semantic-card.v1", "table_name": "nsw.csv",
            "column_name": column, "display_name": column, "concept_id": f"c:{column}",
            "timing": "post_treatment" if column == "earnings" else "pre_treatment",
            "slots": {slot: {"value": column, "status": "hypothesis", "evidence_ids": []}
                      for slot in COLUMN_CARD_SLOTS},
            "claims": [], "alternatives": [], "conflicts": []}


def cards_payload(envelope: AgentTaskEnvelopeV1) -> dict[str, Any]:
    return {"items": [card(str(name)) for name in envelope.scope_ids]}


def roles_payload(envelope: AgentTaskEnvelopeV1) -> dict[str, Any]:
    return {"schema_version": "role-evidence.v1", "assigned_scope": list(envelope.scope_ids),
            "edge_hypotheses": [], "role_hypotheses": [], "competing_mechanisms": [],
            "claims": []}


def context_payload(envelope: AgentTaskEnvelopeV1) -> dict[str, Any]:
    return {"schema_version": "causal-context.v1", "frame": FRAME,
            "concept_ids": ["c:treatment", "c:outcome", "c:unit"], "edges": [EDGE],
            "alternatives": [], "selection_notes": "one mechanism, no live alternative",
            "claims": []}


def ledger_payload(envelope: AgentTaskEnvelopeV1) -> dict[str, Any]:
    return {"schema_version": "role-ledger.v1", "frame": FRAME, "claims": [
        role("treatment", "c:treatment", ["group"], "pre_treatment"),
        role("outcome", "c:outcome", ["earnings"], "post_treatment"),
        role("unit_identifier", "c:unit", ["unit_id"], "pre_treatment")]}


def design_payload(envelope: AgentTaskEnvelopeV1) -> dict[str, Any]:
    parents: Any = envelope.payload["parents"]
    return {"schema_version": "experiment-design.v1",
            "causal_question": "Does the programme raise earnings?",
            "intended_decision": "whether to expand the programme",
            "selected_csv": parents["TableSelection"], "method_id": "randomized_experiment",
            "method_pack_version": PACK.pack_version, "frame": FRAME,
            "rejected_methods": {name: "required roles are absent" for name in METHOD_IDS
                                 if name != "randomized_experiment"},
            "comparator": "applicants not enrolled", "unit": "one applicant", "estimand": "itt",
            "measurement_map": parents["MeasurementMap"],
            "causal_context": parents["CausalContext"], "role_ledger": parents["RoleLedger"],
            "assumptions": ["assignment was randomised"],
            "identification_risks": ["differential attrition"], "eligibility_rules": [],
            "mandatory_repair_boundaries": [], "forbidden_repair_boundaries": [],
            "imputation_eligible_columns": [], "imputation_forbidden_columns": ["group",
                                                                                "earnings"],
            "deletion_impact_dimensions": list(PACK.deletion_impact_dimensions),
            "invalidation_conditions": ["randomisation is broken"],
            "required_prerepair_diagnostics": ["arm_counts", "assignment_unit_uniqueness"],
            "required_postrepair_diagnostics": list(PACK.required_postrepair_diagnostic_ids),
            "required_visual_evidence": list(PACK.required_visual_evidence_ids),
            "primary_contrasts": ["treated vs control"], "multiplicity_policy": None,
            "capacity_check": None, "sensitivity_requirements": [],
            "visualization_catalog_version": "visualization-catalog.v1",
            "capacity_registry_version": "delivery-capacity.v1",
            "registry_versions": dict(graph.REGISTRY_VERSIONS)}


def contract_payload(envelope: AgentTaskEnvelopeV1) -> dict[str, Any]:
    parents: Any = envelope.payload["parents"]
    held: Any = envelope.payload["experiment_design"]
    return {"schema_version": "runnable-frame-contract.v1",
            "selected_csv": parents["TableSelection"], "output_grain": "one row per applicant",
            "key_columns": ["unit_id"],
            "required_roles": ["treatment", "outcome", "unit_identifier"],
            "allowed_roles": [], "forbidden_roles": [], "type_constraints": {},
            "uniqueness_constraints": ["unit_id is unique"], "eligibility_rules": [],
            "exclusion_reason_vocabulary": list(PACK.eligibility_rule_vocabulary),
            "treatment_missingness_rule": "drop the row",
            "outcome_missingness_rule": "drop the row",
            "method_structure": {"design": "randomized_experiment"}, "imputation_permitted": [],
            "imputation_forbidden": ["group", "earnings", "unit_id"],
            "required_missingness_indicators": [],
            "deletion_impact_dimensions": list(PACK.deletion_impact_dimensions),
            "revision_required_conditions": ["randomisation is broken"],
            "feasibility_gates": ["arm_counts >= 2"],
            "required_final_diagnostics": list(PACK.required_postrepair_diagnostic_ids),
            "estimator_input_schema": PACK.reserved_estimator_id,
            "experiment_design_hash": held["content_hash"]}


BUILDERS = {"intent": intent_payload, "semantic_batch": cards_payload,
            "role_evidence": roles_payload, "causal_context": context_payload,
            "role_ledger": ledger_payload, "experiment_design": design_payload,
            "runnable_frame_contract": contract_payload}
ARTIFACTS = {"intent": ("DesignIntent", "design-intent.v1"),
             "semantic_batch": ("ColumnSemanticCard", "column-semantic-card.v1"),
             "role_evidence": ("RoleEvidence", "role-evidence.v1"),
             "causal_context": ("CausalContext", "causal-context.v1"),
             "role_ledger": ("RoleLedger", "role-ledger.v1"),
             "experiment_design": ("ExperimentDesign", "experiment-design.v1"),
             "runnable_frame_contract": ("RunnableFrameContract", "runnable-frame-contract.v1")}
TIMING_REQUIREMENT: dict[str, Any] = {
    "requirement_id": "column.measurement_timing", "registry_version": "context-requirements.v1",
    "scope_id": "earnings", "decisions_blocked": ["role_assignment"], "attempted_evidence": [
        {"evidence_id": "ev:doc/readme.md", "availability_status": "not_offered"}],
    **{key: value for key, value in dict(TEMPLATES["column.measurement_timing"]).items()
       if key != "requirement_id"}}


def script_key(envelope: AgentTaskEnvelopeV1) -> str:
    """The canned-response key: the task kind, or the artifact each two-step task builds."""
    return (str(envelope.scope_ids[0])
            if envelope.task_kind in ("causal_synthesis", "method_design")
            else envelope.task_kind)


class ScriptedGateway:
    """A gateway whose every reply is a valid `AgentTaskResultV1` body built from fixtures."""

    def __init__(self, overrides: Mapping[str, Sequence[Any]] | None = None) -> None:
        self.overrides: dict[str, list[Any]] = {
            key: list(value) for key, value in (overrides or {}).items()}
        self.calls: list[AgentTaskEnvelopeV1] = []

    def invoke(self, envelope: AgentTaskEnvelopeV1, prompt: str,
               response_schema: dict[str, object]) -> GatewayResultV1:
        self.calls.append(envelope)
        key = script_key(envelope)
        queue = self.overrides.get(key)
        body = queue.pop(0) if queue else self.default(key, envelope)
        text = json.dumps(body)
        return GatewayResultV1(text=text, parsed=body, token_usage={}, attempts=1, seed=1)

    def default(self, key: str, envelope: AgentTaskEnvelopeV1,
                **over: Any) -> dict[str, Any]:
        artifact_type, schema_version = ARTIFACTS[key]
        return {"envelope_id": envelope.envelope_id, "schema_version": "agent-task-result.v1",
                "task_id": envelope.task_id, "status": "complete",
                "artifact_type": artifact_type, "artifact_schema_version": schema_version,
                "parent_artifact_ids": [], "payload": BUILDERS[key](envelope), "claims": [],
                "missing_requirements": [], "conflicts": [], "warnings": [], "evidence_ids": [],
                "tool_receipts": [], "output_hash": None,
                "validation_target": f"{schema_version}-validator"} | over


# --- fixtures -------------------------------------------------------------

def run_intake(conn: Any, objects: ObjectStore, files: dict[str, bytes] | None = None,
               declared: dict[str, Any] | None = None) -> Any:
    """One real intake run over the frozen Kaggle fixtures (see tests/intake/test_coordinator)."""
    emitter = EventEmitter(io.StringIO())
    products = ProductStore(conn)
    coordinator = IntakeCoordinator(
        FrozenKaggleClient(files=files, files_response=declared),
        ArtifactCommitter(objects, products, REGISTRY, emitter),
        products, CatalogStore(conn), objects, REGISTRY, CLASSES, emitter, clock=lambda: NOW)
    return coordinator.run(IntakeSubmissionV1(
        schema_version="intake-submission.v1", question_text="Does the programme raise earnings?",
        context_text=None, kaggle_ref="lalonde/nsw", idempotency_key="design-key"))


def make_deps(conn: Any, objects: ObjectStore, gateway: ScriptedGateway,
              sink: io.StringIO) -> graph.DesignDeps:
    """One `DesignDeps` over the dockerized stores and the frozen registries."""
    emitter = EventEmitter(sink)
    products = ProductStore(conn)
    return graph.DesignDeps(
        conn=conn, catalog=PsycopgCatalogReader(conn),
        committer=ArtifactCommitter(objects, products, REGISTRY, emitter), products=products,
        objects=objects, registry=REGISTRY, gateway=gateway, emitter=emitter, clock=lambda: NOW,
        checkpointer=graph.build_checkpointer(sibling(conn)), packs=PACKS, templates=TEMPLATES,
        tool_registry=TOOLS, task_table=TASKS, rules=RULES, capacity_registry=CAPACITY,
        prompts_root=ROOT, repo_root=ROOT)


def sibling(conn: Any) -> Any:
    """A second connection to the same database: the checkpointer owns its own (pipelines)."""
    info = conn.info
    return psycopg.connect(host=info.host, port=info.port, dbname=info.dbname, user=info.user,
                           password=info.password, autocommit=True)


def payload_of(conn: Any, objects: ObjectStore, artifact_id: str) -> dict[str, Any]:
    envelope: ArtifactEnvelopeV1 = ProductStore(conn).load_envelope(artifact_id)
    body = json.loads(objects.get(envelope.payload_locator))
    assert isinstance(body, dict)
    return body


def start(conn: Any, objects: ObjectStore, gateway: ScriptedGateway | None = None,
          files: dict[str, bytes] | None = None, declared: dict[str, Any] | None = None,
          thread: str = "gt:design:1",
          ) -> tuple[graph.DesignDeps, graph.DesignRunResult, io.StringIO]:
    """Run intake, then one design revision on a fresh thread."""
    intake = run_intake(conn, objects, files, declared)
    sink = io.StringIO()
    deps = make_deps(conn, objects, gateway or ScriptedGateway(), sink)
    result = graph.run_design(
        deps, analysis_id=intake.analysis_id, thread_id=thread,
        intake_outcome_artifact_id=str(intake.outcome_artifact_id))
    return deps, result, sink


# --- tests ----------------------------------------------------------------

class TestApprovedRun:
    @needs_dot
    def test_full_run_reaches_an_approved_outcome_and_handoff(
        self, conn: Any, object_store: ObjectStore
    ) -> None:
        deps, opened, sink = start(conn, object_store)
        assert opened.status == graph.NEEDS_USER_INPUT
        assert opened.interrupt_kind == "approval"
        done = graph.resume_design(deps, thread_id=opened.thread_id, resume_value=approval(opened))
        assert done.status == "approved"
        outcome = payload_of(conn, object_store, str(done.outcome_artifact_id))
        assert outcome["status"] == "approved"
        assert outcome["counts"]["columns_carded"] == 3
        manifest = graph.open_design_handoff(
            deps, done.analysis_id, str(done.outcome_artifact_id), "sr:prep")
        kinds = [ProductStore(conn).load_envelope(row.artifact_id).artifact_type
                 for row in manifest.entries]
        assert kinds == ["TableSelection", "ExperimentDesign", "RunnableFrameContract",
                         "DeliveryCapacityCheck"]
        assert done.handoff_id == manifest.handoff_id
        assert '"event_name":"stage.completed"' in sink.getvalue()
        rows = conn.execute(
            "SELECT state, method_id FROM design.design_runs WHERE analysis_id = %s",
            (done.analysis_id,)).fetchone()
        assert rows == ("completed", "randomized_experiment")

    def test_full_run_with_a_stubbed_renderer(
        self, conn: Any, object_store: ObjectStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The no-binary path: a stub returns a real validated view, nothing else changes."""
        stub_renderer(monkeypatch)
        deps, opened, _ = start(conn, object_store)
        done = graph.resume_design(deps, thread_id=opened.thread_id, resume_value=approval(opened))
        assert done.status == "approved" and done.handoff_id is not None
        view = payload_of(conn, object_store, view_id(conn))
        assert view["renderer_version"] == "graphviz-stub" and len(view["nodes"]) == 3
        tasks = conn.execute(
            "SELECT count(*) FROM design.design_tasks WHERE analysis_id = %s",
            (done.analysis_id,)).fetchone()
        assert tasks is not None and tasks[0] == 7


def approval(opened: graph.DesignRunResult) -> dict[str, Any]:
    """The typed approval decision answering exactly the open interrupt (PRD-002 §11.1)."""
    return {"schema_version": "design-approval-decision.v1",
            "interrupt_id": str(opened.interrupt_artifact_id),
            "expected_interrupt_hash": str(opened.interrupt_hash),
            "expected_revision": opened.design_revision, "decision": "approved",
            "approved_artifacts": [{"artifact_id": str(opened.interrupt_artifact_id),
                                    "content_hash": str(opened.interrupt_hash)}],
            "change_requests": [], "idempotency_key": "idem-approve"}


def view_id(conn: Any) -> str:
    row = conn.execute("SELECT artifact_id FROM design.causal_graph_views").fetchone()
    assert row is not None
    return str(row[0])


def stub_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the Graphviz call with a validated `CausalGraphViewV1`, not with a skip."""
    from causal.design.frame import CausalGraphViewV1
    from causal.design.renderer import LEGEND_TEXT, build_graph_spec

    def render(**kwargs: Any) -> CausalGraphViewV1:
        nodes, edges, _, dot = build_graph_spec(
            kwargs["context"], kwargs["ledger"], kwargs["measurement_map"])
        table = "\n".join(f"{node.concept_id} | {node.status.value}" for node in nodes)
        return CausalGraphViewV1(
            parents=kwargs["parents"], nodes=nodes, edges=edges, selected_alternative_id=None,
            layout_direction="TB", renderer_profile="design-graph-renderer.v1",
            legend_text=LEGEND_TEXT, disclosure_text="one graph, no alternatives",
            spec_hash="0" * 64, svg=f"<svg><!--{len(dot)}--></svg>",
            accessible_summary=f"{len(nodes)} concepts and {len(edges)} edges",
            node_edge_table=table, renderer_version="graphviz-stub",
            theme_version="design-graph-theme.v1", validator_version="design-validators.v1",
            validation_status="validated")

    monkeypatch.setattr(graph, "render_causal_graph", render)


class TestInterrupts:
    def test_two_csvs_interrupt_then_resume_with_a_decision(
        self, conn: Any, object_store: ObjectStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub_renderer(monkeypatch)
        files = {"nsw.csv": CSV, "psid.csv": CSV, "readme.md": README}
        deps, opened, sink = start(conn, object_store, files=files, declared=TWO_FILES)
        assert opened.status == graph.NEEDS_USER_INPUT
        assert opened.interrupt_kind == "table_selection"
        chosen = graph.resume_design(deps, thread_id=opened.thread_id, resume_value={
            "schema_version": "table-selection-decision.v1",
            "interrupt_id": str(opened.interrupt_artifact_id),
            "expected_interrupt_hash": str(opened.interrupt_hash),
            "expected_revision": 1, "selected_table": "psid.csv",
            "idempotency_key": "idem-table"})
        assert chosen.interrupt_kind == "approval", sink.getvalue()[-400:]
        selection = payload_of(conn, object_store, selected_id(conn))
        assert selection["logical_name"] == "psid.csv"
        assert selection["selection_source"] == "user_decision"

    def test_resume_from_a_second_coordinator_over_the_same_thread(
        self, conn: Any, object_store: ObjectStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub_renderer(monkeypatch)
        _, opened, _ = start(conn, object_store)
        assert opened.status == graph.NEEDS_USER_INPUT
        fresh = make_deps(conn, object_store, ScriptedGateway(), io.StringIO())
        done = graph.resume_design(
            fresh, thread_id=opened.thread_id, resume_value=approval(opened))
        assert done.status == "approved" and done.thread_id == opened.thread_id

    def test_declined_approval_closes_the_revision_without_a_handoff(
        self, conn: Any, object_store: ObjectStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub_renderer(monkeypatch)
        deps, opened, _ = start(conn, object_store)
        done = graph.resume_design(deps, thread_id=opened.thread_id, resume_value={
            **approval(opened), "decision": "declined", "approved_artifacts": [],
            "idempotency_key": "idem-decline"})
        assert done.status == "declined" and done.handoff_id is None


def selected_id(conn: Any) -> str:
    row = conn.execute(
        "SELECT artifact_id FROM causal.artifacts WHERE artifact_type = 'TableSelection'"
    ).fetchone()
    assert row is not None
    return str(row[0])


class TestAskGate:
    def test_blocking_requirement_asks_once_and_then_continues(
        self, conn: Any, object_store: ObjectStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub_renderer(monkeypatch)
        deps, opened, sink = start(conn, object_store, gateway=asking_gateway())
        assert opened.interrupt_kind == "clarification"
        assert '"event_name":"user_interrupt.created"' in sink.getvalue()
        answered = graph.resume_design(deps, thread_id=opened.thread_id, resume_value={
            "schema_version": "user-context-answer.v1", "packet_id": "qp:1:1",
            "answers": [{"question_id": "q:column.measurement_timing",
                         "answer_kind": "value", "value": "post_treatment"}],
            "provenance": "user"})
        assert answered.interrupt_kind == "approval"
        done = graph.resume_design(
            deps, thread_id=answered.thread_id, resume_value=approval(answered))
        outcome = payload_of(conn, object_store, str(done.outcome_artifact_id))
        assert outcome["clarification_rounds_used"] == 1
        state = conn.execute(
            "SELECT state FROM design.context_requirements WHERE requirement_id = %s",
            ("column.measurement_timing",)).fetchone()
        assert state == ("resolved",)


    def test_two_unknown_rounds_end_the_revision_as_needs_context(
        self, conn: Any, object_store: ObjectStore
    ) -> None:
        deps, first, _ = start(conn, object_store, gateway=asking_gateway())
        second = graph.resume_design(
            deps, thread_id=first.thread_id, resume_value=unknown_answer("qp:1:1"))
        assert second.interrupt_kind == "clarification"
        done = graph.resume_design(
            deps, thread_id=second.thread_id, resume_value=unknown_answer("qp:1:2"))
        assert done.status == "needs_context"
        outcome = payload_of(conn, object_store, str(done.outcome_artifact_id))
        assert outcome["clarification_rounds_used"] == 2
        assert outcome["open_requirement_ids"] == ["column.measurement_timing"]


def unknown_answer(packet_id: str) -> dict[str, Any]:
    """`unknown` is always offered; for an `ask_user` requirement it leaves the row open."""
    return {"schema_version": "user-context-answer.v1", "packet_id": packet_id,
            "answers": [{"question_id": "q:column.measurement_timing",
                         "answer_kind": "unknown", "value": None}], "provenance": "user"}


def asking_gateway() -> ScriptedGateway:
    """An intent reply that raises one blocking, user-answerable requirement."""
    gateway = ScriptedGateway()
    original = gateway.default

    def default(key: str, envelope: AgentTaskEnvelopeV1, **over: Any) -> dict[str, Any]:
        extra = {"missing_requirements": [TIMING_REQUIREMENT]} if key == "intent" else {}
        return original(key, envelope, **extra | over)

    gateway.default = default  # type: ignore[method-assign]
    return gateway


class TestCorrectionLoop:
    def test_schema_failure_corrects_and_the_second_reply_is_committed(
        self, conn: Any, object_store: ObjectStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub_renderer(monkeypatch)
        gateway = ScriptedGateway({"semantic_batch": [{"not": "a result"}]})
        deps, opened, sink = start(conn, object_store, gateway=gateway)
        assert '"event_name":"agent.schema_failed"' in sink.getvalue()
        assert '"event_name":"agent.correction_requested"' in sink.getvalue()
        done = graph.resume_design(deps, thread_id=opened.thread_id, resume_value=approval(opened))
        outcome = payload_of(conn, object_store, str(done.outcome_artifact_id))
        assert outcome["status"] == "approved" and outcome["counts"]["corrections"] == 1
        assert [call.task_kind for call in gateway.calls].count("semantic_batch") == 2

    def test_exhausted_corrections_end_the_revision(
        self, conn: Any, object_store: ObjectStore
    ) -> None:
        broken = [{"not": "a result"}] * 3
        gateway = ScriptedGateway({"semantic_batch": broken})
        _, done, sink = start(conn, object_store, gateway=gateway)
        assert done.status == "failed" and done.error_code == "correction_exhausted"
        assert '"event_name":"retry.exhausted"' in sink.getvalue()
        outcome = payload_of(conn, object_store, str(done.outcome_artifact_id))
        assert outcome["error_code"] == "correction_exhausted"
        assert outcome["experiment_design"] is None


class TestRefusal:
    def test_no_admitted_csv_refuses_without_a_handoff(
        self, conn: Any, object_store: ObjectStore
    ) -> None:
        _, done, _ = start(conn, object_store, files={"nsw.tsv": TSV, "readme.md": README})
        assert done.status == "refused" and done.refusal_code == "NO_ANALYSIS_CSV"
        assert done.handoff_id is None
        outcome = payload_of(conn, object_store, str(done.outcome_artifact_id))
        assert outcome["refusal_code"] == "NO_ANALYSIS_CSV"
        assert outcome["causal_graph_view"] is None


def test_state_carries_only_allowlisted_scalars(
    conn: Any, object_store: ObjectStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§19.1: no payload, prompt, or model text ever reaches a checkpoint."""
    stub_renderer(monkeypatch)
    deps, opened, _ = start(conn, object_store)
    graph.resume_design(deps, thread_id=opened.thread_id, resume_value=approval(opened))
    serde = deps.checkpointer.serde
    assert getattr(serde, "pickle_fallback", True) is False
    assert getattr(serde, "_allowed_msgpack_modules", "permissive") is None
    snapshot = graph.build_graph(deps).get_state(
        {"configurable": {"thread_id": opened.thread_id}}).values
    assert set(snapshot) <= set(graph.DesignState.__annotations__)
    for value in snapshot.values():
        assert isinstance(value, str | int | list | dict)
    assert "training programme" not in json.dumps(snapshot)


def test_registry_versions_are_the_closed_manifest_key_set() -> None:
    assert set(graph.REGISTRY_VERSIONS) == set(REGISTRY_VERSION_KEYS)
