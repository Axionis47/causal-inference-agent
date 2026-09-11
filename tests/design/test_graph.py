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
from causal.design.contracts import REGISTRY_VERSION_KEYS, DesignIntentV1
from causal.design.entry import PsycopgCatalogReader
from causal.design.packs import (
    METHOD_IDS,
    load_method_packs,
    load_requirement_templates,
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
from tests.infrastructure import requires_docker
from tests.intake.conftest import CSV, FILES_RESPONSE, README, FrozenKaggleClient

pytestmark = requires_docker

ROOT = Path(__file__).resolve().parents[2]
REGISTRIES = ROOT / "registries"
REGISTRY = load_artifact_type_registry(REGISTRIES / "artifact-types.v1.json")
CLASSES = load_field_classes(REGISTRIES / "kaggle-field-classes.v1.json")
PACKS = load_method_packs(REGISTRIES / "method-packs.v1.json")
TEMPLATES = load_requirement_templates(REGISTRIES / "context-requirements.v1.json")
RULES = load_validation_rules(REGISTRIES / "design-validation-rules.v1.json")
TASKS = load_task_table(REGISTRIES / "design-tasks.v1.json")
CAPACITY = load_capacity_registry(REGISTRIES / "delivery-capacity.v1.json")
NOW = datetime(2026, 8, 25, 12, 0, 0, tzinfo=UTC)
STUDY_CONTEXT = (
    "The program used randomized assignment. The target estimand is intention-to-treat "
    "(ITT), comparing assigned training to control. Each row is one applicant."
)
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
    return {"role": name, "concept_id": concept, "column_refs": columns,
            "evidence_ids": ["ev:doc/readme.md"],
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
            "candidate_grain": "one_row_per_unit",
            "source_interpretations": [{
                "fact_key": "grain", "value": "one_row_per_unit",
                "evidence_id": "ua:context/text", "verbatim_excerpt": STUDY_CONTEXT,
                "relation": "direct"}],
            "mandatory_concepts": []}


def card(column: str) -> dict[str, Any]:
    return {"schema_version": "column-semantic-card.v1", "table_name": "nsw.csv",
            "column_name": column, "display_name": column, "concept_id": f"c:{column}",
            "timing": "post_treatment" if column == "earnings" else "pre_treatment",
            "slots": {slot: {"value": column, "status": "hypothesis", "evidence_ids": []}
                      for slot in COLUMN_CARD_SLOTS},
            "alternatives": [], "conflicts": []}


def cards_payload(envelope: AgentTaskEnvelopeV1) -> dict[str, Any]:
    return {"items": [card(str(name)) for name in envelope.scope_ids]}


def roles_payload(envelope: AgentTaskEnvelopeV1) -> dict[str, Any]:
    return {"schema_version": "role-evidence.v1", "assigned_scope": list(envelope.scope_ids),
            "edge_hypotheses": [], "role_hypotheses": [], "competing_mechanisms": []}


def context_payload(envelope: AgentTaskEnvelopeV1) -> dict[str, Any]:
    return {"schema_version": "causal-context.v1", "frame": FRAME,
            "concept_ids": ["c:treatment", "c:outcome", "c:unit"], "edges": [EDGE],
            "alternatives": [], "selection_notes": "one mechanism, no live alternative"}


def ledger_payload(envelope: AgentTaskEnvelopeV1) -> dict[str, Any]:
    return {"schema_version": "role-ledger.v1", "frame": FRAME, "claims": [
        role("treatment", "c:treatment", ["group"], "concurrent"),
        role("outcome", "c:outcome", ["earnings"], "post_treatment"),
        role("unit_identifier", "c:unit", ["unit_id"], "pre_treatment")]}


def method_payload(envelope: AgentTaskEnvelopeV1) -> dict[str, Any]:
    source = "ua:context/text"
    observed = [str(row["diagnostic_result_id"])
                for row in envelope.payload.get("diagnostic_results", [])]
    return {"schema_version": "agent-design-proposal.v2",
            "assignment_mechanism": "randomized", "requested_estimand": "itt",
            "comparator": "control", "source_interpretations": [
                {"fact_key": fact_key, "value": value, "evidence_id": source,
                 "verbatim_excerpt": STUDY_CONTEXT, "relation": "direct"}
                for fact_key, value in (("assignment_mechanism", "randomized"),
                                        ("estimand", "itt"),
                                        ("comparator", "control"))],
            "ranked_method_ids": list(METHOD_IDS), "method_facts": [],
            "optional_assumption_ids": [], "optional_risk_ids": ["rct.noncompliance_risk"], "optional_sensitivity_ids": [],
            "requested_diagnostic_ids": [] if observed else ["arm_counts"],
            "diagnostic_assessments": [{"diagnostic_result_id": name,
                                        "judgment": "supports"} for name in observed]}


BUILDERS = {"intent": intent_payload, "semantic_batch": cards_payload,
            "role_evidence": roles_payload, "causal_context": context_payload,
            "role_ledger": ledger_payload, "agent_design_proposal": method_payload}
ARTIFACTS = {"intent": ("DesignIntent", "design-intent.v1"), "semantic_batch": ("ColumnSemanticCard", "column-semantic-card.v1"),
             "role_evidence": ("RoleEvidence", "role-evidence.v1"), "causal_context": ("CausalContext", "causal-context.v1"),
             "role_ledger": ("RoleLedger", "role-ledger.v1"),
             "agent_design_proposal": ("AgentDesignProposal", "agent-design-proposal.v2")}
def requirement(name: str, scope: str, blocked: str) -> dict[str, Any]:
    return {"requirement_id": name, "registry_version": "context-requirements.v1",
            "scope_id": scope, "decisions_blocked": [blocked], "attempted_evidence": [],
            **TEMPLATES[name].model_dump(
                mode="json", exclude={"requirement_id", "accepted_fact"})}


TIMING_REQUIREMENT = requirement("column.measurement_timing", "earnings", "role_assignment")
GRAIN_REQUIREMENT = requirement("design.table_grain", "nsw.csv", "method_eligibility")


def script_key(envelope: AgentTaskEnvelopeV1) -> str:
    """The canned-response key: the task kind, or the artifact each two-step task builds."""
    return (str(envelope.scope_ids[0])
            if envelope.task_kind in ("causal_context", "role_ledger", "method_design")
            else envelope.task_kind)


class ScriptedGateway:
    """A gateway whose every reply is a valid `AgentTaskResultV1` body built from fixtures."""

    def __init__(self, overrides: Mapping[str, Sequence[Any]] | None = None) -> None:
        self.overrides: dict[str, list[Any]] = {
            key: list(value) for key, value in (overrides or {}).items()}
        self.calls: list[AgentTaskEnvelopeV1] = []
        self.schemas: dict[str, Any] = {}
        self.prompts: dict[str, str] = {}

    def invoke(self, envelope: AgentTaskEnvelopeV1, prompt: str,
               response_schema: dict[str, object]) -> GatewayResultV1:
        self.calls.append(envelope)
        key = script_key(envelope)
        self.schemas[key] = response_schema
        self.prompts[key] = prompt
        queue = self.overrides.get(key)
        body = queue.pop(0) if queue else self.default(key, envelope)
        text = json.dumps(body)
        return GatewayResultV1(text=text, parsed=body, token_usage={}, attempts=1, seed=1)

    def default(self, key: str, envelope: AgentTaskEnvelopeV1,
                **over: Any) -> dict[str, Any]:
        return {"status": "complete", "payload": BUILDERS[key](envelope),
                "missing_requirements": [], "conflicts": [], "warnings": []} | over


class RejectedRequirementGateway(ScriptedGateway):
    """A corrected draft must not leave its rejected requirement in durable state."""

    def default(self, key: str, envelope: AgentTaskEnvelopeV1,
                **over: Any) -> dict[str, Any]:
        if key == "role_ledger" and envelope.attempt_id.endswith(":1"):
            over["missing_requirements"] = [TIMING_REQUIREMENT]
        return super().default(key, envelope, **over)


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
        context_text=STUDY_CONTEXT, kaggle_ref="lalonde/nsw", idempotency_key="design-key"))


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
        task_table=TASKS, rules=RULES, capacity_registry=CAPACITY,
        prompts_root=ROOT)


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
    def test_full_run_reaches_an_approved_outcome_and_handoff(self, conn: Any, object_store: ObjectStore) -> None:
        gateway = ScriptedGateway()
        deps, opened, sink = start(conn, object_store, gateway)
        assert opened.status == graph.NEEDS_USER_INPUT
        assert opened.interrupt_kind == "approval"
        done = graph.resume_design(deps, thread_id=opened.thread_id, resume_value=approval(opened))
        assert done.status == "approved"
        outcome = payload_of(conn, object_store, str(done.outcome_artifact_id))
        assert outcome["status"] == "approved"
        manifest = graph.open_design_handoff(
            deps, done.analysis_id, str(done.outcome_artifact_id), "sr:prep")
        kinds = [ProductStore(conn).load_envelope(row.artifact_id).artifact_type
                 for row in manifest.entries]
        assert kinds == ["TableSelection", "CompiledDesign", "DiagnosticReport",
                         "CapacityReport", "DesignReviewBundle", "DesignApproval"]
        assert done.handoff_id == manifest.handoff_id
        events = sink.getvalue()
        assert all(name in events for name in ("agent.diagnostic_requested", "diagnostic.completed", "agent.design_revised"))
        method_calls = [call for call in gateway.calls if call.task_kind == "method_design"]
        assert len(method_calls) == 2 and all(call.allowed_tool_ids == ("run_statistical_diagnostic",) for call in method_calls)
        observed = method_calls[1].payload["diagnostic_results"][0]
        assert observed["registered_diagnostic_id"] == "arm_counts"
        assert str(observed["diagnostic_result_id"]).startswith("dr:arm_counts:")
        observation_id = str(conn.execute(
            "SELECT artifact_id FROM causal.artifacts WHERE analysis_id=%s "
            "AND artifact_type='DiagnosticObservationSet'", (done.analysis_id,)).fetchone()[0])
        proposals = conn.execute(
            "SELECT artifact_id FROM causal.artifacts WHERE analysis_id=%s "
            "AND artifact_type='AgentDesignProposal' ORDER BY created_at_utc, artifact_id",
            (done.analysis_id,)).fetchall()
        committed = [ProductStore(conn).load_envelope(str(row[0])) for row in proposals]
        assert any(observation_id in {ref.artifact_id for ref in item.parent_artifacts}
                   for item in committed)
        assert '"event_name":"stage.completed"' in events
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
        assert tasks is not None and tasks[0] == 11


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
    from causal.design.renderer import LEGEND_TEXT, CausalGraphViewV1, build_graph_spec

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
    def test_requirement_from_a_rejected_draft_never_reaches_the_gate(
        self, conn: Any, object_store: ObjectStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub_renderer(monkeypatch)
        _, opened, _ = start(conn, object_store, gateway=RejectedRequirementGateway())
        assert opened.interrupt_kind == "approval"
        rows = conn.execute(
            "SELECT requirement_id FROM design.context_requirements WHERE requirement_id = %s",
            ("column.measurement_timing",)).fetchall()
        assert rows == []

    def test_compiler_proven_fact_settles_an_earlier_model_requirement(
        self, conn: Any, object_store: ObjectStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub_renderer(monkeypatch)
        _, opened, _ = start(conn, object_store, gateway=asking_gateway(GRAIN_REQUIREMENT))
        assert opened.interrupt_kind == "approval"
        row = conn.execute("SELECT state FROM design.context_requirements WHERE requirement_id = %s",
                           ("design.table_grain",)).fetchone()
        assert row == ("resolved",)

    def test_blocking_requirement_asks_once_and_then_continues(
        self, conn: Any, object_store: ObjectStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub_renderer(monkeypatch)
        deps, opened, sink = start(conn, object_store, gateway=asking_gateway())
        assert opened.interrupt_kind == "clarification"
        assert all(call.task_kind != "method_design" for call in deps.gateway.calls)
        attempted = conn.execute(
            "SELECT attempted_evidence FROM design.context_requirements"
            " WHERE requirement_id = 'column.measurement_timing'"
        ).fetchone()
        assert attempted is not None
        availability = {row["evidence_id"]: row["availability_status"]
                        for row in attempted[0]}
        assert availability["ev:doc/readme.md"] == "evidenced"
        assert "not_offered" not in availability.values()
        assert '"event_name":"user_interrupt.created"' in sink.getvalue()
        answered = graph.resume_design(deps, thread_id=opened.thread_id, resume_value={
            "schema_version": "user-context-answer.v1", "packet_id": "qp:1:1",
            "answers": [{"question_id": "q:column.measurement_timing", "answer_kind": "value",
                         "value": "earnings=post_treatment"}],
            "provenance": "user"})
        assert answered.interrupt_kind == "approval"
        semantic_calls = [call for call in deps.gateway.calls
                          if call.task_kind == "semantic_batch"]
        assert sum("earnings" in call.scope_ids for call in semantic_calls) == 2
        resumed = next(call for call in semantic_calls
                       if any(item.startswith("ua:usercontextanswer:")
                              for item in call.allowed_evidence_ids))
        answer_evidence = [item for item in resumed.allowed_evidence_ids
                           if item.startswith("ua:usercontextanswer:")]
        assert len(answer_evidence) == 1
        assert answer_evidence[0] in deps.gateway.prompts["semantic_batch"]
        assert any(row["requirement_id"] == "column.measurement_timing"
                   and row["scope_id"] == "earnings" and row["value"] == "post_treatment"
                   for row in resumed.payload["prerequisite_context"]["accepted_facts"])
        done = graph.resume_design(
            deps, thread_id=answered.thread_id, resume_value=approval(answered))
        outcome = payload_of(conn, object_store, str(done.outcome_artifact_id))
        assert outcome["status"] == "approved"
        state = conn.execute(
            "SELECT state FROM design.context_requirements WHERE requirement_id = %s",
            ("column.measurement_timing",)).fetchone()
        assert state == ("resolved",)

    def test_supporting_unknowns_are_retained_without_replaying_completed_columns(
        self, conn: Any, object_store: ObjectStore, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class SupportingGateway(ScriptedGateway):
            def default(self, key: str, envelope: AgentTaskEnvelopeV1,
                        **over: Any) -> dict[str, Any]:
                body = super().default(key, envelope, **over)
                if key != "semantic_batch":
                    return body
                settled = envelope.payload.get("prerequisite_context", {}).get(
                    "settled_requirements", [])
                known = {(row["requirement_id"], row["scope_id"]) for row in settled}
                missing = [requirement("column.source_process", str(column), "semantic")
                           for column in envelope.scope_ids
                           if ("column.source_process", column) not in known]
                return body | {"missing_requirements": missing,
                               "status": "needs_context" if missing else "complete"}

        stub_renderer(monkeypatch)
        gateway = SupportingGateway()
        deps, opened, sink = start(conn, object_store, gateway=gateway)
        assert opened.interrupt_kind == "approval"
        calls = [call for call in gateway.calls if call.task_kind == "semantic_batch"]
        columns = [column for call in calls for column in call.scope_ids]
        assert len(calls) == len(set(columns)) == 3
        assert not any(json.loads(line).get("status") == "clarification"
                       for line in sink.getvalue().splitlines())
        rows = conn.execute(
            "SELECT scope_id, state, resolving_fact_id FROM design.context_requirements"
            " WHERE requirement_id = 'column.source_process' ORDER BY scope_id").fetchall()
        assert rows == [(column, "unknown_accepted", None) for column in sorted(columns)]
        final_call = gateway.calls[-1].payload["prerequisite_context"]
        assert {row["scope_id"] for row in final_call["settled_requirements"]
                if row["requirement_id"] == "column.source_process"} == set(columns)
        assert not any(row["requirement_id"] == "column.source_process"
                       for row in final_call["accepted_facts"])
        done = graph.resume_design(deps, thread_id=opened.thread_id, resume_value=approval(opened))
        assert done.status == "approved"


    def test_a_prefixed_column_scope_is_folded(
        self, conn: Any, object_store: ObjectStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A namespace-qualified copy of the selected column resolves to one scope (D-100)."""
        stub_renderer(monkeypatch)
        start(conn, object_store, gateway=asking_gateway(
            TIMING_REQUIREMENT | {"scope_id": "nsw.csv::earnings"}))
        rows = conn.execute(
            "SELECT requirement_id, scope_id FROM design.context_requirements"
            " ORDER BY requirement_id, scope_id").fetchall()
        assert rows == [("column.measurement_timing", "earnings")]

    def test_the_registry_owns_a_design_requirement_scope(
        self, conn: Any, object_store: ObjectStore
    ) -> None:
        raised = requirement("design.assignment_mechanism", "invented-concept", "method") | {
            "scope_kind": "concept"}
        start(conn, object_store, gateway=asking_gateway(raised))
        row = conn.execute("SELECT scope_kind, scope_id FROM design.context_requirements"
                           " WHERE requirement_id='design.assignment_mechanism'").fetchone()
        assert row == ("design", "design")

    def test_an_invented_requirement_cannot_mutate_state_from_a_rejected_draft(
        self, conn: Any, object_store: ObjectStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub_renderer(monkeypatch)
        start(conn, object_store, gateway=asking_gateway(
            TIMING_REQUIREMENT | {"requirement_id": "req:invented_timing"}))
        assert conn.execute("SELECT requirement_id FROM design.context_requirements").fetchall() == []

    def test_unknown_answer_ends_without_reasking_the_same_fact(
        self, conn: Any, object_store: ObjectStore
    ) -> None:
        deps, first, _ = start(conn, object_store, gateway=asking_gateway())
        done = graph.resume_design(deps, thread_id=first.thread_id,
            resume_value=unknown_answer("qp:1:1"))
        assert done.status == "needs_context"
        assert done.error_code == "user_answer_unknown"
        outcome = payload_of(conn, object_store, str(done.outcome_artifact_id))
        assert outcome["status"] == "needs_context"
        assert outcome["issues"][0]["required_input_ids"] == ["column.measurement_timing"]

    def test_a_new_revision_inherits_an_accepted_answer_without_reasking(
        self, conn: Any, object_store: ObjectStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub_renderer(monkeypatch)
        deps, first, _ = start(conn, object_store, gateway=asking_gateway())
        answered = graph.resume_design(deps, thread_id=first.thread_id, resume_value={
            "schema_version": "user-context-answer.v1", "packet_id": "qp:1:1",
            "answers": [{"question_id": "q:column.measurement_timing",
                         "answer_kind": "value", "value": "earnings=post_treatment"}],
            "provenance": "user"})
        changed = graph.resume_design(deps, thread_id=answered.thread_id, resume_value={
            **approval(answered), "decision": "changes_requested", "approved_artifacts": [],
            "change_requests": ["Recheck the design while preserving confirmed timing."],
            "idempotency_key": "idem-change"})
        assert changed.status == "changes_requested"
        intake = conn.execute(
            "SELECT artifact_id FROM causal.artifacts WHERE analysis_id = %s"
            " AND artifact_type = 'IntakeOutcome'", (changed.analysis_id,)).fetchone()
        assert intake is not None
        revised = graph.run_design(
            deps, analysis_id=changed.analysis_id,
            intake_outcome_artifact_id=str(intake[0]), thread_id="gt:design:2",
            design_revision=2)
        assert revised.interrupt_kind == "approval"
        facts = conn.execute(
            "SELECT design_revision, inherited_from_fact_id FROM design.accepted_facts"
            " WHERE requirement_id = 'column.measurement_timing' AND is_current"
            " ORDER BY design_revision").fetchall()
        assert facts[0][0] == 1 and facts[0][1] is None
        assert facts[1][0] == 2 and facts[1][1] is not None


def unknown_answer(packet_id: str) -> dict[str, Any]:
    """`unknown` is always offered; for an `ask_user` requirement it leaves the row open."""
    return {"schema_version": "user-context-answer.v1", "packet_id": packet_id,
            "answers": [{"question_id": "q:column.measurement_timing",
                         "answer_kind": "unknown", "value": None}], "provenance": "user"}


def asking_gateway(*raised: dict[str, Any]) -> ScriptedGateway:
    """Raise requirements only from a task whose registry allowlist contains them."""
    gateway = ScriptedGateway()
    original = gateway.default

    def default(key: str, envelope: AgentTaskEnvelopeV1, **over: Any) -> dict[str, Any]:
        rows = list(raised) or [TIMING_REQUIREMENT]
        target = ("semantic_batch" if any(
            row["requirement_id"].startswith("column.") for row in rows) else "intent")
        settled = envelope.payload.get("prerequisite_context", {}).get(
            "settled_requirements", [])
        extra = {"missing_requirements": rows} if key == target and not settled else {}
        return original(key, envelope, **extra | over)

    gateway.default = default  # type: ignore[method-assign]
    return gateway


class UnwrappedFirstBatch(ScriptedGateway):
    """D-066: the first `many` reply is one card, not `{"items": [...]}`."""

    def default(self, key: str, envelope: AgentTaskEnvelopeV1, **over: Any) -> dict[str, Any]:
        body = super().default(key, envelope, **over)
        if key == "semantic_batch" and envelope.attempt_id.endswith(":1"):
            return body | {"payload": card(str(envelope.scope_ids[0]))}
        return body


class TestCorrectionLoop:
    @pytest.mark.parametrize("defect", ("empty", "duplicate", "wrong_column"))
    def test_semantic_card_count_and_scope_are_checked_before_commit(
        self, defect: str, conn: Any, object_store: ObjectStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class InvalidFirstBatch(ScriptedGateway):
            changed = False

            def default(self, key: str, envelope: AgentTaskEnvelopeV1,
                        **over: Any) -> dict[str, Any]:
                body = super().default(key, envelope, **over)
                if key == "semantic_batch" and not self.changed:
                    self.changed = True
                    assigned = str(envelope.scope_ids[0])
                    columns = [] if defect == "empty" else (
                        [assigned, assigned] if defect == "duplicate" else
                        ["earnings" if assigned != "earnings" else "group"])
                    return body | {"payload": {"items": [card(name) for name in columns]}}
                return body

        stub_renderer(monkeypatch)
        gateway = InvalidFirstBatch()
        deps, opened, sink = start(conn, object_store, gateway=gateway)
        code = "semantic_batch_scope_mismatch" if defect == "wrong_column" else "schema_invalid"
        assert f'"error_code":"{code}"' in sink.getvalue()
        items = gateway.schemas["semantic_batch"]["properties"]["payload"]["properties"]["items"]
        assert items["minItems"] == items["maxItems"] == 1
        done = graph.resume_design(deps, thread_id=opened.thread_id, resume_value=approval(opened))
        assert done.status == "approved"

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
        assert outcome["status"] == "approved"
        semantic = [call for call in gateway.calls if call.task_kind == "semantic_batch"]
        assert len(semantic) == len({call.task_id for call in semantic}) + 1

    def test_a_many_reply_without_items_corrects_instead_of_crashing(
        self, conn: Any, object_store: ObjectStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub_renderer(monkeypatch)
        gateway = UnwrappedFirstBatch()
        deps, opened, sink = start(conn, object_store, gateway=gateway)
        assert '"error_code":"schema_invalid"' in sink.getvalue()
        done = graph.resume_design(deps, thread_id=opened.thread_id, resume_value=approval(opened))
        outcome = payload_of(conn, object_store, str(done.outcome_artifact_id))
        assert outcome["status"] == "approved"

    def test_exhausted_corrections_end_the_revision(
        self, conn: Any, object_store: ObjectStore
    ) -> None:
        broken = [{"not": "a result"}] * 3
        gateway = ScriptedGateway({"semantic_batch": broken})
        _, done, sink = start(conn, object_store, gateway=gateway)
        assert done.status == "system_failure" and done.error_code == "agent_output_invalid"
        assert '"event_name":"retry.exhausted"' in sink.getvalue()
        outcome = payload_of(conn, object_store, str(done.outcome_artifact_id))
        assert outcome["issues"][0]["code"] == "agent_output_invalid"
        assert outcome["compiled_design"] is None


def test_the_gateway_gets_the_draft_result_schema(conn: Any, object_store: ObjectStore) -> None:
    """D-063: constrained decoding sees the real result, not a bare `{"type": "object"}`."""
    gateway = ScriptedGateway({"semantic_batch": [{"not": "a result"}] * 3})
    start(conn, object_store, gateway=gateway)
    schema = gateway.schemas["intent"]
    assert schema["properties"]["status"] == {"$ref": "#/$defs/TaskStatus"}
    payload = schema["properties"]["payload"]
    assert payload != {"type": "object"}
    assert set(payload["properties"]) == set(DesignIntentV1.model_fields)
    assert "ConceptProposalV1" in schema["$defs"]


def test_the_prompt_carries_the_evidence_allowlist(conn: Any, object_store: ObjectStore) -> None:
    """D-065/D-068/D-098: every closed set a wall enforces is rendered into the prompt."""
    gateway = ScriptedGateway({"semantic_batch": [{"not": "a result"}] * 3})
    start(conn, object_store, gateway=gateway)
    evidence, rest = gateway.prompts["intent"].split("## parent_artifacts\n")
    parents, requirements = rest.split("## registered_requirement_ids\n")
    section = evidence.split("## allowed_evidence\n")[-1]
    allowed = set(gateway.calls[0].allowed_evidence_ids)
    assert "ua:question/text" in allowed
    lines = section.splitlines()
    rendered = [line.rstrip(":") for line in lines if line and not line.startswith(" ")]
    # D-103: the intent decision receives the bounded column evidence it needs to select
    # candidate roles and establish row identity. The prompt and enforced envelope use one exact
    # task-local allowlist; neither hides references that validation would nevertheless accept.
    scoped = {i for i in allowed if "/column/" in i or "#/columns/" in i}
    assert rendered == sorted(allowed, key=lambda item: (not item.startswith("ua:"), item))
    assert any("/column/" in i for i in scoped) and any("#/columns/" in i for i in scoped)
    # D-102: the document, not only its name. Shown an id alone, a worker truthfully reports the
    # source `not_offered`, SC §6.2 condition 2 holds trivially, and the gate asks the user for a
    # fact this analysis already committed. A data dictionary is multi-line, so it renders whole.
    assert "ua:question/text:" in lines and "  Does the programme raise earnings?" in lines
    ref = gateway.calls[0].parent_artifacts[0]
    assert parents.split() == [ref.artifact_id]
    # Wall 2 rejects an unregistered requirement id, so the legal vocabulary must be shown.
    registered = set(load_requirement_templates(REGISTRIES / "context-requirements.v1.json"))
    allowed = set(TASKS["intent"].allowed_requirement_ids)
    assert allowed <= registered
    assert requirements.split() == sorted(allowed)
    assert "design.treatment_meaning" in registered


def test_the_semantic_prompt_carries_the_measured_facts_of_its_columns(
    conn: Any, object_store: ObjectStore
) -> None:
    """D-103: the harness profiled every column, then asked the model what a column name meant."""
    gateway = ScriptedGateway()
    start(conn, object_store, gateway=gateway)
    prompt = gateway.prompts["semantic_batch"]
    assigned = [call for call in gateway.calls if call.task_kind == "semantic_batch"][-1].scope_ids
    section = prompt.split("## allowed_evidence\n")[-1].split("## parent_artifacts")[0]
    measured = [line.rstrip(":") for line in section.splitlines() if "#/columns/" in line]
    assert measured, "no measured facts reached the semantic worker"
    assert {line.rsplit("/", 1)[-1] for line in measured} == set(assigned)
    # A card slot citing one of these is a measured observation, not the model's inference.
    assert '"cardinality"' in section and '"dtype"' in section


def test_the_role_prompt_carries_the_cards_of_its_assigned_columns(
    conn: Any, object_store: ObjectStore
) -> None:
    """D-104: PRD-002 §9.5 routes the cited cards here; the worker was sent column names."""
    gateway = ScriptedGateway()
    start(conn, object_store, gateway=gateway)
    prompt = gateway.prompts["role_evidence"]
    body = json.loads(prompt.split("## allowed_evidence")[0].split("## payload\n")[-1]) if (
        "## payload\n" in prompt) else None
    assert '"column-semantic-card.v1"' in prompt, "no card payload reached the role worker"
    # The slots are what the worker was missing: meaning, units, levels and timing per column.
    assert '"slots"' in prompt and '"measurement_window"' in prompt
    assert body is None or {card["column_name"] for card in body["cards"]}


def test_the_role_ledger_prompt_carries_the_frame_it_must_copy(
    conn: Any, object_store: ObjectStore
) -> None:
    gateway = ScriptedGateway()
    start(conn, object_store, gateway=gateway)
    payload = next(call.payload for call in gateway.calls if call.task_kind == "role_ledger")
    assert payload["causal_context"]["frame"] == FRAME  # type: ignore[index]
    prompt = gateway.prompts["role_ledger"]
    assert "never bind both roles to the same CSV column" in prompt and "decision-sufficient claims" in prompt
    assert "concept -> frame.outcome" in prompt and "selected base graph" in prompt


def test_the_approval_interrupt_shows_the_design_it_asks_a_person_to_approve(
    conn: Any, object_store: ObjectStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-104: approving an artifact id and a sha256 is not informed consent."""
    stub_renderer(monkeypatch)
    bodies: list[dict[str, Any]] = []
    real = graph.interrupt

    def capture(body: dict[str, Any]) -> Any:
        bodies.append(body)
        return real(body)

    monkeypatch.setattr(graph, "interrupt", capture)
    start(conn, object_store)
    shown = [body for body in bodies if body["kind"] == "approval"]
    assert shown, "the approval gate never opened"
    design = shown[0]["design"]
    assert design["schema_version"] == "design-review-bundle.v2"
    assert "assumptions" in design and "identification_risks" in design


class TestRefusal:
    def test_no_admitted_csv_refuses_without_a_handoff(
        self, conn: Any, object_store: ObjectStore
    ) -> None:
        _, done, _ = start(conn, object_store, files={"nsw.tsv": TSV, "readme.md": README})
        assert done.status == "needs_data" and done.error_code == "NO_ANALYSIS_CSV"
        assert done.handoff_id is None
        outcome = payload_of(conn, object_store, str(done.outcome_artifact_id))
        assert outcome["issues"][0]["code"] == "NO_ANALYSIS_CSV"
        assert outcome["compiled_design"] is None


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
