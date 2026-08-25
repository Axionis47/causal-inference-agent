"""Design tool router enforcement and retrieval handlers (T-011 §7, §8; EV-SYS-002)."""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from causal.design.contracts import (
    REGISTRY_VERSION_KEYS,
    AvailabilityRowV1,
    DesignContextManifestV1,
    StructuralFieldV1,
)
from causal.design.entry import RETRIEVAL_SURFACES
from causal.design.tools import (
    RETRIEVAL_TOOL_IDS,
    ToolError,
    ToolHandler,
    ToolRouter,
    make_retrieval_handlers,
    tool_allowlists,
)
from causal.shared.contracts import ArtifactEnvelopeV1, ArtifactRef, SensitivityClass
from causal.shared.envelope import AgentTaskEnvelopeV1, TaskBudgets, TaskStatus
from causal.shared.events import EventEmitter

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 8, 24, 12, 0, 0, tzinfo=UTC)
HASH = "a" * 64
REF = ArtifactRef(artifact_id="art-1", content_hash=HASH)
TABLE = "nsw.csv"

ALLOWLISTS: dict[str, tuple[str, ...]] = {
    "list_intake_inventory": ("intent", "semantic_batch"),
    "get_semantic_evidence": ("intent", "semantic_batch"),
    "get_measured_facts": ("semantic_batch",),
    "get_provenance": ("semantic_batch",),
    "get_method_contract": ("method_design",),
    "run_preflight_diagnostic": ("method_design",),
    "validate_causal_model": ("semantic_batch",),
}
REGISTERED = dict.fromkeys(ALLOWLISTS, True) | {"run_preflight_diagnostic": False}

EVIDENCE: dict[str, Any] = {"schema_version": "evidence-bundle.v1", "items": [
    {"evidence_id": "ev-1", "scope_kind": "column", "value": "participant id"},
    {"evidence_id": "ev-2", "scope_kind": "column", "value": "yearly earnings"}]}
PROFILES: dict[str, dict[str, Any]] = {
    TABLE: {"schema_version": "table-profile.v1", "row_count": 5, "columns": {
        "earnings": {
            "dtype": "Float64", "null_count": 1, "null_rate": 0.2, "cardinality": 4,
            "all_null": False, "constant": False, "numeric": {"min": 0.0, "max": 999.0},
            "hypotheses": [{"kind": "missing_sentinel", "detail": "999 may be a sentinel"}]},
        "unit_id": {"dtype": "Int64", "null_count": 0, "null_rate": 0.0, "cardinality": 5,
                    "all_null": False, "constant": False, "hypotheses": []}}}}
PACKS: dict[str, dict[str, Any]] = {"aipw": {
    "method_id": "aipw", "pack_version": "aipw-pack.v1", "supported_estimands": ["ate", "att"]}}


def availability(column: str, slot: str, status: str) -> AvailabilityRowV1:
    return AvailabilityRowV1(
        scope_kind="column", table_name=TABLE, column_name=column, field_or_slot_name=slot,
        status=status, evidence_count=1, json_pointer=f"/columns/0/slots/{slot}")


MANIFEST = DesignContextManifestV1(
    design_revision=1, question_artifact=REF, intake_outcome_artifact=REF,
    table_selection_artifact=REF, selected_table=TABLE,
    structural_inventory=(
        StructuralFieldV1(table_name=TABLE, column_name="earnings", dtype="declared", ordinal=0),
        StructuralFieldV1(table_name=TABLE, column_name="unit_id", dtype="declared", ordinal=1),
        StructuralFieldV1(table_name="psid.csv", column_name="x", dtype="undeclared", ordinal=0),
    ),
    semantic_available=(availability("earnings", "meaning", "evidenced"),),
    semantic_missing=(availability("unit_id", "encoding", "not_offered"),),
    measured_surface=(availability("earnings", "profile", "evidenced"),),
    provenance_surface=(availability("earnings", "capture", "evidenced"),),
    retrieval_surfaces=RETRIEVAL_SURFACES,
    registry_versions=dict.fromkeys(REGISTRY_VERSION_KEYS, "v1"),
    recipient_map={"intent": ("list_intake_inventory",)},
)
PROVENANCE_ENVELOPE = ArtifactEnvelopeV1(
    artifact_id="tp-1", artifact_type="TableProfile", schema_version="table-profile.v1",
    content_hash=HASH, analysis_id="an-1", stage_run_id="run-1",
    producer_component="intake-coordinator", producer_version="0.1.0",
    parent_artifacts=(REF,), sensitivity_class=SensitivityClass.INTERNAL,
    created_at_utc=NOW, payload_locator=f"objects/{HASH}")


class FakeProducts:
    """A ProductsReader over one committed envelope."""

    def load_envelope(self, artifact_id: str) -> ArtifactEnvelopeV1:
        if artifact_id != PROVENANCE_ENVELOPE.artifact_id:
            raise KeyError(artifact_id)
        return PROVENANCE_ENVELOPE


class FakePacks:
    """A method-pack registry that fails closed on an unknown method."""

    def get(self, method_id: str) -> object:
        if method_id not in PACKS:
            raise ValueError(f"unsupported_method: {method_id!r}")
        return PACKS[method_id]


def handlers() -> dict[str, ToolHandler]:
    return make_retrieval_handlers(MANIFEST, FakeProducts(), PROFILES, EVIDENCE, FakePacks())


def make_envelope(
    task_kind: str = "semantic_batch", tools: tuple[str, ...] = RETRIEVAL_TOOL_IDS,
    evidence_ids: tuple[str, ...] = ("ev-1", "ev-9"),
) -> AgentTaskEnvelopeV1:
    return AgentTaskEnvelopeV1(
        envelope_id="env-1", schema_version="agent-task-envelope.v1", analysis_id="an-1",
        stage_run_id="run-1", task_id="task-1", attempt_id="attempt-1", context_manifest=REF,
        task_kind=task_kind, scope_kind="column", scope_ids=("nsw.csv.earnings",),
        parent_artifacts=(REF,), allowed_evidence_ids=evidence_ids,
        allowed_retrieval_ids=RETRIEVAL_SURFACES, allowed_tool_ids=tools,
        output_schema_version="column-semantic-card.v1",
        validator_version="column-semantic-card-validator.v1",
        prompt_version="column-card.v1", model_profile_version="vertex-model-profile.v1",
        budgets=TaskBudgets(token_budget=8000, tool_call_budget=4),
        allowed_stopping_states=(TaskStatus.COMPLETE,), error_vocabulary=("SCHEMA_INVALID",),
        forbidden_payload_classes=("raw_rows",), payload_type="column-card-request.v1",
        payload={"column_name": "earnings"})


def make_router(bound: dict[str, ToolHandler]) -> tuple[ToolRouter, io.StringIO]:
    sink = io.StringIO()
    return ToolRouter(ALLOWLISTS, REGISTERED, bound, EventEmitter(sink), lambda: NOW), sink


def events(sink: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in sink.getvalue().splitlines()]


DENIALS = [
    ("no_such_tool", "semantic_batch", ("no_such_tool",), "unknown_tool"),
    ("run_preflight_diagnostic", "method_design", ("run_preflight_diagnostic",), "tool_denied"),
    ("get_method_contract", "semantic_batch", ("get_method_contract",), "tool_denied"),
    ("get_provenance", "semantic_batch", ("get_measured_facts",), "tool_denied"),
    ("validate_causal_model", "semantic_batch", ("validate_causal_model",), "unknown_tool"),
]


class TestEnforcementOrder:
    @pytest.mark.parametrize(("tool_id", "task_kind", "tools", "code"), DENIALS)
    def test_violations_deny_without_calling_a_handler(
        self, tool_id: str, task_kind: str, tools: tuple[str, ...], code: str
    ) -> None:
        calls: list[str] = []

        def spy(envelope: AgentTaskEnvelopeV1, arguments: Any) -> dict[str, Any]:
            calls.append(tool_id)
            return {}

        router, sink = make_router(dict.fromkeys(RETRIEVAL_TOOL_IDS, spy))
        with pytest.raises(ToolError) as error:
            router.call(make_envelope(task_kind, tools), tool_id, {})
        assert error.value.code == code
        assert calls == []
        emitted = events(sink)
        assert [row["event_name"] for row in emitted] == ["tool.denied"]
        assert emitted[0]["event_id"] == "evt:env-1:tool:1"
        assert emitted[0]["error_code"] == code
        assert emitted[0]["status"] == "denied"
        assert emitted[0]["stage"] == "design"
        assert emitted[0]["component_id"] == "design-harness"
        assert emitted[0]["required_eval_ids"] == ["EV-SYS-002"]
        assert emitted[0]["safe_dimensions"]["tool_id"] == tool_id

    def test_success_emits_started_then_completed_with_stable_ids(self) -> None:
        router, sink = make_router(handlers())
        result = router.call(make_envelope(), "list_intake_inventory", {"table_name": TABLE})
        assert (result.tool_id, result.status) == ("list_intake_inventory", "completed")
        router.call(make_envelope(), "get_measured_facts",
                    {"table_name": TABLE, "column_names": ["unit_id"]})
        emitted = events(sink)
        assert [row["event_name"] for row in emitted] == [
            "tool.started", "tool.completed", "tool.started", "tool.completed"]
        assert [row["event_id"] for row in emitted] == [
            f"evt:env-1:tool:{index}" for index in (1, 2, 3, 4)]
        assert emitted[1]["versions"] == {"tool": "design-tools.v1"}

    def test_handler_exception_emits_tool_failed(self) -> None:
        router, sink = make_router(handlers())
        with pytest.raises(ToolError) as error:
            router.call(make_envelope(), "list_intake_inventory", {"table_name": "psid.csv"})
        assert error.value.code == "tool_failed"
        emitted = events(sink)
        assert [row["event_name"] for row in emitted] == ["tool.started", "tool.failed"]
        assert emitted[1]["exception_class"] == "ValueError"
        assert emitted[1]["error_code"] == "tool_failed"


class TestRetrievalHandlers:
    def test_inventory_is_the_frozen_manifest_surface(self) -> None:
        router, _ = make_router(handlers())
        result = router.call(
            make_envelope(), "list_intake_inventory", {"table_name": TABLE}).result
        assert set(result) == {
            "table_name", "structural", "available", "missing", "measured", "provenance"}
        assert result["structural"] == [
            {"column_name": "earnings", "dtype": "declared", "ordinal": 0},
            {"column_name": "unit_id", "dtype": "declared", "ordinal": 1}]
        assert [row["field_or_slot_name"] for row in result["missing"]] == ["encoding"]
        assert [row["status"] for row in result["available"]] == ["evidenced"]

    def test_semantic_evidence_reports_unreadable_ids(self) -> None:
        router, _ = make_router(handlers())
        result = router.call(
            make_envelope(), "get_semantic_evidence",
            {"evidence_ids": ["ev-1", "ev-2", "ev-9"]}).result
        assert [item["evidence_id"] for item in result["items"]] == ["ev-1"]
        # ev-2 exists but is outside the envelope; ev-9 is allowed but does not exist.
        assert result["missing"] == ["ev-2", "ev-9"]

    def test_measured_facts_are_bounded_scalars(self) -> None:
        router, _ = make_router(handlers())
        result = router.call(
            make_envelope(), "get_measured_facts",
            {"table_name": TABLE, "column_names": ["earnings", "gone"]}).result
        assert result["missing"] == ["gone"]
        assert result["facts"] == [{
            "column_name": "earnings", "dtype": "Float64", "null_count": 1, "null_rate": 0.2,
            "cardinality": 4, "all_null": False, "constant": False,
            "hypotheses": ["missing_sentinel"]}]

    def test_provenance_returns_envelope_fields_only(self) -> None:
        router, _ = make_router(handlers())
        result = router.call(make_envelope(), "get_provenance", {"artifact_id": "tp-1"}).result
        assert result["artifact_type"] == "TableProfile"
        assert result["content_hash"] == HASH
        assert result["payload_locator"] == f"objects/{HASH}"
        assert result["parent_artifact_ids"] == ["art-1"]
        assert "payload" not in result

    def test_method_contract_returns_one_pack(self) -> None:
        router, _ = make_router(handlers())
        envelope = make_envelope("method_design", ("get_method_contract",))
        result = router.call(envelope, "get_method_contract", {"method_id": "aipw"}).result
        assert result["pack_version"] == "aipw-pack.v1"
        with pytest.raises(ToolError) as error:
            router.call(envelope, "get_method_contract", {"method_id": "nope"})
        assert error.value.code == "tool_failed"


class TestToolRegistry:
    def test_allowlists_split_a_registry_document(self) -> None:
        document = {"registry_version": "design-tools.v1", "tools": [
            {"tool_id": "get_provenance", "allowed_task_kinds": ["semantic_batch"],
             "registered": True},
            {"tool_id": "render_causal_graph", "allowed_task_kinds": [], "registered": True}]}
        allowlists, registered = tool_allowlists(document)
        assert allowlists == {
            "get_provenance": ("semantic_batch",), "render_causal_graph": ()}
        assert registered == {"get_provenance": True, "render_causal_graph": True}

    def test_shipped_registry_registers_every_retrieval_tool(self) -> None:
        path = ROOT / "registries" / "design-tools.v1.json"
        if not path.exists():
            pytest.skip("design-tools.v1.json lands with the registry task")
        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["registry_version"] == "design-tools.v1"
        allowlists, registered = tool_allowlists(document)
        assert set(RETRIEVAL_TOOL_IDS) <= set(allowlists)
        assert all(registered[tool_id] for tool_id in RETRIEVAL_TOOL_IDS)
