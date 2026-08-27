# The §9 curator: the bounded context, the ONE call and its two targeted corrections, the one
# allowlisted layout-fact operation, and the deterministic gate-2 plan validator (T-031 §2).

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

import pytest

from causal.presentation import contracts as pc
from causal.presentation import curate as cu
from causal.shared import persistence
from causal.shared.agenttask import TaskRunner
from causal.shared.contracts import ArtifactEnvelopeV1
from causal.shared.envelope import AgentTaskEnvelopeV1
from causal.shared.gateway import GatewayResultV1
from causal.shared.registry import load_artifact_type_registry
from tests.presentation import test_compile as fx

REGISTRY = load_artifact_type_registry(fx.REPO / "registries" / "artifact-types.v1.json")
NOW = datetime(2026, 8, 26, tzinfo=UTC)
FACTS = cu.layout_facts(fx.DATA)
QUALIFICATION = "attrition_above_threshold"
LINEAGE: dict[str, Any] = {"plan_id": "plan:an-1:1", "versions": {"schema": "figure-plan.v1"},
                           "parents": [fx.ref("presentation_context_manifest").model_dump(
                               mode="json")]}
COVERAGE = {"assignment_attrition_flow": "fig_flow", "balance_overview": "fig_balance",
            fx.PRIMARY: "fig_primary", "required_sensitivities": "fig_primary"}
FIGURE_DATA = {name: fx.ref(name) for name in COVERAGE}
# §7.1: one bounded evidence fx.entry per approved question; the sensitivity branch is reported in
# standardized units, so it may never share one panel scale with the primary effect (§11.1).
EVIDENCE = tuple(pc.EvidenceEntryV1(
    visual_evidence_id=name, figure_data=fx.ref(name), suppression_state="none",
    figure_data_schema_id="figure-data-artifact.v1", compatible_template_ids=(template_id,),
    quantities={"x": "effect", "y": "group"}, cardinality=FACTS[name]["series"],
    units={"x": "sd" if name == "required_sensitivities" else "pp", "y": "count"})
    for name, template_id in (("assignment_attrition_flow", "composition_stack.v1"),
                              ("balance_overview", "balance_overview.v1"), (fx.PRIMARY, fx.FOREST),
                              ("required_sensitivities", fx.FOREST)))
MANIFEST = pc.PresentationContextManifestV1(
    parents=(fx.ref("estimation_bundle"),), analysis_id="an-1", stage_run_id="ps:an-1:1",
    versions={"schema": "presentation-context-manifest.v1"}, handoff_manifest=fx.ref("handoff"),
    inputs={key: fx.ref(key) for key in pc.ENTRY_KEYS}, causal_graph_view=fx.ref("graph_view"),
    approved={"question_id": "q1", "estimand_id": "att", "profile_id": fx.RCT.profile_id,
              "method_id": fx.RCT.method_id}, evidence=EVIDENCE,
    required_evidence_ids=fx.RCT.required_evidence_ids, display_profile=fx.CATALOG.display_profile,
    allowlists={"artifact_ids": tuple(sorted(COVERAGE))}, qualification_ids=(QUALIFICATION,),
    claim_status="reportable_with_qualifications", statement_ids=("arm_b_raises_completion",))
CONTEXT = cu.curator_context(MANIFEST, fx.CATALOG, fx.ref("presentation_context_manifest"))
# Everything but the envelope ids and the payload an `AgentTaskResultV1` body always carries.
RESULT: dict[str, Any] = {
    "schema_version": "agent-task-result.v1", "status": "complete", "claims": [], "conflicts": [],
    "artifact_type": cu.ARTIFACT_TYPE, "artifact_schema_version": cu.SCHEMA_VERSION,
    "parent_artifact_ids": [], "missing_requirements": [], "warnings": [], "evidence_ids": [],
    "tool_receipts": [], "output_hash": None, "validation_target": "v"}


def figures(**over: Any) -> tuple[pc.FigureEntryV1, ...]:
    # The golden three-figure plan: flow, balance, and the primary estimate with its sensitivity.
    rows = (fx.entry("fig_flow", "composition_stack.v1", ("assignment_attrition_flow",)),
            fx.entry("fig_balance", "balance_overview.v1", ("balance_overview",)),
            fx.entry("fig_primary", fx.FOREST, (fx.PRIMARY, "required_sensitivities"),
                  qualification_ids=(QUALIFICATION,), text={
                      "title": "How large is the primary effect?", "accessible_description":
                      fx.DESCRIBED, "caption": f"The primary estimate, qualified by {QUALIFICATION}."}))
    return tuple(row.model_copy(update=over.get(row.figure_id, {})) for row in rows)


def draft(rows: tuple[pc.FigureEntryV1, ...] | None = None, **over: Any) -> dict[str, Any]:
    return json.loads(json.dumps({"summary": "the approved figures in evidence order", "figures": [
        row.model_dump(mode="json") for row in (figures() if rows is None else rows)]} | over))


class Gateway:
    # Canned `AgentTaskResultV1` bodies; every construction is counted, so "one call" is an
    # assertion about this list and not about a mock's internals.

    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads, self.calls = list(payloads), []

    def invoke(self, envelope: AgentTaskEnvelopeV1, prompt: str,
               response_schema: dict[str, object]) -> GatewayResultV1:
        self.calls.append(envelope)
        body = RESULT | {"envelope_id": envelope.envelope_id, "task_id": envelope.task_id,
                         "payload": self.payloads.pop(0)}
        return GatewayResultV1(text=json.dumps(body), parsed=body, token_usage={}, attempts=1,
                               seed=1)


class Harness:
    # The injected hooks a real presentation coordinator supplies to the shared TaskRunner. The
    # `validate` hook raises: the curator must bind its own gate-2 validator over every draft.

    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.gateway, self.events = Gateway(payloads), []
        self.committed: list[tuple[str, dict[str, Any]]] = []
        self.state: dict[str, Any] = {
            "analysis_id": "an-1", "stage_run_id": "ps:an-1:1", "design_revision": 1,
            "corrections": {}, "open_requirement_ids": []}
        self.runner = TaskRunner(
            gateway=self.gateway, tasks={cu.TASK_KIND: cu.CURATOR_TASK},
            tools={cu.TASK_KIND: (cu.TOOL_ID, "run_estimator")},
            evals={cu.TASK_KIND: ("EV-P5-002",)}, prompts_root=fx.REPO,
            envelope=cu.build_curator_envelope, prompt=cu.render_curator_prompt,
            validate=self.unbound, context=lambda state: None,
            evidence=lambda state: frozenset(COVERAGE),
            manifest=lambda state: fx.ref("presentation_context_manifest"),
            parents=lambda state, *kinds: (), commit=self.commit,
            emit=lambda state, name, evals, **over: self.events.append((name, over)),
            record=lambda *args: None, upsert=lambda *args: None,
            exhausted=lambda *args: self.events.append(("blocker.raised", {})))

    def unbound(self, *args: Any) -> None:
        raise AssertionError("the curator must bind its own plan validator")

    def commit(self, state: Any, kind: str, payload: Any,
               parents: tuple[Any, ...]) -> ArtifactEnvelopeV1:
        self.committed.append((kind, dict(payload)))
        return persistence.build_envelope(
            REGISTRY, kind, dict(payload), analysis_id=state["analysis_id"], parents=parents,
            stage_run_id=state["stage_run_id"], producer_version="test", created_at_utc=NOW)

    def curate(self) -> cu.CurationOutcome:
        return cu.curate(self.runner, self.state, CONTEXT, profile=fx.RCT, facts=FACTS,
                         lineage=LINEAGE, coverage=COVERAGE, figure_data=FIGURE_DATA)


class TestCuratorContext:
    def test_the_context_carries_bounded_facts_and_structurally_no_rows(self) -> None:
        body = json.dumps(CONTEXT.model_dump(mode="json"))
        assert not {"points", "x_value", "y_value"} & set(re.findall(r'"(\w+)":', body))
        assert {row.template_id for row in CONTEXT.templates} == {
            "composition_stack.v1", "balance_overview.v1", fx.FOREST}
        assert CONTEXT.claim_status == "reportable_with_qualifications"
        assert FACTS[fx.PRIMARY] == {"series": 1, "points": 1, "labels": 2, "label_characters": 11,
                                  "intervals": 1, "denominators": 0}

    def test_the_layout_tool_answers_once_and_denies_the_second_call(self) -> None:
        tool = cu.LayoutFactTool(FACTS)
        assert tool.resolve(["balance_overview"]) == {"balance_overview": FACTS["balance_overview"]}
        with pytest.raises(pc.PresentationError) as excinfo:
            tool.resolve(["balance_overview"])
        assert excinfo.value.code == cu.TOOL_DENIED and tool.calls == 2


class TestPlanValidator:
    @pytest.mark.parametrize(("patch", "code"), [
        ({}, ""),
        ({"template_id": "sankey.v9"}, cu.UNREGISTERED_TEMPLATE),
        ({"template_id": "composition_stack.v1"}, cu.INCOMPATIBLE_TEMPLATE),
        ({"choices": fx.choices(fx.FOREST, legends="floating")}, cu.UNREGISTERED_CHOICE),
        ({"choices": fx.choices(fx.FOREST, reference_lines="")}, cu.DROPPED_ENCODING),
        ({"panel_groups": ((fx.PRIMARY, "required_sensitivities"),)}, cu.INCOMPATIBLE_PANEL),
        ({"annotation_ids": tuple(f"note_{index}" for index in range(9))}, cu.OVER_CAPACITY),
        ({"qualification_ids": ()}, cu.MISPLACED_QUALIFICATION),
        ({"text": {"title": "the primary effect", "caption": f"see {QUALIFICATION}",
                   "accessible_description": fx.DESCRIBED}}, cu.INACCESSIBLE)])
    def test_gate_two_accepts_the_golden_plan_and_names_every_other_failure(
            self, patch: dict[str, Any], code: str) -> None:
        rows = figures(fig_primary=patch)
        found = cu.plan_codes(pc.FigurePlanDraftV1(figures=rows, summary="s"), CONTEXT, fx.RCT, FACTS)
        assert (found == () if not code
                else any(row.split(":", 1)[0] == code for row in found)), found

    def test_hidden_required_evidence_is_uncovered_and_a_typed_inability_is_not(self) -> None:
        rows = tuple(row for row in figures() if row.figure_id != "fig_balance")
        found = cu.plan_codes(pc.FigurePlanDraftV1(figures=rows, summary="s"), CONTEXT, fx.RCT, FACTS)
        stuck = pc.FigurePlanDraftV1(inability_code="none_fit", implicated_evidence_ids=("bal",))
        assert f"{cu.UNCOVERED}:balance_overview" in found
        assert cu.terminal_status(found) == "needs_template"
        assert cu.terminal_status([f"{cu.OVER_CAPACITY}:fig:labels"]) == "needs_layout_revision"
        assert cu.plan_codes(stuck, CONTEXT, fx.RCT, FACTS) == ()


class TestCuratorCall:
    def test_a_golden_draft_commits_one_plan_through_one_bounded_call(self) -> None:
        harness = Harness([draft()])
        found, built = harness.curate(), harness.gateway.calls[0]
        assert found.plan is not None and found.status is None and found.plan.plan_id
        assert len(harness.gateway.calls) == 1 and len(harness.committed) == 1
        assert found.plan.coverage == COVERAGE
        assert found.plan.versions["prompt"] == cu.PINNED_VERSIONS["prompt"]
        assert harness.committed[0][1]["schema_version"] == cu.SCHEMA_VERSION
        assert built.allowed_tool_ids == (cu.TOOL_ID,)
        assert built.forbidden_payload_classes == cu.FORBIDDEN_PAYLOAD_CLASSES
        assert (built.budgets.tool_call_budget, built.budgets.correction_budget) == (
            cu.TOOL_CALL_BUDGET, cu.CORRECTION_BUDGET)

    def test_a_hidden_evidence_draft_is_corrected_and_then_accepted(self) -> None:
        thin = draft(tuple(row for row in figures() if row.figure_id != "fig_balance"))
        harness = Harness([thin, draft()])
        found = harness.curate()
        assert found.plan is not None and len(harness.gateway.calls) == 2
        assert harness.state["corrections"] == {f"{cu.ARTIFACT_TYPE}:{cu.UNCOVERED}": 1}
        assert harness.gateway.calls[1].payload["correction"]["issues"][0]["code"] == cu.UNCOVERED

    def test_an_unregistered_template_is_refused_for_every_permitted_attempt(self) -> None:
        broken = draft(figures(fig_primary={"template_id": "sankey.v9"}))
        harness = Harness([broken, broken, broken])
        found = harness.curate()
        assert found.plan is None and found.status == "needs_template"
        assert found.error_code == cu.CORRECTION_EXHAUSTED and len(harness.gateway.calls) == 3
        assert any(code.startswith(cu.UNREGISTERED_TEMPLATE) for code in found.detail_codes)

    def test_a_typed_inability_becomes_needs_template(self) -> None:
        harness = Harness([draft([], inability_code="no_honest_template",
                                 implicated_evidence_ids=["balance_overview"])])
        found = harness.curate()
        assert found.plan is None and found.status == "needs_template"
        assert found.error_code == "no_honest_template"
        assert found.detail_codes == ("balance_overview",)
