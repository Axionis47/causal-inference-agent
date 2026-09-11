"""Walls 1–7: one passing case and the named failing fixtures per wall (T-012 §6)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import pytest
from pydantic import BaseModel

# isort: off
from causal.design.contracts import (
    REGISTRY_VERSION_KEYS,
    ConceptProposalV1,
    DesignContextManifestV1,
    DesignIntentV1,
    QuestionKind,
    SourceInterpretationV1,
    StructuralFieldV1,
)
from causal.design.packs import (
    METHOD_IDS,
    PackRegistryError,
    load_method_packs,
    load_requirement_templates,
)
from causal.design.semantics import (
    COLUMN_CARD_SLOTS, CausalContextV1, CausalEdgeV1, ColumnSemanticCardV1,
    GraphAlternativeV1, RoleClaimV1, RoleLedgerV1, RoleName, SlotAssertionV1, TimingClass)
from causal.design.v2 import AgentDesignProposalV2
from causal.design.validators import (
    ACTIONS,
    RULE_KINDS,
    ValidationContext,
    ValidationReport,
    canonical_requirement_scope,
    evidence_class,
    load_validation_rules,
    validate_result,
    wall_causal,
    wall_evidence,
    wall_method,
    wall_references,
    wall_shape,
    wall_temporal,
)
from causal.shared.contracts import ArtifactRef, ReferenceKind, reference_field
from causal.shared.envelope import (
    AgentTaskEnvelopeV1, AgentTaskResultV1, AttemptedEvidenceV1,
    CausalFrameV1,
    ContextRequirementV1,
    EpistemicStatus,
    EvidenceClass,
    SupportClass,
    TaskBudgets,
    TaskStatus,
)
# isort: on

REGISTRIES = Path(__file__).resolve().parents[2] / "registries"
RULES = load_validation_rules(REGISTRIES / "design-validation-rules.v1.json")
PACKS = load_method_packs(REGISTRIES / "method-packs.v1.json")
TEMPLATES = load_requirement_templates(REGISTRIES / "context-requirements.v1.json")
HASH = "a" * 64
REF = ArtifactRef(artifact_id="art-1", content_hash=HASH)
FRAME = CausalFrameV1(treatment="c-treat", outcome="c-earn", population="p", timeframe="t")
COLUMNS = ("person_id", "treat", "re78", "age", "period")
MANIFEST = DesignContextManifestV1(
    design_revision=1, question_artifact=REF, intake_outcome_artifact=REF,
    table_selection_artifact=REF, selected_table="nsw.csv",
    structural_inventory=tuple(
        StructuralFieldV1(table_name="nsw.csv", column_name=name, dtype="float64", ordinal=index)
        for index, name in enumerate(COLUMNS)),
    semantic_available=(), semantic_missing=(), measured_surface=(), provenance_surface=(),
    retrieval_surfaces=("s-1",), registry_versions=dict.fromkeys(REGISTRY_VERSION_KEYS, "v1"))
ENVELOPE = AgentTaskEnvelopeV1(
    envelope_id="env-1", schema_version="agent-task-envelope.v1", analysis_id="an-1",
    stage_run_id="sr-1", task_id="t-1", attempt_id="at-1", context_manifest=REF,
    task_kind="causal_context", scope_kind="design", scope_ids=("d-1",), parent_artifacts=(REF,),
    allowed_evidence_ids=(), allowed_retrieval_ids=(), allowed_tool_ids=("validate_causal_model",),
    output_schema_version="causal-context.v1", validator_version="v1", prompt_version="v1",
    model_profile_version="v1", budgets=TaskBudgets(token_budget=1, tool_call_budget=1),
    allowed_stopping_states=(TaskStatus.COMPLETE,), error_vocabulary=(),
    forbidden_payload_classes=(), payload_type="causal-synthesis", payload={})


def codes(report: ValidationReport) -> set[str]:
    """The issue codes one wall raised."""
    return {issue.code for issue in report.issues}


def context(**overrides: Any) -> ValidationContext:
    base: dict[str, Any] = {"manifest": MANIFEST, "rules": RULES, "packs": PACKS,
                            "templates": TEMPLATES,
                            "user_answer_evidence_ids": frozenset({"ua:answer-1"}),
                            "concept_ids": frozenset({"c-treat", "c-earn", "c-x", "c-g"})}
    return ValidationContext(**base | overrides)


def role(name: RoleName, concept: str = "c-treat", *, columns: tuple[str, ...] = (),
         timing: TimingClass = TimingClass.PRE_TREATMENT, edges: tuple[str, ...] = (),
         evidence: tuple[str, ...] = ("ua:answer-1",)) -> RoleClaimV1:
    return RoleClaimV1(role=name, concept_id=concept, column_refs=columns, evidence_ids=evidence,
                       timing=timing, graph_edge_ids=edges, alternatives=(), methods=("did",),
                       support_class=SupportClass.DIRECT_USER_CONFIRMATION,
                       status=EpistemicStatus.EVIDENCED)


def ledger(*claims: RoleClaimV1) -> RoleLedgerV1:
    return RoleLedgerV1(frame=FRAME, claims=(
        role(RoleName.TREATMENT, columns=("treat",), timing=TimingClass.CONCURRENT),
        role(RoleName.OUTCOME, "c-earn", columns=("re78",), timing=TimingClass.POST_TREATMENT),
        *claims))


def edge(edge_id: str, source: str, target: str,
         status: EpistemicStatus = EpistemicStatus.HYPOTHESIS) -> CausalEdgeV1:
    return CausalEdgeV1(edge_id=edge_id, source_concept_id=source, target_concept_id=target,
                        timeframe="t", mechanism_summary="m", supporting_evidence_ids=(),
                        contrary_evidence_ids=(), status=status, differing_alternative_ids=())


def graph(*edges: CausalEdgeV1, alternatives: tuple[GraphAlternativeV1, ...] = (),
          target: bool = True) -> CausalContextV1:
    base = (edge("e-target", "c-treat", "c-earn", EpistemicStatus.EVIDENCED),) if target else ()
    return CausalContextV1(frame=FRAME, concept_ids=("c-treat", "c-earn", "c-x"), edges=base + edges,
                           alternatives=alternatives, selection_notes="n")


def result(payload: Any = None, **overrides: Any) -> AgentTaskResultV1:
    body = payload if isinstance(payload, dict | type(None)) else payload.model_dump(mode="json")
    base: dict[str, Any] = {
        "envelope_id": "env-1", "schema_version": "agent-task-result.v1", "task_id": "t-1",
        "status": TaskStatus.COMPLETE, "artifact_type": "CausalContext",
        "artifact_schema_version": "causal-context.v1", "parent_artifact_ids": (),
        "payload": body or {}, "missing_requirements": (), "conflicts": (), "warnings": (),
        "validation_target": "causal-context-validator.v1"}
    return AgentTaskResultV1(**base | overrides)


def requirement(requirement_id: str) -> ContextRequirementV1:
    template = {key: value for key, value in dict(TEMPLATES["column.meaning"]).items()
                if key not in {"requirement_id", "accepted_fact"}}
    return ContextRequirementV1(
        requirement_id=requirement_id, registry_version="context-requirements.v1", scope_id="re78",
        decisions_blocked=(), attempted_evidence=(), **template)


def registered_requirement(requirement_id: str, scope_id: str) -> ContextRequirementV1:
    template = TEMPLATES[requirement_id]
    return ContextRequirementV1(
        requirement_id=requirement_id, registry_version="context-requirements.v1",
        scope_kind=template.scope_kind, scope_id=scope_id, decisions_blocked=("decision",),
        attempted_evidence=(AttemptedEvidenceV1(
            evidence_id="ua:answer-1", availability_status="not_offered"),),
        **{key: value for key, value in dict(template).items()
           if key not in {"requirement_id", "scope_kind", "accepted_fact"}})


class TypedReferenceProbe(BaseModel):
    source: Annotated[str, reference_field(ReferenceKind.EVIDENCE)]
    diagnostic: Annotated[str, reference_field(ReferenceKind.DIAGNOSTIC)]
    artifact: ArtifactRef


class UntypedSuffixProbe(BaseModel):
    future_fact_evidence_ids: tuple[str, ...]


def proposal(**overrides: Any) -> AgentDesignProposalV2:
    base: dict[str, Any] = {
        "assignment_mechanism": "time_of_adoption", "requested_estimand": "att_group_time_aggregate",
        "comparator": "not yet treated", "ranked_method_ids": METHOD_IDS,
        "method_facts": (), "optional_assumption_ids": (), "optional_risk_ids": (),
        "optional_sensitivity_ids": ()}
    body = base | overrides
    body.setdefault("source_interpretations", tuple(SourceInterpretationV1(
        fact_key=fact_key, value=value, evidence_id="ev:doc/x",
        verbatim_excerpt="documented assignment and comparator", relation="direct")
        for fact_key, value in (
            ("assignment_mechanism", body["assignment_mechanism"]),
            ("estimand", body["requested_estimand"]),
            ("comparator", body["comparator"]),
        ) if value != "unknown"))
    return AgentDesignProposalV2(**body)


def rule_file(**overrides: Any) -> str:
    row = {"rule_id": "r-1", "wall": 4, "kind": "temporal", "params": {}, "code": "c",
           "allowed_actions": [], "user_resolvable": False} | overrides
    return json.dumps({"registry_version": "design-validators.v1", "rules": [row]})


# --- wall 1: shape ---
def test_wall_shape_accepts_a_valid_payload() -> None:
    assert wall_shape(CausalContextV1, result(graph())).passed


def test_causal_context_frame_anchors_must_be_graph_concept_ids() -> None:
    payload = graph().model_dump(mode="json")
    payload["frame"]["treatment"] = "human-readable treatment label"
    parsed = CausalContextV1.model_validate_json(json.dumps(payload))
    report = wall_references(parsed, result(parsed), context())
    assert codes(report) == {"unresolved_concept"}
    assert {issue.json_path for issue in report.issues} == {"/frame/treatment"}


def test_causal_context_rejects_concepts_outside_the_measurement_map() -> None:
    payload = graph(edge("e-1", "missing-source", "missing-target")).model_copy(update={"concept_ids": (*graph().concept_ids, "missing-source", "missing-target")})
    report = wall_references(payload, result(payload), context(concept_ids=frozenset(graph().concept_ids)))
    assert codes(report) == {"unresolved_concept"}
    assert {issue.json_path for issue in report.issues} == {"/concept_ids/3", "/concept_ids/4"}
    assert all("allowed:" in issue.detail and not issue.user_resolvable
               for issue in report.issues)


def test_wall_shape_maps_error_locations_to_json_paths() -> None:
    report = wall_shape(CausalContextV1, result({"schema_version": "causal-context.v1"}))
    assert codes(report) == {"shape_invalid"}
    assert "/payload/frame" in {issue.json_path for issue in report.issues}


def test_wall_shape_carries_the_pydantic_message() -> None:
    report = wall_shape(CausalContextV1, result({"schema_version": "causal-context.v1"}))
    assert all(issue.detail for issue in report.issues)
    assert "Field required" in {issue.detail for issue in report.issues}


# --- wall 2: references ---
def test_wall_references_accepts_resolving_ids() -> None:
    payload = ledger(role(RoleName.GROUP, "c-g", columns=("age",), evidence=("ua:answer-1",)))
    assert wall_references(payload, result(), context(user_answer_evidence_ids={"ua:answer-1"})
                           ).passed


def test_wall_references_flags_an_unknown_column() -> None:
    report = wall_references(ledger(role(RoleName.GROUP, "c-g", columns=("nope",))), result(),
                             context())
    assert codes(report) == {"unresolved_column"}
    assert not report.issues[0].user_resolvable
    assert report.issues[0].artifact_ids == ("nope",)
    assert report.issues[0].json_path == "/claims/2/column_refs/0"


def test_wall_references_flags_evidence_artifact_and_requirement_ids() -> None:
    report = wall_references(
        TypedReferenceProbe(
            source="ev:doc/x", diagnostic="arm_counts",
            artifact=ArtifactRef(artifact_id="art-9", content_hash=HASH)),
        result(missing_requirements=(requirement("nope"),)),
        context())
    assert codes(report) == {
        "unresolved_evidence", "unresolved_diagnostic", "unresolved_artifact",
        "unresolved_requirement"}
    assert {issue.json_path for issue in report.issues} == {
        "/source", "/diagnostic", "/artifact/artifact_id",
        "/missing_requirements/0/requirement_id"}


def test_wall_references_checks_metadata_typed_fields_regardless_of_name() -> None:
    report = wall_references(
        TypedReferenceProbe(
            source="table_facts", diagnostic="arm_counts",
            artifact=ArtifactRef(artifact_id="art-1", content_hash=HASH)),
        result(), context(diagnostic_ids=frozenset({"arm_counts"}), parents={"art-1": HASH}))
    assert codes(report) == {"unresolved_evidence"}
    assert report.issues[0].json_path == "/source"


def test_wall_references_does_not_use_a_suffix_exception_list() -> None:
    payload = UntypedSuffixProbe(future_fact_evidence_ids=("invented",))
    assert wall_references(payload, result(), context()).passed


def test_requirement_scope_is_registry_and_context_owned() -> None:
    ctx = context(dataset_id="dataset-1", concept_ids=frozenset({"c-x"}),
                  relationship_ids=frozenset({"e-x"}))
    assert canonical_requirement_scope(
        registered_requirement("design.assignment_mechanism", "invented"), ctx) == "design"
    assert canonical_requirement_scope(
        registered_requirement("dataset.sampling_mechanism", "invented"), ctx) == "dataset-1"
    assert canonical_requirement_scope(
        registered_requirement("column.meaning", "nsw.csv::age"), ctx) == "age"
    assert canonical_requirement_scope(
        registered_requirement("design.concept_mapping", "c-x"), ctx) == "c-x"
    assert canonical_requirement_scope(
        registered_requirement("design.treatment_descendants", "e-x"), ctx) == "e-x"
    assert canonical_requirement_scope(
        registered_requirement("design.concept_mapping", "invented"), ctx) is None


def test_wall_evidence_rejects_an_invented_registered_scope() -> None:
    sealed = result(graph(), missing_requirements=(
        registered_requirement("design.concept_mapping", "invented"),))
    report = wall_evidence(graph(), sealed, context(concept_ids=frozenset({"c-x"})))
    assert codes(report) == {"invalid_requirement_scope"}


# --- wall 3: evidence ---
def test_wall_evidence_rejects_an_unsupported_role_binding() -> None:
    payload = ledger(role(RoleName.CONFOUNDER_CANDIDATE, "c-x", evidence=()), role(RoleName.GROUP, "c-x").model_copy(update={"status": EpistemicStatus.HYPOTHESIS}))
    assert codes(wall_evidence(payload, result(), context())) == {"role_claim_unsupported"}
    asked = result(missing_requirements=(requirement("design.treatment_meaning"), requirement("column.measurement_timing").model_copy(update={"attempted_evidence": (AttemptedEvidenceV1(evidence_id="ua:answer-1", availability_status="evidenced"),)})))
    report = wall_evidence(ledger(), asked, context())
    assert codes(report) == {"invalid_context_requirement"} and len(report.issues) == 2


def test_wall_evidence_derives_support_class_from_the_citation() -> None:
    claim = role(RoleName.GROUP, "c-g", columns=("age",), evidence=("ua:answer-1",)
                 ).model_copy(update={"support_class": SupportClass.MEASURED_OBSERVATION})
    report = wall_evidence(ledger(claim), result(), context())
    assert codes(report) == {"role_support_class_mismatch"}


@pytest.mark.parametrize(("kind", "timing"), [("identifier", EpistemicStatus.UNKNOWN), ("continuous", EpistemicStatus.EVIDENCED)])
def test_wall_evidence_rejects_redundant_timing_asks(kind: str, timing: EpistemicStatus) -> None:
    slots = dict.fromkeys(COLUMN_CARD_SLOTS, SlotAssertionV1(value=None, status=EpistemicStatus.UNKNOWN, evidence_ids=()))
    slots["kind"] = SlotAssertionV1(value=kind, status=EpistemicStatus.EVIDENCED,
                                     evidence_ids=("ev:kaggle/column/nsw.csv/age/description",))
    slots["timing"] = SlotAssertionV1(value="pre_treatment" if timing.value == "evidenced" else None, status=timing, evidence_ids=("ua:answer-1",) if timing.value == "evidenced" else ())
    card = ColumnSemanticCardV1(table_name="nsw.csv", column_name="age", display_name="age",
                                concept_id="c-x", timing=TimingClass.UNKNOWN, slots=slots,
                                alternatives=(), conflicts=())
    req = requirement("column.measurement_timing").model_copy(update={"attempted_evidence": (
        AttemptedEvidenceV1(evidence_id="ua:answer-1", availability_status="not_offered"),)})
    sealed = result(card, missing_requirements=(req,))
    assert codes(validate_result(3, "semantic_batch", ColumnSemanticCardV1, sealed,
                                 context(evidence_ids={"ev:kaggle/column/nsw.csv/age/description"}))) == {"invalid_context_requirement"}
    assert sealed.payload["slots"]["kind"]["evidence_ids"] == ["ev:kaggle/column/nsw.csv/age/description"]


# --- wall 4: temporal rows ---
@pytest.mark.parametrize(("name", "timing", "expected"), [
    (RoleName.CONFOUNDER_CANDIDATE, TimingClass.POST_TREATMENT, {"post_treatment_role_forbidden"}),
    (RoleName.ASSIGNMENT_VARIABLE, TimingClass.POST_TREATMENT, {"post_treatment_role_forbidden"}),
    (RoleName.MEDIATOR, TimingClass.PRE_TREATMENT, {"mediator_timing_invalid"}),
    (RoleName.RUNNING_VARIABLE, TimingClass.POST_TREATMENT, {"running_variable_timing_invalid"}),
    (RoleName.CONFOUNDER_CANDIDATE, TimingClass.PRE_TREATMENT, set()),
])
def test_wall_temporal_rows(name: RoleName, timing: TimingClass, expected: set[str]) -> None:
    payload = ledger(role(name, "c-x", timing=timing))
    assert codes(wall_temporal(payload, None, context())) == expected


def test_validation_canonicalizes_only_typed_reference_syntax() -> None:
    raw = ledger().model_dump(mode="json")
    raw["claims"][0].update(timing="pre_treatment", column_refs=["nsw.csv/treat"],
                            evidence_ids=["ua:answer-1:"])
    sealed = result().model_copy(update={"payload": raw})
    report = validate_result(5, "role_ledger", RoleLedgerV1, sealed,
                             context(causal_context=graph()))
    assert codes(report) == {"treatment_timing_invalid"}
    assert tuple(sealed.payload["claims"][0][key]
                 for key in ("timing", "column_refs", "evidence_ids")) == (
                     "pre_treatment", ["treat"], ["ua:answer-1"])


@pytest.mark.parametrize("bad", ("UA:ANSWER-1", "prefix/ua:answer-1"))
def test_evidence_identity_rejects_fuzzy_matches(bad: str) -> None:
    raw = ledger().model_dump(mode="json")
    raw["claims"][0]["evidence_ids"] = [bad]
    sealed = result().model_copy(update={"payload": raw})
    report = validate_result(5, "role_ledger", RoleLedgerV1, sealed,
                             context(causal_context=graph()))
    assert codes(report) == {"unresolved_evidence"}
    assert report.issues[0].allowed_actions == ("revise_field",)
    assert "allowed: ua:answer-1" in report.issues[0].detail


# --- wall 5: causal graph ---
CYCLE = graph(edge("e-1", "c-treat", "c-earn"), edge("e-2", "c-earn", "c-treat"))


def test_wall_causal_accepts_an_acyclic_graph() -> None:
    assert wall_causal(graph(edge("e-1", "c-treat", "c-earn")), None, context()).passed
    assert codes(wall_causal(graph(target=False), None, context())) == {"causal_target_edge_missing"}


def test_wall_causal_finds_a_cycle() -> None:
    assert codes(wall_causal(CYCLE, None, context())) == {"graph_cycle"}


def test_wall_causal_reads_the_selected_alternative_edges() -> None:
    alternative = GraphAlternativeV1(alternative_id="alt-1", label="reverse", edges=CYCLE.edges)
    base = graph(edge("e-1", "c-treat", "c-earn"), alternatives=(alternative,))
    assert wall_causal(base, None, context()).passed
    assert codes(wall_causal(base, None, context(selected_alternative_id="alt-1"))) == {
        "graph_cycle"}


def test_wall_causal_requires_a_disputed_edge_to_appear_in_an_alternative() -> None:
    disputed = graph(edge("e-1", "c-treat", "c-earn", EpistemicStatus.DISPUTED))
    assert codes(wall_causal(disputed, None, context())) == {"disputed_edge_without_alternative"}


def test_opposite_alternative_edges_cannot_share_the_rct_relation_identity() -> None:
    shared_id = "e:randomization_stratum_influences_treatment_assignment"
    forward = edge(shared_id, "c-x", "c-treat", EpistemicStatus.EVIDENCED)
    reverse = edge(shared_id, "c-treat", "c-x", EpistemicStatus.DISPUTED)
    alternative = GraphAlternativeV1(
        alternative_id="alt:treatment_causes_stratum", label="reverse direction",
        edges=(forward, reverse))
    report = wall_causal(graph(forward, alternatives=(alternative,)), None, context())
    assert codes(report) == {"duplicate_graph_edge_id", "conflicting_graph_edge_identity"}
    assert all(issue.json_path == "/alternatives/0/edges/1/edge_id" for issue in report.issues)
    assert "c-x -> c-treat" in report.issues[1].detail
    assert "c-treat -> c-x" in report.issues[1].detail
    corrected = alternative.model_copy(update={"edges": (forward,)})
    assert wall_causal(graph(forward, alternatives=(corrected,)), None, context()).passed


def test_duplicate_identical_edges_inside_one_graph_are_rejected_without_rewriting() -> None:
    relation = edge("repeated", "c-x", "c-treat")
    graph_body = graph(relation, relation)
    before = graph_body.canonical_payload()
    report = wall_causal(graph_body, None, context())
    assert codes(report) == {"duplicate_graph_edge_id"}
    assert graph_body.canonical_payload() == before


def test_nhefs_outcome_change_is_not_an_observed_time_coordinate() -> None:
    payload = ledger(role(RoleName.TIME, "c-x", columns=("weight_change",),
                          timing=TimingClass.CONCURRENT))
    payload = payload.model_copy(update={"claims": tuple(
        row.model_copy(update={"column_refs": ("weight_change",)})
        if row.role is RoleName.OUTCOME else row for row in payload.claims)})
    before = payload.canonical_payload()
    report = wall_causal(payload, None, context(causal_context=graph()))
    assert codes(report) == {"time_role_is_outcome_measurement"}
    assert report.issues[0].json_path == "/claims/2/column_refs"
    assert "Preserve the outcome role and frame.timeframe" in report.issues[0].detail
    assert payload.canonical_payload() == before
    corrected = payload.model_copy(update={"claims": payload.claims[:2]})
    assert wall_causal(corrected, None, context(causal_context=graph())).passed


def test_nhefs_study_window_stays_metadata_when_no_time_column_is_observed() -> None:
    common = ConceptProposalV1(name="cohort", description="", candidate_columns=())
    timeframe = ConceptProposalV1(name="1971 to 1982", description="Study follow-up window",
                                   candidate_columns=("wt71", "weight_change"))
    outcome = ConceptProposalV1(name="weight change", description="Kilograms from 1971 to 1982",
                                 candidate_columns=("weight_change",))
    intent = DesignIntentV1(
        question_kind=QuestionKind.CAUSAL, causal_claim="Quitting affects weight change",
        intended_decision="estimate ATE", treatment=common, outcome=outcome, population=common,
        comparator=common, unit=common, timeframe=timeframe,
        candidate_grain="one_row_per_unit", mandatory_concepts=())
    report = wall_evidence(intent, None, context())
    assert codes(report) == {"timeframe_is_outcome_measurement"}
    assert report.issues[0].json_path == "/timeframe/candidate_columns"
    fixed = intent.model_copy(update={"timeframe": timeframe.model_copy(update={"candidate_columns": ()})})
    assert wall_evidence(fixed, None, context()).passed
    assert fixed.timeframe.name == intent.timeframe.name and fixed.outcome == intent.outcome


def test_wall_causal_resolves_every_role_edge_id() -> None:
    payload = ledger(role(RoleName.GROUP, "c-x", edges=("e-missing",)))
    ctx = context(causal_context=graph())
    report = wall_references(payload, result(payload), ctx)
    assert codes(report) == {"unresolved_graph_edge"}
    assert report.issues[0].json_path.endswith("/graph_edge_ids/0")


def test_wall_causal_rejects_role_ledger_frame_drift_and_unbound_anchors() -> None:
    changed = CausalFrameV1(treatment="c-other", outcome="c-earn", population="p", timeframe="t")
    payload = RoleLedgerV1(frame=changed, claims=(
        role(RoleName.TREATMENT, "c-other"), role(RoleName.OUTCOME, "c-earn",
                                                  timing=TimingClass.POST_TREATMENT)))
    assert codes(wall_causal(payload, None, context(causal_context=graph()))) == {
        "causal_frame_changed", "causal_target_edge_missing", "required_role_columns_missing"}


def test_wall_causal_rejects_multiple_columns_for_a_single_role() -> None:
    payload = ledger(role(RoleName.GROUP, "c-x", columns=("age",)),
                     role(RoleName.GROUP, "c-g", columns=("period",)))
    assert codes(wall_causal(payload, None, context(causal_context=graph()))) == {
        "ambiguous_single_role"}


@pytest.mark.parametrize(("name", "pairs", "expected"), [
    (RoleName.CONFOUNDER_CANDIDATE, ("c-x>c-treat",), {"confounder_edges_missing"}),
    (RoleName.CONFOUNDER_CANDIDATE, ("c-x>c-treat", "c-x>c-earn"), set()),
    (RoleName.MEDIATOR, ("c-treat>c-x",), {"mediator_edges_missing"}),
    (RoleName.MEDIATOR, ("c-treat>c-x", "c-x>c-earn"), set()),
    (RoleName.COLLIDER, ("c-treat>c-x",), {"collider_edges_missing"}),
    (RoleName.COLLIDER, ("c-treat>c-x", "c-earn>c-x"), set()),
    (RoleName.INSTRUMENT_CANDIDATE, ("c-x>c-treat", "c-x>c-earn"), {"instrument_exclusion_violated"}),
    (RoleName.INSTRUMENT_CANDIDATE, ("c-x>c-treat",), set()),
])
def test_role_graph_rows(name: RoleName, pairs: tuple[str, ...], expected: set[str]) -> None:
    built = graph(*(edge(f"e-{index}", source, target, EpistemicStatus.EVIDENCED)
                    for index, (source, target) in enumerate(pair.split(">") for pair in pairs)))
    ctx = context(role_ledger=ledger(role(name, "c-x")))
    assert codes(wall_causal(built, None, ctx)) == expected


def test_confounder_graph_correction_explains_the_required_relationships() -> None:
    report = wall_causal(graph(edge("e-1", "c-x", "c-treat")), None,
                         context(role_ledger=ledger(role(RoleName.CONFOUNDER_CANDIDATE, "c-x"))))
    assert report.issues[0].code == "confounder_edges_missing"
    assert "both the frame treatment and frame outcome" in report.issues[0].detail


# --- wall 6: bounded model proposal ---
def test_wall_method_accepts_a_complete_ranked_proposal() -> None:
    assert wall_method(proposal(), None, context()).passed


def test_wall_method_requires_every_registered_method_exactly_once() -> None:
    report = wall_method(proposal(ranked_method_ids=("did", "did")), None, context())
    assert codes(report) == {"method_ranking_incomplete"}


def test_wall_method_requires_an_explicit_estimand() -> None:
    report = wall_method(proposal(requested_estimand="unknown"), None, context())
    assert codes(report) == {"estimand_missing"}


def test_wall_method_rejects_estimand_drift_in_the_preferred_compatible_method() -> None:
    ranked = ("did", "aipw", "randomized_experiment", "sharp_rdd")
    report = wall_method(
        proposal(ranked_method_ids=ranked, requested_estimand="att"), None, context())
    assert codes(report) == {"preferred_method_estimand_mismatch"}


# --- the declarative registry ---
def test_every_row_uses_a_known_kind_and_correction_actions() -> None:
    assert {rule.kind for rule in RULES} == set(RULE_KINDS)
    assert all(set(rule.allowed_actions) <= set(ACTIONS) for rule in RULES)


@pytest.mark.parametrize(("body", "code"), [
    (rule_file(kind="vibes"), "unknown_rule_kind"),
    (rule_file(wall=9), "unknown_wall"),
    ('{"registry_version": "other.v1", "rules": []}', "invalid_registry_file"),
    ("{not json", "invalid_registry_file"),
])
def test_the_rule_loader_fails_closed(tmp_path: Path, body: str, code: str) -> None:
    path = tmp_path / "rules.json"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(PackRegistryError) as error:
        load_validation_rules(path)
    assert error.value.code == code


# --- evidence classes and wall order ---
@pytest.mark.parametrize(("evidence_id", "expected"), [
    ("ua:answer-1", EvidenceClass.USER_CONFIRMATION),
    ("ev:kaggle/column/nsw/treat", EvidenceClass.DATA_DICTIONARY),
    ("ev:kaggle/file/nsw.csv", EvidenceClass.DATA_DICTIONARY),
    ("ev:kaggle/dataset/nsw", EvidenceClass.SOURCE_STATEMENT),
    ("ev:doc/readme", EvidenceClass.SOURCE_STATEMENT),
    ("ev:profile/nsw.csv#/columns/3", EvidenceClass.MEASURED_OBSERVATION),
    ("mystery", None),
])
def test_evidence_class_prefix_map(evidence_id: str, expected: EvidenceClass | None) -> None:
    assert evidence_class(evidence_id) == expected


def test_validate_result_stops_at_the_first_failing_wall() -> None:
    payload = ledger(role(RoleName.CONFOUNDER_CANDIDATE, "c-x", columns=("nope",),
                          timing=TimingClass.POST_TREATMENT))
    report = validate_result(4, "role_evidence", RoleLedgerV1, result(payload), context())
    assert report.wall == 2 and codes(report) == {"unresolved_column"}


def test_validate_result_runs_every_wall_the_task_kind_allows() -> None:
    passing = validate_result(4, "role_evidence", RoleLedgerV1, result(ledger()), context())
    assert passing.passed and passing.wall == 4
    capped = validate_result(7, "role_ledger", RoleLedgerV1, result(ledger()), context())
    assert capped.wall == 5
