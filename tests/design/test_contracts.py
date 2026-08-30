"""Design payload contracts: happy paths and boundary rules (T-009 §6; PRD-002 §11–§18)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from causal.design.contracts import (
    REGISTRY_VERSION_KEYS,
    AnswerItemV1,
    AnswerKind,
    ApprovalDecision,
    AvailabilityRowV1,
    ConceptProposalV1,
    DesignApprovalDecisionV1,
    DesignContextManifestV1,
    DesignIntentV1,
    InterruptKind,
    QuestionItemV1,
    QuestionKind,
    SelectionSource,
    StructuralFieldV1,
    TableSelectionDecisionV1,
    TableSelectionV1,
    UserContextAnswerV1,
    UserQuestionPacketV1,
    _Payload,
)
from causal.design.frame import (
    CAPACITY_DIMENSIONS,
    DESIGN_COUNT_KEYS,
    CapacityStatus,
    CausalGraphViewV1,
    DeliveryCapacityCheckV1,
    DesignOutcomeStatus,
    DesignOutcomeV1,
    DiagnosticResultV1,
    DiagnosticStatus,
    ExperimentDesignV1,
    GraphEdgeViewV1,
    GraphNodeViewV1,
    PreRepairFeasibilityReportV1,
    RunnableFrameContractV1,
)
from causal.design.semantics import (
    COLUMN_CARD_SLOTS,
    CausalContextV1,
    CausalEdgeV1,
    ColumnSemanticCardV1,
    ConceptStatus,
    ConceptV1,
    GraphAlternativeV1,
    MeasurementLinkV1,
    MeasurementMapV1,
    MeasurementRelation,
    RoleClaimV1,
    RoleEvidenceV1,
    RoleLedgerV1,
    RoleName,
    SlotAssertionV1,
    TimingClass,
)
from causal.shared.canonical import content_hash
from causal.shared.contracts import ArtifactRef
from causal.shared.envelope import CausalFrameV1, ClaimV1, EpistemicStatus, SupportClass

HASH = "a" * 64
REF = ArtifactRef(artifact_id="art-1", content_hash=HASH)
FRAME = CausalFrameV1(
    treatment="c-treat", outcome="c-earnings", population="applicants", timeframe="1975-1978"
)
CLAIM = ClaimV1(
    claim_id="claim-1", subject_kind="column", subject_id="nsw.csv:re78", predicate="measures",
    value="earnings in 1978", epistemic_status=EpistemicStatus.EVIDENCED,
    supporting_evidence_ids=("ev-1",), contrary_evidence_ids=(),
    support_class=SupportClass.DIRECT_SOURCE_STATEMENT, alternatives=(), causal_frame=None,
)
SLOT = SlotAssertionV1(
    value="1978 earnings", status=EpistemicStatus.EVIDENCED, evidence_ids=("ev-1",)
)
SLOTS = dict.fromkeys(COLUMN_CARD_SLOTS, SLOT)
REGISTRY_VERSIONS = dict.fromkeys(REGISTRY_VERSION_KEYS, "artifact-types.v1")
CARDINALITIES = dict.fromkeys(CAPACITY_DIMENSIONS, 2)
COUNTS = dict.fromkeys(DESIGN_COUNT_KEYS, 1)


def build(model: type[BaseModel], base: dict[str, object], **overrides: object) -> BaseModel:
    """Construct `model` from a minimal valid kwargs dict, with per-test overrides."""
    return model(**{**base, **overrides})


# --- fragment fixtures (plain rows, reused inside the payload bases) ---
FIELD = StructuralFieldV1(table_name="nsw.csv", column_name="re78", dtype="float64", ordinal=0)
AVAILABILITY = AvailabilityRowV1(
    scope_kind="column", table_name="nsw.csv", column_name="re78", field_or_slot_name="meaning",
    status="present", evidence_count=1, json_pointer="/columns/0",
)
PROPOSAL = ConceptProposalV1(
    name="treatment", description="the training program", candidate_columns=("treat",)
)
QUESTION = QuestionItemV1(
    question_id="q-1", requirement_ids=("req-1",), question_text="When is re78 measured?",
    why_it_matters="timing decides whether re78 can be the outcome",
    blocked_decisions=("role_assignment",), expected_answer_schema="free-text.v1",
)
ANSWER = AnswerItemV1(question_id="q-1", answer_kind=AnswerKind.VALUE, value="1978")
CONCEPT = ConceptV1(
    concept_id="c-earnings", name="earnings", description="post-program earnings",
    status=ConceptStatus.OBSERVED,
)
LINK = MeasurementLinkV1(
    concept_id="c-earnings", table_name="nsw.csv", column_name="re78",
    relation=MeasurementRelation.MEASURES, timing=TimingClass.POST_TREATMENT,
    notes="the column states the year",
)
EDGE = CausalEdgeV1(
    edge_id="e-1", source_concept_id="c-treat", target_concept_id="c-earnings",
    timeframe="1975-1978", mechanism_summary="training raises skill and so earnings",
    supporting_evidence_ids=("ev-1",), contrary_evidence_ids=(),
    status=EpistemicStatus.HYPOTHESIS, differing_alternative_ids=(),
)
STRAY_EDGE = EDGE.model_copy(update={"edge_id": "e-2", "target_concept_id": "c-unknown"})
ALTERNATIVE = GraphAlternativeV1(alternative_id="alt-1", label="selection story", edges=(EDGE,))
TREATMENT_ROLE = RoleClaimV1(
    role=RoleName.TREATMENT, concept_id="c-treat", column_refs=("treat",), evidence_ids=("ev-1",),
    timing=TimingClass.PRE_TREATMENT, graph_edge_ids=("e-1",),
    support_class=SupportClass.DIRECT_SOURCE_STATEMENT, alternatives=(),
    status=EpistemicStatus.EVIDENCED, methods=("difference-in-differences",),
)
OUTCOME_ROLE = TREATMENT_ROLE.model_copy(
    update={"role": RoleName.OUTCOME, "concept_id": "c-earnings", "column_refs": ("re78",)}
)
NODES = (
    GraphNodeViewV1(
        concept_id="c-treat", label="Program", status=ConceptStatus.OBSERVED,
        roles=(RoleName.TREATMENT,),
    ),
    GraphNodeViewV1(
        concept_id="c-earnings", label="Earnings", status=ConceptStatus.OBSERVED,
        roles=(RoleName.OUTCOME,),
    ),
)
EDGE_VIEW = GraphEdgeViewV1(
    edge_id="e-1", source_concept_id="c-treat", target_concept_id="c-earnings",
    status=EpistemicStatus.HYPOTHESIS,
)
REJECTED = {
    "regression-discontinuity": "no running variable",
    "instrumental-variables": "no instrument survives exclusion",
    "propensity-score-matching": "weaker identification under the same overlap",
}
# still exactly three entries, but one of them is the selected method
REJECTED_WITH_SELECTED = {**REJECTED, "difference-in-differences": "kept anyway"}
REJECTED_WITH_SELECTED.pop("regression-discontinuity")

# --- minimal valid kwargs per model; every test overrides one field of one of these ---
DIAGNOSTIC_KWARGS: dict[str, object] = {
    "diagnostic_id": "overlap", "diagnostic_version": "diagnostics.v1",
    "status": DiagnosticStatus.COMPUTED, "csv_artifact": REF, "columns_read": ("treat", "re78"),
    "total_rows": 445, "used_rows": 445, "unused_reason_counts": {"missing_outcome": 0},
    "row_set_hash": HASH, "values": {"overlap_share": 0.82}, "warnings": (),
    "implementation_version": "diagnostics-0.1.0",
}
DIAGNOSTIC = build(DiagnosticResultV1, DIAGNOSTIC_KWARGS)

TABLE_SELECTION: dict[str, object] = {
    "dataset_id": "kaggle:lalonde/nsw@3", "logical_name": "nsw.csv",
    "resource_object_locator": f"objects/{HASH}", "resource_sha256": HASH, "candidate_count": 1,
    "selection_source": SelectionSource.ONLY_CANDIDATE, "decision_artifact_id": None,
}
MANIFEST: dict[str, object] = {
    "design_revision": 1, "question_artifact": REF, "intake_outcome_artifact": REF,
    "table_selection_artifact": REF, "selected_table": "nsw.csv",
    "structural_inventory": (FIELD,), "semantic_available": (AVAILABILITY,),
    "semantic_missing": (), "measured_surface": (), "provenance_surface": (),
    "retrieval_surfaces": ("surface-1",), "registry_versions": REGISTRY_VERSIONS,
    "recipient_map": {"column_card": ("surface-1",)},
}
INTENT: dict[str, object] = {
    "question_kind": QuestionKind.CAUSAL, "causal_claim": "the program raises earnings",
    "intended_decision": "whether to expand the program", "treatment": PROPOSAL,
    "outcome": PROPOSAL, "population": PROPOSAL, "comparator": PROPOSAL, "unit": PROPOSAL,
    "timeframe": PROPOSAL, "candidate_grain": "person", "mandatory_concepts": (PROPOSAL,),
    "claims": (CLAIM,),
}
PACKET: dict[str, object] = {
    "packet_id": "pk-1", "design_revision": 1, "round_number": 1, "questions": (QUESTION,),
}
ANSWER_ITEM: dict[str, object] = {
    "question_id": "q-1", "answer_kind": AnswerKind.VALUE, "value": "1978",
}
ANSWERS: dict[str, object] = {"packet_id": "pk-1", "answers": (ANSWER,)}
TABLE_DECISION: dict[str, object] = {
    "interrupt_id": "int-1", "expected_interrupt_hash": HASH, "expected_revision": 1,
    "selected_table": "nsw.csv", "idempotency_key": "idem-1",
}
APPROVAL: dict[str, object] = {
    "interrupt_id": "int-2", "expected_interrupt_hash": HASH, "expected_revision": 1,
    "decision": ApprovalDecision.APPROVED, "approved_artifacts": (REF,), "change_requests": (),
    "idempotency_key": "idem-2",
}
CARD: dict[str, object] = {
    "table_name": "nsw.csv", "column_name": "re78", "display_name": "Earnings 1978",
    "concept_id": "c-earnings", "timing": TimingClass.POST_TREATMENT, "slots": SLOTS,
    "claims": (CLAIM,), "alternatives": (), "conflicts": (),
}
MEASUREMENT_MAP: dict[str, object] = {
    "concepts": (CONCEPT,), "links": (LINK,), "claims": (CLAIM,),
}
CAUSAL_CONTEXT: dict[str, object] = {
    "frame": FRAME, "concept_ids": ("c-treat", "c-earnings"), "edges": (EDGE,),
    "alternatives": (ALTERNATIVE,), "selection_notes": "one mechanism, one live alternative",
    "claims": (CLAIM,),
}
ROLE_LEDGER: dict[str, object] = {
    "frame": FRAME, "claims": (TREATMENT_ROLE, OUTCOME_ROLE),
}
ROLE_EVIDENCE: dict[str, object] = {
    "assigned_scope": ("re78",), "edge_hypotheses": (EDGE,), "role_hypotheses": (TREATMENT_ROLE,),
    "competing_mechanisms": ("selection into the program",), "claims": (CLAIM,),
}
PRE_REPAIR: dict[str, object] = {
    "method_id": "difference-in-differences", "results": (DIAGNOSTIC,),
}
EXPERIMENT_DESIGN: dict[str, object] = {
    "causal_question": "Does the program raise earnings?",
    "intended_decision": "whether to expand the program", "selected_csv": REF,
    "method_id": "difference-in-differences", "method_pack_version": "method-packs.v1",
    "rejected_methods": REJECTED, "frame": FRAME, "comparator": "non-participants",
    "unit": "person", "estimand": "att", "measurement_map": REF, "causal_context": REF,
    "role_ledger": REF, "assumptions": ("parallel trends",),
    "identification_risks": ("selection on gains",), "eligibility_rules": ("age >= 18",),
    "mandatory_repair_boundaries": ("no outcome imputation",),
    "forbidden_repair_boundaries": ("no row deletion on treatment",),
    "imputation_eligible_columns": ("age",), "imputation_forbidden_columns": ("re78",),
    "deletion_impact_dimensions": ("treatment_arm",),
    "invalidation_conditions": ("the pre-trend fails",),
    "required_prerepair_diagnostics": ("overlap",),
    "required_postrepair_diagnostics": ("balance",), "required_visual_evidence": ("event-study",),
    "primary_contrasts": ("treated vs control",), "multiplicity_policy": None,
    "capacity_check": None, "visualization_catalog_version": "visual-catalog.v1",
    "capacity_registry_version": "capacity.v1", "sensitivity_requirements": ("placebo outcome",),
    "registry_versions": REGISTRY_VERSIONS,
}
RUNNABLE_FRAME: dict[str, object] = {
    "selected_csv": REF, "output_grain": "one row per person", "key_columns": ("person_id",),
    "required_roles": (RoleName.TREATMENT, RoleName.OUTCOME),
    "allowed_roles": (RoleName.CONFOUNDER_CANDIDATE,), "forbidden_roles": (RoleName.COLLIDER,),
    "type_constraints": {"re78": "float64"}, "uniqueness_constraints": ("person_id is unique",),
    "eligibility_rules": ("age >= 18",), "exclusion_reason_vocabulary": ("missing_outcome",),
    "treatment_missingness_rule": "drop the row", "outcome_missingness_rule": "drop the row",
    "method_structure": {"design": "difference-in-differences"},
    "imputation_permitted": ("age",), "imputation_forbidden": ("re78",),
    "required_missingness_indicators": ("re78_missing",),
    "deletion_impact_dimensions": ("treatment_arm",),
    "revision_required_conditions": ("overlap fails after repair",),
    "feasibility_gates": ("overlap_share >= 0.1",), "required_final_diagnostics": ("balance",),
    "estimator_input_schema": "estimator-input.v1", "experiment_design_hash": HASH,
}
GRAPH_VIEW: dict[str, object] = {
    "parents": (REF,), "nodes": NODES, "edges": (EDGE_VIEW,), "selected_alternative_id": None,
    "layout_direction": "LR", "renderer_profile": "graphviz.v1",
    "legend_text": "dashed edges are hypotheses", "disclosure_text": "not a validated graph",
    "spec_hash": HASH, "svg": "<svg/>", "accessible_summary": "two nodes and one edge",
    "node_edge_table": "c-treat -> c-earnings", "renderer_version": "graphviz-0.21",
    "theme_version": "theme.v1", "validator_version": "causal-graph-view-validator.v1",
    "validation_status": "passed",
}
CAPACITY: dict[str, object] = {
    "method_id": "difference-in-differences", "method_profile_id": "did-standard",
    "cardinalities": CARDINALITIES, "required_visual_evidence": ("event-study",),
    "compatible_templates": ("two-arm",), "template_limits": {"two-arm": 4},
    "accessible_table_capacity": 20, "execution_concurrency": 2, "render_concurrency": 2,
    "status": CapacityStatus.PASS, "failure_codes": (),
    "visualization_catalog_version": "visual-catalog.v1",
    "capacity_registry_version": "capacity.v1", "method_registry_version": "method-packs.v1",
}
OUTCOME: dict[str, object] = {
    "status": DesignOutcomeStatus.APPROVED, "design_revision": 1, "refusal_code": None,
    "error_code": None, "experiment_design": REF, "runnable_frame_contract": REF,
    "causal_graph_view": REF, "capacity_check": REF, "approval": REF,
    "open_requirement_ids": (), "clarification_rounds_used": 1, "counts": COUNTS,
}

PAYLOADS: list[tuple[type[_Payload], dict[str, object]]] = [
    (TableSelectionV1, TABLE_SELECTION),
    (DesignContextManifestV1, MANIFEST),
    (DesignIntentV1, INTENT),
    (UserQuestionPacketV1, PACKET),
    (UserContextAnswerV1, ANSWERS),
    (TableSelectionDecisionV1, TABLE_DECISION),
    (DesignApprovalDecisionV1, APPROVAL),
    (ColumnSemanticCardV1, CARD),
    (MeasurementMapV1, MEASUREMENT_MAP),
    (CausalContextV1, CAUSAL_CONTEXT),
    (RoleLedgerV1, ROLE_LEDGER),
    (RoleEvidenceV1, ROLE_EVIDENCE),
    (PreRepairFeasibilityReportV1, PRE_REPAIR),
    (ExperimentDesignV1, EXPERIMENT_DESIGN),
    (RunnableFrameContractV1, RUNNABLE_FRAME),
    (CausalGraphViewV1, GRAPH_VIEW),
    (DeliveryCapacityCheckV1, CAPACITY),
    (DesignOutcomeV1, OUTCOME),
]

BOUNDARIES: list[tuple[type[BaseModel], dict[str, object], dict[str, object], str]] = [
    (TableSelectionV1, TABLE_SELECTION, {"media_type": "text/tsv"}, "text/csv"),
    (UserQuestionPacketV1, PACKET, {"questions": (QUESTION,) * 6}, "at most 5"),
    (UserQuestionPacketV1, PACKET, {"round_number": 3}, "should be 1 or 2"),
    (ColumnSemanticCardV1, CARD, {"slots": {**SLOTS, "vibe": SLOT}}, "slots key mismatch"),
    (AnswerItemV1, ANSWER_ITEM, {"answer_kind": AnswerKind.UNKNOWN}, "value must be None"),
    (AnswerItemV1, ANSWER_ITEM, {"value": None}, "value must be None"),
    (DesignApprovalDecisionV1, APPROVAL, {"approved_artifacts": ()},
     "approved_artifacts is non-empty"),
    (DesignApprovalDecisionV1, APPROVAL,
     {"decision": ApprovalDecision.CHANGES_REQUESTED, "approved_artifacts": ()},
     "change_requests is non-empty"),
    (DesignOutcomeV1, OUTCOME, {"capacity_check": None}, "an approved outcome requires"),
    (DesignOutcomeV1, OUTCOME, {"status": DesignOutcomeStatus.REFUSED},
     "requires a refusal_code"),
    (DeliveryCapacityCheckV1, CAPACITY, {"failure_codes": ("CAPACITY_EXCEEDED",)},
     "failure_codes must be empty"),
    (DeliveryCapacityCheckV1, CAPACITY, {"status": CapacityStatus.FAIL},
     "failure_codes must be empty"),
    (RunnableFrameContractV1, RUNNABLE_FRAME, {"experiment_design_hash": "a" * 63},
     "should match pattern"),
    (ExperimentDesignV1, EXPERIMENT_DESIGN, {"rejected_methods": {"instrumental-variables": "no"}},
     "exactly the three unselected"),
    (ExperimentDesignV1, EXPERIMENT_DESIGN, {"rejected_methods": REJECTED_WITH_SELECTED},
     "cannot also be a rejected method"),
    (CausalContextV1, CAUSAL_CONTEXT, {"edges": (EDGE, STRAY_EDGE)}, "cites unknown concept ids"),
    (RoleLedgerV1, ROLE_LEDGER, {"claims": (TREATMENT_ROLE,)}, "missing a claim for"),
    (DiagnosticResultV1, DIAGNOSTIC_KWARGS, {"used_rows": 446}, "used_rows cannot exceed"),
]


@pytest.mark.parametrize(("model", "base"), PAYLOADS)
class TestCommittedPayloads:
    def test_happy_path_roundtrip(self, model: type[_Payload], base: dict[str, object]) -> None:
        built = model(**base)
        again = model.model_validate(built.model_dump())
        assert again == built
        assert content_hash(again.canonical_payload()) == content_hash(built.canonical_payload())

    def test_extra_field_rejected(self, model: type[_Payload], base: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            build(model, base, surprise="x")


@pytest.mark.parametrize(("model", "base", "overrides", "match"), BOUNDARIES)
def test_boundary_rejected(
    model: type[BaseModel], base: dict[str, object], overrides: dict[str, object], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        build(model, base, **overrides)


def test_interrupt_kinds_cover_the_three_pauses() -> None:
    assert tuple(kind.value for kind in InterruptKind) == (
        "table_selection", "clarification", "approval"
    )
