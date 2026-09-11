"""Boundary tests for retained intake contracts and the V2 compiled-design handoff."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from causal.design.contracts import (
    DiagnosticResultV1,
    DiagnosticStatus,
    InterruptKind,
    SelectionSource,
    SourceInterpretationV1,
    TableSelectionV1,
)
from causal.design.semantics import RoleName
from causal.design.v2 import (
    AgentDesignProposalV2,
    BoundDiagnosticInputV2,
    CapacityReportV2,
    CapacityValueV2,
    CompiledDesignV2,
    DesignApprovalV2,
    DesignFactSetV2,
    DesignFactV2,
    DesignOutcomeV2,
    DesignReviewBundleV2,
    DiagnosticPlanItemV2,
    DiagnosticPlanV2,
    DiagnosticReportV2,
    EvidenceRelation,
    FactAcceptanceStatus,
    FactSource,
    GraphViewSetV2,
    PreparationPolicyV2,
    RoleBindingV2,
)
from causal.shared.canonical import content_hash
from causal.shared.contracts import ArtifactRef
from causal.shared.envelope import CausalFrameV1, EpistemicStatus, EvidenceClass

HASH = "a" * 64
REF = ArtifactRef(artifact_id="artifact:1", content_hash=HASH)
FRAME = CausalFrameV1(
    treatment="treated", outcome="outcome", population="people", timeframe="2025")

TABLE = TableSelectionV1(
    dataset_id="dataset:1", logical_name="analysis.csv",
    resource_object_locator=f"objects/{HASH}", resource_sha256=HASH, candidate_count=1,
    selection_source=SelectionSource.ONLY_CANDIDATE, decision_artifact_id=None)
PROPOSAL = AgentDesignProposalV2(
    assignment_mechanism="self_selected", requested_estimand="att",
    comparator="untreated",
    ranked_method_ids=("aipw", "randomized_experiment", "did", "sharp_rdd"),
    method_facts=(), optional_assumption_ids=(), optional_risk_ids=(),
    optional_sensitivity_ids=(),
    source_interpretations=(
        SourceInterpretationV1(
            fact_key="assignment_mechanism", value="self_selected",
            evidence_id="ev:doc/assignment",
            verbatim_excerpt="self-selected ATT assignment", relation="direct"),
        SourceInterpretationV1(
            fact_key="estimand", value="att", evidence_id="ev:doc/assignment",
            verbatim_excerpt="self-selected ATT assignment", relation="direct"),
        SourceInterpretationV1(
            fact_key="comparator", value="untreated", evidence_id="ev:doc/comparator",
            verbatim_excerpt="untreated comparator", relation="direct"),
    ),
)
FACT = DesignFactV2(
    fact_id="assignment_mechanism", requirement_id="design.assignment_mechanism",
    scope_id="design", value="self_selected", source=FactSource.DOCUMENT,
    source_artifact_ids=("ev:doc/assignment",), evidence_class=EvidenceClass.SOURCE_STATEMENT,
    relation=EvidenceRelation.DIRECT, epistemic_status=EpistemicStatus.EVIDENCED,
    acceptance_status=FactAcceptanceStatus.ACCEPTED,
    executable=True)
BINDING = RoleBindingV2(
    role=RoleName.TREATMENT, columns=("treated",), concept_id="concept:treatment",
    source_artifact_ids=("ledger:1",), epistemic_status=EpistemicStatus.EVIDENCED)
FACTS = DesignFactSetV2(
    selected_csv=REF, grain="one_row_per_unit", facts=(FACT,), role_bindings=(BINDING,),
    conflicts=())
INPUT = BoundDiagnosticInputV2(
    parameter="columns", source_kind="role", source_id="treatment",
    columns=("treated",))
ITEM = DiagnosticPlanItemV2(
    diagnostic_id="arm_counts", primitive="count_by", required_for_eligibility=True,
    inputs=(INPUT,))
PLAN = DiagnosticPlanV2(
    selected_csv=REF, candidate_method_id="aipw", items=(ITEM,), issues=())
RESULT = DiagnosticResultV1(
    diagnostic_id="arm_counts", diagnostic_version="diagnostics.v1",
    status=DiagnosticStatus.COMPUTED, csv_artifact=REF, columns_read=("treated",),
    total_rows=100, used_rows=100, unused_reason_counts={}, row_set_hash=HASH,
    values={"group_count": 2}, warnings=(), implementation_version="diagnostics-0.1.0")
REPORT = DiagnosticReportV2(
    selected_csv=REF, candidate_method_id="aipw", results=(RESULT,), issues=(),
    computable=True)
PREPARATION = PreparationPolicyV2(
    output_grain="one_row_per_unit", key_columns=("_row_unit_id",),
    required_roles=(RoleName.TREATMENT, RoleName.OUTCOME),
    protected_columns=("treated", "outcome"), imputation_permitted=("age",),
    eligibility_rule_ids=("treatment_observed", "outcome_observed"),
    unusable_row_rule_ids=("treatment_observed", "outcome_observed"),
    required_missingness_indicators=("outcome_observed",), method_structure={},
    deletion_impact_dimensions=("treatment_arm",),
    required_final_diagnostic_ids=("balance",), estimator_input_schema_id="aipw.v1")
COMPILED = CompiledDesignV2(
    selected_csv=REF, causal_question="Does treatment change the outcome?",
    intended_decision="whether to expand treatment", frame=FRAME,
    method_id="aipw", method_pack_version="aipw.v1",
    estimand="att", comparator="untreated", unit="person",
    primary_contrasts=("treated_vs_untreated",), role_bindings=(BINDING,),
    preparation=PREPARATION, assumptions=("exchangeability",),
    identification_risks=("limited overlap",), sensitivity_requirements=("overlap trim",),
    rejected_methods={"did": "assignment incompatible"},
    required_visual_evidence=("primary_estimate",), multiplicity_policy_id=None,
    registry_versions={"method_packs": "method-packs.v1"})
CAPACITY = CapacityReportV2(
    compiled_design=REF,
    dimensions=(CapacityValueV2(
        dimension="arms", value=2, applicability="applicable", source="measured"),),
    compatible_template_ids=("estimate_forest.v1",), issues=(), status="pass")
VIEWS = GraphViewSetV2(
    base_view=REF, alternative_views=(), accessible_summaries=("base graph",))
REVIEW = DesignReviewBundleV2(
    compiled_design=REF, diagnostic_report=REF, capacity_report=REF, graph_views=REF,
    rejected_methods=COMPILED.rejected_methods, assumptions=COMPILED.assumptions,
    identification_risks=COMPILED.identification_risks,
    sensitivity_requirements=COMPILED.sensitivity_requirements)
APPROVAL = DesignApprovalV2(
    decision="approved", design_revision=1, review_bundle=REF,
    approved_bundle_hash=REF.content_hash, change_requests=())
OUTCOME = DesignOutcomeV2(
    status="approved", design_revision=1, compiled_design=REF, diagnostic_report=REF,
    capacity_report=REF, review_bundle=REF, approval=REF, issues=())


@pytest.mark.parametrize("payload", [
    TABLE, PROPOSAL, FACTS, PLAN, REPORT, COMPILED, CAPACITY, VIEWS, REVIEW, APPROVAL, OUTCOME,
])
def test_payload_roundtrip_and_hash_are_stable(payload: object) -> None:
    model = type(payload)
    again = model.model_validate(payload.model_dump())
    assert again == payload
    assert content_hash(again.canonical_payload()) == content_hash(payload.canonical_payload())


@pytest.mark.parametrize("payload", [PROPOSAL, FACTS, PLAN, REPORT, COMPILED, CAPACITY,
                                      VIEWS, REVIEW, APPROVAL, OUTCOME])
def test_v2_payloads_reject_extra_fields(payload: object) -> None:
    with pytest.raises(ValidationError):
        type(payload).model_validate(payload.model_dump() | {"surprise": True})


def test_executable_model_fact_is_rejected() -> None:
    with pytest.raises(ValidationError, match="non-model"):
        DesignFactV2(**(FACT.model_dump() | {"source": FactSource.MODEL}))


def test_capacity_applicability_and_value_must_agree() -> None:
    with pytest.raises(ValidationError, match="present exactly"):
        CapacityValueV2(
            dimension="arms", value=None, applicability="applicable", source="profile")


def test_accessible_graph_summary_is_prose_not_a_short_identity() -> None:
    assert GraphViewSetV2(
        base_view=REF, alternative_views=(), accessible_summaries=("x" * 201,)
    ).accessible_summaries == ("x" * 201,)


def test_approval_binds_the_exact_review_bundle_hash() -> None:
    with pytest.raises(ValidationError, match="exact review bundle hash"):
        DesignApprovalV2(
            decision="approved", design_revision=1, review_bundle=REF,
            approved_bundle_hash="b" * 64, change_requests=())


def test_non_approved_outcome_cannot_smuggle_a_complete_handoff() -> None:
    with pytest.raises(ValidationError, match="only an approved outcome"):
        DesignOutcomeV2(**(OUTCOME.model_dump() | {"status": "needs_context"}))


def test_interrupt_kinds_cover_the_three_human_pauses() -> None:
    assert tuple(kind.value for kind in InterruptKind) == (
        "table_selection", "clarification", "approval")
