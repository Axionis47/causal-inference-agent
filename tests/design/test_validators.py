"""Walls 1–7: one passing case and the named failing fixtures per wall (T-012 §6)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from causal.design.contracts import (
    REGISTRY_VERSION_KEYS,
    DesignContextManifestV1,
    StructuralFieldV1,
)
from causal.design.frame import ExperimentDesignV1, RunnableFrameContractV1
from causal.design.packs import (
    METHOD_IDS,
    PackRegistryError,
    load_method_packs,
    load_requirement_templates,
)
from causal.design.semantics import (
    CausalContextV1,
    CausalEdgeV1,
    GraphAlternativeV1,
    RoleClaimV1,
    RoleLedgerV1,
    RoleName,
    TimingClass,
)
from causal.design.triage import ColumnTriageRecordV1
from causal.design.validators import (
    ACTIONS,
    RULE_KINDS,
    ValidationContext,
    ValidationReport,
    evidence_class,
    load_validation_rules,
    make_causal_model_handler,
    validate_result,
    wall_causal,
    wall_evidence,
    wall_frame,
    wall_method,
    wall_references,
    wall_shape,
    wall_temporal,
)
from causal.shared.contracts import ArtifactRef
from causal.shared.envelope import (
    AgentTaskEnvelopeV1,
    AgentTaskResultV1,
    CausalFrameV1,
    ClaimV1,
    ContextRequirementV1,
    EpistemicStatus,
    EvidenceClass,
    SupportClass,
    TaskBudgets,
    TaskStatus,
)

REGISTRIES = Path(__file__).resolve().parents[2] / "registries"
RULES = load_validation_rules(REGISTRIES / "design-validation-rules.v1.json")
PACKS = load_method_packs(REGISTRIES / "method-packs.v1.json")
TEMPLATES = load_requirement_templates(REGISTRIES / "context-requirements.v1.json")
PACK = PACKS.get("did")
BINARY = next(row for row in PACKS.all() if "treatment_binary" in row.structural_requirements)
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
    retrieval_surfaces=("s-1",), registry_versions=dict.fromkeys(REGISTRY_VERSION_KEYS, "v1"),
    recipient_map={})
TRIAGE = ColumnTriageRecordV1(
    table_name="nsw.csv",
    tiers={"critical": ("treat",), "plausible_adjustment": (), "supporting": ("age",),
           "unused": ()},
    batches=(), deferred=("age",),
    match_trace={"treat": "intent_candidate", "age": "profiled_only"})
ENVELOPE = AgentTaskEnvelopeV1(
    envelope_id="env-1", schema_version="agent-task-envelope.v1", analysis_id="an-1",
    stage_run_id="sr-1", task_id="t-1", attempt_id="at-1", context_manifest=REF,
    task_kind="causal_synthesis", scope_kind="design", scope_ids=("d-1",), parent_artifacts=(REF,),
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
                            "templates": TEMPLATES, "method_id": "did"}
    return ValidationContext(**base | overrides)


def role(name: RoleName, concept: str = "c-treat", *, columns: tuple[str, ...] = (),
         timing: TimingClass = TimingClass.PRE_TREATMENT, edges: tuple[str, ...] = (),
         evidence: tuple[str, ...] = ()) -> RoleClaimV1:
    return RoleClaimV1(role=name, concept_id=concept, column_refs=columns, evidence_ids=evidence,
                       timing=timing, graph_edge_ids=edges, alternatives=(), methods=("did",),
                       support_class=SupportClass.DIRECT_SOURCE_STATEMENT,
                       status=EpistemicStatus.EVIDENCED)


def ledger(*claims: RoleClaimV1) -> RoleLedgerV1:
    return RoleLedgerV1(frame=FRAME, claims=(
        role(RoleName.TREATMENT, columns=("treat",)),
        role(RoleName.OUTCOME, "c-earn", columns=("re78",), timing=TimingClass.POST_TREATMENT),
        *claims))


FULL = ledger(role(RoleName.GROUP, "c-g", columns=("age",)),
              role(RoleName.TIME, "c-t", columns=("period",)),
              role(RoleName.CLUSTER, "c-c", columns=("person_id",)),
              role(RoleName.UNIT_IDENTIFIER, "c-u", columns=("person_id",)))
RESOLVED = dict.fromkeys(PACK.required_context_requirement_ids, "resolved")


def edge(edge_id: str, source: str, target: str,
         status: EpistemicStatus = EpistemicStatus.HYPOTHESIS) -> CausalEdgeV1:
    return CausalEdgeV1(edge_id=edge_id, source_concept_id=source, target_concept_id=target,
                        timeframe="t", mechanism_summary="m", supporting_evidence_ids=(),
                        contrary_evidence_ids=(), status=status, differing_alternative_ids=())


def graph(*edges: CausalEdgeV1,
          alternatives: tuple[GraphAlternativeV1, ...] = ()) -> CausalContextV1:
    return CausalContextV1(frame=FRAME, concept_ids=("c-treat", "c-earn", "c-x"), edges=edges,
                           alternatives=alternatives, selection_notes="n", claims=())


def claim(predicate: str, support: SupportClass = SupportClass.DIRECT_SOURCE_STATEMENT,
          evidence: tuple[str, ...] = ("ev:doc/a",)) -> ClaimV1:
    return ClaimV1(claim_id="cl-1", subject_kind="column", subject_id="re78", predicate=predicate,
                   value="x", epistemic_status=EpistemicStatus.EVIDENCED, alternatives=(),
                   supporting_evidence_ids=evidence, contrary_evidence_ids=(),
                   support_class=support, causal_frame=None)


def result(payload: Any = None, **overrides: Any) -> AgentTaskResultV1:
    body = payload if isinstance(payload, dict | type(None)) else payload.model_dump(mode="json")
    base: dict[str, Any] = {
        "envelope_id": "env-1", "schema_version": "agent-task-result.v1", "task_id": "t-1",
        "status": TaskStatus.COMPLETE, "artifact_type": "CausalContext",
        "artifact_schema_version": "causal-context.v1", "parent_artifact_ids": (),
        "payload": body or {}, "claims": (), "missing_requirements": (), "conflicts": (),
        "warnings": (), "evidence_ids": (), "tool_receipts": (), "output_hash": None,
        "validation_target": "causal-context-validator.v1"}
    return AgentTaskResultV1(**base | overrides)


def requirement(requirement_id: str) -> ContextRequirementV1:
    template = {key: value for key, value in dict(TEMPLATES["column.meaning"]).items()
                if key != "requirement_id"}
    return ContextRequirementV1(
        requirement_id=requirement_id, registry_version="context-requirements.v1", scope_id="re78",
        decisions_blocked=(), attempted_evidence=(), **template)


def design(**overrides: Any) -> ExperimentDesignV1:
    base: dict[str, Any] = {
        "causal_question": "q", "intended_decision": "d", "selected_csv": REF, "method_id": "did",
        "method_pack_version": "did-pack.v1", "frame": FRAME, "comparator": "c", "unit": "u",
        "rejected_methods": {name: "no" for name in METHOD_IDS if name != "did"},
        "estimand": "att", "measurement_map": REF, "causal_context": REF, "role_ledger": REF,
        "assumptions": (), "identification_risks": (), "eligibility_rules": (),
        "mandatory_repair_boundaries": (), "forbidden_repair_boundaries": (),
        "imputation_eligible_columns": (), "imputation_forbidden_columns": ("treat", "re78"),
        "deletion_impact_dimensions": PACK.deletion_impact_dimensions,
        "invalidation_conditions": (), "required_prerepair_diagnostics": ("group_time_counts",),
        "required_postrepair_diagnostics": PACK.required_postrepair_diagnostic_ids,
        "required_visual_evidence": PACK.required_visual_evidence_ids, "primary_contrasts": (),
        "multiplicity_policy": None, "capacity_check": None, "sensitivity_requirements": (),
        "visualization_catalog_version": "v1", "capacity_registry_version": "v1",
        "registry_versions": dict.fromkeys(REGISTRY_VERSION_KEYS, "v1")}
    return ExperimentDesignV1(**base | overrides)


def contract(**overrides: Any) -> RunnableFrameContractV1:
    base: dict[str, Any] = {
        "selected_csv": REF, "output_grain": "one row per unit-period",
        "key_columns": ("person_id", "period"), "required_roles": (RoleName.TREATMENT,),
        "allowed_roles": (), "forbidden_roles": (), "type_constraints": {},
        "uniqueness_constraints": (), "eligibility_rules": (),
        "exclusion_reason_vocabulary": PACK.eligibility_rule_vocabulary,
        "treatment_missingness_rule": "drop", "outcome_missingness_rule": "drop",
        "method_structure": {}, "imputation_permitted": ("person_id",),
        "imputation_forbidden": ("treat", "re78", "age", "period"),
        "required_missingness_indicators": (),
        "deletion_impact_dimensions": PACK.deletion_impact_dimensions,
        "revision_required_conditions": (), "feasibility_gates": (),
        "required_final_diagnostics": PACK.required_postrepair_diagnostic_ids,
        "estimator_input_schema": PACK.reserved_estimator_id, "experiment_design_hash": HASH}
    return RunnableFrameContractV1(**base | overrides)


def rule_file(**overrides: Any) -> str:
    row = {"rule_id": "r-1", "wall": 4, "kind": "temporal", "params": {}, "code": "c",
           "allowed_actions": [], "user_resolvable": False} | overrides
    return json.dumps({"registry_version": "design-validators.v1", "rules": [row]})


# --- wall 1: shape ---
def test_wall_shape_accepts_a_valid_payload() -> None:
    assert wall_shape(CausalContextV1, result(graph())).passed


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
    assert report.issues[0].user_resolvable and report.issues[0].artifact_ids == ("nope",)


def test_wall_references_flags_evidence_parent_and_requirement_ids() -> None:
    report = wall_references(
        ledger(role(RoleName.GROUP, "c-g", columns=("age",), evidence=("ev:doc/x",))),
        result(parent_artifact_ids=("art-9",), missing_requirements=(requirement("nope"),)),
        context())
    assert codes(report) == {"unresolved_evidence", "uncommitted_parent", "unknown_requirement_id"}


def test_wall_references_catches_a_claim_that_cites_itself() -> None:
    payload = {"claims": [{"claim_id": "cl-1", "supporting_evidence_ids": ["cl-1"]}]}
    assert "claim_cites_itself" in codes(wall_references(payload, None, context()))


# --- wall 3: evidence ---
@pytest.mark.parametrize(("built", "expected"), [
    (claim("meaning"), set()),
    (claim("meaning", evidence=("ev:none/x",)), {"blocking_claim_unsupported"}),
    (claim("meaning", SupportClass.MODEL_HYPOTHESIS), {"hypothesis_support_for_blocking_claim"}),
    (claim("source_process", SupportClass.MODEL_HYPOTHESIS), set()),
])
def test_wall_evidence_rules(built: ClaimV1, expected: set[str]) -> None:
    assert codes(wall_evidence(None, result(claims=(built,)), context())) == expected


# --- wall 4: temporal rows ---
@pytest.mark.parametrize(("name", "timing", "expected"), [
    (RoleName.CONFOUNDER_CANDIDATE, TimingClass.POST_TREATMENT, {"post_treatment_role_forbidden"}),
    (RoleName.ASSIGNMENT_VARIABLE, TimingClass.POST_TREATMENT, {"post_treatment_role_forbidden"}),
    (RoleName.MEDIATOR, TimingClass.PRE_TREATMENT, {"mediator_timing_invalid"}),
    (RoleName.RUNNING_VARIABLE, TimingClass.POST_TREATMENT, {"running_variable_timing_invalid"}),
    (RoleName.CONFOUNDER_CANDIDATE, TimingClass.PRE_TREATMENT, set()),
    (RoleName.MEDIATOR, TimingClass.POST_TREATMENT, set()),
])
def test_wall_temporal_rows(name: RoleName, timing: TimingClass, expected: set[str]) -> None:
    payload = ledger(role(name, "c-x", timing=timing))
    assert codes(wall_temporal(payload, None, context())) == expected


def test_wall_temporal_rejects_a_pre_treatment_outcome() -> None:
    payload = RoleLedgerV1(frame=FRAME, claims=(role(RoleName.TREATMENT),
                                                role(RoleName.OUTCOME, "c-earn")))
    assert codes(wall_temporal(payload, None, context())) == {"outcome_timing_invalid"}


# --- wall 5: causal graph ---
CYCLE = graph(edge("e-1", "c-treat", "c-earn"), edge("e-2", "c-earn", "c-treat"))


def test_wall_causal_accepts_an_acyclic_graph() -> None:
    assert wall_causal(graph(edge("e-1", "c-treat", "c-earn")), None, context()).passed


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


def test_wall_causal_resolves_every_role_edge_id() -> None:
    ctx = context(role_ledger=ledger(role(RoleName.GROUP, "c-x", edges=("e-missing",))))
    assert codes(wall_causal(graph(), None, ctx)) == {"unresolved_graph_edge"}


@pytest.mark.parametrize(("name", "pairs", "expected"), [
    (RoleName.CONFOUNDER_CANDIDATE, ("c-x>c-treat",), {"confounder_edges_missing"}),
    (RoleName.CONFOUNDER_CANDIDATE, ("c-x>c-treat", "c-x>c-earn"), set()),
    (RoleName.MEDIATOR, ("c-treat>c-x",), {"mediator_edges_missing"}),
    (RoleName.MEDIATOR, ("c-treat>c-x", "c-x>c-earn"), set()),
    (RoleName.COLLIDER, ("c-treat>c-x",), {"collider_edges_missing"}),
    (RoleName.COLLIDER, ("c-treat>c-x", "c-earn>c-x"), set()),
    (RoleName.INSTRUMENT_CANDIDATE, ("c-x>c-treat", "c-x>c-earn"),
     {"instrument_exclusion_violated"}),
    (RoleName.INSTRUMENT_CANDIDATE, ("c-x>c-treat",), set()),
])
def test_role_graph_rows(name: RoleName, pairs: tuple[str, ...], expected: set[str]) -> None:
    ends = [pair.split(">") for pair in pairs]
    built = graph(*(edge(f"e-{index}", source, target, EpistemicStatus.EVIDENCED)
                    for index, (source, target) in enumerate(ends)))
    ctx = context(role_ledger=ledger(role(name, "c-x")))
    assert codes(wall_causal(built, None, ctx)) == expected


# --- wall 6: method ---
def method_ctx(**overrides: Any) -> ValidationContext:
    base: dict[str, Any] = {"role_ledger": FULL, "resolved_requirements": RESOLVED}
    return context(**base | overrides)


def test_wall_method_accepts_a_complete_design() -> None:
    assert wall_method(design(), None, method_ctx()).passed


def test_wall_method_flags_a_missing_role_and_its_deferred_columns() -> None:
    thin = ledger(role(RoleName.TIME, "c-t"), role(RoleName.UNIT_IDENTIFIER, "c-u"))
    report = wall_method(design(), None, method_ctx(role_ledger=thin, triage=TRIAGE))
    assert {"required_role_missing", "deferred_column_blocks_role"} <= codes(report)
    blocked = [issue for issue in report.issues if issue.code == "deferred_column_blocks_role"]
    assert blocked[0].user_resolvable and "age" in blocked[0].artifact_ids


def test_wall_method_flags_a_forbidden_adjustment_member() -> None:
    claims = (*FULL.claims, role(RoleName.CONFOUNDER_CANDIDATE, "c-m"),
              role(RoleName.MEDIATOR, "c-m", timing=TimingClass.POST_TREATMENT))
    ctx = method_ctx(role_ledger=RoleLedgerV1(frame=FRAME, claims=claims))
    assert codes(wall_method(design(), None, ctx)) == {"forbidden_adjustment_role"}


def test_wall_method_flags_every_unresolved_requirement() -> None:
    report = wall_method(design(), None, method_ctx(resolved_requirements={}))
    assert codes(report) == {"unresolved_requirement"}
    assert len(report.issues) == len(PACK.required_context_requirement_ids) - 1


def test_wall_method_flags_a_structural_requirement_without_diagnostics() -> None:
    report = wall_method(design(required_prerepair_diagnostics=()), None, method_ctx())
    assert codes(report) == {"structural_panel_grain_unmet"}


def test_wall_method_checks_the_binary_treatment_requirement() -> None:
    rejected = {name: "no" for name in METHOD_IDS if name != BINARY.method_id}
    built = design(method_id=BINARY.method_id, rejected_methods=rejected)
    assert "structural_treatment_binary_unmet" in codes(wall_method(built, None, method_ctx()))


# --- wall 7: frame ---
def frame_ctx(**overrides: Any) -> ValidationContext:
    base: dict[str, Any] = {"role_ledger": FULL, "design": design(), "parents": {"art-1": HASH}}
    return context(**base | overrides)


def test_wall_frame_accepts_a_complete_contract() -> None:
    assert wall_frame(contract(), None, frame_ctx()).passed


@pytest.mark.parametrize(("overrides", "code"), [
    ({"exclusion_reason_vocabulary": ()}, "exclusion_vocabulary_incomplete"),
    ({"deletion_impact_dimensions": ()}, "deletion_dimensions_incomplete"),
    ({"required_final_diagnostics": ()}, "postrepair_diagnostics_incomplete"),
    ({"imputation_permitted": ("treat",)}, "imputation_lists_overlap"),
    ({"imputation_forbidden": ("treat",)}, "imputation_forbidden_incomplete"),
    ({"key_columns": ("nope",)}, "unknown_key_column"),
    ({"output_grain": "  "}, "frame_grain_missing"),
    ({"experiment_design_hash": "b" * 64}, "design_hash_mismatch"),
    ({"estimator_input_schema": "other.v1"}, "estimator_schema_mismatch"),
])
def test_wall_frame_rows(overrides: dict[str, Any], code: str) -> None:
    assert codes(wall_frame(contract(**overrides), None, frame_ctx())) == {code}


def test_wall_frame_reads_visual_evidence_from_the_design() -> None:
    ctx = frame_ctx(design=design(required_visual_evidence=("trends",)))
    assert codes(wall_frame(contract(), None, ctx)) == {"visual_evidence_incomplete"}


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


# --- evidence classes, wall order, and the tool handler ---
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
    capped = validate_result(7, "causal_synthesis", RoleLedgerV1, result(ledger()), context())
    assert capped.wall == 5


def test_causal_model_handler_returns_wall_five_issue_dicts() -> None:
    handler = make_causal_model_handler(lambda envelope: context())
    failing = handler(ENVELOPE, {"causal_context": CYCLE.model_dump(mode="json")})
    assert failing["passed"] is False
    assert failing["issues"][0]["code"] == "graph_cycle"
    assert handler(ENVELOPE, {"causal_context": graph().model_dump(mode="json")}) == {
        "issues": [], "passed": True}
