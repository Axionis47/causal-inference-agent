"""Design V2 compiler, computability, and actor-routing invariants (T-036)."""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from causal.design.askgate import AcceptedFactV1
from causal.design.compiler_v2 import (
    EligibilityResult,
    assess_candidates,
    compile_contrasts,
    compile_design,
    compile_diagnostic_plan,
    diagnostic_parameters,
    structural_coverage,
)
from causal.design.contracts import SourceInterpretationV1
from causal.design.diagnostics import BytesFrameSource, run_diagnostic
from causal.design.harness_nodes import PipelineNodes
from causal.design.packs import load_method_packs, load_requirement_templates
from causal.design.resolution_v2 import (
    accepted_facts_for_consumer,
    compile_fact_set,
    compile_proposal_facts,
    evaluate_diagnostics,
    route_candidates,
    route_issues,
)
from causal.design.semantics import RoleClaimV1, RoleLedgerV1, RoleName, TimingClass
from causal.design.statistics_v2 import panel_structure, power_precision, rough_overlap
from causal.design.v2 import (
    AgentDesignProposalV2,
    DesignFactV2,
    EvidenceRelation,
    FactAcceptanceStatus,
    FactSource,
    ResolutionCategory,
    ResponsibleActor,
    ValidationIssueV2,
)
from causal.shared.contracts import ArtifactRef
from causal.shared.envelope import CausalFrameV1, EpistemicStatus, EvidenceClass, SupportClass
from causal.shared.frames import ROW_UNIT_COLUMN

PACKS = load_method_packs(__import__("pathlib").Path("registries/method-packs.v1.json"))
TEMPLATES = load_requirement_templates(
    __import__("pathlib").Path("registries/context-requirements.v1.json"))
REF = ArtifactRef(artifact_id="table:1", content_hash="a" * 64)
LEDGER_REF = ArtifactRef(artifact_id="ledger:1", content_hash="b" * 64)
FRAME = CausalFrameV1(
    treatment="c:treatment", outcome="c:outcome", population="people", timeframe="now")


def accepted_fact(
    requirement_id: str, scope_id: str, value: str | float | bool | tuple[str, ...],
    *, accepted_fact_id: str | None = None,
) -> AcceptedFactV1:
    template = TEMPLATES[requirement_id]
    identity = accepted_fact_id or f"af:{requirement_id}:{scope_id}"
    return AcceptedFactV1(
        accepted_fact_id=identity, analysis_id="analysis:1", design_revision=1,
        requirement_id=requirement_id, scope_id=scope_id, value=value,
        value_schema=template.expected_answer_schema, source_kind="user",
        evidence_ids=("ua:answer:1",), evidence_class=EvidenceClass.USER_CONFIRMATION,
        relation="direct", origin_revision=1, origin_reference_id="answer:1",
        origin_reference_hash="c" * 64)


def claim(role: RoleName, *columns: str) -> RoleClaimV1:
    return RoleClaimV1(
        role=role, concept_id=f"c:{role.value}", column_refs=columns, evidence_ids=("ev:doc/x",),
        timing=(TimingClass.PRE_TREATMENT if role is not RoleName.OUTCOME
                else TimingClass.POST_TREATMENT),
        graph_edge_ids=(), support_class=SupportClass.DIRECT_SOURCE_STATEMENT,
        alternatives=(), status=EpistemicStatus.EVIDENCED, methods=())


def fact(name: str, value: str | float | bool) -> DesignFactV2:
    requirements = {"assignment_mechanism": "design.assignment_mechanism",
                    "estimand": "design.estimand", "grain": "design.table_grain",
                    "comparator": "design.population_comparator", "cutoff": "design.cutoff",
                    "sharp_assignment": "design.sharp_assignment",
                    "adoption_time": "design.adoption_time"}
    return DesignFactV2(
        fact_id=name, requirement_id=requirements[name], scope_id="design", value=value,
        source=FactSource.USER, source_artifact_ids=("answer:1",),
        evidence_class=EvidenceClass.USER_CONFIRMATION, relation=EvidenceRelation.DIRECT,
        epistemic_status=EpistemicStatus.EVIDENCED,
        acceptance_status=FactAcceptanceStatus.ACCEPTED, executable=True)


def facts(mechanism: str, roles: tuple[RoleClaimV1, ...], *, grain: str = "one_row_per_unit",
          estimand: str | None = None, extra: tuple[DesignFactV2, ...] = ()) -> Any:
    ledger = RoleLedgerV1(frame=FRAME, claims=roles)
    chosen_estimand = estimand or {
        "randomized": "itt", "self_selected": "att", "time_of_adoption":
        "att_group_time_aggregate", "policy_cutoff": "late_at_cutoff"}[mechanism]
    return compile_fact_set(selected_csv=REF, ledger=ledger,
                            ledger_ref=LEDGER_REF, facts=(
                                fact("assignment_mechanism", mechanism),
                                fact("estimand", chosen_estimand), fact("grain", grain),
                                fact("comparator", "control"), *extra))


def profile(**columns: tuple[int, str]) -> dict[str, Any]:
    return {"columns": {name: {"cardinality": card, "dtype": dtype}
                        for name, (card, dtype) in columns.items()}}


def proposal(method: str, estimand: str) -> AgentDesignProposalV2:
    source = (
        "observational self-selected ATT ATE ITT group-time local effect "
        "with untreated controls")
    return AgentDesignProposalV2(
        assignment_mechanism="self_selected", requested_estimand=estimand,
        comparator="control",
        ranked_method_ids=(method, *(name for name in (
            "randomized_experiment", "aipw", "did", "sharp_rdd") if name != method)),
        method_facts=(), optional_assumption_ids=(), optional_risk_ids=(),
        optional_sensitivity_ids=(), source_interpretations=tuple(
            SourceInterpretationV1(
                fact_key=fact_key, value=value, evidence_id="ev:doc/x",
                verbatim_excerpt=source, relation="direct")
            for fact_key, value in (
                ("assignment_mechanism", "self_selected"),
                ("estimand", estimand), ("comparator", "control"))))


def test_model_fact_is_inert_until_an_allowlisted_source_supports_it() -> None:
    draft = proposal("aipw", "att")
    unsupported = compile_proposal_facts(
        draft, evidence={}, requirement_templates=TEMPLATES)
    assert not next(row for row in unsupported
                    if row.fact_id == "assignment_mechanism").executable
    supported = compile_proposal_facts(
        draft, evidence={"ev:doc/x": (
            "observational self-selected ATT ATE ITT group-time local effect "
            "with untreated controls")}, requirement_templates=TEMPLATES)
    assert next(row for row in supported
                if row.fact_id == "assignment_mechanism").executable


def test_same_citation_cannot_make_contradictory_assignment_values_executable() -> None:
    source = {"ev:doc/x": "This is an observational, self-selected exposure."}
    self_selected = proposal("aipw", "att").model_copy(update={
        "source_interpretations": (SourceInterpretationV1(
            fact_key="assignment_mechanism", value="self_selected",
            evidence_id="ev:doc/x", verbatim_excerpt=source["ev:doc/x"],
            relation="direct"),)})
    randomized = AgentDesignProposalV2.model_validate(
        self_selected.model_dump() | {
            "assignment_mechanism": "randomized",
            "source_interpretations": ({
                "fact_key": "assignment_mechanism", "value": "randomized",
                "evidence_id": "ev:doc/x", "verbatim_excerpt": source["ev:doc/x"],
                "relation": "conflicting"},)})
    accepted = compile_proposal_facts(
        self_selected, evidence=source, requirement_templates=TEMPLATES)
    rejected = compile_proposal_facts(
        randomized, evidence=source, requirement_templates=TEMPLATES)
    assert next(row for row in accepted
                if row.fact_id == "assignment_mechanism").executable
    assert not next(row for row in rejected
                    if row.fact_id == "assignment_mechanism").executable


def test_user_answer_overrides_the_model_assignment_proposal() -> None:
    rows = compile_proposal_facts(
        proposal("aipw", "att"), evidence={}, requirement_templates=TEMPLATES,
        accepted_facts=(accepted_fact(
            "design.assignment_mechanism", "design", "randomized"),))
    chosen = next(row for row in rows if row.fact_id == "assignment_mechanism")
    assert chosen.value == "randomized" and chosen.source is FactSource.USER
    assert chosen.executable


def test_every_registered_answer_is_routed_with_its_exact_scope_and_typed_value() -> None:
    def value(requirement_id: str) -> str | float | bool | tuple[str, ...]:
        template = TEMPLATES[requirement_id]
        value_type = template.accepted_fact.value_type
        if value_type == "boolean":
            return True
        if value_type == "number":
            return 0.0
        if value_type == "duration":
            return "2 years"
        if value_type == "mapping":
            return "post_treatment"
        if value_type == "column_list":
            return ("age", "earnings")
        if template.expected_answer_schema.startswith("choice:"):
            return template.expected_answer_schema.removeprefix("choice:").split("|")[0]
        return "documented value"

    scopes = {"design": "design", "dataset": "dataset:1", "column": "earnings",
              "concept": "concept:earnings", "relationship": "edge:treatment:outcome"}
    facts_by_requirement = {
        requirement_id: accepted_fact(
            requirement_id, scopes[template.scope_kind.value], value(requirement_id))
        for requirement_id, template in TEMPLATES.items()}
    for requirement_id, template in TEMPLATES.items():
        fact = facts_by_requirement[requirement_id]
        key = (template.accepted_fact.fact_key, fact.scope_id)
        for consumer in template.accepted_fact.consumer_ids:
            routed = accepted_facts_for_consumer(
                tuple(facts_by_requirement.values()), TEMPLATES, consumer)
            assert routed[key].accepted_fact_id == fact.accepted_fact_id
            assert routed[key].value == fact.value
    timing = accepted_facts_for_consumer(
        tuple(facts_by_requirement.values()), TEMPLATES, "role_ledger")
    assert timing[("column_measurement_timing", "earnings")].value == "post_treatment"


def test_structural_registry_has_exact_total_coverage() -> None:
    assert structural_coverage(PACKS) == ()


def test_assignment_mechanism_filters_before_model_ranking() -> None:
    built = facts("self_selected", (
        claim(RoleName.TREATMENT, "treat"), claim(RoleName.OUTCOME, "y"),
        claim(RoleName.UNIT_IDENTIFIER, "id"), claim(RoleName.CONFOUNDER_CANDIDATE, "age"), claim(RoleName.GROUP, "age")))
    results = assess_candidates(PACKS, built, profile(
        treat=(2, "Int64"), y=(20, "Float64"), id=(20, "String"), age=(10, "Int64")))
    assert [row.method_id for row in results if row.eligible] == ["aipw"]
    rct = next(row for row in results if row.method_id == "randomized_experiment")
    assert "assignment_mechanism_incompatible" in {issue.code for issue in rct.issues}


def test_requested_estimand_is_a_compiler_eligibility_constraint() -> None:
    built = facts("self_selected", (
        claim(RoleName.TREATMENT, "treat"), claim(RoleName.OUTCOME, "y"),
        claim(RoleName.UNIT_IDENTIFIER, "id"),
        claim(RoleName.CONFOUNDER_CANDIDATE, "age")), estimand="itt")
    results = assess_candidates(PACKS, built, profile(
        treat=(2, "Int64"), y=(20, "Float64"), id=(20, "String"), age=(10, "Int64")))
    aipw = next(row for row in results if row.method_id == "aipw")
    assert not aipw.eligible
    assert "estimand_not_supported" in {issue.code for issue in aipw.issues}


def test_row_unit_is_derived_before_method_validation() -> None:
    built = facts("randomized", (
        claim(RoleName.TREATMENT, "treat"), claim(RoleName.OUTCOME, "y")))
    binding = next(row for row in built.role_bindings
                   if row.role is RoleName.UNIT_IDENTIFIER)
    assert binding.columns == (ROW_UNIT_COLUMN,) and binding.derived


def test_derived_row_unit_is_materialized_for_bound_diagnostics() -> None:
    built = facts("randomized", (
        claim(RoleName.TREATMENT, "treat"), claim(RoleName.OUTCOME, "y")))
    plan = compile_diagnostic_plan(PACKS.get("randomized_experiment"), built)
    item = next(row for row in plan.items if row.diagnostic_id == "assignment_unit_uniqueness")
    source = BytesFrameSource("rows.csv", b"treat,y\n0,1\n1,2\n", REF)
    result = run_diagnostic(item.diagnostic_id, source, diagnostic_parameters(item))
    assert result.status.value == "computed" and result.values["is_unique"] is True


def test_row_unit_is_not_invented_for_panel_data() -> None:
    built = facts("time_of_adoption", (
        claim(RoleName.TREATMENT, "treat"), claim(RoleName.OUTCOME, "y"),
        claim(RoleName.GROUP, "state"), claim(RoleName.TIME, "year"),
        claim(RoleName.CLUSTER, "state")), grain="one_row_per_group_time",
        extra=(fact("adoption_time", "policy_year"),))
    assert not built.columns(RoleName.UNIT_IDENTIFIER)


def test_rdd_diagnostics_bind_running_variable_cutoff_and_outcome() -> None:
    built = facts("policy_cutoff", (
        claim(RoleName.TREATMENT, "treated"), claim(RoleName.OUTCOME, "outcome"),
        claim(RoleName.UNIT_IDENTIFIER, "id"), claim(RoleName.RUNNING_VARIABLE, "score")),
        extra=(fact("cutoff", 50.0), fact("sharp_assignment", True)))
    plan = compile_diagnostic_plan(PACKS.get("sharp_rdd"), built)
    assert not plan.issues
    params = {item.diagnostic_id: diagnostic_parameters(item) for item in plan.items}
    assert params["distance_to_cutoff_support"]["column"] == "score"
    assert params["distance_to_cutoff_support"]["cutoff"] == 50.0
    assert params["missingness_by_side_and_distance"]["target"] == "outcome"


def test_zero_is_a_valid_fixed_cutoff() -> None:
    built = facts("policy_cutoff", (
        claim(RoleName.TREATMENT, "treated"), claim(RoleName.OUTCOME, "y"),
        claim(RoleName.RUNNING_VARIABLE, "score")),
        extra=(fact("cutoff", 0.0), fact("sharp_assignment", True)))
    sharp = PACKS.get("sharp_rdd")
    assert next(row for row in assess_candidates(PACKS, built, profile(
        treated=(2, "int64"), y=(10, "float64"), score=(10, "float64")))
                if row.method_id == sharp.method_id).eligible


def test_group_time_grain_binds_group_time_as_observation_key() -> None:
    built = facts("time_of_adoption", (
        claim(RoleName.TREATMENT, "treated"), claim(RoleName.OUTCOME, "outcome"),
        claim(RoleName.GROUP, "state"), claim(RoleName.TIME, "year"),
        claim(RoleName.CLUSTER, "state")), grain="one_row_per_group_time",
        extra=(fact("adoption_time", "policy_year"),))
    plan = compile_diagnostic_plan(PACKS.get("did"), built)
    item = next(row for row in plan.items if row.diagnostic_id == "unit_period_uniqueness")
    assert diagnostic_parameters(item)["key_columns"] == ("state", "year")


def test_did_uses_the_unit_as_the_default_cluster() -> None:
    built = facts("time_of_adoption", (
        claim(RoleName.TREATMENT, "treated"), claim(RoleName.OUTCOME, "outcome"),
        claim(RoleName.GROUP, "state"), claim(RoleName.TIME, "year"),
        claim(RoleName.UNIT_IDENTIFIER, "state")), grain="one_row_per_group_time",
        extra=(fact("adoption_time", "policy_year"),))
    profile_data = profile(
        treated=(2, "Int64"), outcome=(20, "Float64"), state=(5, "String"),
        year=(4, "Int64"))
    did = next(row for row in assess_candidates(PACKS, built, profile_data)
               if row.method_id == "did")
    assert did.eligible
    plan = compile_diagnostic_plan(PACKS.get("did"), built)
    cluster = next(row for row in plan.items
                   if row.diagnostic_id == "clustering_feasibility")
    assert not plan.issues
    assert diagnostic_parameters(cluster)["columns"] == ("state",)


def test_group_time_grain_compiles_group_time_preparation_key() -> None:
    built = facts("time_of_adoption", (
        claim(RoleName.TREATMENT, "treated"), claim(RoleName.OUTCOME, "outcome"),
        claim(RoleName.GROUP, "state"), claim(RoleName.TIME, "year"),
        claim(RoleName.CLUSTER, "state")), grain="one_row_per_group_time",
        extra=(fact("adoption_time", "policy_year"),))
    design = compile_design(
        pack=PACKS.get("did"), facts=built,
        proposal=proposal("did", "att_group_time_aggregate"), frame=FRAME,
        causal_question="Does adoption change the outcome?", intended_decision="adopt policy",
        comparator="control", unit="state", contrasts=("adopters_vs_comparison",),
        rejected_methods={})
    assert design.preparation.key_columns == ("state", "year") and design.comparator == "comparison states"
    assert design.required_visual_evidence == ()


def test_att_is_copied_into_estimator_parameters() -> None:
    built = facts("self_selected", (
        claim(RoleName.TREATMENT, "treat"), claim(RoleName.OUTCOME, "y"),
        claim(RoleName.UNIT_IDENTIFIER, "id"), claim(RoleName.CONFOUNDER_CANDIDATE, "age")))
    design = compile_design(
        pack=PACKS.get("aipw"), facts=built, proposal=proposal("aipw", "att"), frame=FRAME,
        causal_question="Does treatment change y?", intended_decision="treat people",
        comparator="control", unit="person", contrasts=("treated_vs_control",),
        rejected_methods={})
    assert design.estimand == "att" and RoleName.GROUP not in {row.role for row in design.role_bindings}
    assert design.preparation.unusable_row_rule_ids == PACKS.get("aipw").unusable_row_rule_ids
    assert design.required_visual_evidence == ()


def test_measurement_units_require_evidenced_slots_and_retain_card_provenance() -> None:
    from causal.design.compiler_v2 import compile_measurements
    from causal.design.semantics import SlotAssertionV1
    from tests.design.test_compile import card

    original = card("weight_change", "c:weight")
    supported = SlotAssertionV1(value="kilograms", status=EpistemicStatus.EVIDENCED,
                                evidence_ids=("ev:source/units",))
    speculative = SlotAssertionV1(value="pounds", status=EpistemicStatus.HYPOTHESIS,
                                  evidence_ids=())
    known = original.model_copy(update={"slots": original.slots | {"units": supported}})
    unknown = original.model_copy(update={"slots": original.slots | {"units": speculative}})
    result = compile_measurements(((REF, known),), {"weight_change"})["weight_change"]
    assert (result.units, result.source_card, result.supporting_evidence_ids) == (
        "kilograms", REF, ("ev:source/units",))
    assert compile_measurements(((REF, unknown),), {"weight_change"})["weight_change"].units is None
    assert compile_measurements(((REF, known),), {"different_column"}) == {}


def issue(category: ResolutionCategory) -> ValidationIssueV2:
    return ValidationIssueV2.build(
        code=f"x_{category.value}", category=category, path="/x", rule_id="r.x",
        actual="bad", expected="good", why="blocking",
        actor={ResolutionCategory.MODEL_FIX: ResponsibleActor.MODEL,
               ResolutionCategory.HUMAN_INPUT: ResponsibleActor.USER,
               ResolutionCategory.NEEDS_DATA: ResponsibleActor.DATA_OWNER,
               ResolutionCategory.UNSUPPORTED: ResponsibleActor.PRODUCT,
               ResolutionCategory.SYSTEM_FAILURE: ResponsibleActor.SYSTEM}[category],
        actions=("fix",))


def test_only_model_fix_retries_and_exhaustion_is_system_failure() -> None:
    found = issue(ResolutionCategory.MODEL_FIX)
    assert route_issues((found,)).action == "retry_model"
    exhausted = route_issues((found,), corrections_used=2)
    repeated = route_issues((found,), previous_fingerprints=(found.fingerprint,))
    assert exhausted.terminal_status == repeated.terminal_status == "system_failure"
    assert exhausted.error_code == "agent_output_invalid"


def test_repeated_human_requirement_stops_without_another_ask() -> None:
    found = issue(ResolutionCategory.HUMAN_INPUT)
    first = route_issues((found,))
    repeated = route_issues((found,), previous_fingerprints=(found.fingerprint,))
    assert first.action == "ask_human"
    assert repeated.action == "stop" and repeated.terminal_status == "needs_context"
    assert repeated.error_code == "compiler_requirement_unresolved"


def test_harness_passes_the_previous_compiler_fingerprint_to_the_router() -> None:
    found = issue(ResolutionCategory.HUMAN_INPUT)
    state: Any = {"compiler_issues": [found.model_dump(mode="json")]}
    routed = object.__new__(PipelineNodes)._route_compiler(state, (found,))
    assert routed["status"] == "needs_context"
    assert routed["error_code"] == "compiler_requirement_unresolved"


@pytest.mark.parametrize(("category", "status"), [
    (ResolutionCategory.HUMAN_INPUT, "needs_context"),
    (ResolutionCategory.NEEDS_DATA, "needs_data"),
    (ResolutionCategory.UNSUPPORTED, "unsupported"),
    (ResolutionCategory.SYSTEM_FAILURE, "system_failure"),
])
def test_non_model_errors_have_distinct_routes(category: ResolutionCategory, status: str) -> None:
    assert route_issues((issue(category),)).terminal_status == status


def test_candidate_router_prefers_a_human_rescuable_method() -> None:
    human = issue(ResolutionCategory.HUMAN_INPUT)
    unsupported = issue(ResolutionCategory.UNSUPPORTED)
    routed = route_candidates((EligibilityResult("a", False, (unsupported,)),
                               EligibilityResult("b", False, (human,))))
    assert routed.action == "ask_human" and routed.terminal_status == "needs_context"


def test_required_uncomputed_diagnostic_becomes_needs_data() -> None:
    built = facts("randomized", (
        claim(RoleName.TREATMENT, "treat"), claim(RoleName.OUTCOME, "y"),
        claim(RoleName.UNIT_IDENTIFIER, "id")))
    plan = compile_diagnostic_plan(PACKS.get("randomized_experiment"), built)
    report = evaluate_diagnostics(plan, ())
    assert not report.computable
    assert {row.category for row in report.issues} == {ResolutionCategory.NEEDS_DATA}


def test_statistical_primitives_report_power_overlap_and_panel_structure() -> None:
    frame = pl.DataFrame({
        "id": [1, 1, 2, 2], "time": [0, 1, 0, 1], "treat": [0, 0, 1, 1],
        "y": [1.0, 1.2, 2.0, 2.2], "age": [20, 21, 22, 23]})
    power, _, _, _ = power_precision(frame, {"columns": ("treat",), "target": "y"})
    overlap, _, _, _ = rough_overlap(frame, {"target": "treat", "columns": ("age",)})
    panel, _, _, _ = panel_structure(frame, {"key_columns": ("id", "time")})
    assert power["mde_80pct_95ci"] is not None
    assert "minimum_common_support_share" in overlap
    assert panel["cell_completeness"] == 1.0


def _run_plan(frame: pl.DataFrame, method: str, built: Any) -> Any:
    plan = compile_diagnostic_plan(PACKS.get(method), built)
    source = BytesFrameSource("fixture.csv", frame.write_csv().encode(), REF)
    results = tuple(run_diagnostic(item.diagnostic_id, source, diagnostic_parameters(item))
                    for item in plan.items)
    return evaluate_diagnostics(plan, results)


def test_explicit_binary_contrast_resolves_its_rhs_comparator_level() -> None:
    built = facts("randomized", (claim(RoleName.TREATMENT, "treat"),
                                 claim(RoleName.OUTCOME, "y")))
    report = _run_plan(pl.DataFrame({"treat": [0, 1], "y": [1.0, 2.0]}),
                       "randomized_experiment", built)
    contrasts, issues = compile_contrasts(
        PACKS.get("randomized_experiment"), "treatment=1 vs treatment=0", report)
    assert contrasts == ("1_vs_0",) and not issues and compile_contrasts(PACKS.get("randomized_experiment"), "foo=1 vs bar=0", report)[1]


def test_did_requires_measured_group_time_and_pre_post_support() -> None:
    built = facts("time_of_adoption", (
        claim(RoleName.TREATMENT, "treated"), claim(RoleName.OUTCOME, "y"),
        claim(RoleName.GROUP, "group"), claim(RoleName.TIME, "period"),
        claim(RoleName.CLUSTER, "group")), grain="one_row_per_group_time",
        extra=(fact("adoption_time", "adoption"),))
    supported = pl.DataFrame({
        "group": ["control", "control", "treated", "treated"],
        "period": [1, 2, 1, 2], "adoption": [0, 0, 2, 2],
        "treated": [0, 0, 0, 1], "y": [1.0, 1.1, 1.2, 1.5]})
    supported_report = _run_plan(supported, "did", built)
    assert supported_report.computable
    assert next(row for row in supported_report.results if row.diagnostic_id ==
                "pre_period_availability").values["adoption_profile_id"] == "simultaneous"
    one_period = supported.filter(pl.col("period") == 1)
    report = _run_plan(one_period, "did", built)
    assert not report.computable
    assert any(issue.json_path == "/diagnostics/pre_period_availability"
               for issue in report.issues)


def test_did_support_allows_one_column_to_bind_group_and_treatment() -> None:
    built = facts("time_of_adoption", (
        claim(RoleName.TREATMENT, "exposed"), claim(RoleName.OUTCOME, "y"),
        claim(RoleName.GROUP, "exposed"), claim(RoleName.TIME, "period"),
        claim(RoleName.UNIT_IDENTIFIER, "unit")), grain="one_row_per_unit_period",
        extra=(fact("adoption_time", 2),))
    frame = pl.DataFrame({"unit": [1, 1, 2, 2], "exposed": [0, 0, 1, 1], "period": [1, 2, 1, 2],
                          "y": [1.0, 1.1, 1.2, 1.5]})
    support = next(row for row in _run_plan(frame, "did", built).results if row.diagnostic_id == "pre_period_availability")
    assert support.status.value == "computed" and support.values["pre_period_rows"] == 2


def test_rdd_requires_rows_on_both_sides_of_the_approved_cutoff() -> None:
    built = facts("policy_cutoff", (
        claim(RoleName.TREATMENT, "treated"), claim(RoleName.OUTCOME, "y"),
        claim(RoleName.UNIT_IDENTIFIER, "id"), claim(RoleName.RUNNING_VARIABLE, "score")),
        extra=(fact("cutoff", 50.0), fact("sharp_assignment", True)))
    one_sided = pl.DataFrame({
        "id": [1, 2, 3], "treated": [1, 1, 1], "score": [50.0, 51.0, 52.0],
        "y": [1.0, 2.0, 3.0]})
    report = _run_plan(one_sided, "sharp_rdd", built)
    assert not report.computable
    assert any("below=0" in issue.safe_actual_summary for issue in report.issues)
