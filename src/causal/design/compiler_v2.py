"""Deterministic fact, eligibility, diagnostic-binding, and design compilation (T-036)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from causal.design.packs import MethodPackRegistry, MethodPackV1
from causal.design.semantics import ColumnSemanticCardV1, RoleName

# isort: off
from causal.design.v2 import (AgentDesignProposalV2, BoundDiagnosticInputV2, ColumnMeasurementV2, CompiledDesignV2,
                              DesignFactSetV2, DiagnosticPlanItemV2, DiagnosticPlanV2,
                              DiagnosticReportV2, PreparationPolicyV2, ResolutionCategory,
                              ResponsibleActor, ValidationIssueV2)
# isort: on

from causal.shared.contracts import ArtifactRef, encode_contrast
from causal.shared.envelope import CausalFrameV1, EpistemicStatus
from causal.shared.frames import ROW_UNIT_COLUMN


@dataclass(frozen=True)
class BindingRule:
    parameter: str
    roles: tuple[RoleName, ...] = ()
    first_available: bool = False
    fact_id: str | None = None
    literal: str | int | float | bool | None = None


@dataclass(frozen=True)
class DiagnosticRecipe:
    primitive: str
    bindings: tuple[BindingRule, ...]
    required_for_eligibility: bool = True


def _roles(parameter: str, *roles: RoleName) -> BindingRule:
    return BindingRule(parameter, roles=roles)


def _first_role(parameter: str, *roles: RoleName) -> BindingRule:
    return BindingRule(parameter, roles=roles, first_available=True)


def _fact(parameter: str, fact_id: str) -> BindingRule:
    return BindingRule(parameter, fact_id=fact_id)


T, O, U, G, TIME = (RoleName.TREATMENT, RoleName.OUTCOME, RoleName.UNIT_IDENTIFIER,
                     RoleName.GROUP, RoleName.TIME)
COV = (RoleName.CONFOUNDER_CANDIDATE, RoleName.PRECISION_COVARIATE)
RUN, CLUSTER, ASSIGN = (RoleName.RUNNING_VARIABLE, RoleName.CLUSTER,
                        RoleName.ASSIGNMENT_VARIABLE)

# The binding table is the method-neutral compiler IR. A harness never constructs a generic
# parameter dictionary and a primitive never guesses which causal role one of its inputs means.
DIAGNOSTIC_RECIPES: Final[dict[str, DiagnosticRecipe]] = {
    "arm_counts": DiagnosticRecipe("count_by", (_roles("columns", T),)),
    "assignment_unit_uniqueness": DiagnosticRecipe("uniqueness", (_roles("key_columns", U),)),
    "cluster_sizes": DiagnosticRecipe("count_by", (_roles("columns", CLUSTER),), False),
    "baseline_availability": DiagnosticRecipe("availability", (_roles("columns", *COV),), False),
    "outcome_missingness": DiagnosticRecipe("missing_share", (_roles("target", O),)),
    "compliance_availability": DiagnosticRecipe("availability", (_roles("columns", ASSIGN),), False),
    "power_precision_feasibility": DiagnosticRecipe(
        "power_precision", (_roles("columns", T), _roles("target", O))),
    "treatment_prevalence": DiagnosticRecipe("count_by", (_roles("columns", T),)),
    "covariate_availability": DiagnosticRecipe(
        "availability", (_roles("columns", *COV),), False),
    "missingness": DiagnosticRecipe("missing_share", (_roles("target", O), _roles("by", T))),
    "rough_overlap": DiagnosticRecipe(
        "rough_overlap", (_roles("target", T), _roles("columns", *COV))),
    "level_sparsity": DiagnosticRecipe("level_profile", (_roles("column", *COV),), False),
    "effective_sample_feasibility": DiagnosticRecipe("count_by", (_roles("columns", T),)),
    "cross_fitting_feasibility": DiagnosticRecipe("count_by", (_roles("columns", T),)),
    "unit_period_uniqueness": DiagnosticRecipe("uniqueness", (_roles("key_columns", U, TIME),)),
    "group_time_counts": DiagnosticRecipe("count_by", (_roles("columns", G, TIME),)),
    "panel_completeness": DiagnosticRecipe("panel_structure", (_roles("key_columns", U, TIME),)),
    "adoption_cohorts": DiagnosticRecipe(
        "count_by", (_roles("columns", G, ASSIGN),), False),
    "missingness_by_group_time": DiagnosticRecipe(
        "missing_share", (_roles("target", O), _roles("by", G, TIME))),
    "pre_period_availability": DiagnosticRecipe(
        "did_support", (_roles("key_columns", G, TIME), _roles("target", T),
                        _fact("adoption_time", "adoption_time"))),
    "composition": DiagnosticRecipe("count_by", (_roles("columns", G, TIME),)),
    "clustering_feasibility": DiagnosticRecipe(
        "count_by", (_first_role("columns", CLUSTER, U),)),
    "cutoff_side_counts": DiagnosticRecipe(
        "rdd_assignment", (_roles("running_column", RUN), _roles("target", T),
                           _fact("cutoff", "cutoff"))),
    "distance_to_cutoff_support": DiagnosticRecipe(
        "numeric_support", (_roles("column", RUN), _fact("cutoff", "cutoff"))),
    "missingness_by_side_and_distance": DiagnosticRecipe(
        "missing_share", (_roles("target", O), BindingRule("by", literal="side"),
                          _roles("running_column", RUN), _fact("cutoff", "cutoff"))),
    "mass_points": DiagnosticRecipe("numeric_support", (_roles("column", RUN),)),
    "duplicates": DiagnosticRecipe("uniqueness", (_roles("key_columns", U),)),
    "density_manipulation_warnings": DiagnosticRecipe(
        "numeric_support", (_roles("column", RUN), _fact("cutoff", "cutoff"))),
    "bandwidth_feasibility": DiagnosticRecipe(
        "numeric_support", (_roles("column", RUN), _fact("cutoff", "cutoff"))),
}

_RESOLUTION = {
    ResolutionCategory.MODEL_FIX: (ResponsibleActor.MODEL, "revise_proposal"),
    ResolutionCategory.HUMAN_INPUT: (ResponsibleActor.USER, "request_context"),
    ResolutionCategory.NEEDS_DATA: (ResponsibleActor.DATA_OWNER, "upload_replacement_csv"),
    ResolutionCategory.UNSUPPORTED: (ResponsibleActor.PRODUCT, "select_supported_question"),
    ResolutionCategory.SYSTEM_FAILURE: (ResponsibleActor.SYSTEM, "inspect_system")}


def _issue(code: str, category: ResolutionCategory, path: str, actual: str, expected: str,
           why: str, *, candidates: Sequence[str] = (), required: Sequence[str] = (),
           actor: ResponsibleActor | None = None) -> ValidationIssueV2:
    owner, action = _RESOLUTION[category]
    return ValidationIssueV2.build(
        code=code, category=category, path=path, rule_id=f"compiler.{code}", actual=actual,
        expected=expected, why=why, actor=actor or owner, actions=(action,),
        candidates=tuple(candidates), required=tuple(required))


@dataclass(frozen=True)
class EligibilityResult:
    method_id: str
    eligible: bool
    issues: tuple[ValidationIssueV2, ...]


def _profile_for(role: RoleName, facts: DesignFactSetV2,
                 profile: Mapping[str, Any]) -> Mapping[str, Any]:
    columns = facts.columns(role)
    return dict((profile.get("columns") or {}).get(columns[0]) or {}) if columns else {}


def _grain(requirement: str, facts: DesignFactSetV2) -> bool:
    allowed = {
        "one_row_per_unit": {"one_row_per_unit"},
        "one_row_per_randomization_unit": {"one_row_per_unit"},
        "unit_time_or_group_time_rows": {"one_row_per_unit_period", "one_row_per_group_time"},
    }
    return facts.grain in allowed[requirement]


def _fact_present(requirement: str, facts: DesignFactSetV2) -> bool:
    keys = {"defined_comparator": "comparator", "fixed_cutoff": "cutoff",
            "sharp_assignment_at_cutoff": "sharp_assignment",
            "adoption_time_defined": "adoption_time"}
    value = facts.fact(keys[requirement])
    if requirement == "fixed_cutoff":
        try:
            float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
        return True
    return value not in (None, "", False)


def _treatment_cardinality(requirement: str, facts: DesignFactSetV2,
                           profile: Mapping[str, Any]) -> bool:
    count = int(_profile_for(T, facts, profile).get("cardinality") or 0)
    return count == 2 if requirement == "treatment_binary" else count >= 2


_STRUCTURAL_HANDLERS: dict[str, Callable[[str, DesignFactSetV2], bool]] = {
    "one_row_per_unit": _grain, "one_row_per_randomization_unit": _grain,
    "unit_time_or_group_time_rows": _grain,
    "running_variable_with_cutoff": lambda _, facts: bool(
        facts.columns(RUN)) and facts.fact("cutoff") is not None,
    "fixed_cutoff": _fact_present, "sharp_assignment_at_cutoff": _fact_present,
    "defined_comparator": _fact_present, "adoption_time_defined": _fact_present,
    "treatment_and_outcome_observed": lambda _, facts: bool(
        facts.columns(T) and facts.columns(O)),
    "support_on_both_sides_of_cutoff": lambda *_: True,
    "pre_and_post_periods": lambda *_: True,
}


def structural_coverage(packs: MethodPackRegistry) -> tuple[str, ...]:
    declared = {item for pack in packs.all() for item in pack.structural_requirements}
    implemented = set(_STRUCTURAL_HANDLERS) | {"treatment_binary", "arms_ge_2",
                                                "treated_and_comparison_groups"}
    return tuple(sorted(declared ^ implemented))


def assess_candidate(pack: MethodPackV1, facts: DesignFactSetV2,
                     profile: Mapping[str, Any]) -> EligibilityResult:
    issues: list[ValidationIssueV2] = []
    mechanism = facts.fact("assignment_mechanism")
    if mechanism is None:
        issues.append(_issue("assignment_mechanism_unknown", ResolutionCategory.HUMAN_INPUT,
                             "/facts/assignment_mechanism", "unknown",
                             "one supported assignment mechanism", "method eligibility changes",
                             candidates=pack.compatible_assignment_mechanisms,
                             required=("assignment_mechanism",)))
    elif mechanism not in pack.compatible_assignment_mechanisms:
        issues.append(_issue("assignment_mechanism_incompatible", ResolutionCategory.UNSUPPORTED,
                             "/method_id", str(mechanism),
                             ", ".join(pack.compatible_assignment_mechanisms),
                             "this method does not identify the stated assignment process"))
    requested_estimand = facts.fact("estimand")
    if requested_estimand is None:
        issues.append(_issue(
            "estimand_unknown", ResolutionCategory.HUMAN_INPUT, "/facts/estimand", "unknown",
            ", ".join(pack.supported_estimands), "the causal quantity must be approved",
            candidates=pack.supported_estimands, required=("estimand",)))
    elif requested_estimand not in pack.supported_estimands:
        issues.append(_issue(
            "estimand_not_supported", ResolutionCategory.UNSUPPORTED, "/facts/estimand",
            str(requested_estimand), ", ".join(pack.supported_estimands),
            "this method cannot compile the requested causal quantity",
            candidates=pack.supported_estimands))
    held = {row.role.value for row in facts.role_bindings}
    missing = sorted(set(pack.required_roles) - held)
    for role in missing:
        unverified = any(item.endswith(f":{role}") for item in facts.conflicts)
        issues.append(_issue(
            "required_role_unverified" if unverified else "required_role_unbound",
            ResolutionCategory.HUMAN_INPUT if unverified else ResolutionCategory.NEEDS_DATA,
            f"/role_bindings/{role}", role, "one evidenced binding to a CSV column",
            "the estimator input cannot be constructed from a model hypothesis",
            required=(role,)))
    for requirement in pack.structural_requirements:
        if requirement in {"treatment_binary", "arms_ge_2", "treated_and_comparison_groups"}:
            passed = _treatment_cardinality(requirement, facts, profile)
        else:
            passed = _STRUCTURAL_HANDLERS[requirement](requirement, facts)
        if not passed:
            category = (ResolutionCategory.HUMAN_INPUT if requirement in {
                "fixed_cutoff", "sharp_assignment_at_cutoff", "adoption_time_defined",
                "defined_comparator"} else ResolutionCategory.NEEDS_DATA)
            issues.append(_issue(f"structural_{requirement}", category,
                                 f"/structure/{requirement}", "not satisfied", requirement,
                                 "the method cannot be compiled without this condition",
                                 required=(requirement,)))
    return EligibilityResult(pack.method_id, not issues, tuple(issues))


def assess_candidates(packs: MethodPackRegistry, facts: DesignFactSetV2,
                      profile: Mapping[str, Any]) -> tuple[EligibilityResult, ...]:
    if missing := structural_coverage(packs):
        issue = _issue("structural_registry_incomplete", ResolutionCategory.SYSTEM_FAILURE,
                       "/registry/structural", ", ".join(missing), "exact handler coverage",
                       "eligibility cannot be trusted", required=missing)
        return tuple(EligibilityResult(pack.method_id, False, (issue,)) for pack in packs.all())
    return tuple(assess_candidate(pack, facts, profile) for pack in packs.all())


def _treatment_levels(report: DiagnosticReportV2) -> tuple[str, ...]:
    for diagnostic_id in ("arm_counts", "treatment_prevalence", "effective_sample_feasibility"):
        result = next((row for row in report.results
                       if row.diagnostic_id == diagnostic_id), None)
        if result is not None:
            return tuple(key.removeprefix("count:") for key in result.values
                         if key.startswith("count:") and "|" not in key)
    return ()


def compile_contrasts(
    pack: MethodPackV1, comparator: str, report: DiagnosticReportV2,
) -> tuple[tuple[str, ...], tuple[ValidationIssueV2, ...]]:
    if pack.method_id == "did":
        return ("adopters_vs_comparison",), ()
    if pack.method_id == "sharp_rdd":
        return ("above_vs_below_cutoff",), ()
    levels = _treatment_levels(report)
    if len(levels) < 2:
        return (), (_issue(
            "treatment_levels_unavailable", ResolutionCategory.NEEDS_DATA,
            "/diagnostics/treatment_levels", repr(levels), "at least two observed levels",
            "an exact treatment contrast cannot be compiled", required=("treatment",)),)
    sides = tuple(part.strip().partition("=") for part in comparator.casefold().split(" vs "))
    candidate = (sides[1][2] if len(sides) == 2 and sides[0][0] == sides[1][0]
                 and all(name and eq == "=" and value in {level.casefold() for level in levels} for name, eq, value in sides) else comparator.strip())
    matched = [level for level in levels if level.casefold() == candidate.casefold()]
    if len(matched) != 1:
        return (), (_issue(
            "comparator_level_unresolved", ResolutionCategory.HUMAN_INPUT, "/facts/comparator",
            comparator, "one exact observed treatment level",
            "the estimator must not choose a comparator from sorted observed values",
            candidates=levels, required=("comparator",)),)
    control = matched[0]
    return tuple(encode_contrast(level, control) for level in levels if level != control), ()


def _bind(rule: BindingRule, facts: DesignFactSetV2,
          diagnostic_id: str = "") -> BoundDiagnosticInputV2 | None:
    if rule.roles:
        roles = rule.roles
        if rule.first_available:
            roles = next(((role,) for role in roles if facts.columns(role)), roles[:1])
        if (diagnostic_id in {"unit_period_uniqueness", "panel_completeness"}
                and facts.grain == "one_row_per_group_time"):
            roles = (G, TIME)
        columns = tuple(column for role in roles for column in facts.columns(role))
        return (BoundDiagnosticInputV2(parameter=rule.parameter, source_kind="role",
                                       source_id="+".join(role.value for role in roles),
                                       columns=columns) if columns else None)
    if rule.fact_id is not None:
        value = facts.fact(rule.fact_id)
        return (BoundDiagnosticInputV2(parameter=rule.parameter, source_kind="fact",
                                       source_id=rule.fact_id, value=value)
                if isinstance(value, str | int | float | bool) else None)
    return BoundDiagnosticInputV2(parameter=rule.parameter, source_kind="literal",
                                  source_id=f"literal:{rule.parameter}", value=rule.literal)


def compile_diagnostic_plan(pack: MethodPackV1, facts: DesignFactSetV2) -> DiagnosticPlanV2:
    items: list[DiagnosticPlanItemV2] = []
    issues: list[ValidationIssueV2] = []
    for diagnostic_id in pack.allowed_prerepair_diagnostic_ids:
        recipe = DIAGNOSTIC_RECIPES[diagnostic_id]
        inputs = tuple(found for rule in recipe.bindings
                       if (found := _bind(rule, facts, diagnostic_id)) is not None)
        missing = tuple(rule for rule in recipe.bindings if recipe.required_for_eligibility
                        and _bind(rule, facts, diagnostic_id) is None)
        for rule in missing:
            category = (ResolutionCategory.HUMAN_INPUT if rule.fact_id
                        else ResolutionCategory.NEEDS_DATA)
            required = ((rule.fact_id,) if rule.fact_id is not None
                        else tuple(role.value for role in rule.roles))
            issues.append(_issue("diagnostic_binding_missing", category,
                                 f"/diagnostics/{diagnostic_id}/{rule.parameter}", "unresolved",
                                 "a bound role column or approved scalar fact",
                                 "the diagnostic cannot read the intended statistical quantity",
                                 required=required))
        items.append(DiagnosticPlanItemV2(
            diagnostic_id=diagnostic_id, primitive=recipe.primitive,
            required_for_eligibility=recipe.required_for_eligibility, inputs=inputs))
    return DiagnosticPlanV2(selected_csv=facts.selected_csv, candidate_method_id=pack.method_id,
                            items=tuple(items), issues=tuple(issues))


def diagnostic_parameters(item: DiagnosticPlanItemV2) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for bound in item.inputs:
        value: Any = bound.columns if bound.columns else bound.value
        if bound.parameter in {"target", "column", "running_column"} and bound.columns:
            value = bound.columns[0]
        if bound.parameter in {"columns", "by", "key_columns"} and bound.value is not None:
            value = (bound.value,)
        params[bound.parameter] = value
    return params


def compile_measurements(
    cards: Sequence[tuple[ArtifactRef, ColumnSemanticCardV1]], columns: set[str],
) -> dict[str, ColumnMeasurementV2]:
    measurements = {}
    for source, card in cards:
        if card.column_name not in columns:
            continue
        supported = {name: slot for name in ("units", "scale")
                     if (slot := card.slots[name]).status is EpistemicStatus.EVIDENCED
                     and slot.evidence_ids and slot.value}
        measurements[card.column_name] = ColumnMeasurementV2(
            label=card.display_name, units=supported["units"].value if "units" in supported else None,
            scale=supported["scale"].value if "scale" in supported else None,
            source_card=source, supporting_evidence_ids=tuple(sorted({
                item for slot in supported.values() for item in slot.evidence_ids})))
    return measurements


def compile_design(
    *, pack: MethodPackV1, facts: DesignFactSetV2, proposal: AgentDesignProposalV2,
    frame: CausalFrameV1, causal_question: str, intended_decision: str,
    comparator: str, unit: str, contrasts: Sequence[str], rejected_methods: Mapping[str, str],
    method_structure: Mapping[str, str | int | float | bool] | None = None,
    registry_versions: Mapping[str, str] | None = None,
    column_cards: Sequence[tuple[ArtifactRef, ColumnSemanticCardV1]] = (),
) -> CompiledDesignV2:
    estimand = facts.fact("estimand")
    if not isinstance(estimand, str):
        raise TypeError("an evidenced or user-confirmed estimand is required")
    if estimand not in pack.supported_estimands:
        raise ValueError(f"estimand {estimand!r} is not supported by {pack.method_id}")
    key_roles = ((G, TIME) if facts.grain == "one_row_per_group_time" else
                 (U, TIME) if facts.grain == "one_row_per_unit_period" else (U,))
    keys = tuple(column for role in key_roles for column in facts.columns(role))
    keys = keys or ((ROW_UNIT_COLUMN,) if facts.grain == "one_row_per_unit" else ())
    protected = tuple(sorted({column for role in pack.imputation_forbidden_roles
                              for column in facts.columns(role)} | set(keys)))
    eligible = tuple(sorted({column for role in pack.imputation_eligible_roles
                             for column in facts.columns(role)} - set(protected)))
    selected = {
        "assumptions": (pack.mandatory_assumption_ids, pack.optional_assumption_ids,
                        proposal.optional_assumption_ids),
        "risks": (pack.mandatory_risk_ids, pack.optional_risk_ids, proposal.optional_risk_ids),
        "sensitivities": (pack.mandatory_sensitivity_ids, pack.optional_sensitivity_ids,
                          proposal.optional_sensitivity_ids)}
    for kind, (_, allowed, chosen) in selected.items():
        if unknown := sorted(set(chosen) - set(allowed)):
            raise ValueError(f"unregistered optional {kind}: {unknown}")
    rendered = {kind: tuple(pack.statements[name] for name in (*required, *chosen))
                for kind, (required, _, chosen) in selected.items()}
    return CompiledDesignV2(
        selected_csv=facts.selected_csv, causal_question=causal_question,
        intended_decision=intended_decision, frame=frame, method_id=pack.method_id,
        method_pack_version=pack.pack_version, estimand=estimand, comparator={"sharp_rdd": "below cutoff", "did": "comparison states"}.get(pack.method_id, comparator), unit=unit,
        primary_contrasts=tuple(contrasts), role_bindings=tuple(row for row in facts.role_bindings if row.role.value in {*pack.required_roles, *pack.optional_roles}),
        column_measurements=compile_measurements(column_cards, {
            column for row in facts.role_bindings for column in row.columns
            if row.role.value in {*pack.required_roles, *pack.optional_roles}}),
        preparation=PreparationPolicyV2(
            output_grain=facts.grain, key_columns=keys, required_roles=tuple(
                RoleName(role) for role in pack.required_roles), protected_columns=protected,
            imputation_permitted=eligible, eligibility_rule_ids=pack.eligibility_rule_vocabulary,
            unusable_row_rule_ids=pack.unusable_row_rule_ids,
            required_missingness_indicators=("outcome_observed",),
            method_structure=dict(method_structure or {}),
            deletion_impact_dimensions=pack.deletion_impact_dimensions,
            required_final_diagnostic_ids=pack.required_postrepair_diagnostic_ids,
            estimator_input_schema_id=pack.reserved_estimator_id),
        assumptions=rendered["assumptions"], identification_risks=rendered["risks"],
        sensitivity_requirements=rendered["sensitivities"],
        rejected_methods=dict(rejected_methods),
        required_visual_evidence=(),
        multiplicity_policy_id=("holm_step_down.v1" if len(contrasts) > 1 else None),
        registry_versions=dict(registry_versions or {}))
