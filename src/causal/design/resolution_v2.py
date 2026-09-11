"""Fact compilation, empirical eligibility, and actor-specific routing (T-036)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast, get_args

from pydantic import BaseModel, ConfigDict

from causal.design.askgate import AcceptedFactV1
from causal.design.compiler_v2 import EligibilityResult, _issue
from causal.design.contracts import DiagnosticResultV1, SourceInterpretationV1, TableGrain
from causal.design.packs import AcceptedFactConsumer, RequirementTemplateV1
from causal.design.semantics import RoleLedgerV1, RoleName
from causal.design.v2 import (
    AgentDesignProposalV2,
    DesignFactSetV2,
    DesignFactV2,
    DiagnosticPlanV2,
    DiagnosticReportV2,
    EvidenceRelation,
    FactAcceptanceStatus,
    FactSource,
    ResolutionCategory,
    RoleBindingV2,
    ValidationIssueV2,
)
from causal.shared.contracts import ArtifactRef
from causal.shared.envelope import EpistemicStatus, EvidenceClass, SupportClass
from causal.shared.frames import ROW_UNIT_COLUMN
from causal.shared.validation import evidence_class

_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True)

class RepairDecisionV2(BaseModel):
    model_config = _CONFIG
    action: Literal["continue", "retry_model", "ask_human", "stop"]
    terminal_status: Literal["needs_context", "needs_data", "unsupported",
                             "system_failure"] | None
    error_code: str | None
    issues: tuple[ValidationIssueV2, ...]


def _normalized(value: object) -> str:
    return " ".join(str(value).casefold().split())


def _supported_interpretations(
    fact_id: str, value: object, interpretations: Sequence[SourceInterpretationV1],
    evidence: Mapping[str, str], template: RequirementTemplateV1,
) -> tuple[tuple[str, ...], EvidenceRelation]:
    """Admit explicit model judgments whose exact source span and fact value are reviewable."""
    allowed = set(template.acceptable_evidence_types)
    matching = tuple(row for row in interpretations
                     if row.fact_key == fact_id and row.value == value
                     and row.relation in {"direct", "corroborating"}
                     and row.evidence_id in evidence
                     and evidence_class(row.evidence_id) in allowed
                     and _normalized(row.verbatim_excerpt)
                     in _normalized(evidence[row.evidence_id]))
    ids = tuple(dict.fromkeys(row.evidence_id for row in matching))
    relation = (EvidenceRelation.DIRECT if any(row.relation == "direct" for row in matching)
                else EvidenceRelation.CORROBORATING if matching
                else EvidenceRelation.SPECULATIVE)
    return ids, relation


def accepted_facts_for_consumer(
    accepted_facts: Sequence[AcceptedFactV1],
    requirement_templates: Mapping[str, RequirementTemplateV1],
    consumer_id: AcceptedFactConsumer,
) -> dict[tuple[str, str], AcceptedFactV1]:
    """Expose registry-routed facts without losing their persisted scope or provenance."""
    routed: dict[tuple[str, str], AcceptedFactV1] = {}
    for fact in accepted_facts:
        template = requirement_templates.get(fact.requirement_id)
        if template is None:
            raise ValueError(f"accepted fact has unknown requirement {fact.requirement_id!r}")
        if consumer_id not in template.accepted_fact.consumer_ids:
            continue
        key = (template.accepted_fact.fact_key, fact.scope_id)
        if key in routed:
            raise ValueError(f"duplicate current accepted fact {key!r}")
        routed[key] = fact
    return routed


def compile_proposal_facts(
    proposal: AgentDesignProposalV2, *, evidence: Mapping[str, str],
    requirement_templates: Mapping[str, RequirementTemplateV1],
    accepted_facts: Sequence[AcceptedFactV1] = (),
    source_artifact_id: str = "agent-proposal",
    candidate_grain: str | None = None,
    source_interpretations: Sequence[SourceInterpretationV1] = (),
    verified_grain_source: str | None = None,
) -> tuple[DesignFactV2, ...]:
    contracts = {template.accepted_fact.fact_key: template
                 for template in requirement_templates.values()}
    answered = {fact_key: fact for (fact_key, scope_id), fact in
                accepted_facts_for_consumer(
                    accepted_facts, requirement_templates, "method_compiler").items()
                if scope_id == "design"}
    proposed: dict[str, str | int | float | bool | None] = {
        row.fact_id: row.value
        for row in proposal.method_facts
    }
    proposed["assignment_mechanism"] = proposal.assignment_mechanism
    proposed["estimand"] = proposal.requested_estimand
    proposed["comparator"] = proposal.comparator
    if candidate_grain is not None:
        proposed["grain"] = candidate_grain
    interpretations = (*proposal.source_interpretations, *source_interpretations)
    rows: list[DesignFactV2] = []
    for fact_id, value in proposed.items():
        template = contracts[fact_id]
        if fact_id in answered:
            answer = answered[fact_id]
            rows.append(DesignFactV2(
                fact_id=fact_id, requirement_id=answer.requirement_id,
                scope_id=answer.scope_id, value=answer.value,
                source=FactSource(answer.source_kind),
                source_artifact_ids=(answer.origin_reference_id,),
                supporting_evidence_ids=answer.evidence_ids,
                evidence_class=answer.evidence_class,
                relation=EvidenceRelation(answer.relation),
                epistemic_status=EpistemicStatus.EVIDENCED,
                acceptance_status=FactAcceptanceStatus.ACCEPTED,
                executable=True))
            continue
        if fact_id == "grain" and verified_grain_source:
            rows.append(DesignFactV2(
                fact_id=fact_id, requirement_id=template.requirement_id,
                scope_id="design", value=value, source=FactSource.MEASUREMENT,
                source_artifact_ids=(verified_grain_source,),
                supporting_evidence_ids=(),
                evidence_class=EvidenceClass.MEASURED_OBSERVATION,
                relation=EvidenceRelation.DIRECT, epistemic_status=EpistemicStatus.EVIDENCED,
                acceptance_status=FactAcceptanceStatus.ACCEPTED, executable=True))
            continue
        valid, relation = _supported_interpretations(
            fact_id, value, interpretations, evidence, template)
        supported = bool(valid)
        source = (FactSource.MEASUREMENT if supported and evidence_class(valid[0])
                  is EvidenceClass.MEASURED_OBSERVATION else
                  FactSource.INTERPRETATION if supported else FactSource.MODEL)
        rows.append(DesignFactV2(
            fact_id=fact_id, requirement_id=template.requirement_id, scope_id="design",
            value=value, source=source,
            source_artifact_ids=(source_artifact_id,), supporting_evidence_ids=valid,
            evidence_class=(evidence_class(valid[0]) if valid else None),
            relation=relation,
            epistemic_status=EpistemicStatus.EVIDENCED if supported
            else EpistemicStatus.HYPOTHESIS,
            acceptance_status=(FactAcceptanceStatus.ACCEPTED if supported
                               else FactAcceptanceStatus.UNACCEPTED),
            executable=supported))
    return tuple(rows)


_EXECUTABLE_ROLE_SUPPORT = frozenset({
    SupportClass.DIRECT_USER_CONFIRMATION, SupportClass.DIRECT_SOURCE_STATEMENT,
    SupportClass.CORROBORATED_SOURCE_INFERENCE, SupportClass.MEASURED_OBSERVATION})
_MULTI_COLUMN_ROLES = frozenset({
    RoleName.CONFOUNDER_CANDIDATE, RoleName.PRECISION_COVARIATE,
    RoleName.EFFECT_MODIFIER, RoleName.SELECTION_VARIABLE})


def compile_fact_set(
    *, selected_csv: ArtifactRef, facts: Sequence[DesignFactV2],
    ledger: RoleLedgerV1, ledger_ref: ArtifactRef,
) -> DesignFactSetV2:
    grain_value = next((row.value for row in facts
                        if row.fact_id == "grain" and row.executable), None)
    if not isinstance(grain_value, str) or grain_value not in get_args(TableGrain):
        raise ValueError("an executable, registered table grain is required")
    grain = cast(TableGrain, grain_value)
    grouped: dict[RoleName, list[Any]] = {}
    unresolved: set[RoleName] = set()
    for claim in ledger.claims:
        if not claim.column_refs or claim.status is EpistemicStatus.UNKNOWN:
            continue
        if (claim.status is EpistemicStatus.EVIDENCED and claim.evidence_ids
                and claim.support_class in _EXECUTABLE_ROLE_SUPPORT):
            grouped.setdefault(claim.role, []).append(claim)
        else:
            unresolved.add(claim.role)
    conflicts = {f"unverified_role:{role.value}" for role in unresolved if role not in grouped}
    for role, claims in tuple(grouped.items()):
        columns = {column for claim in claims for column in claim.column_refs}
        if role not in _MULTI_COLUMN_ROLES and len(columns) != 1:
            conflicts.add(f"ambiguous_role:{role.value}")
            del grouped[role]
    bindings = [RoleBindingV2(
        role=role, columns=tuple(dict.fromkeys(
            column for claim in claims for column in claim.column_refs)),
        concept_id=(claims[0].concept_id if len(claims) == 1 else f"compiled:{role.value}"),
        source_artifact_ids=(ledger_ref.artifact_id,),
        epistemic_status=(EpistemicStatus.EVIDENCED if all(
            claim.status is EpistemicStatus.EVIDENCED for claim in claims)
            else EpistemicStatus.HYPOTHESIS))
        for role, claims in sorted(grouped.items(), key=lambda row: row[0].value)]
    if grain == "one_row_per_unit" and RoleName.UNIT_IDENTIFIER not in grouped:
        bindings.append(RoleBindingV2(
            role=RoleName.UNIT_IDENTIFIER, columns=(ROW_UNIT_COLUMN,),
            concept_id="derived:row_unit", source_artifact_ids=(selected_csv.artifact_id,),
            epistemic_status=EpistemicStatus.EVIDENCED, derived=True))
    compiled_facts = [row for row in facts if row.fact_id != "unit"]
    unit_binding = next((row for row in bindings if row.role is RoleName.UNIT_IDENTIFIER), None)
    unit_binding = unit_binding or next((row for row in bindings if row.role is RoleName.GROUP), None)
    if unit_binding is not None:
        compiled_facts.append(DesignFactV2(
            fact_id="unit", requirement_id="design.unit_identity", scope_id="design",
            value=unit_binding.concept_id, source=FactSource.DERIVED,
            source_artifact_ids=unit_binding.source_artifact_ids,
            supporting_evidence_ids=(),
            evidence_class=EvidenceClass.MEASURED_OBSERVATION,
            relation=EvidenceRelation.DIRECT,
            epistemic_status=unit_binding.epistemic_status,
            acceptance_status=FactAcceptanceStatus.ACCEPTED, executable=True))
    conflicts |= {fact.fact_id for fact in compiled_facts
                  if fact.relation.value == "conflicting"}
    return DesignFactSetV2(selected_csv=selected_csv, grain=grain, facts=tuple(compiled_facts),
                           role_bindings=tuple(bindings), conflicts=tuple(sorted(conflicts)))


def route_issues(
    issues: Sequence[ValidationIssueV2], *, corrections_used: int = 0,
    previous_fingerprints: Sequence[str] = (),
) -> RepairDecisionV2:
    rows = tuple(issues)
    if not rows:
        return RepairDecisionV2(action="continue", terminal_status=None,
                                error_code=None, issues=())
    categories = {row.category for row in rows}
    if categories == {ResolutionCategory.MODEL_FIX}:
        repeated = any(row.fingerprint in previous_fingerprints for row in rows)
        if corrections_used < 2 and not repeated:
            return RepairDecisionV2(action="retry_model", terminal_status=None,
                                    error_code=rows[0].code, issues=rows)
        return RepairDecisionV2(action="stop", terminal_status="system_failure",
                                error_code="agent_output_invalid", issues=rows)
    if ResolutionCategory.SYSTEM_FAILURE in categories or ResolutionCategory.MODEL_FIX in categories:
        return RepairDecisionV2(action="stop", terminal_status="system_failure",
                                error_code=rows[0].code, issues=rows)
    if ResolutionCategory.NEEDS_DATA in categories:
        return RepairDecisionV2(action="stop", terminal_status="needs_data",
                                error_code=rows[0].code, issues=rows)
    if ResolutionCategory.HUMAN_INPUT in categories:
        human = tuple(row for row in rows
                      if row.category is ResolutionCategory.HUMAN_INPUT)
        if human and all(row.fingerprint in previous_fingerprints for row in human):
            return RepairDecisionV2(
                action="stop", terminal_status="needs_context",
                error_code="compiler_requirement_unresolved", issues=rows)
        return RepairDecisionV2(action="ask_human", terminal_status="needs_context",
                                error_code=rows[0].code, issues=rows)
    return RepairDecisionV2(action="stop", terminal_status="unsupported",
                            error_code=rows[0].code, issues=rows)


def route_candidates(results: Sequence[EligibilityResult]) -> RepairDecisionV2:
    if any(row.eligible for row in results):
        return route_issues(())
    reports = [row.issues for row in results]
    shared_context = next((issue for rows in reports for issue in rows
                           if issue.code == "assignment_mechanism_unknown"), None)
    if shared_context is not None:
        return route_issues((shared_context,))
    human_only = next((rows for rows in reports if rows and all(
        issue.category is ResolutionCategory.HUMAN_INPUT for issue in rows)), None)
    if human_only is not None:
        return route_issues(human_only)
    model_only = next((rows for rows in reports if rows and all(
        issue.category is ResolutionCategory.MODEL_FIX for issue in rows)), None)
    if model_only is not None:
        return route_issues(model_only)
    data_only = next((rows for rows in reports if rows and all(
        issue.category in {ResolutionCategory.NEEDS_DATA, ResolutionCategory.HUMAN_INPUT}
        for issue in rows)), None)
    if data_only is not None:
        return route_issues(data_only)
    system = tuple(issue for rows in reports for issue in rows
                   if issue.category is ResolutionCategory.SYSTEM_FAILURE)
    if system:
        return route_issues(system)
    unsupported = tuple(issue for rows in reports for issue in rows
                        if issue.category is ResolutionCategory.UNSUPPORTED)
    return route_issues(unsupported or tuple(issue for rows in reports for issue in rows))


_MINIMUM: dict[str, tuple[str, str, float]] = {
    "arm_counts": ("group_count", "ge", 2),
    "treatment_prevalence": ("group_count", "eq", 2),
    "effective_sample_feasibility": ("group_count", "eq", 2),
    "cross_fitting_feasibility": ("group_count", "eq", 2),
    "group_time_counts": ("group_count", "ge", 4),
    "composition": ("group_count", "ge", 4),
    "cutoff_side_counts": ("group_count", "eq", 2),
}


def _result_issue(diagnostic_id: str, actual: str, expected: str, *,
                  code: str = "diagnostic_not_computable") -> ValidationIssueV2:
    return _issue(code, ResolutionCategory.NEEDS_DATA,
                  f"/diagnostics/{diagnostic_id}", actual, expected,
                  "the candidate method lacks measured empirical support",
                  required=(diagnostic_id,))


def _computed_issues(diagnostic_id: str, values: Mapping[str, Any],
                     ) -> list[ValidationIssueV2]:
    issues: list[ValidationIssueV2] = []
    if diagnostic_id in _MINIMUM:
        key, operation, threshold = _MINIMUM[diagnostic_id]
        observed = float(values.get(key) or 0)
        passed = observed >= threshold if operation == "ge" else observed == threshold
        if not passed:
            issues.append(_result_issue(diagnostic_id, f"{key}={observed:g}",
                                        f"{key} {operation} {threshold:g}",
                                        code="diagnostic_support_insufficient"))
    if diagnostic_id in {"assignment_unit_uniqueness", "unit_period_uniqueness",
                          "duplicates"} and values.get("is_unique") is not True:
        issues.append(_result_issue(diagnostic_id, "keys are repeated", "unique keys",
                                    code="diagnostic_support_insufficient"))
    if (diagnostic_id == "rough_overlap"
            and float(values.get("minimum_common_support_share") or 0) <= 0):
        issues.append(_result_issue(diagnostic_id, "no common support",
                                    "positive common support",
                                    code="diagnostic_support_insufficient"))
    if diagnostic_id == "power_precision_feasibility" and values.get("mde_80pct_95ci") is None:
        issues.append(_result_issue(
            diagnostic_id, "outcome variance is not estimable in at least two arms",
            "2+ observed outcomes in at least two arms", code="diagnostic_support_insufficient"))
    if diagnostic_id in {"effective_sample_feasibility", "cross_fitting_feasibility"}:
        minimum = int(values.get("minimum_group_count") or 0)
        if minimum < 2:
            issues.append(_result_issue(
                diagnostic_id, f"minimum_group_count={minimum}",
                "at least two rows in every treatment group",
                code="diagnostic_support_insufficient"))
    if (diagnostic_id == "panel_completeness"
            and float(values.get("cell_completeness") or 0) <= 0):
        issues.append(_result_issue(
            diagnostic_id, "no usable unit-period cells", "positive panel completeness",
            code="diagnostic_support_insufficient"))
    if diagnostic_id in {"distance_to_cutoff_support", "bandwidth_feasibility"}:
        below, above = (int(values.get(key) or 0)
                        for key in ("below_count", "at_or_above_count"))
        if below < 1 or above < 1:
            issues.append(_result_issue(
                diagnostic_id, f"below={below}, at_or_above={above}",
                "at least one observed row on each side of the approved cutoff",
                code="diagnostic_support_insufficient"))
    if (diagnostic_id == "cutoff_side_counts"
            and (values.get("assignment_direction") not in {"above", "below"}
                 or int(values.get("contradiction_count") or 0))):
        issues.append(_result_issue(
            diagnostic_id,
            f"direction={values.get('assignment_direction')}, "
            f"contradictions={int(values.get('contradiction_count') or 0)}",
            "one exact sharp assignment direction with zero contradictions",
            code="diagnostic_support_insufficient"))
    if diagnostic_id == "pre_period_availability":
        support = {name: int(values.get(name) or 0) for name in (
            "group_count", "period_count", "treatment_level_count",
            "treatment_history_count", "pre_period_rows", "post_period_rows")}
        if (support["group_count"] < 2 or support["period_count"] < 2
                or support["treatment_level_count"] != 2
                or support["treatment_history_count"] < 2
                or support["pre_period_rows"] < 1 or support["post_period_rows"] < 1):
            actual = ", ".join(f"{key}={value}" for key, value in support.items())
            issues.append(_result_issue(
                diagnostic_id, actual,
                "2+ groups, 2+ periods, binary treatment, distinct group histories, and "
                "observed pre/post rows", code="diagnostic_support_insufficient"))
    return issues


def evaluate_diagnostics(plan: DiagnosticPlanV2, results: Sequence[DiagnosticResultV1],
                         ) -> DiagnosticReportV2:
    issues = list(plan.issues)
    indexed = {row.diagnostic_id: row for row in results}
    for item in (row for row in plan.items if row.required_for_eligibility):
        result = indexed.get(item.diagnostic_id)
        if result is None:
            issues.append(_result_issue(item.diagnostic_id, "missing", "a computed result"))
        elif result.status.value != "computed":
            issues.append(_result_issue(item.diagnostic_id, result.status.value, "status computed"))
        else:
            issues.extend(_computed_issues(item.diagnostic_id, result.values))
    return DiagnosticReportV2(
        selected_csv=plan.selected_csv, candidate_method_id=plan.candidate_method_id,
        results=tuple(results), issues=tuple(issues), computable=not issues)


def compile_method_structure(
    facts: DesignFactSetV2, report: DiagnosticReportV2,
) -> dict[str, str | int | float | bool]:
    structure = {key: value for key in ("cutoff", "adoption_time")
                 if isinstance((value := facts.fact(key)), str | int | float | bool)}
    fields = {"cutoff_side_counts": ("assignment_direction", "treated_value", "comparator_value"),
              "pre_period_availability": ("adoption_profile_id",)}
    for result in report.results:
        for key in fields.get(result.diagnostic_id, ()):
            if isinstance((value := result.values.get(key)), str | int | float | bool):
                structure[key] = value
    return structure
