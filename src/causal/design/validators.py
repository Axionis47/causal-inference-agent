"""Validation walls over model-authored design evidence and proposals."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel

# isort: off
from causal.design.contracts import DesignContextManifestV1, DesignIntentV1
from causal.design.packs import MethodPackRegistry, MethodPackV1, PackRegistryError, RequirementTemplateV1
from causal.design.semantics import (
    CausalContextV1, CausalEdgeV1, ColumnSemanticCardV1, MeasurementMapV1, RoleClaimV1, RoleLedgerV1, RoleName,
)
from causal.design.v2 import AgentDesignProposalV2
from causal.shared.envelope import (
    AgentTaskResultV1, CausalFrameV1, ContextRequirementV1, EpistemicStatus, EvidenceClass,
    SupportClass,
)
from causal.shared.registry import RegistryError
from causal.shared.validation import (
    ACTIONS, DROP_ACTIONS, EVIDENCE_ACTIONS, EVIDENCE_CLASS_PREFIXES, FIX_ACTIONS,
    ReferenceKind, ValidationIssueV1, ValidationReport, ValidationRuleV1, evidence_class,
    has_cycle, load_rules, make_issue, parse_strict, rewrite_references, shape_report,
    validate_references,
)
# isort: on

__all__ = [
    "ACTIONS", "EVIDENCE_CLASS_PREFIXES", "REGISTRY_VERSION", "RULE_KINDS", "UNKNOWN_WALL",
    "ValidationContext", "ValidationIssueV1", "ValidationReport", "ValidationRuleV1",
    "canonical_requirement_scope", "evidence_class", "load_validation_rules",
    "validate_result", "wall_causal", "wall_evidence", "wall_method", "wall_references",
    "wall_shape", "wall_temporal"]

REGISTRY_VERSION: Final = "design-validators.v1"
RULE_KINDS: Final = ("temporal", "role_graph")
UNKNOWN_WALL: Final = "unknown_wall"
_FIX: Final = FIX_ACTIONS
_EVID: Final = EVIDENCE_ACTIONS
_DROP: Final = DROP_ACTIONS
_LIVE: Final = (EpistemicStatus.EVIDENCED, EpistemicStatus.HYPOTHESIS)
_SINGLE_ROLES: Final = frozenset({RoleName.TREATMENT, RoleName.OUTCOME, RoleName.UNIT_IDENTIFIER, RoleName.TIME, RoleName.ASSIGNMENT_VARIABLE, RoleName.GROUP, RoleName.CLUSTER, RoleName.STRATUM, RoleName.RUNNING_VARIABLE})
# The highest wall each §5.4 task kind can reach; later walls have no payload to read.
_TASK_MAX_WALL: Final = {"intent": 3, "semantic_batch": 4, "role_evidence": 4, "causal_context": 5, "role_ledger": 5, "method_design": 6}


type Result = AgentTaskResultV1 | None


def load_validation_rules(path: Path) -> tuple[ValidationRuleV1, ...]:
    """The design wall rows; failures keep the packs-registry error type (T-012 API)."""
    try:
        return load_rules(path, registry_version=REGISTRY_VERSION, kinds=RULE_KINDS, max_wall=7)
    except RegistryError as error:
        raise PackRegistryError(str(error), error.code) from error


@dataclass(frozen=True)
class ValidationContext:
    """Everything the walls may read; a validator never loads anything itself."""

    manifest: DesignContextManifestV1; intent: DesignIntentV1 | None = None; measurement_map: MeasurementMapV1 | None = None
    rules: tuple[ValidationRuleV1, ...] = ()
    parents: Mapping[str, str] = field(default_factory=dict)
    evidence_ids: frozenset[str] = frozenset(); concept_ids: frozenset[str] = frozenset()
    evidence_text: Mapping[str, str] = field(default_factory=dict)
    user_answer_evidence_ids: frozenset[str] = frozenset()
    templates: Mapping[str, RequirementTemplateV1] = field(default_factory=dict)
    packs: MethodPackRegistry | None = None
    causal_context: CausalContextV1 | None = None
    selected_alternative_id: str | None = None
    role_ledger: RoleLedgerV1 | None = None
    resolved_requirements: Mapping[str, str] = field(default_factory=dict)
    diagnostic_ids: frozenset[str] = frozenset()
    diagnostic_result_ids: frozenset[str] = frozenset()
    dataset_id: str = ""
    relationship_ids: frozenset[str] = frozenset()

    def rules_of(self, kind: str) -> tuple[ValidationRuleV1, ...]:
        return tuple(rule for rule in self.rules if rule.kind == kind)

def _role_claims(payload: Any, ctx: ValidationContext) -> tuple[tuple[int, RoleClaimV1], ...]:
    found = [item for name in ("claims", "role_hypotheses")
             for item in getattr(payload, name, ()) if isinstance(item, RoleClaimV1)]
    return tuple(enumerate(found or list(ctx.role_ledger.claims if ctx.role_ledger else ())))


wall_shape = shape_report


def canonical_requirement_scope(
    requirement: Any, ctx: ValidationContext, payload: Any = None,
) -> str | None:
    """Resolve one requirement scope from registry kind plus the bounded task context."""
    template = ctx.templates.get(requirement.requirement_id)
    if template is None:
        return None
    kind = template.scope_kind.value
    raw = str(requirement.scope_id)
    if kind == "design":
        return "design"
    if kind == "dataset":
        return ctx.dataset_id or None
    if kind == "table":
        return ctx.manifest.selected_table
    if kind == "column":
        candidate = raw.rsplit("::", 1)[-1]
        return candidate if candidate in {row.column_name for row in ctx.manifest.structural_inventory} else None
    concepts = set(ctx.concept_ids)
    relationships = set(ctx.relationship_ids)
    if isinstance(payload, CausalContextV1):
        concepts.update(payload.concept_ids)
        relationships.update(edge.edge_id for edge in payload.edges)
        relationships.update(edge.edge_id for alt in payload.alternatives for edge in alt.edges)
    elif isinstance(payload, Mapping) and payload.get("schema_version") == "causal-context.v1":
        concepts.update(str(item) for item in payload.get("concept_ids", ()))
        relationships.update(str(edge.get("edge_id")) for edge in payload.get("edges", ())
                             if isinstance(edge, Mapping) and edge.get("edge_id"))
    legal = concepts if kind == "concept" else relationships if kind == "relationship" else set()
    return raw if raw in legal else None


def _reference_catalogs(ctx: ValidationContext) -> dict[ReferenceKind, tuple[str, ...]]:
    """The namespaces the current validation context closes authoritatively."""
    catalogs: dict[ReferenceKind, tuple[str, ...]] = {
        ReferenceKind.EVIDENCE: tuple(sorted(
            ctx.evidence_ids | ctx.user_answer_evidence_ids)),
        ReferenceKind.REQUIREMENT: tuple(sorted(ctx.templates)),
        ReferenceKind.COLUMN: tuple(sorted(
            row.column_name for row in ctx.manifest.structural_inventory)),
        ReferenceKind.DIAGNOSTIC: tuple(sorted(ctx.diagnostic_ids)),
        ReferenceKind.DIAGNOSTIC_RESULT: tuple(sorted(ctx.diagnostic_result_ids)),
        ReferenceKind.ARTIFACT: tuple(sorted(ctx.parents)),
    }
    concepts = set(ctx.concept_ids)
    graph_edges = set(ctx.relationship_ids)
    if ctx.causal_context is not None:
        concepts.update(ctx.causal_context.concept_ids)
        graph_edges.update(edge.edge_id for edge in ctx.causal_context.edges)
        graph_edges.update(
            edge.edge_id for alternative in ctx.causal_context.alternatives
            for edge in alternative.edges)
        catalogs[ReferenceKind.ALTERNATIVE] = tuple(sorted(
            alternative.alternative_id for alternative in ctx.causal_context.alternatives))
    if concepts:
        catalogs[ReferenceKind.CONCEPT] = tuple(sorted(concepts))
    if graph_edges:
        catalogs[ReferenceKind.GRAPH_EDGE] = tuple(sorted(graph_edges))
    if ctx.packs is not None:
        catalogs[ReferenceKind.METHOD] = tuple(sorted(
            pack.method_id for pack in ctx.packs.all()))
    return catalogs


def _canonicalize_payload(
    model_cls: type[BaseModel], payload: dict[str, Any], ctx: ValidationContext,
) -> None:
    """Rewrite uniquely resolvable syntax while preserving every semantic decision."""
    catalogs = _reference_catalogs(ctx)

    def normalize(kind: ReferenceKind, value: str) -> str:
        legal = catalogs.get(kind, ())
        candidate = value.removesuffix(":")
        if kind is ReferenceKind.EVIDENCE:
            return candidate if candidate in legal else value
        if kind is not ReferenceKind.COLUMN:
            return value
        folded = candidate.casefold()
        matches = [item for item in legal if item.casefold() == folded or any(
            folded.endswith(separator + item.casefold())
            for separator in ("/", ".", "::"))]
        return matches[0] if len(matches) == 1 else value

    payload.update(rewrite_references(model_cls, payload, normalize))


def wall_references(payload: Any, result: Result, ctx: ValidationContext) -> ValidationReport:
    if not isinstance(payload, BaseModel):
        return ValidationReport(wall=2, issues=())
    catalogs = _reference_catalogs(ctx)
    issues = validate_references(
        type(payload), payload.model_dump(mode="python"), catalogs)
    if result is not None:
        issues += tuple(
            issue for index, requirement in enumerate(result.missing_requirements)
            for issue in validate_references(
                ContextRequirementV1, requirement.model_dump(mode="python"), catalogs,
                prefix=f"/missing_requirements/{index}",
            ))
    return ValidationReport(wall=2, issues=issues)


def _compatible_methods(payload: AgentDesignProposalV2,
                        ctx: ValidationContext) -> tuple[MethodPackV1, ...]:
    by_id = {pack.method_id: pack for pack in ctx.packs.all()} if ctx.packs else {}
    return tuple(by_id[method_id] for method_id in payload.ranked_method_ids
                 if method_id in by_id and payload.assignment_mechanism
                 in by_id[method_id].compatible_assignment_mechanisms)


def wall_evidence(payload: Any, result: Result, ctx: ValidationContext) -> ValidationReport:
    issues: list[ValidationIssueV1] = []
    if isinstance(payload, DesignIntentV1):
        overlap = set(payload.timeframe.candidate_columns) & set(payload.outcome.candidate_columns)
        if overlap:
            issues.append(make_issue(
                "timeframe_is_outcome_measurement", "/timeframe/candidate_columns",
                "wall3.timeframe_metadata", _FIX, False, tuple(sorted(overlap)),
                "An outcome measured over a study window is not an observed time coordinate. "
                "Preserve the stated window in timeframe.name/description; use empty "
                "timeframe.candidate_columns when the source supplies only window metadata. "
                "Retain the outcome binding and do not choose another measure as a time proxy."))
    for index, interpretation in enumerate(
            getattr(payload, "source_interpretations", ())):
        source = ctx.evidence_text.get(interpretation.evidence_id)
        normalized = " ".join(interpretation.verbatim_excerpt.casefold().split())
        if source is not None and normalized in " ".join(source.casefold().split()):
            continue
        issues.append(make_issue(
            "evidence_quote_mismatch",
            f"/source_interpretations/{index}/verbatim_excerpt",
            "wall3.evidence_quote", _FIX, False, (interpretation.evidence_id,),
            "Copy a short exact span from the cited task-local evidence."))
    unsupported = {SupportClass.MODEL_HYPOTHESIS, SupportClass.CONFLICTING, SupportClass.UNKNOWN}
    expected_classes = {
        SupportClass.DIRECT_USER_CONFIRMATION: {EvidenceClass.USER_CONFIRMATION},
        SupportClass.MEASURED_OBSERVATION: {EvidenceClass.MEASURED_OBSERVATION},
        SupportClass.DIRECT_SOURCE_STATEMENT: {
            EvidenceClass.DATA_DICTIONARY, EvidenceClass.STUDY_PROTOCOL,
            EvidenceClass.SOURCE_STATEMENT, EvidenceClass.TIMESTAMP_RELATIONSHIP},
        SupportClass.CORROBORATED_SOURCE_INFERENCE: {
            EvidenceClass.DATA_DICTIONARY, EvidenceClass.STUDY_PROTOCOL,
            EvidenceClass.SOURCE_STATEMENT, EvidenceClass.TIMESTAMP_RELATIONSHIP}}
    for index, claim in _role_claims(payload, ctx):
        if (claim.status is EpistemicStatus.EVIDENCED and (not claim.evidence_ids or claim.support_class in unsupported)) or (claim.status is EpistemicStatus.HYPOTHESIS and bool(claim.evidence_ids) and claim.support_class not in unsupported):
            issues.append(make_issue(
                "role_claim_unsupported", f"/claims/{index}", "wall3.role_evidence", _EVID, True, (claim.role.value,), "Role status, citations, and support class must agree: direct or corroborated cited support is evidenced; a hypothesis uses a non-evidentiary support class."))
        cited_classes = {evidence_class(item) for item in claim.evidence_ids}
        allowed = expected_classes.get(claim.support_class)
        if (claim.status is EpistemicStatus.EVIDENCED and claim.evidence_ids
                and allowed is not None and not (cited_classes & allowed)):
            issues.append(make_issue(
                "role_support_class_mismatch", f"/claims/{index}/support_class",
                "wall3.support_class", _FIX, False, tuple(claim.evidence_ids),
                f"The cited evidence classes do not support {claim.support_class.value}; "
                f"use one of {sorted(item.value for item in allowed)} or revise the status."))
    preferred = (next(iter(_compatible_methods(payload, ctx)), None)
                 if isinstance(payload, AgentDesignProposalV2) else None)
    for index, requirement in enumerate(result.missing_requirements if result else ()):
        scope_id = canonical_requirement_scope(requirement, ctx, payload)
        if scope_id is None:
            issues.append(make_issue(
                "invalid_requirement_scope", f"/missing_requirements/{index}/scope_id",
                "wall3.requirement_scope", _FIX, False, (requirement.scope_id,),
                "Use the registry-owned scope for this requirement and an exact scope id from "
                "the task reference catalog."))
            continue
        prior = ctx.resolved_requirements.get(
            f"{requirement.requirement_id}::{scope_id}")
        settled = (isinstance(payload, ColumnSemanticCardV1) and requirement.requirement_id == "column.measurement_timing" and (payload.slots["timing"].status is EpistemicStatus.EVIDENCED
                        or (payload.slots["kind"].status is EpistemicStatus.EVIDENCED and payload.slots["kind"].value == "identifier")))
        role = RoleName.TREATMENT if requirement.requirement_id == "design.treatment_meaning" else None
        settled = settled or (isinstance(payload, RoleLedgerV1) and role is not None and any(
            claim.role is role and claim.status is EpistemicStatus.EVIDENCED
            for claim in payload.claims))
        settled = settled or (requirement.requirement_id == "column.measurement_timing" and any(requirement.scope_id in claim.column_refs and claim.status is EpistemicStatus.EVIDENCED and claim.timing.value != "unknown" for _, claim in _role_claims(payload, ctx)))
        invalid = prior in {"resolved", "unknown_accepted", "refused"} or settled
        if invalid:
            detail = (f"{requirement.requirement_id} at canonical scope {scope_id} is already "
                      f"{prior}; remove it from missing_requirements."
                      if prior else "Remove a requirement already settled by the payload.")
            issues.append(make_issue("invalid_context_requirement", f"/missing_requirements/{index}", "wall3.requirement_search", _FIX, False, (requirement.requirement_id,), detail))
        template = ctx.templates.get(requirement.requirement_id)
        if (preferred is not None and template is not None
                and preferred.method_id not in template.methods_required_for):
            issues.append(make_issue(
                "method_requirement_outside_preferred_design",
                f"/missing_requirements/{index}", "wall3.preferred_method_requirements",
                _FIX, False, (requirement.requirement_id, preferred.method_id),
                f"{requirement.requirement_id} is not a prerequisite of the first "
                f"assignment-compatible method in your ranking ({preferred.method_id}). "
                "Ranking every registered method does not activate all their prerequisites. "
                "Remove this irrelevant request; preserve the ranking and scientific facts. "
                "Use existing supported facts, and request only unresolved facts needed by "
                "the preferred design; return complete when no such question remains."))
    return ValidationReport(wall=3, issues=tuple(issues))


def wall_temporal(payload: Any, result: Result, ctx: ValidationContext) -> ValidationReport:
    """Wall 4: role timing obeys the declarative `temporal` rows."""
    return ValidationReport(wall=4, issues=tuple(
        rule.issue(f"/claims/{index}/timing", (claim.concept_id,))
        for index, claim in _role_claims(payload, ctx) for rule in ctx.rules_of("temporal")
        if claim.role.value in rule.params["roles"]
        and (claim.timing.value in rule.params["timings"])
        == (rule.params["relation"] == "forbidden")))


def _role_graph_issues(index: int, claim: RoleClaimV1, frame: CausalFrameV1,
                       edges: Sequence[CausalEdgeV1],
                       ctx: ValidationContext) -> Iterator[ValidationIssueV1]:
    def present(pair: Sequence[str], statuses: Sequence[str]) -> bool:
        ends = [claim.concept_id if n == "concept" else str(getattr(frame, n)) for n in pair]
        return any(e.source_concept_id == ends[0] and e.target_concept_id == ends[1]
                   and e.status.value in statuses for e in edges)
    for rule in (row for row in ctx.rules_of("role_graph") if row.params["role"] == claim.role):
        bad = False
        for key, offending in (("required_edges", False), ("forbidden_edges", True)):
            for pair in rule.params.get(key) or ():
                bad = bad or offending is present(pair, rule.params["statuses"])
        if bad:
            yield rule.issue(f"/claims/{index}/graph_edge_ids", (claim.concept_id,)).model_copy(update={"detail": str(rule.params.get("correction_detail", ""))})


def _graph_identity_issues(context: CausalContextV1) -> list[ValidationIssueV1]:
    issues: list[ValidationIssueV1] = []
    endpoints: dict[str, tuple[str, str]] = {}
    alternatives: set[str] = set()
    groups = [("/edges", context.edges)]
    for index, alternative in enumerate(context.alternatives):
        if alternative.alternative_id in alternatives:
            issues.append(make_issue(
                "duplicate_graph_alternative_id", f"/alternatives/{index}/alternative_id",
                "wall5.graph_identity", _FIX, False, (alternative.alternative_id,),
                "Each alternative needs its own identity; retain only distinct competing graphs."))
        alternatives.add(alternative.alternative_id)
        groups.append((f"/alternatives/{index}/edges", alternative.edges))
    for prefix, edges in groups:
        local: set[str] = set()
        for index, edge in enumerate(edges):
            pair = (edge.source_concept_id, edge.target_concept_id)
            path = f"{prefix}/{index}/edge_id"
            if edge.edge_id in local:
                issues.append(make_issue(
                    "duplicate_graph_edge_id", path, "wall5.graph_identity", _FIX, False,
                    (edge.edge_id,), "An edge_id may appear only once within one graph edge set. "
                    "Resolve the duplicate in the causal context; do not silently drop evidence."))
            local.add(edge.edge_id)
            if edge.edge_id in endpoints and endpoints[edge.edge_id] != pair:
                old = endpoints[edge.edge_id]
                issues.append(make_issue(
                    "conflicting_graph_edge_identity", path, "wall5.graph_identity", _FIX, False,
                    (edge.edge_id,), f"This edge_id already names {old[0]} -> {old[1]}, but "
                    f"now names {pair[0]} -> {pair[1]}. Reuse an edge_id across alternatives only "
                    "for the same directed relation. Reconsider conflicting directions using "
                    "the supplied evidence; distinct retained relations need distinct ids."))
            else:
                endpoints[edge.edge_id] = pair
    return issues


def wall_causal(payload: Any, result: Result, ctx: ValidationContext) -> ValidationReport:
    context = payload if isinstance(payload, CausalContextV1) else ctx.causal_context
    if context is None:
        return ValidationReport(wall=5, issues=())
    # A selected alternative replaces the base edge set outright (GraphAlternativeV1 semantics).
    picked = [alt.edges for alt in context.alternatives
              if alt.alternative_id == ctx.selected_alternative_id]
    edges = picked[0] if picked else context.edges
    live = {edge.edge_id for alt in context.alternatives for edge in alt.edges}
    live_edges = tuple(e for e in edges if e.status in _LIVE)
    cycle = tuple((e.source_concept_id, e.target_concept_id) for e in live_edges)
    issues = _graph_identity_issues(context)
    issues += [make_issue(
        "graph_cycle", "/edges", "wall5.acyclic", _DROP, False,
        tuple(e.edge_id for e in live_edges), "Remove or mark unknown at least one edge in each "
        "directed cycle. Live endpoints: " + "; ".join(f"{a} -> {b}" for a, b in cycle))
        ] if has_cycle(cycle) else []
    issues += [make_issue("causal_target_edge_missing", "/frame", "wall5.causal_target", _FIX, False, (context.frame.treatment, context.frame.outcome), "Align the causal frame anchors with a supported selected-graph edge from treatment to outcome; do not invent an edge.")] if isinstance(payload, CausalContextV1) and not any(e.source_concept_id == context.frame.treatment and e.target_concept_id == context.frame.outcome and e.status in _LIVE for e in edges) else []
    if isinstance(payload, RoleLedgerV1) and payload.frame != context.frame:
        issues.append(make_issue(
            "causal_frame_changed", "/frame", "wall5.frame_immutable", _FIX, False, (),
            "Copy the validated causal-context frame exactly; the role ledger cannot redefine "
            "the treatment, outcome, population, or timeframe."))
    if isinstance(payload, RoleLedgerV1):
        for role in _SINGLE_ROLES:
            columns = {column for claim in payload.claims if claim.role is role
                       and claim.status in _LIVE for column in claim.column_refs}
            if len(columns) > 1:
                issues.append(make_issue(
                    "ambiguous_single_role", "/claims", "wall5.single_role", _FIX, False,
                    tuple(sorted(columns)), f"Bind {role.value} to one column only; keep other "
                    "candidates in alternatives instead of making duplicate role claims."))
    issues += [make_issue("disputed_edge_without_alternative", f"/edges/{edge.edge_id}",
                      "wall5.alternatives", ("revise_field", "add_evidence"), False,
                      (edge.edge_id,), "A disputed edge must appear with the same edge_id in at "
                      "least one graph alternative; otherwise revise its status.")
               for edge in context.edges
               if edge.status is EpistemicStatus.DISPUTED and edge.edge_id not in live]
    for index, claim in _role_claims(payload, ctx):
        if isinstance(payload, RoleLedgerV1) and claim.role is RoleName.TIME:
            outcomes = {column for row in payload.claims if row.role is RoleName.OUTCOME
                        for column in row.column_refs}
            if overlap := outcomes & set(claim.column_refs):
                issues.append(make_issue(
                    "time_role_is_outcome_measurement", f"/claims/{index}/column_refs",
                    "wall5.time_coordinate", _FIX, False, tuple(sorted(overlap)),
                    "The outcome column measures the response, not the observation time. "
                    "Preserve the outcome role and frame.timeframe. When the source gives only "
                    "a study window, omit the optional time claim; a longitudinal time role "
                    "must instead bind a distinct source-supported observation coordinate."))
        field = {RoleName.TREATMENT: "treatment", RoleName.OUTCOME: "outcome", RoleName.UNIT_IDENTIFIER: "unit", RoleName.TIME: "timeframe", RoleName.GROUP: "comparator"}.get(claim.role)
        proposed = getattr(ctx.intent, field, None) if ctx.intent is not None and field else None
        if isinstance(payload, RoleLedgerV1) and proposed and claim.column_refs and not set(claim.column_refs) <= set(proposed.candidate_columns):
            issues.append(make_issue("role_column_outside_intent", f"/claims/{index}/column_refs", "wall5.intent_binding", _FIX, False, tuple(sorted(claim.column_refs)), f"Bind {claim.role.value} only to columns proposed for intent.{field}: {list(proposed.candidate_columns)}."))
        if isinstance(payload, RoleLedgerV1) and claim.role is RoleName.TREATMENT and ctx.measurement_map and claim.column_refs:
            concurrent = {row.column_name for row in ctx.measurement_map.links if row.concept_id == payload.frame.treatment and row.timing.value == "concurrent"}
            if concurrent and not set(claim.column_refs) <= concurrent:
                issues.append(make_issue("treatment_column_not_active_exposure", f"/claims/{index}/column_refs", "wall5.active_exposure", _FIX, False, tuple(sorted(claim.column_refs)), f"Treatment must be the concurrent realized-exposure measurement. Compatible columns: {sorted(concurrent)}."))
        if (isinstance(payload, RoleLedgerV1)
                and claim.role in {RoleName.TREATMENT, RoleName.OUTCOME}
                and not claim.column_refs):
            issues.append(make_issue(
                "required_role_columns_missing", f"/claims/{index}/column_refs",
                "wall5.required_role_columns", _FIX, False, (claim.role.value,),
                "Bind the treatment and outcome anchors to one or more columns from the "
                "structural inventory."))
        issues += _role_graph_issues(index, claim, context.frame, edges, ctx)
    return ValidationReport(wall=5, issues=tuple(issues))


def wall_method(payload: Any, result: Result, ctx: ValidationContext) -> ValidationReport:
    """Wall 6: the proposal ranks only registered methods and names a valid estimand."""
    if isinstance(payload, AgentDesignProposalV2):
        registered = tuple(pack.method_id for pack in ctx.packs.all()) if ctx.packs else ()
        ranked = payload.ranked_method_ids
        issues = []
        if len(ranked) != len(set(ranked)) or set(ranked) != set(registered):
            issues.append(make_issue(
                "method_ranking_incomplete", "/ranked_method_ids", "wall6.method_ranking",
                _FIX, False, (*ranked, *registered),
                "Rank every registered method exactly once; do not invent or omit a method."))
        if payload.requested_estimand == "unknown":
            issues.append(make_issue(
                "estimand_missing", "/requested_estimand", "wall6.estimand", _FIX, False, (),
                "State the requested estimand; the compiler must never choose one implicitly."))
        elif ctx.packs and not any(payload.requested_estimand in pack.supported_estimands
                                   for pack in ctx.packs.all()):
            issues.append(make_issue(
                "unsupported_estimand", "/requested_estimand", "wall6.estimand",
                _FIX, False, (payload.requested_estimand,),
                "Choose an estimand declared by at least one registered method pack."))
        compatible = _compatible_methods(payload, ctx)
        if compatible:
            pack = compatible[0]
            choices = (("optional_assumption_ids", pack.optional_assumption_ids),
                       ("optional_risk_ids", pack.optional_risk_ids),
                       ("optional_sensitivity_ids", pack.optional_sensitivity_ids))
            for field, allowed in choices:
                if unknown := sorted(set(getattr(payload, field)) - set(allowed)):
                    issues.append(make_issue(
                        "unregistered_method_statement", f"/{field}",
                        "wall6.registered_statements", _FIX, False, tuple(unknown),
                        f"Select only registered optional ids: {sorted(allowed)}."))
        if (payload.assignment_mechanism != "unknown" and compatible
                and payload.requested_estimand != "unknown"
                and payload.requested_estimand not in compatible[0].supported_estimands):
            issues.append(make_issue(
                "preferred_method_estimand_mismatch", "/ranked_method_ids/0",
                "wall6.estimand_compatibility", _FIX, False,
                (compatible[0].method_id, payload.requested_estimand),
                "Rank an assignment-compatible method that supports the requested estimand first."))
        return ValidationReport(wall=6, issues=tuple(issues))
    return ValidationReport(wall=6, issues=())


type Wall = Callable[[Any, Result, ValidationContext], ValidationReport]

_WALL_ORDER: Final[tuple[Wall, ...]] = (wall_references, wall_evidence, wall_temporal,
                                        wall_causal, wall_method)


def validate_result(wall: int, task_kind: str, model_cls: type[BaseModel],
                    result: AgentTaskResultV1, ctx: ValidationContext) -> ValidationReport:
    """Run walls 1..`wall` in §16.3 order; the first failing wall stops the run (§16.4)."""
    _canonicalize_payload(model_cls, result.payload, ctx)
    report = wall_shape(model_cls, result)
    highest = min(wall, _TASK_MAX_WALL.get(task_kind, wall))
    if not report.passed or highest < 2:
        return report
    payload = parse_strict(model_cls, result.payload)
    for run in _WALL_ORDER[:highest - 1]:
        report = run(payload, result, ctx)
        if not report.passed:
            return report
    return ValidationReport(wall=highest, issues=())
