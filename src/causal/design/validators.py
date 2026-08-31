"""Validation walls 1–7 over declarative rows and pure graph code (PRD-002 §16.3–§16.4)."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel

from causal.design.contracts import DesignContextManifestV1
from causal.design.frame import ExperimentDesignV1, RunnableFrameContractV1
from causal.design.packs import (
    MethodPackRegistry,
    MethodPackV1,
    PackRegistryError,
    RequirementTemplateV1,
)
from causal.design.semantics import CausalContextV1, CausalEdgeV1, RoleClaimV1, RoleLedgerV1
from causal.design.triage import ColumnTriageRecordV1
from causal.shared.envelope import (
    AgentTaskResultV1,
    CausalFrameV1,
    ClaimV1,
    EpistemicStatus,
)
from causal.shared.registry import RegistryError
from causal.shared.validation import (
    ACTIONS,
    ASK_ACTIONS,
    DROP_ACTIONS,
    EVIDENCE_ACTIONS,
    EVIDENCE_CLASS_PREFIXES,
    FIX_ACTIONS,
    ValidationIssueV1,
    ValidationReport,
    ValidationRuleV1,
    as_tuple,
    collect_ids,
    evidence_class,
    has_cycle,
    load_rules,
    make_issue,
    parse_strict,
    self_citation_issues,
    shape_report,
    unresolved_issues,
)

__all__ = [
    "ACTIONS", "EVIDENCE_CLASS_PREFIXES", "REGISTRY_VERSION", "RULE_KINDS", "UNKNOWN_RULE_KIND",
    "UNKNOWN_WALL", "ValidationContext", "ValidationIssueV1", "ValidationReport",
    "ValidationRuleV1", "evidence_class", "load_validation_rules",
    "validate_result", "wall_causal", "wall_evidence", "wall_frame", "wall_method",
    "wall_references", "wall_shape", "wall_temporal"]

REGISTRY_VERSION: Final = "design-validators.v1"
RULE_KINDS: Final = ("temporal", "role_graph", "frame_subset", "method_structural")
UNKNOWN_RULE_KIND: Final = "unknown_rule_kind"
UNKNOWN_WALL: Final = "unknown_wall"
RESOLVED_STATES: Final = ("resolved", "unknown_accepted")
_FIX: Final = FIX_ACTIONS
_ASK: Final = ASK_ACTIONS
_EVID: Final = EVIDENCE_ACTIONS
_DROP: Final = DROP_ACTIONS
_ADJUSTMENT: Final = ("confounder_candidate", "precision_covariate")
_COLUMN_KEYS: Final = ("column_name", "column_refs", "key_columns", "imputation_permitted",
                       "imputation_forbidden", "imputation_eligible_columns")
_EVIDENCE_KEYS: Final = ("evidence_ids", "supporting_evidence_ids", "contrary_evidence_ids")
_LIVE: Final = (EpistemicStatus.EVIDENCED, EpistemicStatus.HYPOTHESIS)
# The highest wall each §5.4 task kind can reach; later walls have no payload to read.
_TASK_MAX_WALL: Final = {"intent": 3, "semantic_batch": 4, "role_evidence": 4,
                         "causal_synthesis": 5, "method_design": 7}


type Issues = Iterator[ValidationIssueV1]
type Result = AgentTaskResultV1 | None
type Claims = Sequence[RoleClaimV1]
type Edges = Sequence[CausalEdgeV1]


def load_validation_rules(path: Path) -> tuple[ValidationRuleV1, ...]:
    """The design wall rows; failures keep the packs-registry error type (T-012 API)."""
    try:
        return load_rules(path, registry_version=REGISTRY_VERSION, kinds=RULE_KINDS, max_wall=7)
    except RegistryError as error:
        raise PackRegistryError(str(error), error.code) from error


@dataclass(frozen=True)
class ValidationContext:
    """Everything the walls may read; a validator never loads anything itself."""

    manifest: DesignContextManifestV1
    rules: tuple[ValidationRuleV1, ...] = ()
    triage: ColumnTriageRecordV1 | None = None
    parents: Mapping[str, str] = field(default_factory=dict)
    evidence_ids: frozenset[str] = frozenset()
    user_answer_evidence_ids: frozenset[str] = frozenset()
    templates: Mapping[str, RequirementTemplateV1] = field(default_factory=dict)
    packs: MethodPackRegistry | None = None
    method_id: str | None = None
    causal_context: CausalContextV1 | None = None
    selected_alternative_id: str | None = None
    role_ledger: RoleLedgerV1 | None = None
    design: ExperimentDesignV1 | None = None
    resolved_requirements: Mapping[str, str] = field(default_factory=dict)

    def rules_of(self, kind: str) -> tuple[ValidationRuleV1, ...]:
        return tuple(rule for rule in self.rules if rule.kind == kind)

    def pack(self, method_id: str | None = None) -> MethodPackV1 | None:
        chosen = method_id or self.method_id
        return None if self.packs is None or chosen is None else self.packs.get(chosen)


_issue = make_issue
_parse = parse_strict
_gather = collect_ids
_unresolved = unresolved_issues
_self_citations = self_citation_issues
_seq = as_tuple


def _role_claims(payload: Any, ctx: ValidationContext) -> tuple[tuple[int, RoleClaimV1], ...]:
    found = [item for name in ("claims", "role_hypotheses")
             for item in getattr(payload, name, ()) if isinstance(item, RoleClaimV1)]
    return tuple(enumerate(found or list(ctx.role_ledger.claims if ctx.role_ledger else ())))


wall_shape = shape_report


def wall_references(payload: Any, result: Result, ctx: ValidationContext) -> ValidationReport:
    """Wall 2: every column, evidence, parent, and requirement id resolves."""
    dumped = payload.model_dump(mode="python") if isinstance(payload, BaseModel) else dict(payload)
    inventory = {row.column_name for row in ctx.manifest.structural_inventory}
    known = ctx.evidence_ids | ctx.user_answer_evidence_ids
    parents = set(result.parent_artifact_ids) if result else set()
    raised = {req.requirement_id for req in result.missing_requirements} if result else set()
    return ValidationReport(wall=2, issues=(
        *_unresolved(_gather(dumped, _COLUMN_KEYS) - inventory, "unresolved_column", "/columns"),
        *_unresolved(_gather(dumped, _EVIDENCE_KEYS) - known, "unresolved_evidence", "/evidence"),
        *_unresolved(parents - set(ctx.parents), "uncommitted_parent", "/parent_artifact_ids"),
        *_unresolved(raised - set(ctx.templates), "unknown_requirement_id", "/requirements",
                     _ASK, sorted(ctx.templates)),
        *_self_citations(result.payload if result else dumped)))


def _claim_issues(index: int, claim: ClaimV1, ctx: ValidationContext) -> Issues:
    blocking = [row for row in ctx.templates.values() if row.criticality == "blocking"
                and claim.predicate in (row.requirement_id, row.requirement_id.rsplit(".", 1)[-1])]
    if not blocking or claim.epistemic_status is not EpistemicStatus.EVIDENCED:
        return
    guessed = claim.support_class == "model_hypothesis"
    classes = {evidence_class(found) for found in claim.supporting_evidence_ids}
    for template in blocking:
        if guessed or not classes & set(template.acceptable_evidence_types):
            yield _issue("hypothesis_support_for_blocking_claim" if guessed
                         else "blocking_claim_unsupported", f"/claims/{index}", "wall3.evidence",
                         _EVID, True, (claim.claim_id, template.requirement_id))


def wall_evidence(payload: Any, result: Result, ctx: ValidationContext) -> ValidationReport:
    """Wall 3: a blocking fact stated as evidenced needs an acceptable class (§10.2)."""
    claims = (*(item for item in getattr(payload, "claims", ()) if isinstance(item, ClaimV1)),
              *(result.claims if result else ()))
    return ValidationReport(wall=3, issues=tuple(
        issue for index, claim in enumerate(claims) for issue in _claim_issues(index, claim, ctx)))


def _violates(rule: ValidationRuleV1, role: str, timing: str) -> bool:
    inside = timing in rule.params["timings"]
    return role in rule.params["roles"] and inside == (rule.params["relation"] == "forbidden")


def wall_temporal(payload: Any, result: Result, ctx: ValidationContext) -> ValidationReport:
    """Wall 4: role timing obeys the declarative `temporal` rows."""
    return ValidationReport(wall=4, issues=tuple(
        rule.issue(f"/claims/{index}/timing", (claim.concept_id,))
        for index, claim in _role_claims(payload, ctx) for rule in ctx.rules_of("temporal")
        if _violates(rule, claim.role.value, claim.timing.value)))


def _has_cycle(edges: Edges) -> bool:
    return has_cycle((e.source_concept_id, e.target_concept_id) for e in edges
                     if e.status in _LIVE)


def _role_graph_issues(index: int, claim: RoleClaimV1, frame: CausalFrameV1, edges: Edges,
                       ctx: ValidationContext) -> Issues:
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
            yield rule.issue(f"/claims/{index}/graph_edge_ids", (claim.concept_id,))


def wall_causal(payload: Any, result: Result, ctx: ValidationContext) -> ValidationReport:
    """Wall 5: the selected graph is acyclic and role claims match their edges (§12.2)."""
    context = payload if isinstance(payload, CausalContextV1) else ctx.causal_context
    if context is None:
        return ValidationReport(wall=5, issues=())
    # A selected alternative replaces the base edge set outright (GraphAlternativeV1 semantics).
    picked = [alt.edges for alt in context.alternatives
              if alt.alternative_id == ctx.selected_alternative_id]
    edges = picked[0] if picked else context.edges
    live = {edge.edge_id for alt in context.alternatives for edge in alt.edges}
    known = {edge.edge_id for edge in context.edges} | live
    issues = [_issue("graph_cycle", "/edges", "wall5.acyclic", _DROP)] if _has_cycle(edges) else []
    issues += [_issue("disputed_edge_without_alternative", f"/edges/{edge.edge_id}",
                      "wall5.alternatives", ("revise_field", "add_evidence"), False,
                      (edge.edge_id,))
               for edge in context.edges
               if edge.status is EpistemicStatus.DISPUTED and edge.edge_id not in live]
    for index, claim in _role_claims(payload, ctx):
        issues += [_issue("unresolved_graph_edge", f"/claims/{index}/graph_edge_ids",
                          "wall5.edge_ids", _FIX, False, (edge_id,))
                   for edge_id in claim.graph_edge_ids if edge_id not in known]
        issues += _role_graph_issues(index, claim, context.frame, edges, ctx)
    return ValidationReport(wall=5, issues=tuple(issues))


def _method_issues(pack: MethodPackV1, design: ExperimentDesignV1, claims: Claims,
                   ctx: ValidationContext) -> Issues:
    held = {claim.role.value for claim in claims if claim.status is not EpistemicStatus.UNKNOWN}
    deferred = ctx.triage.deferred if ctx.triage else ()
    adjusted = {claim.concept_id for claim in claims if claim.role.value in _ADJUSTMENT
                and design.method_id in claim.methods}
    for role in (name for name in pack.required_roles if name not in held):
        yield _issue("required_role_missing", f"/role_ledger/{role}", "wall6.roles", _ASK, False,
                     (role,))
        if deferred:
            yield _issue("deferred_column_blocks_role", f"/role_ledger/{role}", "wall6.roles",
                         _ASK, True, (role, *deferred))
    for claim in (row for row in claims if row.concept_id in adjusted
                  and row.role.value in set(pack.forbidden_adjustment_roles)):
        yield _issue("forbidden_adjustment_role", f"/adjustment_set/{claim.concept_id}",
                     "wall6.adjustment", _DROP, False, (claim.concept_id, claim.role.value))
    requested = set(design.required_prerepair_diagnostics)
    for rule in (row for row in ctx.rules_of("method_structural")
                 if row.params["requirement"] in pack.structural_requirements):
        absent = sorted(set(rule.params["roles"]) - held)
        if absent or not requested & set(rule.params["diagnostics"]):
            yield rule.issue(f"/structural/{rule.params['requirement']}", tuple(absent))


def wall_method(payload: Any, result: Result, ctx: ValidationContext) -> ValidationReport:
    """Wall 6: the selected pack's roles, structure, and context are satisfied (§13)."""
    design = payload if isinstance(payload, ExperimentDesignV1) else ctx.design
    pack = ctx.pack(design.method_id if design else None)
    if design is None or pack is None:
        return ValidationReport(wall=6, issues=())
    claims = ctx.role_ledger.claims if ctx.role_ledger else ()
    unresolved = [found for found in pack.required_context_requirement_ids
                  if ctx.resolved_requirements.get(found) not in RESOLVED_STATES
                  and getattr(ctx.templates.get(found), "missing_action", None)
                  != "retain_as_sensitivity"]
    return ValidationReport(wall=6, issues=(
        *_method_issues(pack, design, claims, ctx),
        *(_issue("unresolved_requirement", f"/required_context/{found}", "wall6.requirements",
                 ("add_evidence", "request_context"), True, (found,)) for found in unresolved)))


def _frame_issues(contract: RunnableFrameContractV1, pack: MethodPackV1,
                  ctx: ValidationContext) -> Issues:
    # Every `holder.field` a frame row may name, flattened to id tuples.
    ledger = ctx.role_ledger.claims if ctx.role_ledger else ()
    holders: dict[str, BaseModel | None] = {"contract": contract, "pack": pack, "design": ctx.design}
    values = {f"{name}.{key}": tuple(str(item) for item in _seq(held) if item is not None)
              for name, holder in holders.items() if holder is not None
              for key, held in dict(holder).items()} | {
        "manifest.inventory_columns": tuple(row.column_name
                                            for row in ctx.manifest.structural_inventory),
        "parents.content_hashes": tuple(ctx.parents.values()),
        "ledger.imputation_forbidden_columns": tuple(
            column for claim in ledger for column in claim.column_refs
            if claim.role.value in pack.imputation_forbidden_roles)}
    for rule in ctx.rules_of("frame_subset"):
        left = set(values.get(rule.params["left"], ()))
        right = set(values.get(rule.params.get("right", ""), ()))
        if rule.params["op"] == "nonempty":
            bad = set() if any(item.strip() for item in left) else {rule.params["left"]}
        else:
            bad = (left & right) if rule.params["op"] == "disjoint" else (right - left)
        if bad:
            yield rule.issue(rule.params["path"], tuple(sorted(bad)))


def wall_frame(payload: Any, result: Result, ctx: ValidationContext) -> ValidationReport:
    """Wall 7: the runnable frame carries what the pack and the design demand (§18)."""
    pack = ctx.pack()
    if not isinstance(payload, RunnableFrameContractV1) or pack is None:
        return ValidationReport(wall=7, issues=())
    return ValidationReport(wall=7, issues=tuple(_frame_issues(payload, pack, ctx)))


type Wall = Callable[[Any, Result, ValidationContext], ValidationReport]

_WALL_ORDER: Final[tuple[Wall, ...]] = (wall_references, wall_evidence, wall_temporal,
                                        wall_causal, wall_method, wall_frame)


def validate_result(wall: int, task_kind: str, model_cls: type[BaseModel],
                    result: AgentTaskResultV1, ctx: ValidationContext) -> ValidationReport:
    """Run walls 1..`wall` in §16.3 order; the first failing wall stops the run (§16.4)."""
    report = wall_shape(model_cls, result)
    highest = min(wall, _TASK_MAX_WALL.get(task_kind, wall))
    if not report.passed or highest < 2:
        return report
    payload = _parse(model_cls, result.payload)
    for run in _WALL_ORDER[:highest - 1]:
        report = run(payload, result, ctx)
        if not report.passed:
            return report
    return ValidationReport(wall=highest, issues=())


