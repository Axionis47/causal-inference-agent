# The fifteen §18 estimation walls over declarative rows (PRD-004 §18, §26.2). One rule kind per
# wall, dispatched through a predicate table; no later wall waives an earlier failure. A wall never
# loads, recomputes, or repairs anything — it reads the context the coordinator handed it.

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from causal.estimation import contracts as ec
from causal.estimation.packs import EstimationPackV1
from causal.estimation.plancompile import DesignConflictDraftV1
from causal.shared import validation
from causal.shared.contracts import ArtifactRef
from causal.shared.registry import RegistryError
from causal.shared.validation import ValidationIssueV1, ValidationReport, ValidationRuleV1

REGISTRY_VERSION: Final = "estimation-validators.v1"
# One rule kind per §18 wall, in the order the walls must pass.
RULE_KINDS: Final = ("handoff", "plan", "capacity", "input", "contribution", "preprocessing",
                     "estimator", "uncertainty", "diagnostic", "sensitivity", "lineage",
                     "ceiling", "claim", "figure", "bundle")
MAX_WALL: Final = len(RULE_KINDS)
UNKNOWN_RULE_CHECK, UNKNOWN_WALL = "unknown_rule_check", "unknown_wall"
CLAIM_VALIDATOR_MISSING: Final = "claim_validator_missing"
ClaimValidator = Callable[["WallContext"], tuple[str, ...]]


# T-025 supplies the real §16.3 claim validator; until then wall 13 refuses, never passes.
def _no_claim_validator(ctx: WallContext) -> tuple[str, ...]:
    return (CLAIM_VALIDATOR_MISSING,)


# Everything the fifteen walls may read.
@dataclass(frozen=True)
class WallContext:
    rules: tuple[ValidationRuleV1, ...] = ()
    handoff_accepted: bool = False
    entry_codes: tuple[str, ...] = ()
    manifest: ec.EstimationContextManifestV1 | None = None
    plan: ec.EstimationPlanV1 | None = None
    pack: EstimationPackV1 | None = None
    capacity_conflict: DesignConflictDraftV1 | None = None
    # Estimator input role to the prepared frame's dtype, and the §4 method-structure codes.
    estimator_input_types: Mapping[str, str] = field(default_factory=dict)
    structure_codes: tuple[str, ...] = ()
    masks: tuple[ec.AnalysisContributionMaskV1, ...] = ()
    mask_refs: tuple[ArtifactRef, ...] = ()
    frozen_row_count: int | None = None
    assignments: tuple[ec.CrossFitAssignmentV1, ...] = ()
    # Fold id to the rows the fold's fit actually read, by fold role (wall 6 leakage evidence).
    fold_fit_counts: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    primary_result: ec.PrimaryAnalysisResultV1 | None = None
    numerical_environment: ArtifactRef | None = None
    diagnostics: tuple[ec.DiagnosticResultV1, ...] = ()
    sensitivities: tuple[ec.SensitivityResultV1, ...] = ()
    # Severities whose non-computed terminal status the method pack approves (§14.2).
    approved_handling: frozenset[str] = frozenset()
    ceiling: ec.JudgmentCeilingV1 | None = None
    claim_validator: ClaimValidator = _no_claim_validator
    figures: tuple[ec.FigureDataArtifactV1, ...] = ()
    bundle: ec.EstimationBundleV1 | None = None
    trace_codes: tuple[str, ...] = ()

    def rules_of(self, kind: str) -> tuple[ValidationRuleV1, ...]:
        return tuple(rule for rule in self.rules if rule.kind == kind)


Check = Callable[[WallContext, ValidationRuleV1], tuple[str, ...]]


def load_validation_rules(path: Path) -> tuple[ValidationRuleV1, ...]:
    # The fifteen-wall estimation rows; an unknown kind or wall fails closed (T-002 machinery).
    return validation.load_rules(path, registry_version=REGISTRY_VERSION, kinds=RULE_KINDS,
                                 max_wall=MAX_WALL)


def _handoff_accepted(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    return () if ctx.handoff_accepted else ("handoff_not_accepted",)


def _entry_codes_clear(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    return tuple(sorted(ctx.entry_codes))


# The manifest's four approved parents and its frozen row set survive unchanged into the plan.
def _handoff_hashes_match(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    manifest, plan = ctx.manifest, ctx.plan
    if manifest is None or plan is None:
        return ()
    refs = (manifest.prepared_bundle, manifest.experiment_design,
            manifest.runnable_frame_contract, manifest.capacity_check)
    drift = () if manifest.row_set_hash == plan.row_set_hash else ("row_set_hash_drift",)
    return drift + tuple(ref.artifact_id for ref in refs if ref not in plan.parents)


def _plan_committed(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    plan, pack = ctx.plan, ctx.pack
    if plan is None:
        return ("plan_missing",)
    want = {} if pack is None else {
        "method_id": pack.method_id, "estimator_id": pack.estimator_id,
        "uncertainty_method": pack.uncertainty_method, "confidence_level": pack.confidence_level}
    return tuple(sorted(name for name, value in want.items() if getattr(plan, name) != value))


def _plan_registered(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    plan, pack = ctx.plan, ctx.pack
    if plan is None or pack is None:
        return ()
    unknown = set(plan.required_diagnostics) - set(pack.severities())
    unknown |= set(plan.required_sensitivity_ids) - {r.branch_id for r in pack.sensitivity_branches}
    unknown |= set(plan.figure_builder_ids) - set(pack.figure_builder_ids)
    return tuple(sorted(unknown | ({plan.primary_mask_rule_id} - set(pack.allowed_mask_rule_ids))))


def _capacity_recheck_clear(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    found = ctx.capacity_conflict
    return () if found is None else (found.conflict_code, found.failed_rule_id)


def _input_schema_satisfied(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    schema = {} if ctx.pack is None else dict(ctx.pack.estimator_input_schema)
    return tuple(sorted(role for role, dtype in ctx.estimator_input_types.items()
                        if role not in schema or dtype not in schema[role]))


def _structure_satisfied(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    return tuple(sorted(ctx.structure_codes))


# Every mask names a permitted rule, hangs off the frozen row set, and accounts for every row.
def _masks_registered(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    allowed = () if ctx.manifest is None else ctx.manifest.contribution_mask_rule_ids
    frozen, total = (None if ctx.plan is None else ctx.plan.row_set_hash), ctx.frozen_row_count
    return tuple(sorted(
        mask.calculation_id for mask in ctx.masks if mask.mask_rule_id not in allowed
        or (frozen is not None and mask.parent_row_set_hash != frozen)
        or (total is not None and mask.included_counts.get("row", 0)
            + mask.noncontributing_counts.get("row", 0) != total)))


# Wall 6: every fold fit reads training rows only, measured against the assignment's own counts.
def _folds_train_only(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    codes: list[str] = []
    for assignment in ctx.assignments:
        for fold_id, counts in sorted(assignment.counts_by_fold.items()):
            fitted = ctx.fold_fit_counts.get(fold_id)
            if fitted is None:
                codes.append(f"{fold_id}:fit_receipt_missing")
            elif fitted.get("validation", 0):
                codes.append(f"{fold_id}:validation_rows_fitted")
            elif fitted.get("train", 0) != counts.get("train", 0):
                codes.append(f"{fold_id}:train_count_mismatch")
    return tuple(codes)


# Wall 7: the primary items are atomically the plan's contrast set, in the plan's order.
def _primary_result_complete(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    result, plan = ctx.primary_result, ctx.plan
    if plan is None or result is None:
        return ("primary_result_missing",) if plan is not None else ()
    codes = () if result.complete else ("primary_result_incomplete",)
    return codes + tuple(sorted(
        set(plan.contrast_ids) ^ {item.contrast_id for item in result.primary_items}))


def _contrasts_ordered(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    result, plan = ctx.primary_result, ctx.plan
    if result is None or plan is None:
        return ()
    return () if result.contrast_order == plan.contrast_ids else ("contrast_order_mismatch",)


def _uncertainty_complete(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    result, plan = ctx.primary_result, ctx.plan
    if result is None or plan is None:
        return ()
    want = (plan.confidence_level, plan.uncertainty_method, plan.finite_sample_correction)
    return tuple(item.contrast_id for item in result.primary_items if want != (
        item.confidence_level, item.uncertainty_method, item.finite_sample_correction))


def _diagnostics_terminal(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    plan = ctx.plan
    if plan is None:
        return ()
    reported = {row.diagnostic_id: row.execution_status for row in ctx.diagnostics}
    return tuple(sorted(
        name for name, severity in plan.required_diagnostics.items()
        if name not in reported or (reported[name] != "computed"
                                    and severity not in ctx.approved_handling)))


def _sensitivities_terminal(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    plan = ctx.plan
    if plan is None:
        return ()
    reported = {row.branch_id for row in ctx.sensitivities}
    return tuple(sorted(set(plan.required_sensitivity_ids) - reported))


def _results_resolve(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    result = ctx.primary_result
    if result is None:
        return ()
    known = {ref.artifact_id for ref in ctx.mask_refs}
    codes = tuple(item.contrast_id for item in result.primary_items
                  if item.contribution_mask.artifact_id not in known)
    codes += () if result.plan in result.parents else ("plan_not_a_parent",)
    return codes + (() if ctx.numerical_environment is not None else ("environment_missing",))


def _ceiling_complete(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    ceiling, plan = ctx.ceiling, ctx.plan
    if plan is None or ceiling is None:
        return ("judgment_ceiling_missing",) if plan is not None else ()
    return tuple(sorted(set(plan.contrast_ids) - {item.contrast_id for item in ceiling.items}))


def _claims_within_ceiling(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    return ctx.claim_validator(ctx)


# Every required builder produced a dataset, and no dataset discloses above the overall ceiling.
def _figures_resolve(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    plan, ceiling = ctx.plan, ctx.ceiling
    if plan is None:
        return ()
    over = () if ceiling is None else tuple(
        row.visual_evidence_id for row in ctx.figures if row.disclosure_status
        != ec.most_restrictive((row.disclosure_status, ceiling.overall_ceiling)))
    return tuple(sorted(set(plan.figure_builder_ids) - {r.builder_id for r in ctx.figures})) + over


def _bundle_complete(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    bundle, manifest = ctx.bundle, ctx.manifest
    if bundle is None or manifest is None:
        return ("estimation_bundle_missing",) if bundle is None else ()
    pairs = ((bundle.row_set_hash, manifest.row_set_hash),
             (bundle.capacity_check, manifest.capacity_check),
             (bundle.prepared_bundle, manifest.prepared_bundle),
             (bundle.experiment_design, manifest.experiment_design))
    return (("bundle_binding_mismatch",) if any(a != b for a, b in pairs)
            else ()) + tuple(sorted(ctx.trace_codes))


# One (kind, checks) row per §18 wall, in the order the walls must pass.
_WALLS: Final[tuple[tuple[str, dict[str, Check]], ...]] = (
    ("handoff", {"handoff_accepted": _handoff_accepted, "entry_codes_clear": _entry_codes_clear,
                 "handoff_hashes_match": _handoff_hashes_match}),
    ("plan", {"plan_committed": _plan_committed, "plan_registered": _plan_registered}),
    ("capacity", {"capacity_recheck_clear": _capacity_recheck_clear}),
    ("input", {"input_schema_satisfied": _input_schema_satisfied,
               "structure_satisfied": _structure_satisfied}),
    ("contribution", {"masks_registered": _masks_registered}),
    ("preprocessing", {"folds_train_only": _folds_train_only}),
    ("estimator", {"primary_result_complete": _primary_result_complete,
                   "contrasts_ordered": _contrasts_ordered}),
    ("uncertainty", {"uncertainty_complete": _uncertainty_complete}),
    ("diagnostic", {"diagnostics_terminal": _diagnostics_terminal}),
    ("sensitivity", {"sensitivities_terminal": _sensitivities_terminal}),
    ("lineage", {"results_resolve": _results_resolve}),
    ("ceiling", {"ceiling_complete": _ceiling_complete}),
    ("claim", {"claims_within_ceiling": _claims_within_ceiling}),
    ("figure", {"figures_resolve": _figures_resolve}),
    ("bundle", {"bundle_complete": _bundle_complete}),
)


def wall(number: int, ctx: WallContext) -> ValidationReport:
    # Run one §18 wall over the rows registered for it; an unknown check or wall fails closed.
    if not 1 <= number <= MAX_WALL:
        raise RegistryError(f"no estimation wall {number}", UNKNOWN_WALL)
    kind, checks = _WALLS[number - 1]
    issues: list[ValidationIssueV1] = []
    for rule in ctx.rules_of(kind):
        name = str(rule.params.get("check", ""))
        check = checks.get(name)
        if check is None:
            raise RegistryError(f"{rule.rule_id} names unknown check {name!r}", UNKNOWN_RULE_CHECK)
        if found := check(ctx, rule):
            issues.append(rule.issue(str(rule.params.get("path", "/")), found))
    return ValidationReport(wall=number, issues=tuple(issues))


def validate(highest: int, ctx: WallContext) -> ValidationReport:
    # Walls 1..`highest` in order; the first failing wall stops the run and is never waived.
    report = ValidationReport(wall=1, issues=())
    for number in range(1, highest + 1):
        report = wall(number, ctx)
        if not report.passed:
            return report
    return report
