"""Deterministic checks for numerical inputs, execution, uncertainty and evidence."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from causal.analysis.integration import contracts as ec
from causal.analysis.integration.packs import EstimationPackV1
from causal.shared import validation
from causal.shared.contracts import ArtifactRef
from causal.shared.registry import RegistryError
from causal.shared.validation import ValidationIssueV1, ValidationReport, ValidationRuleV1

REGISTRY_VERSION: Final = "estimation-validators.v1"
# Preserve numeric rule numbers used by saved validation reports.
_NUMERICAL_WALLS: Final = (1, 2, 4, 5, 6, 7, 8, 9, 10, 11)
RULE_KINDS: Final = ("handoff", "plan", "input", "contribution", "preprocessing",
                    "estimator", "uncertainty", "diagnostic", "sensitivity", "lineage")
MAX_WALL: Final = 11
UNKNOWN_RULE_CHECK, UNKNOWN_WALL = "unknown_rule_check", "unknown_wall"


@dataclass(frozen=True)
class WallContext:
    rules: tuple[ValidationRuleV1, ...] = ()
    handoff_accepted: bool = False
    entry_codes: tuple[str, ...] = ()
    manifest: ec.EstimationContextManifestV1 | None = None
    plan: ec.EstimationPlanV1 | None = None
    pack: EstimationPackV1 | None = None
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
    def rules_of(self, kind: str) -> tuple[ValidationRuleV1, ...]:
        return tuple(rule for rule in self.rules if rule.kind == kind)


Check = Callable[[WallContext, ValidationRuleV1], tuple[str, ...]]


def load_validation_rules(path: Path) -> tuple[ValidationRuleV1, ...]:
    # Old registry rows remain readable; reporting/capacity checks no longer execute here.
    legacy = (*RULE_KINDS, "capacity", "ceiling", "claim", "figure", "bundle")
    rows = validation.load_rules(path, registry_version=REGISTRY_VERSION, kinds=legacy,
                                 max_wall=15)
    return tuple(row for row in rows if row.wall in _NUMERICAL_WALLS)


def _handoff_accepted(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    return () if ctx.handoff_accepted else ("handoff_not_accepted",)


def _entry_codes_clear(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    return tuple(sorted(ctx.entry_codes))


# The manifest's four approved parents and its frozen row set survive unchanged into the plan.
def _handoff_hashes_match(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    manifest, plan = ctx.manifest, ctx.plan
    if manifest is None or plan is None:
        return ()
    refs = (manifest.prepared_bundle, manifest.compiled_design, manifest.capacity_report)
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
    return tuple(sorted(unknown | ({plan.primary_mask_rule_id} - set(pack.allowed_mask_rule_ids))))


def _input_schema_satisfied(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    schema = {} if ctx.pack is None else dict(ctx.pack.estimator_input_schema)
    return tuple(sorted(role for role, dtype in ctx.estimator_input_types.items()
                        if dtype not in schema.get(role.partition("__")[0], ())))


# Every mask names a permitted rule, hangs off the frozen row set, and accounts for every row.
def _masks_registered(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    allowed = () if ctx.pack is None else ctx.pack.allowed_mask_rule_ids
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
    corrections = {plan.finite_sample_correction}
    # Preserve readability of the historical RCT alias while admitting the corrected label
    # for that exact same CRV1/HC2 calculation. No new plan may claim the obsolete CR2 label.
    if (plan.method_id == "randomized_experiment" and plan.finite_sample_correction == "hc2_cr2"
            and plan.estimator_parameters.get("cluster_covariance") == "cluster_robust_cr2"):
        corrections.add("hc2_cr1")
    return tuple(item.contrast_id for item in result.primary_items if (
        plan.confidence_level, plan.uncertainty_method) != (
        item.confidence_level, item.uncertainty_method)
        or item.finite_sample_correction not in corrections)


def _coverage(expected: set[str], actual: tuple[str, ...]) -> tuple[str, ...]:
    seen = set(actual)
    return tuple(sorted((expected ^ seen) | {name for name in actual if actual.count(name) > 1}))


def _diagnostics_terminal(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    # An explicit failure is evidence. Claim consequences belong to post-analysis.
    return () if ctx.plan is None else _coverage(
        set(ctx.plan.required_diagnostics), tuple(row.diagnostic_id for row in ctx.diagnostics))


def _sensitivities_terminal(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    return () if ctx.plan is None else _coverage(
        set(ctx.plan.required_sensitivity_ids), tuple(row.branch_id for row in ctx.sensitivities))


def _results_resolve(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    result = ctx.primary_result
    if result is None:
        return ()
    known = {ref.artifact_id for ref in ctx.mask_refs}
    codes = tuple(item.contrast_id for item in result.primary_items
                  if item.contribution_mask.artifact_id not in known)
    codes += () if result.plan in result.parents else ("plan_not_a_parent",)
    return codes + (() if ctx.numerical_environment is not None else ("environment_missing",))


# One (kind, checks) row per §18 wall, in the order the walls must pass.
_WALLS: Final[tuple[tuple[str, dict[str, Check]], ...]] = (
    ("handoff", {"handoff_accepted": _handoff_accepted, "entry_codes_clear": _entry_codes_clear,
                 "handoff_hashes_match": _handoff_hashes_match}),
    ("plan", {"plan_committed": _plan_committed, "plan_registered": _plan_registered}),
    ("input", {"input_schema_satisfied": _input_schema_satisfied,
               "structure_satisfied": lambda ctx, _: tuple(sorted(ctx.structure_codes))}),
    ("contribution", {"masks_registered": _masks_registered}),
    ("preprocessing", {"folds_train_only": _folds_train_only}),
    ("estimator", {"primary_result_complete": _primary_result_complete,
                   "contrasts_ordered": _contrasts_ordered}),
    ("uncertainty", {"uncertainty_complete": _uncertainty_complete}),
    ("diagnostic", {"diagnostics_terminal": _diagnostics_terminal}),
    ("sensitivity", {"sensitivities_terminal": _sensitivities_terminal}),
    ("lineage", {"results_resolve": _results_resolve}),

)


def wall(number: int, ctx: WallContext) -> ValidationReport:
    # Run one §18 wall over the rows registered for it; an unknown check or wall fails closed.
    if number not in _NUMERICAL_WALLS:
        raise RegistryError(f"no estimation wall {number}", UNKNOWN_WALL)
    kind, checks = _WALLS[_NUMERICAL_WALLS.index(number)]
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
    if not 1 <= highest <= MAX_WALL:
        raise RegistryError(f"no estimation wall {highest}", UNKNOWN_WALL)
    for number in (n for n in _NUMERICAL_WALLS if n <= highest):
        report = wall(number, ctx)
        if not report.passed:
            return report
    return report
