"""The six §24.3 preparation walls over declarative rows (PRD-003 §14 → §24.3, §20)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from causal.preparation.contracts import (
    RETAINING_DISPOSITIONS,
    DiagnosticStatus,
    PreparationContextManifestV1,
    PreparationDiagnosticV1,
    RowDisposition,
    StabilizationRecordV1,
)
from causal.preparation.diagnostics import PreparationDiagnosticRowV1
from causal.preparation.impact import MethodStructureResultV1, StructureVerdict
from causal.preparation.operations import OperationRegistry
from causal.preparation.packs import PreparationPackV1
from causal.preparation.plancompile import TABLE_WIDE_SCOPE, FrameGapV1
from causal.preparation.plans import ItemPhase, PlanItemV1, PreparationPlanV1, has_cycle
from causal.shared.contracts import ArtifactRef
from causal.shared.receipts import (
    ExecutionReceiptV1,
    OutputArtifactLike,
    PostconditionLike,
    tri_agreement,
)
from causal.shared.registry import RegistryError
from causal.shared.validation import (
    ValidationIssueV1,
    ValidationReport,
    ValidationRuleV1,
    load_rules,
)

__all__ = [
    "MAX_WALL", "REGISTRY_VERSION", "RULE_KINDS", "ReceiptCheck", "ValidationIssueV1",
    "ValidationReport", "ValidationRuleV1", "WallContext", "load_validation_rules", "validate",
    "wall",
]

REGISTRY_VERSION: Final = "preparation-validators.v1"
# One rule kind per §24.3 wall; a row's `check` names the predicate inside its wall.
RULE_KINDS: Final = ("entry", "rows", "freeze", "plan", "execution", "final")
MAX_WALL: Final = len(RULE_KINDS)
UNKNOWN_RULE_CHECK, UNKNOWN_WALL = "unknown_rule_check", "unknown_wall"
# `preparation_common` diagnostics dispatch here; the two phases below fill an existing column.
_COMMON, _FILLING = "preparation_common", (ItemPhase.REPAIR, ItemPhase.IMPUTATION)


# One executed plan item as the §17.6 gate sees it: receipt, output, postcondition.
@dataclass(frozen=True)
class ReceiptCheck:
    receipt: ExecutionReceiptV1
    output: OutputArtifactLike
    postcondition: PostconditionLike


# Everything the walls may read; a wall never loads or recomputes anything itself.
@dataclass(frozen=True)
class WallContext:
    rules: tuple[ValidationRuleV1, ...] = ()
    handoff_accepted: bool = False
    entry_codes: tuple[str, ...] = ()
    manifest: PreparationContextManifestV1 | None = None
    record: StabilizationRecordV1 | None = None
    evaluated_rule_ids: tuple[str, ...] = ()
    structure: MethodStructureResultV1 | None = None
    row_set_hash: str | None = None
    plan: PreparationPlanV1 | None = None
    operations: OperationRegistry | None = None
    pack: PreparationPackV1 | None = None
    gaps: tuple[FrameGapV1, ...] = ()
    receipts: tuple[ReceiptCheck, ...] = ()
    source_ref: ArtifactRef | None = None
    diagnostics: tuple[PreparationDiagnosticV1, ...] = ()
    required_diagnostic_ids: tuple[str, ...] = ()
    diagnostic_registry: Mapping[str, PreparationDiagnosticRowV1] = field(default_factory=dict)
    implemented_diagnostic_ids: frozenset[str] = frozenset()
    approved_handling: frozenset[str] = frozenset()
    prepared_schema: Mapping[str, str] = field(default_factory=dict)
    required_schema: Mapping[str, str] = field(default_factory=dict)
    readability: Mapping[str, bool] = field(default_factory=dict)

    def rules_of(self, kind: str) -> tuple[ValidationRuleV1, ...]:
        return tuple(rule for rule in self.rules if rule.kind == kind)

    def items(self) -> tuple[PlanItemV1, ...]:
        return self.plan.items if self.plan is not None else ()


Check = Callable[[WallContext, ValidationRuleV1], tuple[str, ...]]


def load_validation_rules(path: Path) -> tuple[ValidationRuleV1, ...]:
    """The six-wall preparation rows; an unknown kind or wall fails closed (T-002 machinery)."""
    return load_rules(path, registry_version=REGISTRY_VERSION, kinds=RULE_KINDS, max_wall=MAX_WALL)


def _handoff_accepted(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    return () if ctx.handoff_accepted else ("handoff_not_accepted",)


def _entry_codes_clear(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    return tuple(sorted(ctx.entry_codes))


# Every source row carries exactly one terminal disposition (§6.2, walls 2–4).
def _rows_dispositioned(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    record = ctx.record
    if record is None:
        return ()
    total, indexed = (sum(row.row_count for row in record.dispositions.counts),
                      record.source_row_index.row_count)
    codes = [] if total == indexed else [f"dispositioned_{total}_of_{indexed}"]
    unresolved = record.dispositions.total(RowDisposition.UNRESOLVED_CONFLICT)
    return tuple(codes + (["unresolved_conflict_remains"] if unresolved else []))


def _approved_rules_only(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    manifest = ctx.manifest
    if manifest is None:
        return ()
    approved = set(manifest.eligibility_rule_ids) | set(manifest.unusable_row_rule_ids)
    return tuple(sorted(set(ctx.evaluated_rule_ids) - approved))


# Retained rows equal the freeze and every required impact dimension was reported (§9.5).
def _counts_reconcile(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    record, manifest = ctx.record, ctx.manifest
    if record is None:
        return ()
    retained = record.dispositions.total(*RETAINING_DISPOSITIONS)
    codes = [] if retained == record.freeze.retained_row_count else ["retained_count_mismatch"]
    required = set(manifest.deletion_impact_dimensions) if manifest is not None else set()
    return tuple(codes + sorted(required - {row.dimension_id for row in record.impact}))


def _support_gates(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    structure = ctx.structure
    if structure is None or structure.verdict is StructureVerdict.RUNNABLE:
        return ()
    return tuple(structure.codes)


def _row_set_hash_stable(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    frozen = ctx.record.freeze.row_set_hash if ctx.record is not None else None
    return () if frozen is None or ctx.row_set_hash in (None, frozen) else ("row_set_hash_drift",)


def _operations_registered(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    known = ctx.operations.rows if ctx.operations is not None else None
    return () if known is None else tuple(sorted(
        {item.operation_id for item in ctx.items() if item.operation_id not in known}))


def _operations_permitted(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    allowed = ctx.manifest.permitted_operation_ids if ctx.manifest is not None else None
    return () if allowed is None else tuple(sorted(
        {item.operation_id for item in ctx.items() if item.operation_id not in allowed}))


def _items_ordered(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    return ("plan_not_a_dag",) if has_cycle(ctx.items()) else ()


def _gaps_covered(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    items = ctx.items()
    touched = {column for item in items for column in item.target_columns}
    touched |= {item.output_column for item in items if item.output_column is not None}
    return tuple(sorted({gap.column for gap in ctx.gaps
                         if gap.column != TABLE_WIDE_SCOPE and gap.column not in touched}))


def _protected_untouched(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    protected = set(ctx.manifest.protected_columns) if ctx.manifest is not None else set()
    return tuple(sorted({column for item in ctx.items() if item.phase in _FILLING
                         for column in item.target_columns if column in protected}))


def _fit_scopes_permitted(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    manifest, pack = ctx.manifest, ctx.pack
    if manifest is None or pack is None:
        return ()
    scopes = {(target.role, target.fit_scope) for target in pack.permitted_imputation_targets}
    return tuple(sorted({
        column for item in ctx.items() if item.phase is ItemPhase.IMPUTATION
        for column in item.target_columns
        if (manifest.column_roles.get(column), item.fit_scope) not in scopes}))


def _tri_agreement_clean(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    return tuple(f"{check.receipt.plan_item_id}:{code}" for check in ctx.receipts
                 for code in tri_agreement(check.receipt, check.output, check.postcondition))


# Every receipt opens on the previous receipt's output; the first opens on the source.
def _hash_chain_unbroken(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    expected, broken = ctx.source_ref, []
    for check in ctx.receipts:
        if expected is not None and check.receipt.input_ref != expected:
            broken.append(check.receipt.plan_item_id)
        expected = check.receipt.output_ref
    return tuple(broken)


def _row_invariance(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    frozen = ctx.row_set_hash
    return tuple(
        check.receipt.plan_item_id for check in ctx.receipts
        if check.receipt.row_set_hash_before != check.receipt.row_set_hash_after
        or (frozen is not None and check.receipt.row_set_hash_after != frozen))


# A required diagnostic passed, or its terminal status carries approved handling. An
# unimplemented method-pack diagnostic must be `not_computable`, never a silent pass (§15).
def _handled(found: PreparationDiagnosticV1, dispatchable: bool, approved: frozenset[str]) -> bool:
    if not dispatchable:
        return found.status is DiagnosticStatus.NOT_COMPUTABLE and found.status.value in approved
    return found.status is DiagnosticStatus.PASS or found.status.value in approved


# Every required diagnostic is terminal with approved handling (§15, wall 12).
def _diagnostics_terminal(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    reported = {found.diagnostic_id: found for found in ctx.diagnostics}
    offending: list[str] = []
    for diagnostic_id in sorted(ctx.required_diagnostic_ids):
        found, row = reported.get(diagnostic_id), ctx.diagnostic_registry.get(diagnostic_id)
        dispatchable = (row is None or row.implementation == _COMMON
                        or diagnostic_id in ctx.implemented_diagnostic_ids)
        if found is None or not _handled(found, dispatchable, ctx.approved_handling):
            offending.append(diagnostic_id)
    return tuple(offending)


def _contract_satisfied(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    return tuple(sorted(column for column, dtype in ctx.required_schema.items()
                        if ctx.prepared_schema.get(column) != dtype))


def _handoff_readable(ctx: WallContext, rule: ValidationRuleV1) -> tuple[str, ...]:
    return tuple(str(name) for name in rule.params.get("conditions") or ()
                 if not ctx.readability.get(str(name)))


# One (kind, checks) row per §24.3 wall, in the order the walls must pass.
_WALLS: Final[tuple[tuple[str, dict[str, Check]], ...]] = (
    ("entry", {"handoff_accepted": _handoff_accepted, "entry_codes_clear": _entry_codes_clear}),
    ("rows", {"rows_dispositioned": _rows_dispositioned,
              "approved_rules_only": _approved_rules_only}),
    ("freeze", {"counts_reconcile": _counts_reconcile, "support_gates": _support_gates,
                "row_set_hash_stable": _row_set_hash_stable}),
    ("plan", {"operations_registered": _operations_registered,
              "operations_permitted": _operations_permitted, "items_ordered": _items_ordered,
              "gaps_covered": _gaps_covered, "protected_untouched": _protected_untouched,
              "fit_scopes_permitted": _fit_scopes_permitted}),
    ("execution", {"tri_agreement_clean": _tri_agreement_clean,
                   "hash_chain_unbroken": _hash_chain_unbroken,
                   "row_invariance": _row_invariance}),
    ("final", {"diagnostics_terminal": _diagnostics_terminal,
               "contract_satisfied": _contract_satisfied,
               "handoff_readable": _handoff_readable}),
)


def wall(number: int, ctx: WallContext) -> ValidationReport:
    """Run one §24.3 wall over the rows registered for it; an unknown check fails closed."""
    if not 1 <= number <= MAX_WALL:
        raise RegistryError(f"no preparation wall {number}", UNKNOWN_WALL)
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
    """Walls 1..`highest` in order; the first failing wall stops the run and is never waived."""
    report = ValidationReport(wall=1, issues=())
    for number in range(1, highest + 1):
        report = wall(number, ctx)
        if not report.passed:
            return report
    return report
