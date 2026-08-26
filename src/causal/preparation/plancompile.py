"""Contract gaps, §7.4 grouping, and the §7.5 fan-in into one plan (PRD-003 §7.4, §7.5, §10.1)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Final, cast

from causal.preparation import contracts as pc
from causal.preparation import plans as pp
from causal.preparation.contracts import PreparationContextManifestV1, PreparationError, _Row
from causal.preparation.operations import OperationRegistry, by_role
from causal.preparation.plans import FitScope, GroupKind, ItemPhase, PlanItemV1, PlanPhase
from causal.shared.contracts import ArtifactRef, Identity
from causal.shared.registry import RegistryError

# The §7.4 concurrency ceiling; raising it never changes membership or fan-in order.
MAX_CONCURRENT_GROUPS: Final = 8
# The pseudo-column a table-wide gap is filed under; a real column never collides with it.
TABLE_WIDE_SCOPE, TABLE_WIDE_GROUP = "__table__", "g:table_wide"
MISSING_PREPARED_COLUMN, TYPE_MISMATCH = "missing_prepared_column", "type_mismatch"
SENTINEL_EVIDENCE, DERIVATION_REQUIRED = "sentinel_evidence_present", "required_derivation_missing"
IMPUTATION_REQUIRED, GROUPS_NOT_A_DAG = "imputation_target_missing", "groups_not_a_dag"
FAN_IN_REJECTED, TASK_NOT_PROPOSED = "fan_in_rejected", "task_not_proposed"
DUPLICATE_PLAN_ITEM, UNKNOWN_DEPENDENCY = "duplicate_plan_item", "unknown_dependency"
PLAN_NOT_A_DAG, UNREGISTERED_OPERATION = "plan_not_a_dag", "unregistered_operation"
OPERATION_NOT_PERMITTED, DUPLICATE_MAPPING = "operation_not_permitted", "duplicate_mapping"
CONFLICTING_OPERATION_ON_COLUMN = "conflicting_operation_on_column"
# §25.1: the gap no registered V1 operation resolves, returned to PRD-002 as a design conflict.
NO_RESOLUTION: Final = "no_registered_resolution"
_REPAIR_POSTCONDITIONS: Final = ("missingness_before_after", "row_set_invariance",
                                 "changed_cells_by_operation_column")
INCONSISTENT_MAPPING, PROTECTED_ROLE_TARGET = "inconsistent_mapping", "protected_role_target"
ILLEGAL_FIT_SCOPE, ROW_MEMBERSHIP_CHANGE = "illegal_fit_scope", "row_membership_change"
GAP_NOT_COVERED = "gap_not_covered"
# The two phases that write into an existing column's value space (§10.3).
_FILLING: Final = (ItemPhase.REPAIR, ItemPhase.IMPUTATION)
_NO_COUNTS: Final[Mapping[str, int]] = {}


class PlanCompileError(PreparationError):
    """The compiler refused. `code` is stable; `detail_codes` carries every rejection."""


# One deterministic mismatch between the current frame and the runnable-frame contract.
class FrameGapV1(_Row):
    gap_code: Identity
    column: Identity
    detail: str = ""


# The runnable-frame facts the deterministic gap computation reads (§7.4, §10.1).
@dataclass(frozen=True)
class ContractSurfaceV1:
    required_dtypes: Mapping[str, str] = field(default_factory=dict)
    sentinel_columns: Mapping[str, str] = field(default_factory=dict)
    required_derivations: Mapping[str, str] = field(default_factory=dict)
    derivation_sources: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    imputation_targets: Mapping[str, str] = field(default_factory=dict)
    imputation_fit_scopes: Mapping[str, FitScope] = field(default_factory=dict)
    coupled_columns: tuple[tuple[str, ...], ...] = ()
    recipe_groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    table_wide_codes: tuple[str, ...] = ()


# Every registered mismatch, in stable order; a satisfied column yields no gap (EV-P3-001).
def contract_gaps(schema: Mapping[str, str], surface: ContractSurfaceV1,
                  missing_counts: Mapping[str, int] = _NO_COUNTS) -> tuple[FrameGapV1, ...]:
    gaps = [FrameGapV1(gap_code=code, column=TABLE_WIDE_SCOPE)
            for code in sorted(surface.table_wide_codes)]
    for column, dtype in sorted(surface.required_dtypes.items()):
        # An absent derived column is reported once, by the derivation family below.
        if column not in schema and column not in surface.required_derivations:
            gaps.append(FrameGapV1(gap_code=MISSING_PREPARED_COLUMN, column=column, detail=dtype))
        elif schema.get(column, dtype) != dtype:
            gaps.append(FrameGapV1(gap_code=TYPE_MISMATCH, column=column, detail=dtype))
    gaps += [FrameGapV1(gap_code=SENTINEL_EVIDENCE, column=column, detail=evidence)
             for column, evidence in sorted(surface.sentinel_columns.items()) if column in schema]
    gaps += [FrameGapV1(gap_code=DERIVATION_REQUIRED, column=column, detail=derivation)
             for column, derivation in sorted(surface.required_derivations.items())
             if column not in schema]
    return tuple(gaps + [FrameGapV1(gap_code=IMPUTATION_REQUIRED, column=column, detail=strategy)
                         for column, strategy in sorted(surface.imputation_targets.items())
                         if missing_counts.get(column, 0)])


# One §7.4 fan-out group: its scope, the gaps it owns, and the groups it waits on.
class TaskGroupPlanV1(_Row):
    group_id: Identity
    group_kind: GroupKind
    columns: tuple[Identity, ...]
    gap_codes: tuple[Identity, ...]
    depends_on: tuple[Identity, ...]


# The frozen task graph plus the deterministic dispatch queue under the concurrency cap.
@dataclass(frozen=True)
class GroupPlan:
    groups: tuple[TaskGroupPlanV1, ...]
    waves: tuple[tuple[str, ...], ...]


# The smallest safe scope one gap column belongs to (§7.4 table).
def _scope_of(column: str, surface: ContractSurfaceV1) -> tuple[GroupKind, str, tuple[str, ...]]:
    if column == TABLE_WIDE_SCOPE:
        return GroupKind.TABLE_WIDE, TABLE_WIDE_GROUP, (TABLE_WIDE_SCOPE,)
    for recipe_id, columns in sorted(surface.recipe_groups.items()):
        if column in columns:
            return GroupKind.RECIPE_GROUP, f"g:recipe:{recipe_id}", tuple(sorted(columns))
    for coupled in surface.coupled_columns:
        if column in coupled:
            members = tuple(sorted(coupled))
            return GroupKind.COUPLED_COLUMNS, "g:coupled:" + "+".join(members), members
    return GroupKind.SINGLE_COLUMN, f"g:single:{column}", (column,)


# Deterministic dispatch waves; overflow queues under the same frozen task graph.
def _waves(groups: Sequence[TaskGroupPlanV1], max_concurrent: int) -> tuple[tuple[str, ...], ...]:
    known = {group.group_id for group in groups}
    pending = {group.group_id: set(group.depends_on) & known for group in groups}
    waves: list[tuple[str, ...]] = []
    done: set[str] = set()
    while pending:
        ready = sorted(group_id for group_id, deps in pending.items() if deps <= done)
        if not ready:
            raise PlanCompileError("task groups must form a DAG", GROUPS_NOT_A_DAG)
        waves.append(tuple(ready[:max_concurrent]))
        done.update(waves[-1])
        for group_id in waves[-1]:
            del pending[group_id]
    return tuple(waves)


# Build the smallest safe task graph over the gaps, then its deterministic queue (§7.4).
def group_gaps(gaps: Sequence[FrameGapV1], surface: ContractSurfaceV1, *,
               max_concurrent: int = MAX_CONCURRENT_GROUPS) -> GroupPlan:
    scopes: dict[str, tuple[GroupKind, tuple[str, ...], list[str]]] = {}
    owner: dict[str, str] = {}
    for gap in gaps:
        kind, group_id, members = _scope_of(gap.column, surface)
        scope = scopes.setdefault(group_id, (kind, members, []))
        scope[2].extend([gap.gap_code] if gap.gap_code not in scope[2] else [])
        owner.update(dict.fromkeys(members, group_id))
    groups: list[TaskGroupPlanV1] = []
    for group_id in sorted(scopes):
        kind, members, codes = scopes[group_id]
        depends = {found for column in members
                   for source in surface.derivation_sources.get(column, ())
                   for found in (owner.get(source),) if found not in (None, group_id)}
        if kind is not GroupKind.TABLE_WIDE and TABLE_WIDE_GROUP in scopes:
            depends.add(TABLE_WIDE_GROUP)
        groups.append(TaskGroupPlanV1(
            group_id=group_id, group_kind=kind, columns=members, gap_codes=tuple(sorted(codes)),
            depends_on=tuple(sorted(found for found in depends if found is not None))))
    return GroupPlan(groups=tuple(groups), waves=_waves(groups, max_concurrent))


# A draft that returns a design conflict; PRD-003 never resolves one itself (§16).
@dataclass(frozen=True)
class ConflictRoute:
    task_id: str
    draft: pc.DesignConflictDraftV1


# Everything the fan-in measures proposals against; it loads nothing itself.
@dataclass(frozen=True)
class ReconcileInputs:
    manifest: PreparationContextManifestV1
    operations: OperationRegistry
    pack: pp.PreparationPackV1
    context_manifest: ArtifactRef
    gaps: tuple[FrameGapV1, ...] = ()
    groups: tuple[TaskGroupPlanV1, ...] = ()
    phase: PlanPhase = PlanPhase.PREPARATION
    plan_revision: int = 1
    stabilized_frame: ArtifactRef | None = None
    versions: Mapping[str, str] = field(default_factory=dict)


# Registered, permitted, protected, fit-scoped, and row-invariant, item by item (§7.5).
def _item_codes(items: Sequence[PlanItemV1], inputs: ReconcileInputs) -> set[str]:
    codes: set[str] = set()
    manifest = inputs.manifest
    scopes = {(target.role, target.fit_scope) for target in inputs.pack.permitted_imputation_targets}
    for item in items:
        try:
            inputs.operations.get(item.operation_id)
        except RegistryError:
            codes.add(UNREGISTERED_OPERATION)
        targets = set(item.target_columns)
        if item.operation_id not in manifest.permitted_operation_ids:
            codes.add(OPERATION_NOT_PERMITTED)
        if item.phase is ItemPhase.STABILIZATION and inputs.phase is PlanPhase.PREPARATION:
            codes.add(ROW_MEMBERSHIP_CHANGE)
        if item.phase in _FILLING and targets & set(manifest.protected_columns):
            codes.add(PROTECTED_ROLE_TARGET)
        if item.phase is ItemPhase.IMPUTATION and any(
            (manifest.column_roles.get(column), item.fit_scope) not in scopes for column in targets
        ):
            codes.add(ILLEGAL_FIT_SCOPE)
    return codes


# Two proposals for one (operation, target) are a duplicate or an inconsistency; two items
# touching one column must be ordered; two items never share an output column (§7.5).
def _column_codes(items: Sequence[PlanItemV1]) -> set[str]:
    seen: dict[tuple[str, tuple[str, ...]], list[Mapping[str, object]]] = {}
    by_column: dict[str, list[PlanItemV1]] = {}
    for item in items:
        seen.setdefault((item.operation_id, item.target_columns), []).append(item.parameters)
        for column in item.target_columns:
            by_column.setdefault(column, []).append(item)
    codes = {DUPLICATE_MAPPING if all(row == found[0] for row in found) else INCONSISTENT_MAPPING
             for found in seen.values() if len(found) > 1}
    outputs = [item.output_column for item in items if item.output_column is not None]
    unordered = any(left.plan_item_id not in right.depends_on
                    and right.plan_item_id not in left.depends_on
                    for members in by_column.values() for left, right in combinations(members, 2))
    return codes | ({CONFLICTING_OPERATION_ON_COLUMN}
                    if unordered or len(set(outputs)) != len(outputs) else set())


def _recipes(items: Sequence[PlanItemV1], manifest: PreparationContextManifestV1
             ) -> tuple[pp.EstimatorScopedRecipeV1, ...]:
    return tuple(
        pp.EstimatorScopedRecipeV1(
            recipe_id=str(item.parameters.get("recipe_id") or item.plan_item_id),
            target_columns=item.target_columns, method_id=manifest.method_id,
            pack_version=manifest.method_pack_version, operation_id=item.operation_id)
        for item in items if item.fit_scope is FitScope.CROSS_FIT_TRAINING_FOLD)


# Bind every committed item to the frozen group whose scope owns one of its columns.
def _plan_groups(items: Sequence[PlanItemV1],
                 groups: Sequence[TaskGroupPlanV1]) -> tuple[pp.TaskGroupV1, ...]:
    owned = {column: group.group_id for group in groups for column in group.columns}
    members: dict[str, list[str]] = {}
    for item in items:
        for column in (*item.target_columns, item.output_column):
            if column is not None and column in owned:
                members.setdefault(owned[column], []).append(item.plan_item_id)
                break
    return tuple(
        pp.TaskGroupV1(group_id=group.group_id, group_kind=group.group_kind,
                       plan_item_ids=tuple(dict.fromkeys(members[group.group_id])))
        for group in groups if members.get(group.group_id))


# Fan proposals into ONE plan, or route the first design conflict back to PRD-002 (§7.5).
def reconcile(drafts: Sequence[pp.PreparationTaskDraftV1],
              inputs: ReconcileInputs) -> pp.PreparationPlanV1 | ConflictRoute:
    ordered = sorted(drafts, key=lambda draft: draft.task_id)
    for draft in ordered:
        if draft.design_conflict is not None:
            return ConflictRoute(task_id=draft.task_id, draft=draft.design_conflict)
    codes = {TASK_NOT_PROPOSED for draft in ordered
             if draft.stop_state is not pp.TaskStopState.PROPOSED}
    items = [item for draft in ordered for item in draft.plan_items]
    declared = [item.plan_item_id for item in items]
    touched = {column for item in items for column in item.target_columns}
    touched |= {item.output_column for item in items if item.output_column is not None}
    stabilizing = inputs.phase is PlanPhase.STABILIZATION and bool(items)
    codes |= {DUPLICATE_PLAN_ITEM} if len(set(declared)) != len(declared) else set()
    codes |= {PLAN_NOT_A_DAG} if pp.has_cycle(tuple(items)) else set()
    codes |= {UNKNOWN_DEPENDENCY} if any(
        set(item.depends_on) - set(declared) for item in items) else set()
    codes |= {GAP_NOT_COVERED} if any(
        not stabilizing if gap.column == TABLE_WIDE_SCOPE else gap.column not in touched
        for gap in inputs.gaps) else set()
    codes |= _item_codes(items, inputs) | _column_codes(items)
    if codes:
        raise PlanCompileError("the fan-in refused the proposed preparation plan",
                               FAN_IN_REJECTED, tuple(sorted(codes)))
    stabilization = inputs.phase is PlanPhase.STABILIZATION
    return pp.PreparationPlanV1(
        plan_revision=inputs.plan_revision, phase=inputs.phase, items=tuple(items),
        eligibility_rule_ids=inputs.manifest.eligibility_rule_ids if stabilization else (),
        unusable_row_rule_ids=inputs.manifest.unusable_row_rule_ids if stabilization else (),
        recipes=() if stabilization else _recipes(items, inputs.manifest),
        groups=_plan_groups(items, inputs.groups), context_manifest=inputs.context_manifest,
        stabilized_frame=inputs.stabilized_frame, versions=dict(inputs.versions))


# The runnable-frame facts the deterministic gap computation reads (§7.4, §10.1).
def contract_surface(book: PreparationContextManifestV1, pack: pp.PreparationPackV1,
                     contract: Mapping[str, object],
                     columns: Mapping[str, str]) -> ContractSurfaceV1:
    roles, schema = by_role(book.column_roles), set(columns)
    derivations: dict[str, str] = {}
    sources: dict[str, tuple[str, ...]] = {}
    indicators = contract.get("required_missingness_indicators") or ()
    for indicator in cast(Sequence[str], indicators):
        source = roles.get(str(indicator).rsplit("_", 1)[0])
        if source is not None and source in schema and str(indicator) not in schema:
            derivations[str(indicator)], sources[str(indicator)] = "nonnull_indicator", (source,)
    targets = {target.role: target for target in pack.permitted_imputation_targets}
    imputed = {column: targets[book.column_roles[column]].strategy_id
               for column in book.permitted_imputation_columns
               if column in schema and book.column_roles.get(column, "") in targets}
    folds = tuple(sorted(column for column in imputed if targets[book.column_roles[column]]
                         .fit_scope is FitScope.CROSS_FIT_TRAINING_FOLD))
    keys = tuple(book.key_columns)
    return ContractSurfaceV1(
        # A dtype mismatch inside the frame has no V1 resolution: a registered operation only
        # ever writes a NEW column (§10.3 lineage), so only an absent column is a gap.
        required_dtypes={str(name): str(dtype) for name, dtype
                         in cast(Mapping[str, str], contract.get("type_constraints") or {}).items()
                         if name not in schema},
        required_derivations=derivations, derivation_sources=sources, imputation_targets=imputed,
        imputation_fit_scopes={column: targets[book.column_roles[column]].fit_scope
                               for column in imputed},
        coupled_columns=(keys,) if len(keys) > 1 else (),
        recipe_groups={"cross_fit": folds} if folds else {})


# One frozen plan item; its id names the gap it closes, so a rerun recompiles it byte-equal.
def _item(gap: FrameGapV1, phase: ItemPhase, operation_id: str, targets: tuple[str, ...],
          output: str | None, parameters: dict[str, str], fit_scope: FitScope,
          postconditions: tuple[str, ...], depends_on: tuple[str, ...]) -> PlanItemV1:
    return PlanItemV1(
        plan_item_id=f"pi:{gap.gap_code}:{gap.column}", phase=phase, operation_id=operation_id,
        operation_version="v1", target_columns=targets, output_column=output,
        parameters=cast(dict[str, pp.ParameterValue], parameters), fit_scope=fit_scope,
        predicted_missingness_change={}, depends_on=depends_on,
        postcondition_ids=postconditions, rationale_evidence=())


# One gap to the one registered operation the approved contract already names (§25.1).
def _compile_gap(gap: FrameGapV1, surface: ContractSurfaceV1,
                 depends_on: tuple[str, ...]) -> PlanItemV1 | None:
    scope = surface.imputation_fit_scopes.get(gap.column, FitScope.NONE)
    operation = pp.strategy_operation(gap.detail, scope)
    if gap.gap_code == DERIVATION_REQUIRED:
        return _item(gap, ItemPhase.DERIVATION, "registered_derivation",
                     surface.derivation_sources[gap.column], gap.column,
                     {"derivation_id": gap.detail}, FitScope.NONE,
                     ("schema_type_validation", "row_set_invariance"), depends_on)
    if gap.gap_code == SENTINEL_EVIDENCE:
        return _item(gap, ItemPhase.REPAIR, "missing_sentinel_normalization", (gap.column,),
                     f"{gap.column}__normalized", {"sentinels": gap.detail}, FitScope.NONE,
                     _REPAIR_POSTCONDITIONS, depends_on)
    if gap.gap_code != IMPUTATION_REQUIRED or operation is None:
        return None
    if operation == pp.CROSS_FIT_OPERATION:
        # PRD-003 never fits a fold-scoped strategy: it registers the recipe PRD-004 fits (§11.2).
        return _item(gap, ItemPhase.IMPUTATION, operation, (gap.column,), None,
                     {"recipe_id": f"r:{gap.column}", "strategy_id": gap.detail}, scope,
                     ("row_set_invariance",), depends_on)
    return _item(gap, ItemPhase.IMPUTATION, operation, (gap.column,), f"{gap.column}__imputed",
                 {"indicator_column": f"{gap.column}__missing"}
                 if operation == "numeric_median_imputation" else {"missing_level": "__missing__"},
                 scope, _REPAIR_POSTCONDITIONS, depends_on)


# Compile one group with no model call; an unresolvable gap becomes a §16 conflict (§25.1).
def compile_drafts(group: TaskGroupPlanV1, gaps: Sequence[FrameGapV1], surface: ContractSurfaceV1
                   ) -> pp.PreparationTaskDraftV1 | pc.DesignConflictDraftV1:
    items: list[PlanItemV1] = []
    for gap in sorted(gaps, key=lambda row: (row.gap_code, row.column)):
        # Two items over one column must be ordered, so each waits on everything already compiled.
        item = _compile_gap(gap, surface, tuple(row.plan_item_id for row in items))
        if item is None:
            return pc.DesignConflictDraftV1(
                conflict_code=NO_RESOLUTION, failed_rule_id=gap.gap_code, affected_row_count=0,
                affected_unit_count=0, affected_dimension_counts={}, evidence_artifact_ids=(),
                why_no_permitted_operation=f"no registered V1 operation resolves {gap.gap_code}",
                material_design_fields=(gap.gap_code, gap.column),
                recommended_action=pc.ConflictAction.REVISE_DESIGN)
        items.append(item)
    return pp.PreparationTaskDraftV1(
        task_id=group.group_id, stop_state=pp.TaskStopState.PROPOSED, design_conflict=None,
        notes=("harness-compiled: the approved contract names every operation in this group",),
        plan_items=tuple(items))
