# Analysis execution: contribution masks, the estimator-adapter seam, sensitivity runs,
# shared balance calculations and the numerical environment.
# Diagnostic applicability, severity, and claim limits live in analysis.diagnostics.

from __future__ import annotations

import math
import platform
from collections.abc import Callable, Mapping, Sequence
from importlib import metadata
from typing import Any, Final, NamedTuple, Protocol, runtime_checkable

import numpy as np
import polars as pl
from numpy.typing import NDArray

from causal.analysis.integration import contracts as ec
from causal.analysis.integration.packs import EstimationPackV1
from causal.shared.contracts import ArtifactRef

BUILDER_VERSION: Final = "estimation-engine.v1"
UNKNOWN_MASK_RULE, MASK_ROW_SET_MISMATCH = "unknown_mask_rule", "mask_row_set_mismatch"
UNKNOWN_COMPARISON_RULE, MISSING_ROLE_COLUMN = "unknown_comparison_rule", "missing_role_column"
BRANCH_FAILED = "sensitivity_branch_failed"
_Bits = NDArray[np.bool_]
# One §6.2 mask shape: the frozen rows it keeps, decided from declared columns only.
MaskShape = Callable[[pl.DataFrame, Mapping[str, str], ec.ValueMap], _Bits]


def _number(params: ec.ValueMap, key: str, default: float) -> float:
    value = params.get(key)
    return default if value is None or isinstance(value, bool) else float(value)


def _column(frame: pl.DataFrame, roles: Mapping[str, str], role: str) -> pl.Series:
    if (name := roles.get(role)) is None or name not in frame.columns:
        raise ec.EstimationError(f"the plan declares no {role} column", MISSING_ROLE_COLUMN)
    return frame[name]


def _all_rows(frame: pl.DataFrame, roles: Mapping[str, str], params: ec.ValueMap) -> _Bits:
    return np.ones(frame.height, dtype=np.bool_)


def _outcome_observed(frame: pl.DataFrame, roles: Mapping[str, str], params: ec.ValueMap) -> _Bits:
    return _column(frame, roles, "outcome").is_not_null().to_numpy().astype(np.bool_)


# §12.2: a window around the approved cutoff, optionally donut-punched or one-sided.
def _bandwidth(frame: pl.DataFrame, roles: Mapping[str, str], params: ec.ValueMap) -> _Bits:
    running = _column(frame, roles, "running_variable").cast(pl.Float64).to_numpy()
    cutoff, side = _number(params, "cutoff", 0.0), params.get("side")
    distance = np.abs(running - cutoff)
    keep = (distance <= _number(params, "bandwidth", math.inf)) & (
        distance >= _number(params, "donut_radius", 0.0))
    if side is not None:
        keep &= running >= cutoff if side == "above" else running < cutoff
    return keep.astype(np.bool_)


# §11.2: a group-time or stratum cell contributes when it carries the registered support.
def _event_cell(frame: pl.DataFrame, roles: Mapping[str, str], params: ec.ValueMap) -> _Bits:
    keys = [roles[name] for name in ("group", "time") if roles.get(name) in frame.columns]
    if not keys:
        return _all_rows(frame, roles, params)
    sized = frame.select(pl.len().over(keys).alias("cell"))
    return (sized["cell"] >= int(_number(params, "min_cell_units", 1))).to_numpy().astype(np.bool_)


# Every registered pack mask rule id and the §6.2 shape that builds it (§9.2, §10.4, §11.2, §12.2).
MASK_RULES: Final[dict[str, MaskShape]] = {
    "all_rows": _all_rows, "arm_membership": _all_rows, "cross_fit_predicted": _all_rows,
    "approved_target_population": _all_rows, "outcome_observed": _outcome_observed,
    "stratum_cell": _event_cell, "group_time_cell": _event_cell, "event_time_cell": _event_cell,
    "balanced_panel_cell": _event_cell, "selected_bandwidth": _bandwidth,
    "donut_exclusion": _bandwidth, "cutoff_side": _bandwidth}


def mask_bits(rule_id: str, frame: pl.DataFrame, roles: Mapping[str, str],
              params: ec.ValueMap) -> _Bits:
    # One registered mask: a declaration of which frozen rows contribute, never a deletion.
    if (shape := MASK_RULES.get(rule_id)) is None:
        raise ec.EstimationError(f"no registered mask rule {rule_id!r}", UNKNOWN_MASK_RULE)
    return shape(frame, roles, params)


def mask_object_payload(rule_id: str, bits: _Bits) -> dict[str, object]:
    # The bit vector itself is a restricted object payload; no envelope or event carries it.
    return {"mask_rule_id": rule_id, "row_count": int(bits.size),
            "builder_version": BUILDER_VERSION, "bits_hex": np.packbits(bits).tobytes().hex()}


def contribution_mask(plan: ec.EstimationPlanV1, rule_id: str, bits: _Bits,
                      mask_object: ec.ObjectRefV1, *, frame_row_set_hash: str,
                      calculation_id: str, parents: tuple[ArtifactRef, ...],
                      unit_ids: pl.Series | None = None) -> ec.AnalysisContributionMaskV1:
    # The mask hangs off the frozen row set: a frame that is not the planned one is refused,
    # and every non-contributing row is accounted for under its registered reason.
    if plan.row_set_hash != frame_row_set_hash:
        raise ec.EstimationError(
            f"{calculation_id} masks a frame outside the frozen row set", MASK_ROW_SET_MISMATCH)
    included, total = int(np.count_nonzero(bits)), int(bits.size)
    kept = 0 if unit_ids is None else int(unit_ids.filter(pl.Series(bits)).n_unique())
    units = {} if unit_ids is None else {"unit": kept}
    dropped = {} if unit_ids is None else {"unit": int(unit_ids.n_unique()) - kept}
    return ec.AnalysisContributionMaskV1(
        parents=parents, versions=dict(plan.versions), parent_row_set_hash=plan.row_set_hash,
        calculation_id=calculation_id, outcome_id=plan.outcome_id,
        estimator_id=plan.estimator_id, mask_rule_id=rule_id, mask_object=mask_object,
        included_counts={"row": included} | units,
        noncontributing_counts={"row": total - included} | dropped,
        reason_counts={f"{rule_id}_excluded": total - included}, builder_version=BUILDER_VERSION)


class AdapterResult(NamedTuple):
    # What one estimator adapter returns: the primary items, the diagnostic values harvested
    # from the same fit, and the fit handle a sensitivity branch may reuse.
    items: tuple[ec.PrimaryContrastResultV1, ...]
    harvest: dict[str, ec.ValueMap]
    fit: object | None = None


@runtime_checkable
class CrossFitFit(Protocol):
    # §10.2: what a cross-fitted adapter hands back on `AdapterResult.fit` so the coordinator can
    # commit what was dealt — the assignment, its restricted row-to-fold mapping, the out-of-fold
    # prediction object, and the receipts wall 6 measures. No coordinator knows which pack
    # cross-fits; it asks the fit it was handed and commits nothing when the answer is no.

    receipts: Mapping[str, Mapping[str, int]]

    def mapping_payload(self) -> dict[str, object]: ...
    def prediction_payload(self) -> dict[str, object]: ...
    def assignment(self, plan: ec.EstimationPlanV1, mapping_object: ec.ObjectRefV1, *,
                   plan_ref: ArtifactRef,
                   parents: tuple[ArtifactRef, ...]) -> ec.CrossFitAssignmentV1: ...


class EstimatorAdapter(Protocol):
    # §19.1: an adapter receives the declared estimator-input view, the frozen plan, its pack,
    # and — for a sensitivity branch — the prespecified parameter delta. Nothing else.

    def fit(self, view: pl.DataFrame, plan: ec.EstimationPlanV1, pack: EstimationPackV1,
            overrides: ec.ValueMap | None = None) -> AdapterResult: ...


class MethodAdapter(EstimatorAdapter, Protocol):
    # The registered per-method seam a coordinator binds to one run's contribution mask: the one
    # fit and the pack's registered multiplicity adjustment. No
    # coordinator holds method-specific statistics of its own (§8, §19.1).

    def multiplicity(self, items: Sequence[ec.PrimaryContrastResultV1],
                     level: float) -> dict[str, ec.ValueMap]: ...


def estimator_view(frame: pl.DataFrame, role_columns: Mapping[str, str]) -> pl.DataFrame:
    # The adapter never sees an undeclared column: the view is built from the plan's roles.
    if missing := sorted({name for name in role_columns.values() if name not in frame.columns}):
        raise ec.EstimationError(f"the prepared frame lacks {missing}", MISSING_ROLE_COLUMN)
    return frame.select(sorted(set(role_columns.values())))


# Every registered comparison rule, the generic comparison it uses, and what an unstable
# branch is called (§15: direction, magnitude, and interval stability).
COMPARISON_RULES: Final[dict[str, str]] = {
    "sign_and_interval_stability": "direction", "interval_width_stability": "interval",
    "influence_stability": "magnitude", "bound_containment": "interval",
    "null_effect_expected": "null", "descriptive_only": "descriptive"}
UNSTABLE: Final[dict[str, str]] = {
    "direction": "direction_changed", "magnitude": "magnitude_shifted",
    "interval": "intervals_disjoint", "null": "null_effect_rejected"}


def compare(rule_id: str, primary: ec.PrimaryContrastResultV1 | None,
            branch: ec.PrimaryContrastResultV1 | None, tolerance: float) -> str:
    # §15 comparison against the primary result; never a selection of the best branch.
    if (kind := COMPARISON_RULES.get(rule_id)) is None:
        raise ec.EstimationError(f"no registered comparison rule {rule_id!r}",
                                 UNKNOWN_COMPARISON_RULE)
    if kind == "descriptive":
        return "not_applicable"
    if branch is None or primary is None:
        return "not_computed"
    stable = {"direction": (primary.estimate >= 0.0) == (branch.estimate >= 0.0),
              "magnitude": abs(branch.estimate - primary.estimate) <= tolerance * max(
                  abs(primary.estimate), 1.0),
              "null": branch.interval_lower <= 0.0 <= branch.interval_upper,
              "interval": primary.interval_lower <= branch.interval_upper
              and branch.interval_lower <= primary.interval_upper}
    return "stable" if stable[kind] else UNSTABLE[kind]


def run_sensitivities(plan: ec.EstimationPlanV1, pack: EstimationPackV1,
                      adapter: EstimatorAdapter, view: pl.DataFrame,
                      primary: ec.PrimaryContrastResultV1 | None,
                      base: Mapping[str, Any]) -> tuple[ec.SensitivityResultV1, ...]:
    # Every prespecified branch runs even when the primary result is favorable, in registered
    # order, at concurrency one (§15, §26.1). A failed branch is reported failed, never omitted.
    tolerance = plan.numerical_tolerances.get("sensitivity_magnitude", 0.25)
    results: list[ec.SensitivityResultV1] = []
    for branch_id in plan.required_sensitivity_ids:
        row, item = pack.branch(branch_id), None
        warnings: tuple[str, ...] = (BRANCH_FAILED,)
        try:
            fitted = adapter.fit(view, plan, pack, dict(row.parameter_delta))
            item = fitted.items[0] if fitted.items else None
        except Exception as error:  # noqa: BLE001 -- §15: report every failed branch
            item = None
            if isinstance(error, ec.EstimationError):
                warnings += (error.code,)
        comparison = compare(row.comparison_rule_id, primary, item, tolerance)
        policy: ec.PolicyResult = ("descriptive" if comparison == "not_applicable"
                                   else "acceptable" if comparison == "stable" else "warning")
        results.append(ec.SensitivityResultV1(
            **base, branch_id=branch_id, purpose=row.purpose, result=item,
            parameter_delta=dict(row.parameter_delta), policy_result=policy,
            execution_status="computed" if item is not None else "failed",
            values={"comparison": comparison}, comparison_result=comparison,
            warnings=() if item is not None else warnings,
            comparison_rule_id=row.comparison_rule_id, interpreting_rule_id=row.comparison_rule_id,
            qualification_rule_ids=(row.comparison_rule_id,) if policy == "warning" else (),
            implementation_version=BUILDER_VERSION))
    return tuple(results)


def balance(frame: pl.DataFrame, group_column: str, columns: Sequence[str], *,
            weight_column: str | None = None) -> dict[str, ec.ValueMap]:
    # The §9.3/§10.5 balance helper: grouped means and the pooled standardized difference,
    # optionally weighted. Shared by every pack that must show comparability.
    weight = pl.lit(1.0) if weight_column is None else pl.col(weight_column).cast(pl.Float64)
    total = pl.len().cast(pl.Float64) if weight_column is None else weight.sum()
    rows: dict[str, ec.ValueMap] = {}
    for column in columns:
        if frame.schema[column].base_type() in (pl.String, pl.Categorical, pl.Enum):
            # Category codes have no numeric distance. Compare each level's group prevalence;
            # null, when present, is an explicit level rather than an invented zero code.
            levels = frame[column].cast(pl.String).unique().sort().to_list()
            # The original-column::repr(level) key preserves provenance for registered summaries.
            values = {f"{column}::{level!r}": (pl.col(column).is_null() if level is None else
                (pl.col(column).cast(pl.String) == level).fill_null(False)).cast(pl.Float64)
                for level in levels}
        else:
            values = {column: pl.col(column).cast(pl.Float64).fill_null(0.0)}
        for name, value in values.items():
            grouped = frame.group_by(group_column).agg(
                ((value * weight).sum() / total).alias("mean"),
                ((value * value * weight).sum() / total).alias("second")).sort(group_column)
            means, seconds = grouped["mean"].to_list(), grouped["second"].to_list()
            spread = math.sqrt(sum(max(s - m * m, 0.0) for m, s in zip(means, seconds, strict=True))
                               / max(len(means), 1))
            difference = float(means[-1]) - float(means[0])
            rows[name] = {"mean_low": float(means[0]), "mean_high": float(means[-1]),
                          "mean_difference": difference, "pooled_std": spread,
                          "standardized_difference": difference / spread if spread else 0.0}
    return rows


def _version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "absent"


def numerical_environment(plan: ec.EstimationPlanV1, packages: Sequence[str], *,
                          build_identifier: str,
                          image_digest: str | None = None) -> ec.NumericalEnvironmentManifestV1:
    # §6.4: the exact runtime this run executed in, read from the installed distributions and
    # the plan's own seed, thread, and tolerance pins. Two builds of one run agree exactly.
    return ec.NumericalEnvironmentManifestV1(
        python_version=platform.python_version(), platform=platform.platform(),
        package_versions={name: _version(name) for name in sorted(set(packages))},
        seeds={"plan": plan.seed}, float_dtype="float64",
        parallelism={key: int(_number(plan.estimator_parameters, key, 1))
                     for key in ("threads", "processes")},
        serialization_policy="canonical-json.v1", build_identifier=build_identifier,
        numerical_tolerances=dict(plan.numerical_tolerances), runtime_image_digest=image_digest)


def evidence_bundle(plan: ec.EstimationPlanV1, kind: ec.EvidenceKind,
                    results: Sequence[tuple[ArtifactRef, ec.ExecutionStatus]],
                    *, plan_ref: ArtifactRef, parents: tuple[ArtifactRef, ...],
                    ) -> ec.EvidenceBundleV1:
    # §26.2: one bundle per kind, counting every result under exactly one terminal status.
    counts: dict[str, int] = {}
    for _, status in results:
        counts[status] = counts.get(status, 0) + 1
    return ec.EvidenceBundleV1(
        parents=parents, versions=dict(plan.versions), kind=kind, plan=plan_ref,
        results=tuple(ref for ref, _ in results), terminal_status_counts=counts)
