# The estimation engine: contribution masks, the estimator-adapter seam, the sequential
# evidence runner, the generic severity applier, the sensitivity loop, the §16.1 judgment
# ceiling, the §6.4 environment manifest, and the frozen §17 figure-data boundary
# (PRD-004 §6.2, §6.4, §14, §15, §16.1, §17, §26.1). Nothing here imports another stage.

from __future__ import annotations

import math
import platform
from collections.abc import Callable, Mapping, Sequence
from importlib import metadata
from typing import Any, Final, NamedTuple, Protocol

import numpy as np
import polars as pl
from numpy.typing import NDArray

from causal.estimation import contracts as ec
from causal.estimation.packs import EstimationPackV1
from causal.shared.contracts import ArtifactRef

BUILDER_VERSION: Final = "estimation-engine.v1"
UNKNOWN_MASK_RULE, MASK_ROW_SET_MISMATCH = "unknown_mask_rule", "mask_row_set_mismatch"
UNKNOWN_COMPARISON_RULE, MISSING_ROLE_COLUMN = "unknown_comparison_rule", "missing_role_column"
BRANCH_FAILED, DIAGNOSTIC_NOT_COMPUTED = "sensitivity_branch_failed", "diagnostic_not_computed"
# The §16.1 table's four capping conditions, in the order it states them.
CEILING_RULES: Final = ("primary_estimator_unavailable", "invalidation_guard_triggered",
                        "required_diagnostic_missing", "qualification_guard_triggered")
_Cap = tuple[ec.JudgmentStatus, str, str]
_Bits = NDArray[np.bool_]
# One §6.2 mask shape: the frozen rows it keeps, decided from declared columns only.
MaskShape = Callable[[pl.DataFrame, Mapping[str, str], ec.ValueMap], _Bits]


def _number(params: ec.ValueMap, key: str, default: float) -> float:
    value = params.get(key)
    return default if value is None or isinstance(value, bool) else float(value)


def _as_float(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


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


class EstimatorAdapter(Protocol):
    # §19.1: an adapter receives the declared estimator-input view, the frozen plan, its pack,
    # and — for a sensitivity branch — the prespecified parameter delta. Nothing else.

    def fit(self, view: pl.DataFrame, plan: ec.EstimationPlanV1, pack: EstimationPackV1,
            overrides: ec.ValueMap | None = None) -> AdapterResult: ...


class MethodAdapter(EstimatorAdapter, Protocol):
    # The registered per-method seam a coordinator binds to one run's contribution mask: the one
    # fit, the pack's registered multiplicity adjustment, and its §17 figure-data builders. No
    # coordinator holds method-specific statistics of its own (§8, §19.1).

    def multiplicity(self, items: Sequence[ec.PrimaryContrastResultV1],
                     level: float) -> dict[str, ec.ValueMap]: ...
    def figures(self, harvest: Mapping[str, ec.ValueMap]) -> Mapping[str, FigureBuilder]: ...
    def visual_evidence(self, builder_id: str) -> str: ...


def estimator_view(frame: pl.DataFrame, role_columns: Mapping[str, str]) -> pl.DataFrame:
    # The adapter never sees an undeclared column: the view is built from the plan's roles.
    if missing := sorted({name for name in role_columns.values() if name not in frame.columns}):
        raise ec.EstimationError(f"the prepared frame lacks {missing}", MISSING_ROLE_COLUMN)
    return frame.select(sorted(set(role_columns.values())))


def _bound(key: str) -> tuple[str, bool]:
    # One registered threshold key as (measured value name, whether it is a lower bound).
    for prefix, lower in (("max_", False), ("min_", True), ("minimum_", True)):
        if key.startswith(prefix):
            return key[len(prefix):], lower
    return (key[:-6], True) if key.endswith("_floor") else (key, False)


def severity_result(severity: ec.DiagnosticSeverity, values: ec.ValueMap,
                    thresholds: ec.ValueMap) -> ec.PolicyResult:
    # §14.2: the prespecified severity and the pack's threshold row decide together. Nothing
    # here may promote, demote, or reinterpret a severity after seeing the value.
    if severity == "descriptive":
        return "descriptive"
    breached = False
    for key, limit in thresholds.items():
        name, lower = _bound(key)
        found, bound = _as_float(values.get(name)), _as_float(limit)
        breached = breached or (found is not None and bound is not None
                                and (found < bound if lower else found > bound))
    if not breached:
        return "acceptable"
    return "invalidating" if severity == "invalidation_guard" else "warning"


def run_diagnostics(pack: EstimationPackV1, harvest: Mapping[str, ec.ValueMap],
                    base: Mapping[str, Any]) -> tuple[ec.DiagnosticResultV1, ...]:
    # §26.1: the registered order, one at a time. Every required row reaches a visible terminal
    # result; an unharvested row is `not_computable` and an empty harvest is `failed`.
    results: list[ec.DiagnosticResultV1] = []
    for row in pack.required_diagnostics:
        values = harvest.get(row.diagnostic_id)
        status: ec.ExecutionStatus = ("not_computable" if values is None
                                      else "computed" if values else "failed")
        policy: ec.PolicyResult = severity_result(
            row.severity, values, row.threshold_params) if values else (
            "descriptive" if row.severity == "descriptive" else "warning")
        results.append(ec.DiagnosticResultV1(
            **base, diagnostic_id=row.diagnostic_id, diagnostic_version="v1",
            severity=row.severity, threshold_context=dict(row.threshold_params),
            execution_status=status, policy_result=policy, values=dict(values or {}),
            warnings=() if status == "computed" else (DIAGNOSTIC_NOT_COMPUTED,),
            interpreting_rule_id=row.diagnostic_id, implementation_version=BUILDER_VERSION))
    return tuple(results)


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
        try:
            fitted = adapter.fit(view, plan, pack, dict(row.parameter_delta))
            item = fitted.items[0] if fitted.items else None
        except Exception:  # noqa: BLE001 -- §15: any adapter failure is a reported failed branch
            item = None
        comparison = compare(row.comparison_rule_id, primary, item, tolerance)
        policy: ec.PolicyResult = ("descriptive" if comparison == "not_applicable"
                                   else "acceptable" if comparison == "stable" else "warning")
        results.append(ec.SensitivityResultV1(
            **base, branch_id=branch_id, purpose=row.purpose, result=item,
            parameter_delta=dict(row.parameter_delta), policy_result=policy,
            execution_status="computed" if item is not None else "failed",
            values={"comparison": comparison}, comparison_result=comparison,
            warnings=() if item is not None else (BRANCH_FAILED,),
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
        value = pl.col(column).cast(pl.Float64).fill_null(0.0)
        grouped = frame.group_by(group_column).agg(
            ((value * weight).sum() / total).alias("mean"),
            ((value * value * weight).sum() / total).alias("second")).sort(group_column)
        means, seconds = grouped["mean"].to_list(), grouped["second"].to_list()
        spread = math.sqrt(sum(max(s - m * m, 0.0) for m, s in zip(means, seconds, strict=True))
                           / max(len(means), 1))
        difference = float(means[-1]) - float(means[0])
        rows[column] = {"mean_low": float(means[0]), "mean_high": float(means[-1]),
                        "mean_difference": difference, "pooled_std": spread,
                        "standardized_difference": difference / spread if spread else 0.0}
    return rows


def _diagnostic_caps(plan: ec.EstimationPlanV1, diagnostics: Sequence[ec.DiagnosticResultV1],
                     approved: frozenset[str]) -> tuple[_Cap, ...]:
    # §16.1 rows 2-4, decided once over the frozen diagnostic set.
    caps: list[_Cap] = []
    reported = {row.diagnostic_id: row for row in diagnostics}
    for name, severity in plan.required_diagnostics.items():
        found = reported.get(name)
        if found is None or (found.execution_status != "computed" and severity not in approved):
            caps.append(("not_reportable", CEILING_RULES[2], name))
        elif found.policy_result == "invalidating":
            caps.append(("not_reportable", CEILING_RULES[1], name))
        elif found.policy_result == "warning" and severity == "qualification_guard":
            caps.append(("reportable_with_qualifications", CEILING_RULES[3], name))
    return tuple(caps)


def judgment_ceiling(plan: ec.EstimationPlanV1, result: ec.PrimaryAnalysisResultV1 | None,
                     diagnostics: Sequence[ec.DiagnosticResultV1],
                     evidence: Mapping[str, ArtifactRef], *, plan_ref: ArtifactRef,
                     primary_ref: ArtifactRef | None, parents: tuple[ArtifactRef, ...],
                     approved_handling: frozenset[str] = frozenset()) -> ec.JudgmentCeilingV1:
    # The §16.1 deterministic ceiling, calculated before any model-authored interpretation and
    # raisable by nobody: one item per prespecified contrast, overall = the most restrictive.
    caps = _diagnostic_caps(plan, diagnostics, approved_handling)
    found = {item.contrast_id: item for item in (result.primary_items if result else ())}
    items: list[ec.CeilingItemV1] = []
    for contrast in plan.contrast_ids:
        item = found.get(contrast)
        rows: tuple[_Cap, ...] = caps if item is not None and (
            item.convergence != "not_converged") else (
            ("not_estimable", CEILING_RULES[0], contrast), *caps)
        items.append(ec.CeilingItemV1(
            contrast_id=contrast, ceiling=ec.most_restrictive(tuple(cap for cap, _, _ in rows)),
            triggering_rule_ids=tuple(f"{rule}:{name}" for _, rule, name in rows),
            evidence=tuple(ref for _, _, name in rows if (ref := evidence.get(name)) is not None)))
    return ec.JudgmentCeilingV1(
        parents=parents, versions=dict(plan.versions), plan=plan_ref, primary_result=primary_ref,
        items=tuple(items),
        overall_ceiling=ec.most_restrictive(tuple(item.ceiling for item in items)))


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


# One registered §17 builder: frozen results in, typed figure-ready points out.
FigureBuilder = Callable[[ec.PrimaryAnalysisResultV1], tuple[ec.FigureDataPointV1, ...]]


def figure_data(plan: ec.EstimationPlanV1, builder_id: str, builder: FigureBuilder,
                result: ec.PrimaryAnalysisResultV1, *, visual_evidence_id: str,
                disclosure: ec.JudgmentStatus, parents: tuple[ArtifactRef, ...],
                counts: Mapping[str, int], mask_hash: str | None = None,
                units: Mapping[str, str] = {}, labels: Mapping[str, str] = {},
                rule_ids: Sequence[str] = ("frozen_result_passthrough",),
                ) -> ec.FigureDataArtifactV1:
    # §17: the builder wraps frozen approved values. It estimates nothing, chooses no binning
    # after seeing a result, and never discloses above the claim judgment's ceiling.
    return ec.FigureDataArtifactV1(
        parents=parents, versions=dict(plan.versions), visual_evidence_id=visual_evidence_id,
        builder_id=builder_id, builder_version=BUILDER_VERSION, points=builder(result),
        units=dict(units), labels=dict(labels), rule_ids=tuple(rule_ids),
        contributing_counts=dict(counts), contribution_mask_hash=mask_hash,
        disclosure_status=disclosure)


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
