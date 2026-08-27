# The sharp regression-discontinuity estimator adapter (PRD-004 §12): one versioned `rdrobust`
# call at the exact approved cutoff, the §12.2 assignment and bandwidth view, the §12.3 diagnostic
# harvest read off that call plus the registered `rddensity` and covariate-continuity adapters,
# the §12.4 prespecified branches, and the §17 row-four figure payloads at a fixed registered
# binning. This module commits nothing, holds no state beyond the mask it was bound to, and never
# imports another stage.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

import numpy as np
import polars as pl
from rddensity import rddensity  # type: ignore[import-untyped]
from rdrobust import rdplot, rdrobust  # type: ignore[import-untyped]

from causal.estimation import contracts as ec
from causal.estimation import engine
from causal.estimation.packs import EstimationPackV1
from causal.shared.contracts import ArtifactRef

ADAPTER_VERSION: Final = "sharp-rdd-adapter.v1"
# The three labelled quantities every call reports; the pack names which one is primary.
QUANTITIES: Final[dict[str, str]] = {"robust_bias_corrected": "Robust",
                                     "bias_corrected": "Bias-Corrected",
                                     "conventional": "Conventional"}
# The approved kernel under `rdrobust`'s own name; triangular is the V1 default profile.
KERNELS: Final[dict[str, str]] = {"triangular": "tri", "uniform": "uni", "epanechnikov": "epa"}
# §12.2: a sharp assignment that contradicts the approved cutoff invalidates the method; no row
# is ever removed to make the rule hold.
CONTRADICTION: Final = "assignment_contradicts_approved_cutoff"
NO_SUPPORT: Final = "cutoff_side_without_support"
# §17 row four: the registered binning is fixed here, before any result is seen, and the fitted
# curve is reported at this many evenly spaced points.
BINS, CURVE = 20, 40
VISUAL_EVIDENCE: Final[dict[str, str]] = {
    "binned_outcome_summary": "binned_outcome_fit", "fitted_curve_points": "cutoff_bandwidth",
    "density_continuity_summary": "density_continuity", "primary_contrast_intervals":
    "primary_local_estimate"}


def _number(params: ec.ValueMap, key: str, default: float) -> float:
    value = params.get(key)
    return default if value is None or isinstance(value, bool) else float(value)


def _column(frame: pl.DataFrame, roles: Mapping[str, str], role: str) -> np.ndarray[Any, Any]:
    return frame[roles[role]].cast(pl.Float64).to_numpy()


def _codes(frame: pl.DataFrame, name: str) -> np.ndarray[Any, Any]:
    return frame[name].cast(pl.String).cast(pl.Categorical).to_physical().to_numpy()


def _assignment(frame: pl.DataFrame, roles: Mapping[str, str], params: ec.ValueMap) -> None:
    # §12.2: the approved cutoff and direction decide treatment exactly. A contradiction
    # invalidates the sharp method and never edits the frozen frame.
    if roles.get("treatment") not in frame.columns:
        return
    above = pl.col(roles["running_variable"]).cast(pl.Float64) >= _number(params, "cutoff", 0.0)
    if str(params.get("assignment_direction") or "above") != "above":
        above = ~above
    broken = int(frame.filter(above != (pl.col(roles["treatment"]).cast(pl.Float64) > 0)).height)
    if broken:
        raise ec.EstimationError(
            f"{broken} rows contradict the approved sharp assignment", CONTRADICTION)


def _covariates(frame: pl.DataFrame, roles: Mapping[str, str]) -> tuple[str, ...]:
    named = roles.get("predetermined_covariate")
    return (named,) if named is not None and named in frame.columns else ()


def _call(frame: pl.DataFrame, roles: Mapping[str, str], plan: ec.EstimationPlanV1,
          params: ec.ValueMap, *, order: float | None = None, width: Sequence[float] | None = None,
          outcome: str | None = None) -> Any:
    # The one versioned `rdrobust` call: the exact approved cutoff, the approved polynomial order
    # and kernel, the registered bandwidth rule, robust bias-corrected inference, and the approved
    # covariate and cluster handling. A donut or placebo branch reaches it as a contribution
    # subset over the unchanged frozen rows, never as a deletion.
    running = _column(frame, roles, "running_variable")
    base = _number(params, "cutoff", 0.0)
    offset = _number(params, "placebo_cutoff_offset", 0.0)
    keep = np.abs(running - base) >= _number(params, "donut_radius", 0.0)
    if offset:
        keep = keep & (running < base if offset < 0.0 else running >= base)
    adjusted = _covariates(frame, roles) if params.get("covariate_adjustment") else ()
    cluster = roles.get("cluster")
    return rdrobust(
        y=_column(frame, roles, outcome or "outcome"), x=running, c=base + offset,
        p=int(order if order is not None else _number(params, "polynomial_order", 1.0)),
        kernel=KERNELS.get(str(params.get("kernel") or "triangular"), "tri"),
        bwselect=str(params.get("bandwidth_selector") or "mserd"),
        h=list(width) if width is not None else None,
        covs=_column(frame, roles, "predetermined_covariate").reshape(-1, 1) if adjusted else None,
        cluster=_codes(frame, cluster) if cluster is not None else None,
        masspoints="check" if params.get("mass_point_handling") == "checked" else "adjust",
        subset=keep if not keep.all() else None,
        level=round(100.0 * plan.confidence_level))


def _widths(found: Any, multiplier: float) -> list[float]:
    # §12.2: a bandwidth multiple is a registered branch over the bandwidth the approved selector
    # produced, never a search for the most favorable window.
    return [float(value) * multiplier for value in found.bws.loc["h"].to_list()]


def _read(found: Any, quantity: str) -> ec.ValueMap:
    # The labelled quantity the pack names primary; the other two stay labelled beside it.
    row = QUANTITIES.get(quantity, "Robust")
    lower, upper = (float(bound) for bound in found.ci.loc[row].to_list())
    return {"estimate": float(found.coef.loc[row].iloc[0]), "interval_lower": lower,
            "standard_error": float(found.se.loc[row].iloc[0]), "interval_upper": upper,
            "p_value": float(found.pv.loc[row].iloc[0])}


def _quantities(found: Any, params: ec.ValueMap) -> ec.ValueMap:
    left, right = found.bws.loc["h"].to_list()
    return {"primary_quantity": str(params.get("primary_quantity") or "robust_bias_corrected"),
            "bandwidth_left": float(left), "bandwidth_right": float(right),
            "effective_left": int(found.N_h[0]), "effective_right": int(found.N_h[1]),
            "kernel": str(found.kernel), "bandwidth_selector": str(found.bwselect),
            "polynomial_order": int(found.p), "cutoff": _number(params, "cutoff", 0.0) + _number(
                params, "placebo_cutoff_offset", 0.0)}


def _item(plan: ec.EstimationPlanV1, params: ec.ValueMap, found: Any, counts: dict[str, int],
          mask: ArtifactRef) -> ec.PrimaryContrastResultV1:
    # §12.1: the one approved local treatment effect at the cutoff (§6.3).
    return ec.PrimaryContrastResultV1(
        **cast(Any, _read(found, str(params.get("primary_quantity") or "robust"))),
        contrast_id=plan.contrast_ids[0], estimand_id=plan.estimand_id,
        estimand_label="local average treatment effect at the approved cutoff",
        estimate_units=plan.outcome_scale, comparator_id=plan.comparator_id,
        effect_direction="above_minus_below", confidence_level=plan.confidence_level,
        uncertainty_method=plan.uncertainty_method,
        finite_sample_correction=plan.finite_sample_correction, contributing_counts=counts,
        contribution_mask=mask, estimator_id=plan.estimator_id,
        estimator_version=plan.estimator_version, estimator_parameters=dict(params),
        adapter_version=ADAPTER_VERSION, convergence="converged",
        method_quantities=_quantities(found, params))


def _density(frame: pl.DataFrame, roles: Mapping[str, str], params: ec.ValueMap
             ) -> tuple[ec.ValueMap, ec.ValueMap]:
    # §12.3: the registered `rddensity` manipulation test. A density test never proves the absence
    # of manipulation, and evidence of it is never repaired away.
    try:
        found = rddensity(X=_column(frame, roles, "running_variable"),
                          c=_number(params, "cutoff", 0.0))
    except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
        return {}, {}
    left, right, difference = (float(value) for value in found.hat.to_list())
    return ({"density_p_value": float(found.test["p_jk"]),
             "density_statistic": float(found.test["t_jk"])},
            {"density_left": left, "density_right": right, "density_difference": difference})


def _continuity(frame: pl.DataFrame, roles: Mapping[str, str], plan: ec.EstimationPlanV1,
                params: ec.ValueMap) -> ec.ValueMap:
    # §12.3: continuity at the cutoff for every approved predetermined covariate, each read from
    # its own `rdrobust` call at the same approved cutoff and bandwidth rule.
    found: ec.ValueMap = {}
    for name in _covariates(frame, roles):
        seen = _call(frame, roles, plan, dict(params) | {"covariate_adjustment": False},
                     outcome="predetermined_covariate")
        jump, error = (float(seen.coef.loc["Robust"].iloc[0]), float(seen.se.loc["Robust"].iloc[0]))
        found[f"jump_{name}"] = jump
        found[f"z_{name}"] = abs(jump) / error if error else 0.0
    found["covariate_jump_z"] = max(
        (abs(float(value or 0.0)) for key, value in found.items() if key.startswith("z_")),
        default=0.0)
    return found


def _probes(frame: pl.DataFrame, roles: Mapping[str, str], plan: ec.EstimationPlanV1,
            params: ec.ValueMap, found: Any) -> ec.ValueMap:
    # §12.3: how far the estimate moves under the registered bandwidth multiples and the
    # registered polynomial-order comparison, measured before any of them can be preferred.
    base = float(found.coef.loc["Robust"].iloc[0])
    order = _number(params, "polynomial_order", 1.0)
    seen: ec.ValueMap = {}
    for name, width, degree in (("half", 0.5, order), ("double", 2.0, order),
                                ("order", 1.0, order + 1.0)):
        probe = _call(frame, roles, plan, params, order=degree, width=_widths(found, width))
        seen[f"{name}_estimate"] = float(probe.coef.loc["Robust"].iloc[0])
    seen["relative_estimate_change"] = max(
        abs(cast(float, value) - base) for value in seen.values()) / max(abs(base), 1.0)
    return seen


def harvest(frame: pl.DataFrame, roles: Mapping[str, str], plan: ec.EstimationPlanV1,
            params: ec.ValueMap, found: Any) -> dict[str, ec.ValueMap]:
    # The §12.3 required diagnostics: support and heaping on both sides of the cutoff, the
    # selected bandwidths and effective observations, the manipulation test, covariate continuity,
    # bandwidth and polynomial sensitivity, near-cutoff leverage, and bias-correction integrity.
    running = _column(frame, roles, "running_variable")
    cutoff = _number(params, "cutoff", 0.0)
    below, above = running < cutoff, running >= cutoff
    effective = min(int(found.N_h[0]), int(found.N_h[1]))
    unique = len(np.unique(running))
    density, hats = _density(frame, roles, params)
    return {
        "cutoff_side_support": {
            "effective_observations_per_side": effective, "rows_below": int(below.sum()),
            "rows_above": int(above.sum()), "unique_running_values": unique},
        "selected_bandwidth_report": {
            "bandwidth_left": float(found.bws.loc["h"].iloc[0]),
            "bandwidth_right": float(found.bws.loc["h"].iloc[1]),
            "bias_bandwidth": float(found.bws.loc["b"].iloc[0]),
            "selector": str(found.bwselect)},
        "mass_points_and_heaping": {
            "repeated_value_share": 1.0 - (unique / float(frame.height or 1)),
            "mass_points_left": int(found.M[0]), "mass_points_right": int(found.M[1])},
        "density_manipulation_test": density,
        "covariate_continuity": _continuity(frame, roles, plan, params),
        "bandwidth_polynomial_sensitivity": _probes(frame, roles, plan, params, found),
        "influence_leverage_near_cutoff": {
            "single_observation_leverage_share": 1.0 / float(effective or 1),
            "effective_observations": effective},
        "sorting_and_other_policy_qualification": dict(hats) | {
            "cutoff": cutoff, "carried_rule_ids": len(plan.numerical_failure_rule_ids)},
        "robust_bias_correction_integrity": {
            "conventional_estimate": float(found.coef.loc["Conventional"].iloc[0]),
            "bias_corrected_estimate": float(found.coef.loc["Bias-Corrected"].iloc[0]),
            "robust_standard_error": float(found.se.loc["Robust"].iloc[0]),
            "confidence_level": float(found.level) / 100.0, "variance_estimator": str(found.vce)}}


def series(frame: pl.DataFrame, roles: Mapping[str, str], params: ec.ValueMap, found: Any,
           hats: ec.ValueMap) -> dict[str, ec.ValueMap]:
    # §17 row four, keyed `category|x|denominator|lower|upper`: binned outcome summaries at the
    # FIXED registered binning above — never a binning chosen after a result was seen — the fitted
    # curve `rdplot` produced, and the density and continuity summary.
    cutoff = _number(params, "cutoff", 0.0)
    plotted = rdplot(y=_column(frame, roles, "outcome"), c=cutoff, nbins=[BINS, BINS],
                     x=_column(frame, roles, "running_variable"), binselect="es",
                     p=int(_number(params, "polynomial_order", 1.0)), hide=True)
    binned, curve = plotted.vars_bins, plotted.vars_poly
    return {
        "figure_binned_outcome": {
            f"{'above' if row.rdplot_mean_bin >= cutoff else 'below'}|{row.rdplot_mean_x:.6f}"
            f"|{int(row.rdplot_N)}|{row.rdplot_ci_l:.6f}|{row.rdplot_ci_r:.6f}":
            float(row.rdplot_mean_y) for row in binned.itertuples()},
        "figure_fitted_curve": {
            f"{'above' if row.rdplot_x >= cutoff else 'below'}|{row.rdplot_x:.6f}|||":
            float(row.rdplot_y) for row in curve.iloc[::max(len(curve) // CURVE, 1)].itertuples()},
        "figure_density_continuity": dict(hats) | {
            "cutoff": cutoff, "bandwidth_left": float(found.bws.loc["h"].iloc[0]),
            "bandwidth_right": float(found.bws.loc["h"].iloc[1])}}


def _points(found: Mapping[str, ec.ValueMap], name: str,
            series_id: str) -> tuple[ec.FigureDataPointV1, ...]:
    return tuple(ec.FigureDataPointV1(
        series_id=series_id, category=parts[0] or None,
        x_value=float(parts[1]) if parts[1] else None,
        y_value=float(value) if isinstance(value, int | float) else None,
        interval_lower=float(parts[3]) if parts[3] else None,
        interval_upper=float(parts[4]) if parts[4] else None,
        denominator=int(parts[2]) if parts[2] else None)
        for key, value in sorted(found.get(name, {}).items())
        for parts in [key.split("|") if key.count("|") == 4 else [key, "", "", "", ""]])


def _intervals(result: ec.PrimaryAnalysisResultV1) -> tuple[ec.FigureDataPointV1, ...]:
    return tuple(ec.FigureDataPointV1(
        series_id=result.estimand_family, category=item.contrast_id, x_value=float(index),
        y_value=item.estimate, interval_lower=item.interval_lower,
        interval_upper=item.interval_upper, denominator=item.contributing_counts.get("row"))
        for index, item in enumerate(result.primary_items))


def figure_builders(found: Mapping[str, ec.ValueMap]) -> dict[str, engine.FigureBuilder]:
    # §17 row four: fixed binned outcome summaries, fitted-curve points, the density and
    # continuity summary with its cutoff and bandwidth metadata, and the primary local estimate.
    return {
        "binned_outcome_summary": lambda result: _points(found, "figure_binned_outcome", "binned"),
        "fitted_curve_points": lambda result: _points(found, "figure_fitted_curve", "fitted"),
        "density_continuity_summary": lambda result: _points(
            found, "figure_density_continuity", "density"),
        "primary_contrast_intervals": _intervals}


@dataclass(frozen=True)
class SharpRegressionDiscontinuityAdapter:
    # §19.1: the adapter receives the declared estimator-input view, the frozen plan, its pack,
    # and — for a sensitivity branch — the prespecified parameter delta. Nothing else. `mask` is
    # the committed §6.2 contribution mask every item it reports contributed through.

    mask: ArtifactRef

    def fit(self, view: pl.DataFrame, plan: ec.EstimationPlanV1, pack: EstimationPackV1,
            overrides: ec.ValueMap | None = None) -> engine.AdapterResult:
        # One local polynomial fit at the approved cutoff inside one adapter call, then the §12.3
        # harvest read off that same call. An override is only ever a prespecified branch delta,
        # never a bandwidth, kernel, or cutoff chosen after seeing a result.
        params = dict(pack.parameter_defaults) | dict(plan.estimator_parameters) | dict(
            overrides or {})
        roles = dict(plan.role_columns)
        frame = view.drop_nulls([roles["running_variable"], roles["outcome"]])
        _assignment(frame, roles, params)
        found = _call(frame, roles, plan, params)
        multiplier = _number(params, "bandwidth_multiplier", 1.0)
        if multiplier != 1.0:
            found = _call(frame, roles, plan, params, width=_widths(found, multiplier))
        if min(int(found.N_h[0]), int(found.N_h[1])) < 1:
            raise ec.EstimationError("a cutoff side has no effective support", NO_SUPPORT)
        counts = {"row": frame.height, "unit": frame[roles["unit_identifier"]].n_unique()}
        harvested = harvest(frame, roles, plan, params, found)
        return engine.AdapterResult(
            (_item(plan, params, found, counts, self.mask),),
            harvested | series(frame, roles, params, found,
                               harvested["sorting_and_other_policy_qualification"]), found)

    def multiplicity(self, items: Sequence[ec.PrimaryContrastResultV1],
                     level: float) -> dict[str, ec.ValueMap]:
        # §12 registers one local estimate: no confirmatory family, no adjustment to make.
        return {item.contrast_id: {"p_value": item.p_value, "adjusted_p_value": item.p_value,
                                   "rank": 1} for item in items}

    def figures(self, harvested: Mapping[str, ec.ValueMap]
                ) -> Mapping[str, engine.FigureBuilder]:
        return figure_builders(harvested)

    def visual_evidence(self, builder_id: str) -> str:
        return VISUAL_EVIDENCE.get(builder_id, builder_id)
