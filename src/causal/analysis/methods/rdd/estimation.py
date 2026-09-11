# The sharp regression-discontinuity estimator adapter (PRD-004 §12): one versioned `rdrobust`
# call at the exact approved cutoff, the §12.2 assignment and bandwidth view, the §12.3 diagnostic
# harvest read off that call plus the registered `rddensity` and covariate-continuity adapters,
# the §12.4 prespecified branches, and scientific support summaries at a fixed registered
# binning. This module commits nothing, holds no state beyond the mask it was bound to, and never
# imports another stage.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

import numpy as np
import polars as pl
from rdrobust import rdplot, rdrobust  # type: ignore[import-untyped]

from causal.analysis.common import legacy_engine as engine
from causal.analysis.integration import contracts as ec
from causal.analysis.integration.packs import EstimationPackV1
from causal.analysis.methods.rdd import diagnostics
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
MISSING_ASSIGNMENT: Final = "missing_approved_assignment"
NO_APPROVED_COVARIATES: Final = "no_approved_predetermined_covariate"
# §17 row four: the registered binning is fixed here, before any result is seen, and the fitted
# curve is reported at this many evenly spaced points.
BINS, CURVE = 20, 40


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
    keys = ("cutoff", "assignment_direction", "treated_value", "comparator_value")
    if roles.get("treatment") not in frame.columns or any(key not in params for key in keys) or (
            params["assignment_direction"] not in {"above", "below"}):
        raise ec.EstimationError("the approved sharp assignment is incomplete", MISSING_ASSIGNMENT)
    above = pl.col(roles["running_variable"]).cast(pl.Float64) >= _number(params, "cutoff", 0.0)
    if params["assignment_direction"] == "below":
        above = ~above
    expected = pl.when(above).then(pl.lit(str(params["treated_value"]))).otherwise(pl.lit(str(params["comparator_value"])))
    broken = int(frame.filter(pl.col(roles["treatment"]).cast(pl.String) != expected).height)
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
    if params.get("covariate_adjustment") and not adjusted:
        raise ec.EstimationError(
            "covariate adjustment is unavailable without an approved predetermined covariate",
            NO_APPROVED_COVARIATES)
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


@dataclass(frozen=True)
class SharpRegressionDiscontinuityAdapter:
    # §19.1: the adapter receives the declared estimator-input view, the frozen plan, its pack,
    # and — for a sensitivity branch — the prespecified parameter delta. Nothing else. `mask` is
    # the committed §6.2 contribution mask every item it reports contributed through.

    mask: ArtifactRef

    def fit(self, view: pl.DataFrame, plan: ec.EstimationPlanV1, pack: EstimationPackV1,
            overrides: ec.ValueMap | None = None, *,
            compute_diagnostics: bool = True) -> engine.AdapterResult:
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
        if not compute_diagnostics:
            return engine.AdapterResult(
                (_item(plan, params, found, counts, self.mask),), {}, found)
        harvested = diagnostics.harvest(frame, roles, plan, params, found)
        return engine.AdapterResult(
            (_item(plan, params, found, counts, self.mask),),
            harvested | series(frame, roles, params, found,
                               harvested["sorting_and_other_policy_qualification"]), found)

    def multiplicity(self, items: Sequence[ec.PrimaryContrastResultV1],
                     level: float) -> dict[str, ec.ValueMap]:
        # §12 registers one local estimate: no confirmatory family, no adjustment to make.
        return {item.contrast_id: {"p_value": item.p_value, "adjusted_p_value": item.p_value,
                                   "rank": 1} for item in items}

