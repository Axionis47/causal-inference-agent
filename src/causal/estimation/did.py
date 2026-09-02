# The difference-in-differences estimator adapter (PRD-004 §11): the registered simultaneous
# and staggered profiles the frozen plan selects, the §11.2 event-cell view, the §11.3 diagnostic
# harvest read off the same fits and one polars pivot, the §11.4 prespecified branches, and the
# §17 row-three figure payloads. `pandas` appears only at the pyfixest boundary. This module
# commits nothing, holds no state beyond the mask it was bound to, and never imports another stage.

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import polars as pl
from pyfixest.estimation import feols
from pyfixest.estimation.feols_ import Feols
from scipy import stats  # type: ignore[import-untyped]

from causal.estimation import contracts as ec
from causal.estimation import engine
from causal.estimation.packs import EstimationPackV1
from causal.shared.contracts import ArtifactRef

ADAPTER_VERSION: Final = "difference-in-differences-adapter.v1"
STAGGERED: Final = "staggered"
TERM, ADOPT, REL, COHORT, WEIGHT = "treated_post", "adopt", "rel_time", "cohort_code", "cell_weight"
# The comparison pool carries cohort code 0 and this relative-time sentinel: `ref2` drops its
# interaction terms, so its rows identify the time effects and estimate no treated cell.
POOL, SENTINEL, REFERENCE = 0.0, 1.0e6, -1.0
# The pack's §11.2 refusals; the estimator never answers an unsupported cell with another cohort.
NO_COMPARISON, NO_PRE = "cohort_without_comparison_cell", "missing_required_pre_period"
NOT_PANEL: Final = "repeated_cross_section_under_panel_profile"
# The approved fixed-effect specification under the role whose column absorbs it.
FIXED_EFFECTS: Final[dict[str, str]] = {"unit_and_time": "unit_identifier",
                                        "group_and_time": "group"}
# One §17 row-three visual-evidence family per registered figure-data builder id.
VISUAL_EVIDENCE: Final[dict[str, str]] = {
    "group_time_means": "trends", "event_time_estimates": "event_time_evidence",
    "support_composition_counts": "support_composition", "primary_contrast_intervals":
    "primary_aggregate"}
# `rel_time::<event time>:cohort_code::<cohort>` — one saturated cohort-relative cell.
CELL = re.compile(r"^rel_time::(-?[\d.]+):cohort_code::([\d.]+)$")


def _params(plan: ec.EstimationPlanV1, pack: EstimationPackV1,
            overrides: ec.ValueMap | None) -> tuple[ec.ValueMap, str]:
    # §11.1: the plan names the adoption profile; the pack's registered profile carries its
    # defaults. An override is only ever a prespecified branch delta.
    merged = dict(plan.estimator_parameters) | dict(overrides or {})
    profile = pack.estimator_profile(str(merged.get("adoption_profile_id")))
    return dict(profile.parameter_defaults) | merged, profile.profile_id


def _number(params: ec.ValueMap, key: str, default: float) -> float:
    value = params.get(key)
    return default if value is None or isinstance(value, bool) else float(value)


def _adoption(view: pl.DataFrame, roles: Mapping[str, str]) -> pl.Expr:
    # The approved adoption time per unit, or the first period the approved treatment indicator
    # turns on when the design declared no adoption column. A never-treated unit stays null.
    named = roles.get("adoption_time")
    if named is not None and named in view.columns:
        return pl.when(pl.col(named) > 0).then(pl.col(named)).otherwise(None).alias(ADOPT)
    return pl.when(pl.col(roles["treatment"]).cast(pl.Float64) > 0).then(
        pl.col(roles["time"]).cast(pl.Float64)).otherwise(None).min().over(
        roles["unit_identifier"]).alias(ADOPT)


def _timing(view: pl.DataFrame, roles: Mapping[str, str], params: ec.ValueMap) -> pl.DataFrame:
    # §11.1: the declared view under the dtypes the fits need, then cohort, event time, and the
    # post indicator under the approved anticipation window and comparison cohort. A placebo
    # branch reads the registered shift and contributes only genuinely pre-treatment rows, so
    # its window can carry no true effect.
    time = roles["time"]
    placebo = _number(params, "placebo_shift_periods", 0.0)
    shift = _number(params, "anticipation_periods", 0.0) - placebo
    frame = view.with_columns(
        pl.col(roles[role]).cast(pl.Float64) for role in ("outcome", "time", "adoption_time")
        if roles.get(role) in view.columns)
    frame = frame.with_columns(_adoption(frame, roles))
    if placebo:
        frame = frame.filter(pl.col(ADOPT).is_null() | (pl.col(time) < pl.col(ADOPT)))
    cohorts = sorted(frame[ADOPT].drop_nulls().unique().to_list())
    never = int(frame[ADOPT].is_null().sum())
    approved = str(params.get("comparison_cohort") or "never_treated")
    if not cohorts or (not never and approved == "never_treated"):
        raise ec.EstimationError("no approved comparison cohort has support", NO_COMPARISON)
    # The not-yet-treated pool adds the last adopting cohort, whose untreated periods identify
    # the earlier cohorts; never-treated units always belong to it.
    pooled = pl.col(ADOPT).is_null()
    if approved != "never_treated":
        pooled = pooled | (pl.col(ADOPT) == cohorts[-1])
    effective = pl.col(ADOPT) - shift
    return frame.with_columns(
        pl.when(pooled).then(pl.lit(POOL)).otherwise(effective).alias(COHORT),
        pl.when(pooled).then(pl.lit(SENTINEL)).otherwise(
            pl.col(time) - effective).alias(REL)).with_columns(
        (~pooled & (pl.col(REL) >= 0.0)).cast(pl.Float64).alias(TERM))


def _panel(frame: pl.DataFrame, roles: Mapping[str, str]) -> bool:
    # §11.1: a declared panel follows the same units across periods; a repeated cross section
    # gives every observation its own unit, and the staggered profile will not pretend otherwise.
    return frame.height > frame[roles["unit_identifier"]].n_unique()


def _contributing(frame: pl.DataFrame, roles: Mapping[str, str], params: ec.ValueMap
                  ) -> pl.DataFrame:
    # §11.2: the rows contributing to this calculation. The balanced-panel branch is a mask over
    # the frozen frame, never a deletion of a period, cohort, or group from it.
    kept, time = frame.drop_nulls(roles["outcome"]), roles["time"]
    if params.get("mask_rule_id") == "balanced_panel_cell":
        kept = kept.filter(pl.col(time).n_unique().over(
            roles["unit_identifier"]) == kept[time].n_unique())
    return kept


def _cells(frame: pl.DataFrame, roles: Mapping[str, str]) -> pl.DataFrame:
    # The one polars pivot behind §11.2 support and §11.3 composition: treated and comparison
    # observations and units per cohort-relative cell.
    return frame.group_by(COHORT, REL).agg(
        pl.len().alias("rows"), pl.col(roles["unit_identifier"]).n_unique().alias("units"),
        pl.col(TERM).sum().alias("treated")).sort(COHORT, REL)


def _boundary(frame: pl.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    # The one pyfixest boundary: only the columns a formula or a covariance names, handed over as
    # numpy arrays because the pinned stack carries no arrow bridge.
    return pd.DataFrame({name: frame[name].to_numpy() for name in sorted(set(names))})


def _vcov(roles: Mapping[str, str]) -> dict[str, str]:
    # §11.3: cluster-robust over the approved cluster, or over the unit when none was approved.
    return {"CRV1": roles.get("cluster") or roles["unit_identifier"]}


def _simultaneous(frame: pl.DataFrame, roles: Mapping[str, str],
                  plan: ec.EstimationPlanV1, params: ec.ValueMap) -> ec.ValueMap:
    # §11.1: the registered common-adoption estimator — the approved group-by-post indicator
    # under the approved fixed effects, cell-size weighting, and cluster specification.
    absorbed = roles[FIXED_EFFECTS.get(str(params.get("fixed_effects")), "unit_identifier")]
    kept = frame.with_columns((
        pl.len().over(roles.get("group") or COHORT, roles["time"]).cast(pl.Float64)
        if params.get("weighting") == "cell_size" else pl.lit(1.0)).alias(WEIGHT))
    names = (roles["outcome"], TERM, WEIGHT, absorbed, roles["time"], _vcov(roles)["CRV1"])
    fitted = cast(Feols, feols(f"{roles['outcome']} ~ {TERM} | {absorbed} + {roles['time']}",
                               data=_boundary(kept, names), vcov=_vcov(roles), weights=WEIGHT))
    lower, upper = (float(bound) for bound in
                    fitted.confint(alpha=1.0 - plan.confidence_level).loc[TERM])
    return {"estimate": float(fitted.coef()[TERM]), "standard_error": float(fitted.se()[TERM]),
            "interval_lower": lower, "interval_upper": upper,
            "p_value": float(fitted.pvalue()[TERM])}


def _event_fit(frame: pl.DataFrame, roles: Mapping[str, str]) -> Feols:
    # The saturated Sun–Abraham event study: one coefficient per cohort-relative cell against the
    # approved reference period, with the approved comparison pool dropped by `ref2`.
    names = (roles["outcome"], REL, COHORT, roles["unit_identifier"], roles["time"],
             _vcov(roles)["CRV1"])
    formula = (f"{roles['outcome']} ~ i({REL}, {COHORT}, ref={REFERENCE}, ref2={POOL})"
               f" | {roles['unit_identifier']} + {roles['time']}")
    return cast(Feols, feols(formula, data=_boundary(frame, names), vcov=_vcov(roles)))


def _coefficients(fitted: Feols) -> list[tuple[int, float, float]]:
    # Every fitted cell as (position, event time, cohort); a cell pyfixest dropped for
    # collinearity is simply absent, and no aggregate may weight it.
    return [(index, float(row.group(1)), float(row.group(2)))
            for index, name in enumerate(fitted._coefnames)
            if (row := CELL.match(str(name))) is not None]


def _shares(cells: pl.DataFrame, params: ec.ValueMap) -> dict[tuple[float, float], float]:
    # §11.1: the registered cohort weights — each cell's share of the treated observations inside
    # the approved event window, fixed before any estimate is read.
    high = _number(params, "event_window_lags", 3.0)
    kept = cells.filter((pl.col(REL) >= 0.0) & (pl.col(REL) <= high) & (pl.col("treated") > 0))
    total = float(kept["treated"].sum() or 1.0)
    return {(float(rel), float(cohort)): float(treated) / total
            for cohort, rel, treated in kept.select(COHORT, REL, "treated").iter_rows()}


def _combination(fitted: Feols, weights: Mapping[tuple[float, float], float],
                 aggregation: str) -> np.ndarray[Any, np.dtype[np.float64]]:
    # The approved aggregation to one primary quantity: cohort-weighted over the whole window, or
    # equal weight per event time with the cohort shares inside it.
    row = np.zeros(len(fitted._coefnames), dtype=np.float64)
    cells = [(index, key) for index, rel, cohort in _coefficients(fitted)
             if (key := (rel, cohort)) in weights]
    inside = {rel: sum(share for (seen, _), share in weights.items() if seen == rel)
              for rel, _ in (key for _, key in cells)}
    for index, key in cells:
        row[index] = (weights[key] / (inside[key[0]] * len(inside))
                      if aggregation == "equal_weighted_event_time" else weights[key])
    total = float(row.sum())
    return row / total if total else row


def _lincomb(fitted: Feols, row: np.ndarray[Any, np.dtype[np.float64]],
             level: float) -> ec.ValueMap:
    # One linear combination of the fitted cells under the fit's own cluster-robust covariance:
    # pyfixest publishes no aggregation for this profile, so the delta method is applied here and
    # nowhere else. Nothing about the coefficients or their covariance is recomputed.
    estimate = float(row @ fitted._beta_hat)
    error = float(np.sqrt(max(float(row @ fitted._vcov @ row), 0.0)))
    half = float(stats.norm.ppf(0.5 + level / 2.0)) * error
    return {"estimate": estimate, "standard_error": error, "interval_lower": estimate - half,
            "interval_upper": estimate + half,
            "p_value": 2.0 * float(stats.norm.sf(abs(estimate) / error)) if error else 1.0}


def _item(plan: ec.EstimationPlanV1, params: ec.ValueMap, found: ec.ValueMap,
          quantities: ec.ValueMap, counts: dict[str, int],
          mask: ArtifactRef) -> ec.PrimaryContrastResultV1:
    # The one primary aggregate this method reports, at the plan's confidence level and under the
    # plan's uncertainty method (§6.3, §11.1).
    return ec.PrimaryContrastResultV1(
        **cast(Any, found), contrast_id=plan.contrast_ids[0], estimand_id=plan.estimand_id,
        estimand_label="average treatment effect on the treated, adopters against comparison",
        estimate_units=plan.outcome_scale, comparator_id=plan.comparator_id,
        effect_direction="treated_minus_comparison", confidence_level=plan.confidence_level,
        uncertainty_method=plan.uncertainty_method,
        finite_sample_correction=plan.finite_sample_correction, contributing_counts=counts,
        contribution_mask=mask, estimator_id=plan.estimator_id,
        estimator_version=plan.estimator_version, estimator_parameters=dict(params),
        adapter_version=ADAPTER_VERSION, convergence="converged", method_quantities=quantities)


def _leads(fitted: Feols, params: ec.ValueMap) -> ec.ValueMap:
    # §11.3: every approved lead estimate and the joint prespecified pre-period Wald test, both
    # read off the same fit. A non-significant test does not prove parallel trends.
    horizon = -_number(params, "event_window_leads", 3.0) + REFERENCE
    cells = [(index, rel, cohort) for index, rel, cohort in _coefficients(fitted)
             if horizon <= rel < REFERENCE]
    found: ec.ValueMap = {"leads": len(cells)} | {
        f"lead_{rel:g}_cohort_{cohort:g}": float(fitted._beta_hat[index])
        for index, rel, cohort in cells}
    if not cells:
        return found
    selector = np.eye(len(fitted._coefnames))[[index for index, _, _ in cells]]
    found["largest_lead_estimate"] = max(abs(float(fitted._beta_hat[i])) for i, _, _ in cells)
    tested = cast(Any, fitted.wald_test)(R=selector, q=np.zeros(len(cells)))
    found["joint_pre_period_p_value"] = float(tested["pvalue"])
    return found


def _event_times(fitted: Feols, level: float) -> ec.ValueMap:
    # §17 row three: the cohort-relative estimate and interval at each event time, read off the
    # fit that already produced them, keyed `category|x|denominator|lower|upper`.
    bounds = fitted.confint(alpha=1.0 - level)
    return {f"cohort_{cohort:g}|{rel:g}||{low:.6f}|{high:.6f}": float(fitted._beta_hat[index])
            for index, rel, cohort in _coefficients(fitted) if abs(rel) < SENTINEL
            for low, high in [bounds.iloc[index].to_list()]}


def _support(cells: pl.DataFrame, frame: pl.DataFrame, roles: Mapping[str, str],
             params: ec.ValueMap) -> dict[str, ec.ValueMap]:
    # §11.2 and §11.3: cohort-relative support, pre and post placement, composition change across
    # time, the cluster count the covariance rests on, and the aggregate's weight spread.
    unit, time = roles["unit_identifier"], roles["time"]
    treated, periods = cells.filter(pl.col("treated") > 0), frame[time].n_unique()
    adopting, weights = cells.filter(pl.col(COHORT) > POOL), _shares(cells, params)
    balanced = int(frame.group_by(unit).agg(
        pl.col(time).n_unique().alias("seen")).filter(pl.col("seen") == periods).height)
    units, cluster = frame[unit].n_unique(), roles.get("cluster") or unit
    return {
        "group_time_cohort_support": {
            "cell_units": int(cast(int, treated["units"].min() or 0)), "cells": cells.height,
            "treated_cells": treated.height, "cohorts": int(treated[COHORT].n_unique())},
        "pre_and_post_period_placement": {
            "pre_periods": int(adopting.filter(pl.col(REL) < 0)[REL].n_unique()),
            "post_periods": int(adopting.filter(
                (pl.col(REL) >= 0) & (pl.col(REL) < SENTINEL))[REL].n_unique()),
            "periods": periods, "reference_period": REFERENCE},
        "attrition_composition_change": {"balanced_units": balanced, "units": units,
                                         "composition_change_share": 1.0 - (
                                             balanced / float(units or 1))},
        "cluster_covariance_adequacy": {
            "clusters": frame[cluster].n_unique(), "contributing_rows": frame.height},
        "aggregate_weight_sensitivity": {
            "weight_cells": len(weights),
            "largest_cell_weight": max(weights.values(), default=0.0)}}


def harvest(frame: pl.DataFrame, cells: pl.DataFrame, roles: Mapping[str, str],
            plan: ec.EstimationPlanV1, params: ec.ValueMap,
            fitted: Feols | None) -> dict[str, ec.ValueMap]:
    # The §11.3 required diagnostics, harvested from the frozen frame, the one pivot above, and
    # the fits already run — plus the frozen §17 series the figure builders wrap.
    unit, time = roles["unit_identifier"], roles["time"]
    pairs, found = frame.select(unit, time).n_unique(), _support(cells, frame, roles, params)
    comparison = str(params.get("comparison_cohort") or "never_treated")
    anticipation = _number(params, "anticipation_periods", 0.0)
    found["unit_period_schema_reconciliation"] = {
        "units": frame[unit].n_unique(), "periods": frame[time].n_unique(), "rows": frame.height,
        "unit_periods": pairs, "duplicate_unit_periods": frame.height - pairs,
        "structure": "panel" if _panel(frame, roles) else "repeated_cross_section"}
    found["treatment_timing_anticipation"] = {
        "anticipation_periods": anticipation, "comparison_cohort": comparison, "cohorts": int(
            cells.filter(pl.col(COHORT) > POOL)[COHORT].n_unique())}
    found["event_study_pre_period_test"] = _leads(fitted, params) if fitted is not None else {}
    found["concurrent_event_qualification"] = {
        "carried_rule_ids": len(plan.numerical_failure_rule_ids), "comparison_cohort": comparison}
    found["reference_period_numerical_integrity"] = {
        "reference_rows": int(frame.filter(pl.col(REL) == REFERENCE).height),
        "fitted_cells": len(_coefficients(fitted)) if fitted is not None else 0,
        "converged": fitted is not None, "aggregation": str(params.get("aggregation") or "")}
    # §17 row three, keyed `category|x|denominator|lower|upper`, so a builder below wraps frozen
    # values and computes nothing of its own.
    means = frame.group_by(roles.get("group") or COHORT, time).agg(
        pl.col(roles["outcome"]).mean().alias("mean"), pl.len().alias("rows")).sort("mean")
    found["figure_group_time_means"] = {f"{name}|{when:g}|{rows}||": float(mean or 0.0)
                                        for name, when, mean, rows in means.iter_rows()}
    found["figure_event_time_estimates"] = (_event_times(fitted, plan.confidence_level)
                                            if fitted is not None else {})
    found["figure_support_counts"] = {
        f"cohort_{cohort:g}|{rel:g}|{rows}||": float(treated)
        for cohort, rel, rows, _, treated in cells.filter(pl.col(REL) < SENTINEL).iter_rows()}
    return found


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
        for parts in [key.split("|")])


def _intervals(result: ec.PrimaryAnalysisResultV1) -> tuple[ec.FigureDataPointV1, ...]:
    return tuple(ec.FigureDataPointV1(
        series_id=result.estimand_family, category=item.contrast_id, x_value=float(index),
        y_value=item.estimate, interval_lower=item.interval_lower,
        interval_upper=item.interval_upper, denominator=item.contributing_counts.get("row"))
        for index, item in enumerate(result.primary_items))


def figure_builders(found: Mapping[str, ec.ValueMap]) -> dict[str, engine.FigureBuilder]:
    # §17 row three: group-time means, event-time estimates and intervals, support and
    # composition counts, and the primary aggregate. Each builder wraps values this run froze.
    return {
        "group_time_means": lambda result: _points(found, "figure_group_time_means", "group_time"),
        "event_time_estimates": lambda result: _points(
            found, "figure_event_time_estimates", "event_time"),
        "support_composition_counts": lambda result: tuple(point.model_copy(update={
            "series_id": str(point.category), "category": str(point.x_value)})
            for point in _points(found, "figure_support_counts", "support")),
        "primary_contrast_intervals": _intervals}


@dataclass(frozen=True)
class DifferenceInDifferencesAdapter:
    # §19.1: the adapter receives the declared estimator-input view, the frozen plan, its pack,
    # and — for a sensitivity branch — the prespecified parameter delta. Nothing else. `mask` is
    # the committed §6.2 contribution mask every item it reports contributed through.

    mask: ArtifactRef

    def fit(self, view: pl.DataFrame, plan: ec.EstimationPlanV1, pack: EstimationPackV1,
            overrides: ec.ValueMap | None = None) -> engine.AdapterResult:
        # One primary aggregate inside one adapter call, then the §11.3 harvest read off the same
        # frozen frame. The plan's profile decides the estimator; an override is only ever a
        # prespecified branch delta, never a choice made after seeing a result.
        params, profile = _params(plan, pack, overrides)
        roles = dict(plan.role_columns)
        frame = _contributing(_timing(view, roles, params), roles, params)
        # §11.1: the staggered profile requires a declared panel, and both registered profiles
        # absorb unit fixed effects, so a repeated cross section is refused here rather than
        # fitted as though its units persisted.
        if not _panel(frame, roles):
            raise ec.EstimationError("this profile requires a declared panel", NOT_PANEL)
        if not frame.filter(pl.col(REL) == REFERENCE).height:
            raise ec.EstimationError("the approved reference period has no support", NO_PRE)
        return self._fitted(frame, roles, plan, params, profile)

    def _fitted(self, frame: pl.DataFrame, roles: Mapping[str, str], plan: ec.EstimationPlanV1,
                params: ec.ValueMap, profile: str) -> engine.AdapterResult:
        # The registered profile's one fit, its §11.4 influence summary when the branch asks for
        # one, and the harvest both profiles share.
        cells, fitted = _cells(frame, roles), _event_fit(frame, roles)
        aggregation = str(params.get("aggregation") or "cohort_weighted_overall")
        quantities: ec.ValueMap = {
            "adoption_profile_id": profile, "fitted_cells": len(_coefficients(fitted)),
            "comparison_cohort": str(params.get("comparison_cohort") or "never_treated"),
            "aggregation": aggregation, "effective_rows": frame.height}
        if profile == STAGGERED:
            row = _combination(fitted, _shares(cells, params), aggregation)
            found = _lincomb(fitted, row, plan.confidence_level)
            quantities["aggregated_cells"] = int(np.count_nonzero(row))
        else:
            found = _simultaneous(frame, roles, plan, params)
        counts = {"row": frame.height, "unit": frame[roles["unit_identifier"]].n_unique()}
        item = _item(plan, params, found, quantities, counts, self.mask)
        if params.get("influence_summary") == "leave_one_cohort_out":
            item = self._influence(frame, roles, plan, params, profile, item)
        return engine.AdapterResult(
            (item,), harvest(frame, cells, roles, plan, params, fitted), fitted)

    def _influence(self, frame: pl.DataFrame, roles: Mapping[str, str],
                   plan: ec.EstimationPlanV1, params: ec.ValueMap, profile: str,
                   base: ec.PrimaryContrastResultV1) -> ec.PrimaryContrastResultV1:
        # §11.4: refit with each treated cohort's contribution withheld in turn and report the
        # most influential omission. The branch summarizes the frozen specification and competes
        # with nothing; the withheld cohort stays in the frozen prepared frame.
        delta = dict(params) | {"influence_summary": "none"}
        worst, moved = base, 0.0
        for cohort in sorted(frame.filter(pl.col(COHORT) > POOL)[COHORT].unique().to_list()):
            kept = frame.filter(pl.col(COHORT) != cohort)
            try:
                found = self._fitted(kept, roles, plan, delta, profile).items[0]
            except (ec.EstimationError, ValueError, KeyError):
                continue
            gap = abs(found.estimate - base.estimate)
            worst, moved = (found, gap) if gap > moved else (worst, moved)
        return worst.model_copy(update={"estimator_parameters": dict(params)})

    def multiplicity(self, items: Sequence[ec.PrimaryContrastResultV1],
                     level: float) -> dict[str, ec.ValueMap]:
        # §11 registers one primary aggregate: no confirmatory family, no adjustment to make.
        return {item.contrast_id: {"p_value": item.p_value, "adjusted_p_value": item.p_value,
                                   "rank": 1} for item in items}

    def figures(self, harvested: Mapping[str, ec.ValueMap]
                ) -> Mapping[str, engine.FigureBuilder]:
        return figure_builders(harvested)

    def visual_evidence(self, builder_id: str) -> str:
        return VISUAL_EVIDENCE.get(builder_id, builder_id)
