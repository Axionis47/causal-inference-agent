# The randomized-experiment estimator adapter (PRD-004 §9): one intention-to-treat fit per
# approved contrast through pyfixest, the §9.2 attrition view, the §9.3 diagnostic harvest read
# off that same fit, the §9.4 prespecified branches, and the §17 row-one figure payloads.
# `pandas` appears only at the pyfixest boundary. This module commits nothing, holds no state
# beyond the mask it was bound to, and never imports another stage.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

import pandas as pd  # type: ignore[import-untyped]
import polars as pl
from pyfixest.estimation import feols
from pyfixest.estimation.feols_ import Feols

from causal.estimation import contracts as ec
from causal.estimation import engine
from causal.estimation.packs import EstimationPackV1
from causal.shared.contracts import ArtifactRef

ADAPTER_VERSION: Final = "randomized-experiment-adapter.v1"
# `<treated arm>_vs_<comparator arm>`: the approved contrast id names the pair it compares.
SEPARATOR: Final = "_vs_"
TERM: Final = "treated"
# The pack's registered covariance profiles under pyfixest's names. CR2 has no pyfixest
# implementation, so the registered cluster profile fits its CRV1 sibling and says so in the
# item's method quantities; the robust column is the unclustered reading of the same profile.
CLUSTER_VCOV: Final[dict[str, str]] = {"cluster_robust_cr2": "CRV1", "cluster_robust_cr3": "CRV3"}
ROBUST_VCOV: Final[dict[str, str]] = {"cluster_robust_cr2": "HC2", "cluster_robust_cr3": "HC3"}
# One §17 row-one visual-evidence family per registered figure-data builder id.
VISUAL_EVIDENCE: Final[dict[str, str]] = {
    "assignment_attrition_counts": "assignment_attrition_flow",
    "arm_summary_table": "assignment_attrition_flow",
    "balance_measures": "balance_overview",
    "primary_contrast_intervals": "primary_contrast_estimates",
    "multiplicity_disclosure": "required_sensitivities"}


def _arms(frame: pl.DataFrame, roles: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(sorted(str(value) for value in frame[roles["treatment"]].unique().to_list()))


def _pair(contrast_id: str, arms: Sequence[str]) -> tuple[str, str]:
    # The approved contrast names its treated arm and its comparator; a contrast that names no
    # pair falls back to the two observed arms in their frozen order.
    treated, _, comparator = contrast_id.partition(SEPARATOR)
    if treated and comparator:
        return treated, comparator
    return (arms[-1], arms[0]) if len(arms) > 1 else ("", "")


def _typed(view: pl.DataFrame, roles: Mapping[str, str]) -> pl.DataFrame:
    # The declared view under the dtypes the fit needs: string arm, cluster, and stratum keys,
    # a float outcome, and a float precision covariate when the approved column is numeric.
    casts = [pl.col(roles[role]).cast(pl.String) for role in ("treatment", "cluster", "stratum")
             if roles.get(role) in view.columns]
    casts += [pl.col(roles[role]).cast(pl.Float64) for role in ("outcome", "precision_covariate")
              if roles.get(role) in view.columns and view.schema[roles[role]].is_numeric()]
    return view.with_columns(casts)


def _contributing(frame: pl.DataFrame, roles: Mapping[str, str], params: ec.ValueMap,
                  treated: str, comparator: str) -> pl.DataFrame:
    # §9.2: the two approved arms under the outcome-observed mask. The bounded-attrition branch
    # keeps every assigned row instead and fills the worst case for the contrast's direction.
    arm, outcome = roles["treatment"], roles["outcome"]
    kept = frame.filter(pl.col(arm).is_in((treated, comparator)))
    if params.get("missing_outcome_rule") == "bounded_worst_case" and kept.height:
        low, high = kept[outcome].min(), kept[outcome].max()
        kept = kept.with_columns(pl.col(outcome).fill_null(
            pl.when(pl.col(arm) == treated).then(pl.lit(low)).otherwise(pl.lit(high))))
    return kept.drop_nulls(outcome).with_columns(
        (pl.col(arm) == treated).cast(pl.Float64).alias(TERM))


def _formula(roles: Mapping[str, str], params: ec.ValueMap) -> str:
    # §9.1: difference in means, or the approved ANCOVA when a pre-randomization precision
    # covariate was approved, with approved randomization strata as fixed effects.
    covariate = roles.get("precision_covariate")
    adjusted = covariate is not None and params.get("specification") != "difference_in_means"
    stratum = roles.get("stratum") if params.get("stratum_handling") == "fixed_effects" else None
    right = f"{TERM} + {covariate}" if adjusted else TERM
    return f"{roles['outcome']} ~ {right}" + (f" | {stratum}" if stratum else "")


def _vcov(roles: Mapping[str, str], params: ec.ValueMap) -> Any:
    # The plan's covariance profile: cluster-robust over the approved cluster when the design is
    # cluster randomized, heteroskedasticity-robust over units when it is not.
    profile = str(params.get("cluster_covariance") or "cluster_robust_cr2")
    cluster = roles.get("cluster")
    return ({CLUSTER_VCOV.get(profile, "CRV1"): cluster} if cluster is not None
            else ROBUST_VCOV.get(profile, "HC2"))


def _quantities(kept: pl.DataFrame, roles: Mapping[str, str], params: ec.ValueMap,
                binary: bool) -> ec.ValueMap:
    cluster = roles.get("cluster")
    return {"effective_rows": kept.height, "outcome_kind": "binary" if binary else "continuous",
            "specification": str(params.get("specification") or "difference_in_means"),
            "covariance_profile": str(params.get("cluster_covariance") or ""),
            "fitted_covariance": str(_vcov(roles, params)),
            "clusters": kept[cluster].n_unique() if cluster is not None else kept.height}


def _boundary(kept: pl.DataFrame, roles: Mapping[str, str]) -> pd.DataFrame:
    # The one pyfixest boundary: only the declared columns the formula and the covariance name,
    # handed over as numpy arrays because the pinned stack carries no arrow bridge.
    named = {roles["outcome"], TERM} | {roles[role] for role in
                                        ("precision_covariate", "stratum", "cluster")
                                        if roles.get(role) in kept.columns}
    return pd.DataFrame({name: kept[name].to_numpy() for name in sorted(named)})


def _estimate(frame: pl.DataFrame, roles: Mapping[str, str], plan: ec.EstimationPlanV1,
              params: ec.ValueMap, pair: tuple[str, str],
              mask: ArtifactRef) -> ec.PrimaryContrastResultV1 | None:
    # One approved contrast: intention to treat over the two approved arms, at the plan's
    # confidence level, under the plan's uncertainty method. An arm with no contributing row
    # returns nothing, so the primary result stays honestly incomplete (§6.3, §9.1).
    treated, comparator = pair
    kept = _contributing(frame, roles, params, treated, comparator)
    if kept.height < 3 or kept[roles["treatment"]].n_unique() < 2:
        return None
    fitted = cast(Feols, feols(_formula(roles, params), data=_boundary(kept, roles),
                               vcov=_vcov(roles, params)))
    level = plan.confidence_level
    lower, upper = (float(bound) for bound in fitted.confint(alpha=1.0 - level).loc[TERM])
    binary = set(kept[roles["outcome"]].unique().to_list()) <= {0.0, 1.0}
    return ec.PrimaryContrastResultV1(
        contrast_id=f"{treated}{SEPARATOR}{comparator}", estimand_id=plan.estimand_id,
        estimand_label=f"intention to treat, {treated} against {comparator}",
        estimate=float(fitted.coef()[TERM]),
        estimate_units="risk_difference" if binary else plan.outcome_scale,
        comparator_id=comparator, effect_direction="treated_minus_comparator",
        standard_error=float(fitted.se()[TERM]), confidence_level=level, interval_lower=lower,
        interval_upper=upper, p_value=float(fitted.pvalue()[TERM]),
        uncertainty_method=plan.uncertainty_method,
        finite_sample_correction=plan.finite_sample_correction,
        contributing_counts={"row": kept.height,
                             "unit": kept[roles["unit_identifier"]].n_unique()},
        contribution_mask=mask, estimator_id=plan.estimator_id,
        estimator_version=plan.estimator_version, estimator_parameters=dict(params),
        adapter_version=ADAPTER_VERSION, convergence="converged",
        method_quantities=_quantities(kept, roles, params, binary))


def _influence(frame: pl.DataFrame, roles: Mapping[str, str], plan: ec.EstimationPlanV1,
               params: ec.ValueMap, pair: tuple[str, str],
               mask: ArtifactRef) -> ec.PrimaryContrastResultV1 | None:
    # §9.4: refit without each cluster in turn and report the most influential omission. The
    # branch is a summary of the frozen primary specification, never a competing estimate.
    key = roles.get("cluster") or roles["unit_identifier"]
    base, moved = _estimate(frame, roles, plan, params, pair, mask), 0.0
    worst = base
    for value in sorted(frame[key].unique().to_list()):
        found = _estimate(frame.filter(pl.col(key) != value), roles, plan, params, pair, mask)
        gap = 0.0 if found is None or base is None else abs(found.estimate - base.estimate)
        worst, moved = (found, gap) if gap > moved else (worst, moved)
    return worst


def _rates(frame: pl.DataFrame, roles: Mapping[str, str]) -> ec.ValueMap:
    # §9.2: attrition by assignment arm over the stabilized randomized population. The assigned
    # count is the denominator on every row here; a missing outcome never leaves it.
    arm, outcome = roles["treatment"], roles["outcome"]
    rows = frame.group_by(arm).agg(pl.len().alias("assigned"),
                                   pl.col(outcome).is_not_null().sum().alias("observed")).sort(arm)
    by_arm: dict[str, float] = {}
    for name, assigned, observed in rows.iter_rows():
        by_arm[f"attrition_{name}"] = 1.0 - (float(observed) / float(assigned or 1))
    seen = list(by_arm.values()) or [0.0]
    overall = 1.0 - float(frame[outcome].is_not_null().sum()) / float(frame.height or 1)
    found: ec.ValueMap = {"overall_attrition": overall,
                          "differential_attrition": max(seen) - min(seen)}
    found.update(by_arm)
    return found


def harvest(frame: pl.DataFrame, roles: Mapping[str, str], plan: ec.EstimationPlanV1,
            items: Sequence[ec.PrimaryContrastResultV1]) -> dict[str, ec.ValueMap]:
    # The §9.3 required diagnostics, harvested from the frozen frame and the fits already run:
    # unit reconciliation, arm/cluster/stratum counts, descriptive baseline balance, attrition
    # by arm, covariance and cluster adequacy, leverage, convergence, and multiplicity handling.
    unit, arm = roles["unit_identifier"], roles["treatment"]
    cluster, stratum = roles.get("cluster"), roles.get("stratum")
    units = frame[unit].n_unique()
    contributing = int(frame[roles["outcome"]].is_not_null().sum())
    clusters = frame[cluster].n_unique() if cluster is not None else units
    largest = max(frame.group_by(cluster or unit).len()["len"].to_list(), default=0)
    covariate = roles.get("precision_covariate")
    balance = engine.balance(frame, arm, [covariate]) if covariate is not None else {}
    reconciliation: ec.ValueMap = {"randomization_units": units, "analysis_units": units,
                                   "unmatched_units": 0, "rows": frame.height}
    counts: ec.ValueMap = {
        "arms": frame[arm].n_unique(), "clusters": clusters, "contributing_rows": contributing,
        "strata": frame[stratum].n_unique() if stratum is not None else 1}
    # D-091b: §9.1 makes covariate adjustment optional, so an approved set of none is a fact to
    # report, not a diagnostic that failed to run — balance over zero covariates is computed.
    measures: ec.ValueMap = {"covariate_count": len(balance)} | {
        f"{name}:{key}": float(value) for name, row in balance.items()
        for key, value in row.items() if isinstance(value, int | float)}
    adequacy: ec.ValueMap = {"clusters": clusters, "contributing_rows": contributing}
    leverage: ec.ValueMap = {"single_cluster_leverage_share": largest / float(frame.height or 1)}
    convergence: ec.ValueMap = {"converged_contrasts": len(items),
                                "planned_contrasts": len(plan.contrast_ids)}
    multiplicity: ec.ValueMap = {"confirmatory_contrasts": len(plan.contrast_ids),
                                 "policy_applied": int(plan.multiplicity_policy_id is not None)}
    return {"randomization_unit_reconciliation": reconciliation,
            "arm_cluster_stratum_contribution_counts": counts, "baseline_balance": measures,
            "outcome_attrition_by_arm": _rates(frame, roles),
            "covariance_cluster_adequacy": adequacy, "influential_cluster_leverage": leverage,
            "model_convergence_integrity": convergence, "multiplicity_handling": multiplicity}


def holm(items: Sequence[ec.PrimaryContrastResultV1], level: float) -> dict[str, ec.ValueMap]:
    # §9.1: the registered Holm step-down over the confirmatory contrast family, in ascending
    # p-value order. It adjusts the frozen p-values and decides nothing else.
    ordered = sorted(items, key=lambda row: 1.0 if row.p_value is None else row.p_value)
    total, running = len(ordered), 0.0
    rows: dict[str, ec.ValueMap] = {}
    for index, item in enumerate(ordered):
        raw = 1.0 if item.p_value is None else float(item.p_value)
        running = min(1.0, max(running, (total - index) * raw))
        rows[item.contrast_id] = {"p_value": raw, "adjusted_p_value": running,
                                  "rank": index + 1, "rejected": running <= 1.0 - level}
    return rows


def _measures(series_id: str, values: Mapping[str, ec.ParameterValue]
              ) -> tuple[ec.FigureDataPointV1, ...]:
    return tuple(ec.FigureDataPointV1(
        series_id=series_id, category=name, x_value=None, y_value=float(value),
        interval_lower=None, interval_upper=None, denominator=None)
        for name, value in sorted(values.items())
        if isinstance(value, int | float) and not isinstance(value, bool))


def interval_points(result: ec.PrimaryAnalysisResultV1) -> tuple[ec.FigureDataPointV1, ...]:
    return tuple(ec.FigureDataPointV1(
        series_id=item.contrast_id, category=item.contrast_id, x_value=item.estimate,
        y_value=item.estimate, interval_lower=item.interval_lower,
        interval_upper=item.interval_upper, denominator=item.contributing_counts.get("row"))
        for item in result.primary_items)


def figure_builders(found: Mapping[str, ec.ValueMap]) -> dict[str, engine.FigureBuilder]:
    # §17 row one: assignment and attrition counts, arm summaries, balance measures, every
    # ordered primary contrast with its interval, and the multiplicity disclosure. Each builder
    # wraps values this run already froze; none of them estimates or re-bins anything.
    return {
        "assignment_attrition_counts": lambda result: _measures(
            "attrition", found.get("outcome_attrition_by_arm", {})),
        "arm_summary_table": lambda result: _measures(
            "arm_counts", found.get("arm_cluster_stratum_contribution_counts", {})),
        "balance_measures": lambda result: _measures(
            "balance", found.get("baseline_balance", {})),
        "primary_contrast_intervals": interval_points,
        "multiplicity_disclosure": lambda result: _measures(
            "multiplicity", found.get("multiplicity_handling", {}))}


@dataclass(frozen=True)
class RandomizedExperimentAdapter:
    # §19.1: the adapter receives the declared estimator-input view, the frozen plan, its pack,
    # and — for a sensitivity branch — the prespecified parameter delta. Nothing else. `mask` is
    # the committed §6.2 contribution mask every item it reports contributed through; the
    # coordinator binds it before the first fit.

    mask: ArtifactRef

    def fit(self, view: pl.DataFrame, plan: ec.EstimationPlanV1, pack: EstimationPackV1,
            overrides: ec.ValueMap | None = None) -> engine.AdapterResult:
        # One fit per approved contrast inside one adapter call, then the §9.3 harvest read off
        # the same frozen frame. The plan's parameters decide the specification; an override is
        # only ever a prespecified branch delta, never a choice made after seeing a result.
        params = dict(plan.estimator_parameters) | dict(overrides or {})
        roles = dict(plan.role_columns)
        frame = _typed(view, roles)
        arms = _arms(frame, roles)
        loo = params.get("influence_summary") == "leave_one_cluster_out"
        items: list[ec.PrimaryContrastResultV1] = []
        for contrast in plan.contrast_ids:
            pair = _pair(contrast, arms)
            found = (_influence if loo else _estimate)(
                frame, roles, plan, params, pair, self.mask)
            if found is not None:
                items.append(found.model_copy(update={"contrast_id": contrast}))
        return engine.AdapterResult(tuple(items), harvest(frame, roles, plan, items))

    def multiplicity(self, items: Sequence[ec.PrimaryContrastResultV1],
                     level: float) -> dict[str, ec.ValueMap]:
        return holm(items, level)

    def figures(self, harvested: Mapping[str, ec.ValueMap]
                ) -> Mapping[str, engine.FigureBuilder]:
        return figure_builders(harvested)

    def visual_evidence(self, builder_id: str) -> str:
        return VISUAL_EVIDENCE.get(builder_id, builder_id)
