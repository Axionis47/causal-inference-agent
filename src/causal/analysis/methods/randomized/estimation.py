# The randomized-experiment estimator adapter (PRD-004 §9): one intention-to-treat fit per
# approved contrast through pyfixest, the §9.2 attrition view, the §9.3 diagnostic harvest read
# off that same fit and the §9.4 prespecified branches.
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

from causal.analysis.common import legacy_engine as engine
from causal.analysis.integration import contracts as ec
from causal.analysis.integration.packs import EstimationPackV1
from causal.analysis.methods.randomized import diagnostics
from causal.shared.contracts import ArtifactRef, decode_contrast, encode_contrast

ADAPTER_VERSION: Final = "randomized-experiment-adapter.v1"
TERM: Final = "treated"
# The registered default names the CRV1 calculation actually used. The historical CR2-named
# alias remains readable, but a new result reports CR1 rather than claiming unavailable CR2.
CLUSTER_VCOV: Final[dict[str, str]] = {"cluster_robust_cr1": "CRV1",
    "cluster_robust_cr2": "CRV1", "cluster_robust_cr3": "CRV3"}
ROBUST_VCOV: Final[dict[str, str]] = {"cluster_robust_cr1": "HC2",
    "cluster_robust_cr2": "HC2", "cluster_robust_cr3": "HC3"}
FINITE_CORRECTIONS: Final[dict[str, str]] = {"cluster_robust_cr1": "hc2_cr1",
    "cluster_robust_cr2": "hc2_cr1", "cluster_robust_cr3": "hc3_cr3"}


def _arms(frame: pl.DataFrame, roles: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(sorted(str(value) for value in frame[roles["treatment"]].unique().to_list()))


def _pair(contrast_id: str, arms: Sequence[str]) -> tuple[str, str]:
    """Resolve only an approved canonical pair; never choose arms after seeing the data."""
    try:
        treated, comparator = decode_contrast(contrast_id)
    except ValueError as error:
        raise ec.EstimationError(str(error), "invalid_approved_contrast") from error
    if treated not in arms or comparator not in arms or treated == comparator:
        raise ec.EstimationError(
            f"approved contrast {contrast_id!r} is absent from observed arms {tuple(arms)!r}",
            "approved_contrast_without_support")
    return treated, comparator


def _typed(view: pl.DataFrame, roles: Mapping[str, str]) -> pl.DataFrame:
    # The declared view under the dtypes the fit needs: string arm, cluster, and stratum keys,
    # a float outcome, and a float precision covariate when the approved column is numeric.
    casts = {roles[role]: pl.col(roles[role]).cast(pl.Float64)
             for role in ("outcome", "precision_covariate")
             if roles.get(role) in view.columns and view.schema[roles[role]].is_numeric()}
    # Multiple causal roles may refer to one physical column. Its identifier role takes
    # precedence over a numeric-looking ID, and each column receives exactly one cast.
    casts.update({roles[role]: pl.col(roles[role]).cast(pl.String)
                  for role in ("treatment", "cluster", "stratum")
                  if roles.get(role) in view.columns})
    return view.with_columns(list(casts.values()))


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
    stratum = roles.get("stratum") if params.get("stratum_handling") == "fixed_effects" else None
    # A covariate already absorbed as the randomization stratum contributes no additional
    # regressor. Keeping it on the right also risks treating its arbitrary IDs as a slope.
    adjusted = (covariate is not None and covariate != stratum
                and params.get("specification") != "difference_in_means")
    right = f"{TERM} + {covariate}" if adjusted else TERM
    return f"{roles['outcome']} ~ {right}" + (f" | {stratum}" if stratum else "")


def _vcov(roles: Mapping[str, str], params: ec.ValueMap) -> Any:
    # The plan's covariance profile: cluster-robust over the approved cluster when the design is
    # cluster randomized, heteroskedasticity-robust over units when it is not.
    profile = str(params.get("cluster_covariance") or "cluster_robust_cr1")
    cluster = roles.get("cluster")
    return ({CLUSTER_VCOV.get(profile, "CRV1"): cluster} if cluster is not None
            else ROBUST_VCOV.get(profile, "HC2"))


def _quantities(kept: pl.DataFrame, roles: Mapping[str, str], params: ec.ValueMap,
                binary: bool) -> ec.ValueMap:
    cluster = roles.get("cluster")
    requested = str(params.get("cluster_covariance") or "cluster_robust_cr1")
    return {"effective_rows": kept.height, "outcome_kind": "binary" if binary else "continuous",
            "specification": str(params.get("specification") or "difference_in_means"),
            "covariance_profile": "cluster_robust_cr1" if requested == "cluster_robust_cr2" else requested,
            "requested_covariance_profile": requested,
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
    binary = (params["outcome_kind"] == "binary" if "outcome_kind" in params else
              set(kept[roles["outcome"]].unique().to_list()) <= {0.0, 1.0})
    return ec.PrimaryContrastResultV1(
        contrast_id=encode_contrast(treated, comparator), estimand_id=plan.estimand_id,
        estimand_label=f"intention to treat, {treated} against {comparator}",
        estimate=float(fitted.coef()[TERM]),
        estimate_units="risk_difference" if binary else plan.outcome_scale,
        comparator_id=comparator, effect_direction="treated_minus_comparator",
        standard_error=float(fitted.se()[TERM]), confidence_level=level, interval_lower=lower,
        interval_upper=upper, p_value=float(fitted.pvalue()[TERM]),
        uncertainty_method=plan.uncertainty_method,
        finite_sample_correction=FINITE_CORRECTIONS.get(
            str(params.get("cluster_covariance") or "cluster_robust_cr1"), plan.finite_sample_correction),
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


@dataclass(frozen=True)
class RandomizedExperimentAdapter:
    # §19.1: the adapter receives the declared estimator-input view, the frozen plan, its pack,
    # and — for a sensitivity branch — the prespecified parameter delta. Nothing else. `mask` is
    # the committed §6.2 contribution mask every item it reports contributed through; the
    # coordinator binds it before the first fit.

    mask: ArtifactRef

    def fit(self, view: pl.DataFrame, plan: ec.EstimationPlanV1, pack: EstimationPackV1,
            overrides: ec.ValueMap | None = None, *,
            compute_diagnostics: bool = True) -> engine.AdapterResult:
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
        return engine.AdapterResult(tuple(items),
            diagnostics.harvest(frame, roles, plan, items) if compute_diagnostics else {})

    def multiplicity(self, items: Sequence[ec.PrimaryContrastResultV1],
                     level: float) -> dict[str, ec.ValueMap]:
        return holm(items, level)

