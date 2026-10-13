"""Frozen certificates executed with the existing DoWhy and statsmodels libraries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from causal_agent.common.contracts import Estimate
from causal_agent.families.adjustment.lane.contracts import Design
from causal_agent.families.adjustment.lane.engine.model import TREATED, build_model, observed_set, valid_model_adjustment
from causal_agent.families.adjustment.lane.knowledge import EstimatorEntry
from causal_agent.families.adjustment.lane.support import verify


class FrozenSupportError(ValueError):
    """A fit or bootstrap draw must not silently redefine the authorized population."""


def _params(entry: EstimatorEntry) -> dict:
    params = dict(entry.params)
    if params.get("glm_family") == "binomial":
        import statsmodels.api as sm

        params["glm_family"] = sm.families.Binomial()
    return params


def _iv_fit(table: pd.DataFrame, d: Design) -> float:
    """Every non-instrumented regressor is repeated in the instrument matrix."""
    d.assert_integrity()
    from statsmodels.sandbox.regression.gmm import IV2SLS

    controls = pd.get_dummies(table[d.estimand.adjustment_set], drop_first=True, dtype=float) if d.estimand.adjustment_set else pd.DataFrame(index=table.index)
    exogenous = np.column_stack([np.ones(len(table)), controls.to_numpy(dtype=float)])
    regressors = np.column_stack([exogenous, table[TREATED].to_numpy(dtype=float)])
    instruments = np.column_stack([exogenous, table[d.estimand.instruments].to_numpy(dtype=float)])
    if np.linalg.matrix_rank(instruments) != instruments.shape[1]:
        raise ValueError("instrument matrix is rank deficient")
    fit = IV2SLS(table[d.graph.outcome].to_numpy(dtype=float), regressors, instruments).fit()
    return float(fit.params[-1])


def _dowhy_fit(table: pd.DataFrame, d: Design, entry: EstimatorEntry):
    d.assert_integrity()
    source = table.copy(deep=True)
    table = source.copy(deep=True)
    model = build_model(table, d.graph, d.graph.outcome)
    e = d.estimand
    if e.kind == "mediation":
        ide = model.identify_effect(estimand_type=f"nonparametric-{e.target}", method_name=e.identification_method, proceed_when_unidentifiable=True)
        ide.default_backdoor_id = "backdoor"
        found = (
            sorted(ide.get_mediator_variables() or []),
            sorted((ide.mediation_first_stage_confounders or {}).get("backdoor") or []),
            sorted((ide.mediation_second_stage_confounders or {}).get("backdoor") or []),
            sorted(ide.get_backdoor_variables() or []),
        )
        want = ([e.mediator], e.first_stage_set, e.second_stage_set, e.total_set)
        if found != want:
            raise ValueError(f"design: mediation certificate differs from freeze: {found!r} != {want!r}")
    else:
        ide = model.identify_effect(method_name=e.identification_method, proceed_when_unidentifiable=True)
        found = observed_set(ide, e.kind)
        want = e.adjustment_set if e.kind in ("backdoor", "general_adjustment") else e.frontdoor_set if e.kind == "frontdoor" else e.instruments
        if found is None or sorted(found) != sorted(want):
            raise ValueError(f"design: identified {found!r}; frozen {want!r}")
        if e.kind == "frontdoor":
            stages = (
                sorted((ide.mediation_first_stage_confounders or {}).get("backdoor") or []),
                sorted((ide.mediation_second_stage_confounders or {}).get("backdoor") or []),
            )
            if stages != (e.first_stage_set, e.second_stage_set):
                raise ValueError("design: front-door stage adjustment differs from freeze")
    if d.modifiers:
        if sorted(set(e.adjustment_set) | set(d.modifiers)) != sorted(e.model_covariates) or not valid_model_adjustment(d.graph, e.model_covariates):
            raise ValueError("design: conditional model covariates do not satisfy the frozen valid adjustment policy")
        if e.kind == "general_adjustment":
            ide.set_general_adjustment_variables(e.model_covariates, key="general_adjustment")
            ide.default_adjustment_set_id = "general_adjustment"
        else:
            ide.set_backdoor_variables(e.model_covariates, key="backdoor")
            ide.default_backdoor_id = "backdoor"
    fitted = model.estimate_effect(
        ide,
        method_name=entry.method,
        control_value=0,
        treatment_value=1,
        target_units="ate" if d.target_units in ("late", "nie", "nde", "conditional") else d.target_units,
        confidence_intervals=False,
        effect_modifiers=d.modifiers,
        method_params=_params(entry),
    )
    if entry.name == "propensity_score_stratification":
        data = fitted._data
        counts = data[TREATED].groupby(data["strata"]).agg(["sum", "count"])
        threshold = fitted.estimator.clipping_threshold
        retained = counts[(counts["sum"] > threshold) & ((counts["count"] - counts["sum"]) > threshold)]
        kept = int(retained["count"].sum())
        if kept != len(source):
            raise FrozenSupportError(f"support: propensity stratification would retain {kept} of {len(source)} frozen rows; target population cannot change")
    value = float(fitted.value)
    if not np.isfinite(value):
        raise ValueError("estimate is not finite")
    # DoWhy propensity fits add score/weight columns to their input. Refutations must
    # start from raw observations so each permuted or perturbed treatment refits scores.
    model._data = source
    return value, (model, ide, fitted)


def estimate(table: pd.DataFrame, d: Design, entry: EstimatorEntry, contrast: str, *, secondary: bool = False) -> tuple[Estimate, Any]:
    kwargs = dict(
        design_id=d.id,
        endpoint_id=d.endpoint_id,
        target={**d.target, "analysis_population_id": d.support.get("by_contrast", {}).get(contrast, {}).get("analysis_population_id")},
        contrast=contrast,
        method=entry.name,
        n_treated=int((table[TREATED] == 1).sum()),
        n_control=int((table[TREATED] == 0).sum()),
        target_units=d.target_units,
        secondary=secondary or d.endpoint_id == "adjustment.linear_conditional",
    )
    try:
        d.assert_integrity()
        verify(table, d, contrast)
        if entry.method == "structural.zero":
            return Estimate(**kwargs, value=0.0), None
        if entry.method == "statsmodels.iv2sls":
            value = _iv_fit(table, d)
            rng = np.random.default_rng(d.bootstrap_seed)
            draws = []
            for _ in range(d.bootstrap_draws):
                try:
                    draws.append(_iv_fit(table.iloc[rng.integers(0, len(table), len(table))].reset_index(drop=True), d))
                except (ValueError, np.linalg.LinAlgError):
                    pass
            if len(draws) < int(0.9 * d.bootstrap_draws):
                return Estimate(**kwargs, value=value, error="bootstrap failed on more than ten percent of frozen IV fits"), None
            low, high = np.quantile(draws, [0.025, 0.975])
            return Estimate(**kwargs, value=value, ci_low=float(low), ci_high=float(high)), {"table": table, "design": d}
        value, bundle = _dowhy_fit(table, d, entry)
        rng = np.random.default_rng(d.bootstrap_seed)
        draws = []
        for _ in range(d.bootstrap_draws):
            sample = table.iloc[rng.integers(0, len(table), len(table))].reset_index(drop=True)
            try:
                draw, _ = _dowhy_fit(sample, d, entry)
                draws.append(draw)
            except FrozenSupportError:
                raise
            except Exception:
                pass
        if len(draws) < int(0.9 * d.bootstrap_draws):
            return Estimate(**kwargs, value=value, error="bootstrap failed on more than ten percent of frozen fits"), bundle
        low, high = np.quantile(draws, [0.025, 0.975])
        return Estimate(**kwargs, value=value, ci_low=float(low), ci_high=float(high)), bundle

    except Exception as ex:
        return Estimate(**kwargs, error=f"{type(ex).__name__}: {str(ex)[:500]}"), None


def conditional_estimates(table: pd.DataFrame, d: Design, contrast: str, *, entry=None, bundle=None, error=None) -> list[Estimate]:
    """Same fitted interaction model; subgroup definitions were frozen before effects."""
    if not d.modifiers:
        return []
    d.assert_integrity()
    verify(table, d, contrast)
    entry = entry or d.estimator
    if entry not in [d.estimator, d.fallback]:
        raise ValueError("design: subgroup method must be an estimator declared before freeze")
    if error is None:
        try:
            if bundle is None:
                _, bundle = _dowhy_fit(table, d, entry)
        except Exception as ex:
            error = f"conditional model fit failed: {type(ex).__name__}: {str(ex)[:500]}"
    fit = bundle[2].estimator if bundle is not None else None
    out = []
    for col, groups in d.modifier_groups.items():
        for group in groups:
            if "values" in group:
                mask = table[col].astype(str).isin(group["values"])
            else:
                mask = table[col].between(group["low"], group["high"], inclusive="both" if group.get("include_low") else "right")
            part = table[mask]
            nt, nc = int((part[TREATED] == 1).sum()), int((part[TREATED] == 0).sum())
            base = dict(
                design_id=d.id,
                endpoint_id=d.endpoint_id,
                target={
                    **d.target,
                    "analysis_population_id": d.support.get("by_contrast", {}).get(contrast, {}).get("analysis_population_id"),
                    "subgroup": {"column": col, "definition": group},
                },
                contrast=contrast,
                method=entry.name,
                target_units="conditional",
                modifier=col,
                level=group["label"],
                n_treated=nt,
                n_control=nc,
            )
            if error:
                out.append(Estimate(**base, error=error))
                continue
            if min(nt, nc) < 20:
                out.append(Estimate(**base, error="fewer than twenty rows in a subgroup arm"))
                continue
            try:
                value = float(fit._do(1, part) - fit._do(0, part))
                if not np.isfinite(value):
                    raise ValueError("conditional estimate is not finite")
            except Exception as ex:
                out.append(Estimate(**base, error=f"subgroup computation failed: {type(ex).__name__}: {str(ex)[:500]}"))
                continue
            rng = np.random.default_rng(d.bootstrap_seed)
            draws = []
            for _ in range(d.bootstrap_draws):
                sample = table.iloc[rng.integers(0, len(table), len(table))].reset_index(drop=True)
                try:
                    _, bb = _dowhy_fit(sample, d, entry)
                    draw = float(bb[2].estimator._do(1, part) - bb[2].estimator._do(0, part))
                    if not np.isfinite(draw):
                        raise ValueError("conditional bootstrap estimate is not finite")
                    draws.append(draw)
                except Exception:
                    pass
            if len(draws) < int(0.9 * d.bootstrap_draws):
                out.append(Estimate(**base, value=value, error="subgroup interval bootstrap exhausted"))
            else:
                low, high = np.quantile(draws, [0.025, 0.975])
                out.append(Estimate(**base, value=value, ci_low=float(low), ci_high=float(high)))
    return out
