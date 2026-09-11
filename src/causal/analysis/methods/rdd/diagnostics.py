"""Numerical diagnostic measurements for rdd; no result-driven selection."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import cache
from typing import Any, cast

import numpy as np
import polars as pl
from rddensity import rddensity  # type: ignore[import-untyped]

from causal.analysis.integration import contracts as ec


def _density(frame: pl.DataFrame, roles: Mapping[str, str], params: ec.ValueMap
             ) -> tuple[ec.ValueMap, ec.ValueMap]:
    # §12.3: the registered `rddensity` manipulation test. A density test never proves the absence
    # of manipulation, and evidence of it is never repaired away.
    from causal.analysis.methods.rdd.estimation import _column, _number

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
    from causal.analysis.methods.rdd.estimation import _call, _covariates

    found: ec.ValueMap = {}
    for name in _covariates(frame, roles):
        seen = _call(frame, roles, plan, dict(params) | {"covariate_adjustment": False},
                     outcome="predetermined_covariate")
        jump, error = (float(seen.coef.loc["Robust"].iloc[0]), float(seen.se.loc["Robust"].iloc[0]))
        found[f"jump_{name}"] = jump
        found[f"z_{name}"] = abs(jump) / error if error else 0.0
    if not found:
        return {}
    found["covariate_jump_z"] = max(
        (abs(float(value or 0.0)) for key, value in found.items() if key.startswith("z_")),
        default=0.0)
    return found


def _probes(frame: pl.DataFrame, roles: Mapping[str, str], plan: ec.EstimationPlanV1,
            params: ec.ValueMap, found: Any) -> ec.ValueMap:
    # §12.3: how far the estimate moves under the registered bandwidth multiples and the
    # registered polynomial-order comparison, measured before any of them can be preferred.
    from causal.analysis.methods.rdd.estimation import _call, _number, _widths

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


def measurement_calls(frame: pl.DataFrame, roles: Mapping[str, str], plan: ec.EstimationPlanV1,
            params: ec.ValueMap, found: Any) -> dict[str, Callable[[], ec.ValueMap]]:
    # The §12.3 required diagnostics: support and heaping on both sides of the cutoff, the
    # selected bandwidths and effective observations, the manipulation test, covariate continuity,
    # bandwidth and polynomial sensitivity, near-cutoff leverage, and bias-correction integrity.
    from causal.analysis.methods.rdd.estimation import _column, _number

    running = _column(frame, roles, "running_variable")
    cutoff = _number(params, "cutoff", 0.0)
    below, above = running < cutoff, running >= cutoff
    effective = min(int(found.N_h[0]), int(found.N_h[1]))
    unique = len(np.unique(running))
    @cache
    def density() -> tuple[ec.ValueMap, ec.ValueMap]:
        return _density(frame, roles, params)

    return {
        "cutoff_side_support": lambda: {
            "effective_observations_per_side": effective, "rows_below": int(below.sum()),
            "rows_above": int(above.sum()), "unique_running_values": unique},
        "selected_bandwidth_report": lambda: {
            "bandwidth_left": float(found.bws.loc["h"].iloc[0]),
            "bandwidth_right": float(found.bws.loc["h"].iloc[1]),
            "bias_bandwidth": float(found.bws.loc["b"].iloc[0]),
            "selector": str(found.bwselect)},
        "mass_points_and_heaping": lambda: {
            "repeated_value_share": 1.0 - (unique / float(frame.height or 1)),
            "mass_points_left": int(found.M[0]), "mass_points_right": int(found.M[1])},
        "density_manipulation_test": lambda: density()[0],
        # An absent approved role is inapplicable, not evidence of zero imbalance. Omitting
        # its harvest yields a visible not_computable diagnostic; the wall checks applicability.
        **({"covariate_continuity": lambda: _continuity(frame, roles, plan, params)}
           if "predetermined_covariate" in roles else {}),
        "bandwidth_polynomial_sensitivity": lambda: _probes(frame, roles, plan, params, found),
        "influence_leverage_near_cutoff": lambda: {
            "single_observation_leverage_share": 1.0 / float(effective or 1),
            "effective_observations": effective},
        "sorting_and_other_policy_qualification": lambda: dict(density()[1]) | {
            "cutoff": cutoff, "carried_rule_ids": len(plan.numerical_failure_rule_ids)},
        "robust_bias_correction_integrity": lambda: {
            "conventional_estimate": float(found.coef.loc["Conventional"].iloc[0]),
            "bias_corrected_estimate": float(found.coef.loc["Bias-Corrected"].iloc[0]),
            "robust_standard_error": float(found.se.loc["Robust"].iloc[0]),
            "confidence_level": float(found.level) / 100.0, "variance_estimator": str(found.vce)}}


def harvest(frame: pl.DataFrame, roles: Mapping[str, str], plan: ec.EstimationPlanV1,
            params: ec.ValueMap, found: Any) -> dict[str, ec.ValueMap]:
    # The legacy pipeline retains its eager harvest. The standalone runner invokes each
    # callback independently so a failed check cannot discard the primary estimate.
    return {name: compute() for name, compute in
            measurement_calls(frame, roles, plan, params, found).items()}

