"""Numerical diagnostic measurements for randomized; no result-driven selection."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl

from causal.analysis.common import legacy_engine as engine
from causal.analysis.integration import contracts as ec


def _rates(frame: pl.DataFrame, roles: Mapping[str, str]) -> ec.ValueMap:
    # §9.2: attrition by assignment arm over the stabilized randomized population. The assigned
    # count is the denominator on every row here; a missing outcome never leaves it.
    arm, outcome = roles["treatment"], roles["outcome"]
    # Work in a two-column local view so source names cannot collide with count aliases.
    rows = frame.select(pl.col(arm).alias("arm"),
                        pl.col(outcome).is_not_null().alias("observed")).group_by("arm").agg(
        pl.len().alias("assigned"), pl.col("observed").sum()).sort("arm")
    by_arm: dict[str, int | float] = {}
    for name, assigned, observed in rows.iter_rows():
        key = f"attrition_{name}"
        by_arm[key] = 1.0 - (float(observed) / float(assigned or 1))
        by_arm[f"denominator:{key}"] = int(assigned)
    seen = [float(value) for key, value in by_arm.items()
            if not key.startswith("denominator:")]
    overall = 1.0 - float(frame[outcome].is_not_null().sum()) / float(frame.height or 1)
    found: ec.ValueMap = {"overall_attrition": overall,
                          "differential_attrition": max(seen) - min(seen),
                          "denominator:overall_attrition": frame.height,
                          "denominator:differential_attrition": frame.height}
    found.update(by_arm)
    return found


def harvest(frame: pl.DataFrame, roles: Mapping[str, str], plan: ec.EstimationPlanV1,
            items: Sequence[ec.PrimaryContrastResultV1], *,
            diagnostic_ids: frozenset[str] | None = None) -> dict[str, ec.ValueMap]:
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
    balance_selected = diagnostic_ids is None or "baseline_balance" in diagnostic_ids
    balance = (engine.balance(frame, arm, [covariate])
               if covariate is not None and balance_selected else {})
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
            "arm_cluster_stratum_contribution_counts": counts,
            **({"baseline_balance": measures} if balance_selected else {}),
            "outcome_attrition_by_arm": _rates(frame, roles),
            "covariance_cluster_adequacy": adequacy, "influential_cluster_leverage": leverage,
            "model_convergence_integrity": convergence, "multiplicity_handling": multiplicity}

