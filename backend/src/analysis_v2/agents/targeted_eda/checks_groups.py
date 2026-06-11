"""Group-comparison checks: observational/matching, interaction, factors.

Raw differences here are descriptive inputs to the plan gate and the
diagnostics stage, never effect estimates.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis_v2.spec import EDACheck, EDACheckStatus

from .common import ArtifactSink, EDAInputs, failed, resolved_col, skipped


def _binary_groups(frame: pd.DataFrame, col: str) -> tuple | None:
    values = frame[col].dropna().unique()
    if len(values) != 2:
        return None
    lo, hi = sorted(values, key=str)
    return lo, hi


def group_outcome_comparison(inputs: EDAInputs, sink: ArtifactSink) -> EDACheck:
    treat = resolved_col(inputs.spec.treatment, inputs.frame)
    outcome = resolved_col(inputs.spec.outcome, inputs.frame)
    if treat is None or outcome is None:
        return skipped("group_outcome_comparison", "treatment or outcome unresolved")
    groups = _binary_groups(inputs.frame, treat)
    if groups is None:
        return skipped("group_outcome_comparison", f"'{treat}' is not two-valued")
    try:
        lo, hi = groups
        y = pd.to_numeric(inputs.frame[outcome], errors="coerce")
        mean_lo = float(y[inputs.frame[treat] == lo].mean())
        mean_hi = float(y[inputs.frame[treat] == hi].mean())
        diff = mean_hi - mean_lo
        fig, ax = plt.subplots(figsize=(4.5, 3))
        ax.bar([str(lo), str(hi)], [mean_lo, mean_hi], color=["#777", "#5f7fb4"])
        ax.set_title(f"Raw mean {outcome} by {treat} (descriptive)")
        artifact = sink.add_plot(
            "group_outcome_comparison", fig, f"Raw mean {outcome} by {treat}"
        )
        return EDACheck(
            name="group_outcome_comparison", status=EDACheckStatus.OK,
            detail=(
                f"raw mean difference ({hi} vs {lo}): {diff:.4g}; descriptive only, "
                "groups may not be comparable"
            ),
            metrics={
                "raw_mean_difference": round(diff, 6),
                f"mean_{lo}": round(mean_lo, 6),
                f"mean_{hi}": round(mean_hi, 6),
            },
            artifact_ids=[artifact],
        )
    except Exception as exc:
        return failed("group_outcome_comparison", exc)


def covariate_balance(inputs: EDAInputs, sink: ArtifactSink) -> EDACheck:
    treat = resolved_col(inputs.spec.treatment, inputs.frame)
    covariates = [c for c in inputs.spec.candidate_confounders if c in inputs.frame.columns]
    if treat is None or not covariates:
        return skipped("covariate_balance", "needs a resolved treatment and confounders")
    groups = _binary_groups(inputs.frame, treat)
    if groups is None:
        return skipped("covariate_balance", f"'{treat}' is not two-valued")
    try:
        lo, hi = groups
        rows = []
        for cov in covariates:
            x = pd.to_numeric(inputs.frame[cov], errors="coerce")
            a, b = x[inputs.frame[treat] == hi], x[inputs.frame[treat] == lo]
            pooled = np.sqrt((a.var() + b.var()) / 2)
            smd = float((a.mean() - b.mean()) / pooled) if pooled and pooled > 0 else 0.0
            rows.append({"covariate": cov, "smd": round(smd, 4)})
        table = pd.DataFrame(rows).sort_values("smd", key=abs, ascending=False)
        table_id = sink.add_table("covariate_balance", table, "Covariate balance (SMD)")
        fig, ax = plt.subplots(figsize=(6, max(2.0, 0.4 * len(rows))))
        ax.barh(table["covariate"], table["smd"], color="#b45f5f")
        for line in (-0.1, 0.1):
            ax.axvline(line, color="#999", linestyle="--", linewidth=0.8)
        ax.set_title("Standardized mean differences")
        plot_id = sink.add_plot("covariate_balance_plot", fig, "Covariate balance plot")
        worst = float(table["smd"].abs().max())
        status = EDACheckStatus.WARNING if worst > 0.1 else EDACheckStatus.OK
        return EDACheck(
            name="covariate_balance", status=status,
            detail=f"largest |SMD| {worst:.3f} ({table.iloc[0]['covariate']}); "
            + ("imbalance present; adjustment is needed" if worst > 0.1 else "groups look balanced"),
            metrics={"max_abs_smd": round(worst, 4)},
            artifact_ids=[table_id, plot_id],
        )
    except Exception as exc:
        return failed("covariate_balance", exc)


def propensity_overlap(inputs: EDAInputs, sink: ArtifactSink) -> EDACheck:
    treat = resolved_col(inputs.spec.treatment, inputs.frame)
    covariates = [c for c in inputs.spec.candidate_confounders if c in inputs.frame.columns]
    if treat is None or len(covariates) < 1:
        return skipped("propensity_overlap", "needs a resolved treatment and confounders")
    groups = _binary_groups(inputs.frame, treat)
    if groups is None:
        return skipped("propensity_overlap", f"'{treat}' is not two-valued")
    try:
        from sklearn.linear_model import LogisticRegression

        lo, hi = groups
        data = inputs.frame[[treat, *covariates]].apply(
            lambda s: pd.to_numeric(s, errors="coerce")
        ).dropna()
        if len(data) < 50:
            return skipped("propensity_overlap", "too few complete rows")
        y = (data[treat] == hi).astype(int) if not pd.api.types.is_numeric_dtype(
            inputs.frame[treat]
        ) else (data[treat] == max(lo, hi)).astype(int)
        model = LogisticRegression(max_iter=200)
        scores = model.fit(data[covariates], y).predict_proba(data[covariates])[:, 1]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(scores[y == 1], bins=30, alpha=0.6, label="treated", color="#5f7fb4")
        ax.hist(scores[y == 0], bins=30, alpha=0.6, label="control", color="#999")
        ax.legend()
        ax.set_title("Propensity score overlap (descriptive)")
        artifact = sink.add_plot("propensity_overlap", fig, "Propensity score overlap")
        off_support = float(
            ((scores[y == 1] > np.percentile(scores[y == 0], 99)).mean())
        )
        status = EDACheckStatus.WARNING if off_support > 0.05 else EDACheckStatus.OK
        return EDACheck(
            name="propensity_overlap", status=status,
            detail=f"{off_support:.1%} of treated above the control 99th percentile score",
            metrics={"treated_off_support_share": round(off_support, 4)},
            artifact_ids=[artifact],
        )
    except Exception as exc:
        return failed("propensity_overlap", exc)


def missingness_by_group(inputs: EDAInputs, sink: ArtifactSink) -> EDACheck:
    treat = resolved_col(inputs.spec.treatment, inputs.frame)
    if treat is None:
        return skipped("missingness_by_group", "treatment unresolved")
    groups = _binary_groups(inputs.frame, treat)
    if groups is None:
        return skipped("missingness_by_group", f"'{treat}' is not two-valued")
    lo, hi = groups
    miss = inputs.frame.drop(columns=[treat]).isna().groupby(inputs.frame[treat]).mean()
    gap = float((miss.loc[hi] - miss.loc[lo]).abs().max()) if len(miss) == 2 else 0.0
    status = EDACheckStatus.WARNING if gap > 0.1 else EDACheckStatus.OK
    return EDACheck(
        name="missingness_by_group", status=status,
        detail=f"largest missingness gap between groups: {gap:.1%}",
        metrics={"max_missingness_gap": round(gap, 4)},
    )


def subgroup_outcome_table(inputs: EDAInputs, sink: ArtifactSink) -> EDACheck:
    """Moderator x treatment outcome means: the interaction/HTE story."""
    treat = resolved_col(inputs.spec.treatment, inputs.frame)
    outcome = resolved_col(inputs.spec.outcome, inputs.frame)
    moderator = resolved_col(inputs.spec.moderator, inputs.frame)
    if moderator is None or outcome is None:
        return skipped("subgroup_outcome_table", "needs a resolved moderator and outcome")
    try:
        frame = inputs.frame.copy()
        mod = frame[moderator]
        if pd.api.types.is_numeric_dtype(mod) and mod.nunique() > 8:
            frame["_mod_band"] = pd.qcut(mod, 4, duplicates="drop").astype(str)
            mod_col = "_mod_band"
        else:
            mod_col = moderator
        keys = [mod_col] if treat is None else [mod_col, treat]
        table = (
            frame.groupby(keys)[outcome].agg(["mean", "count"]).round(4).reset_index()
        )
        artifact = sink.add_table(
            "subgroup_outcome_table", table, f"{outcome} by {moderator} subgroups"
        )
        return EDACheck(
            name="subgroup_outcome_table", status=EDACheckStatus.OK,
            detail=f"outcome means across {table[mod_col].nunique()} {moderator} bands "
            "(descriptive; subgroup gaps are not effect estimates)",
            artifact_ids=[artifact],
        )
    except Exception as exc:
        return failed("subgroup_outcome_table", exc)


def factor_screening(inputs: EDAInputs, sink: ArtifactSink) -> EDACheck:
    outcome = resolved_col(inputs.spec.outcome, inputs.frame)
    factors = [c for c in inputs.spec.candidate_factors if c in inputs.frame.columns]
    if not factors:
        factors = [
            c.name for c in inputs.profile.columns
            if c.semantic_type in ("numeric", "ordinal", "binary")
            and c.name != outcome and not c.is_constant
        ][:15]
    if outcome is None or not factors:
        return skipped("factor_screening", "needs a resolved outcome and candidate factors")
    try:
        numeric = inputs.frame[[outcome, *factors]].apply(
            lambda s: pd.to_numeric(s, errors="coerce") if s.dtype == object else s
        )
        numeric = numeric.select_dtypes("number")
        corr = numeric.corr()[outcome].drop(outcome, errors="ignore").dropna()
        table = (
            corr.abs().sort_values(ascending=False).rename("abs_corr").reset_index()
        )
        table.columns = ["factor", "abs_corr_with_outcome"]
        table_id = sink.add_table("factor_screening", table.round(4), "Factor screening")
        inter = numeric.drop(columns=[outcome], errors="ignore").corr().abs()
        np.fill_diagonal(inter.values, 0.0)
        max_collinearity = float(inter.max().max()) if not inter.empty else 0.0
        status = EDACheckStatus.WARNING if max_collinearity > 0.85 else EDACheckStatus.OK
        return EDACheck(
            name="factor_screening", status=status,
            detail=f"{len(table)} factors screened; max inter-factor |corr| "
            f"{max_collinearity:.2f}; associations are not causal rankings",
            metrics={"max_collinearity": round(max_collinearity, 4)},
            artifact_ids=[table_id],
        )
    except Exception as exc:
        return failed("factor_screening", exc)
