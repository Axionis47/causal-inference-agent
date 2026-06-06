"""Pure helpers for the confounder_discovery agent.

DAG-based role classification, candidate selection, parquet/CSV loading,
the statistical fallback scan, and the agent's initial observation text.
None of these touch the agent class; the heavy ones take exactly what
they need so the logic can be unit-tested without the ReAct stack.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.agents.causal_discovery.output import CausalDAG
from src.analysis.agents.domain_knowledge import DomainKnowledge

# Oracle / ground-truth column names that must never be treated as
# confounder candidates — these leak the answer into the analysis.
ORACLE_PATTERNS = (
    "y0_true", "y1_true", "y0", "y1", "cate_true", "cate", "ite",
    "propensity_true", "propensity_score", "true_propensity", "counterfactual",
)


def classify_variable_role(
    dag: CausalDAG | None,
    variable: str,
    treatment: str,
    outcome: str,
    domain_knowledge: DomainKnowledge | None = None,
) -> str:
    """Topological role of `variable` relative to (treatment, outcome).

    Returns "confounder", "potential_mediator", "potential_collider", or
    "unknown". The domain_knowledge parameter is currently unused but kept
    in the signature so callers don't need to change when the heuristic
    lands.
    """
    del domain_knowledge

    if dag and dag.edges:
        import networkx as nx

        G = nx.DiGraph()
        for edge in dag.edges:
            if edge.edge_type == "directed":
                G.add_edge(edge.source, edge.target)

        if variable in G.nodes and treatment in G.nodes and outcome in G.nodes:
            is_ancestor_of_t = variable in nx.ancestors(G, treatment)
            is_ancestor_of_y = variable in nx.ancestors(G, outcome)
            is_descendant_of_t = variable in nx.descendants(G, treatment)
            is_descendant_of_y = variable in nx.descendants(G, outcome)

            if is_ancestor_of_t and is_ancestor_of_y:
                return "confounder"
            if is_descendant_of_t and is_ancestor_of_y:
                return "potential_mediator"
            if is_descendant_of_t and is_descendant_of_y:
                return "potential_collider"
            if is_ancestor_of_t or is_ancestor_of_y:
                return "confounder"

    return "unknown"


def candidates_from_df(
    df: pd.DataFrame | None,
    treatment_var: str | None,
    outcome_var: str | None,
) -> list[str]:
    """All numeric columns except treatment and outcome."""
    if df is None:
        return []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in (treatment_var, outcome_var)]


def load_dataframe(dataframe_path: str | None, logger) -> pd.DataFrame | None:
    """Load parquet or CSV from disk, swallowing read errors."""
    if not dataframe_path:
        return None
    if not Path(dataframe_path).exists():
        return None
    try:
        if dataframe_path.endswith(".csv"):
            return pd.read_csv(dataframe_path)
        return pd.read_parquet(dataframe_path)
    except Exception as e:
        logger.error("load_failed", error=str(e))
        return None


def _load_thresholds() -> tuple[float, float, float]:
    """(classic correlation, prognostic, balance p-value) from settings."""
    from src.config.settings import get_settings
    s = get_settings()
    return (
        s.confounder_correlation_threshold,
        s.confounder_prognostic_threshold,
        s.balance_pvalue_threshold,
    )


def statistical_confounder_scan(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    logger,
) -> list[dict[str, Any]]:
    """Multi-criterion correlation scan used as the fallback finalizer.

    Any one of three criteria flags a variable: classic confounder, strong
    prognostic predictor of Y, or imbalance between treatment groups.
    Returns dicts sorted by outcome-prediction strength.
    """
    confounders: list[dict[str, Any]] = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    candidate_cols = [
        c for c in numeric_cols
        if c not in (treatment_col, outcome_col)
        and not any(p in c.lower() for p in ORACLE_PATTERNS)
    ]

    treatment_vals = df[treatment_col].values.astype(float)
    outcome_vals = df[outcome_col].values.astype(float)

    treated_mask = treatment_vals == 1
    if treated_mask.sum() == 0 or (~treated_mask).sum() == 0:
        treated_mask = treatment_vals > np.median(treatment_vals)

    corr_thresh, prog_thresh, bal_thresh = _load_thresholds()

    for col in candidate_cols:
        try:
            col_data = df[col].values.astype(float)
            valid = ~(np.isnan(col_data) | np.isnan(treatment_vals) | np.isnan(outcome_vals))
            if valid.sum() < 20:
                continue

            c = col_data[valid]
            t = treatment_vals[valid]
            y = outcome_vals[valid]

            corr_t = abs(np.corrcoef(c, t)[0, 1]) if not np.isnan(np.corrcoef(c, t)[0, 1]) else 0
            corr_y = abs(np.corrcoef(c, y)[0, 1]) if not np.isnan(np.corrcoef(c, y)[0, 1]) else 0

            is_classic = corr_t > corr_thresh and corr_y > corr_thresh
            is_prognostic = corr_y > prog_thresh

            try:
                treated_vals_col = c[treated_mask[valid]]
                control_vals_col = c[~treated_mask[valid]]
                if len(treated_vals_col) > 2 and len(control_vals_col) > 2:
                    _, balance_p = stats.ttest_ind(treated_vals_col, control_vals_col, equal_var=False)
                    is_imbalanced = balance_p < bal_thresh
                else:
                    is_imbalanced = False
                    balance_p = 1.0
            except Exception:
                is_imbalanced = False
                balance_p = 1.0

            if is_classic or is_prognostic or is_imbalanced:
                reasons: list[str] = []
                if is_classic:
                    reasons.append(f"correlated with T ({corr_t:.3f}) and Y ({corr_y:.3f})")
                if is_prognostic:
                    reasons.append(f"prognostic variable (corr_Y={corr_y:.3f})")
                if is_imbalanced:
                    reasons.append(f"imbalanced across groups (p={balance_p:.3f})")

                confounders.append({
                    "variable": col,
                    "corr_with_treatment": round(float(corr_t), 3),
                    "corr_with_outcome": round(float(corr_y), 3),
                    "balance_p_value": round(float(balance_p), 3),
                    "strength": round(float(corr_y), 4),
                    "reasons": reasons,
                    "source": "statistical_fallback",
                })

                logger.info(
                    "statistical_scan_confounder_found",
                    variable=col,
                    corr_t=round(float(corr_t), 3),
                    corr_y=round(float(corr_y), 3),
                    balance_p=round(float(balance_p), 3),
                    reasons=reasons,
                )
        except Exception:
            continue

    confounders.sort(key=lambda x: x["strength"], reverse=True)
    logger.info(
        "statistical_scan_complete",
        n_candidates=len(candidate_cols),
        n_confounders=len(confounders),
    )
    return confounders


def fallback_confounder_identification(
    df: pd.DataFrame,
    treatment_var: str,
    outcome_var: str,
    dag: CausalDAG | None,
    domain_knowledge: DomainKnowledge | None,
    logger,
) -> tuple[list[str], list[str]]:
    """Statistical heuristic used when the ReAct loop never finalises.

    Returns (ranked_confounders, excluded_variables). Excluded vars are
    statistically associated but classified as mediators/colliders by the
    DAG.
    """
    candidates = candidates_from_df(df, treatment_var, outcome_var)
    confounders_with_strength: list[tuple[str, float, list[str]]] = []
    excluded: list[str] = []

    treatment_vals = df[treatment_var].values.astype(float)
    treated_mask = treatment_vals == 1
    if treated_mask.sum() == 0 or (~treated_mask).sum() == 0:
        treated_mask = treatment_vals > np.median(treatment_vals)

    corr_thresh, prog_thresh, bal_thresh = _load_thresholds()

    for col in candidates:
        try:
            corr_t = abs(stats.pearsonr(df[col], df[treatment_var])[0])
            corr_y = abs(stats.pearsonr(df[col], df[outcome_var])[0])

            is_classic = corr_t > corr_thresh and corr_y > corr_thresh
            is_prognostic = corr_y > prog_thresh

            try:
                treated_vals_col = df.loc[treated_mask, col].dropna()
                control_vals_col = df.loc[~treated_mask, col].dropna()
                if len(treated_vals_col) > 2 and len(control_vals_col) > 2:
                    _, balance_p = stats.ttest_ind(treated_vals_col, control_vals_col, equal_var=False)
                    is_imbalanced = balance_p < bal_thresh
                else:
                    is_imbalanced = False
            except Exception:
                is_imbalanced = False

            if is_classic or is_prognostic or is_imbalanced:
                role = classify_variable_role(
                    dag=dag,
                    variable=col,
                    treatment=treatment_var,
                    outcome=outcome_var,
                    domain_knowledge=domain_knowledge,
                )
                if role in ("potential_mediator", "potential_collider"):
                    excluded.append(col)
                    logger.info("fallback_excluded_variable", variable=col, causal_role=role)
                else:
                    reasons: list[str] = []
                    if is_classic:
                        reasons.append(f"correlated with T ({corr_t:.3f}) and Y ({corr_y:.3f})")
                    if is_prognostic:
                        reasons.append(f"prognostic (corr_Y={corr_y:.3f})")
                    if is_imbalanced:
                        reasons.append("imbalanced across groups")
                    confounders_with_strength.append((col, corr_y, reasons))
                    logger.info(
                        "fallback_confounder_found",
                        variable=col,
                        corr_y=round(corr_y, 3),
                        reasons=reasons,
                    )
        except Exception:
            logger.debug("correlation_computation_skipped", column=col, exc_info=True)

    confounders_with_strength.sort(key=lambda x: x[1], reverse=True)
    ranked = [c for c, _, _ in confounders_with_strength]
    return ranked, excluded


def initial_observation_text(
    candidates: list[str],
    treatment: str | None,
    outcome: str | None,
    dag_available: bool,
) -> str:
    dag_note = ""
    if not dag_available:
        dag_note = (
            "\n\nIMPORTANT: No causal graph available (DAG is empty or missing). "
            "Use statistical methods to identify confounders. Focus on correlations "
            "with both treatment and outcome variables."
        )

    return f"""CONFOUNDER DISCOVERY TASK
Treatment variable: {treatment or "unknown"}
Outcome variable: {outcome or "unknown"}
Candidate variables to investigate: {candidates}{dag_note}

Your goal is to identify which candidates are confounders (affect both treatment and outcome).
Use the tools to systematically investigate each candidate:
1. Start by getting the candidate list
2. For each candidate, test if it meets confounder criteria
3. Use correlations and partial correlations to verify
4. Call finalize_confounders when done with your ranked list."""
