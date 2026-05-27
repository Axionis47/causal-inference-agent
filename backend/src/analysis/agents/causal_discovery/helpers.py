"""Pure helpers for the causal_discovery agent.

Column selection, algorithm execution (engine + legacy), graph utilities,
and the auto-finalize heuristic. None of these touch agent.* — they take
exactly what they need, so they can be unit-tested independently.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from .output import CausalDAG, CausalEdge


def load_dataframe(dataframe_path: str | None) -> pd.DataFrame | None:
    if dataframe_path and Path(dataframe_path).exists():
        return pd.read_parquet(dataframe_path)
    return None


def select_columns(state, df: pd.DataFrame, treatment_var: str | None, outcome_var: str | None) -> tuple[pd.DataFrame, list[str]]:
    """Pick the columns we hand to a discovery algorithm.

    Starts from the profiler's hints, falls back to all numeric, then
    injects treatment and outcome (if numeric) and folds low-cardinality
    categoricals in as one-hot dummies. Returns (df_with_dummies, cols).
    """
    import numpy as np
    profile = state.data_profile
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    relevant_cols: list[str] = []
    if profile:
        relevant_cols = (
            profile.treatment_candidates[:2]
            + profile.outcome_candidates[:2]
            + profile.potential_confounders[:10]
        )
        relevant_cols = [c for c in relevant_cols if c in numeric_cols]

    for col in numeric_cols:
        if col not in relevant_cols:
            relevant_cols.append(col)

    if state.treatment_variable and state.treatment_variable in numeric_cols:
        if state.treatment_variable not in relevant_cols:
            relevant_cols.insert(0, state.treatment_variable)
    if state.outcome_variable and state.outcome_variable in numeric_cols:
        if state.outcome_variable not in relevant_cols:
            relevant_cols.insert(1, state.outcome_variable)

    if not relevant_cols:
        relevant_cols = numeric_cols[:15]

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    df_work = df.copy()
    for col in cat_cols:
        if df[col].nunique() <= 10:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            for dcol in dummies.columns:
                df_work[dcol] = dummies[dcol].astype(float)
                relevant_cols.append(dcol)

    relevant_cols = relevant_cols[:20]
    return df_work, relevant_cols


def run_algorithm(
    df: pd.DataFrame,
    algorithm: str,
    alpha: float,
    threshold: float,
    treatment_var: str | None,
    outcome_var: str | None,
    logger,
) -> CausalDAG | None:
    """Try the new CausalDiscovery engine first, fall back to legacy on failure."""
    nodes = df.columns.tolist()
    try:
        from src.causal.dag import CausalDiscovery
        discovery = CausalDiscovery(algorithm=algorithm, alpha=alpha)
        result = discovery.discover(
            df=df,
            variables=nodes,
            treatment=treatment_var,
            outcome=outcome_var,
        )
        edges = [
            CausalEdge(
                source=e.source,
                target=e.target,
                edge_type=e.edge_type,
                confidence=e.confidence,
            )
            for e in result.edges
        ]
        return CausalDAG(
            nodes=result.nodes,
            edges=edges,
            discovery_method=f"{algorithm.upper()} (engine)",
            treatment_variable=treatment_var,
            outcome_variable=outcome_var,
        )
    except Exception as e:
        logger.warning(f"engine_failed_{algorithm}", error=str(e))
        return run_algorithm_legacy(df, algorithm, alpha, threshold, treatment_var, outcome_var, logger)


def run_algorithm_legacy(
    df: pd.DataFrame,
    algorithm: str,
    alpha: float,
    threshold: float,
    treatment_var: str | None,
    outcome_var: str | None,
    logger,
) -> CausalDAG | None:
    """Legacy direct-causallearn fallback, kept verbatim from the monolith."""
    nodes = df.columns.tolist()
    edges = []
    try:
        if algorithm == "pc":
            from causallearn.search.ConstraintBased.PC import pc
            from causallearn.utils.cit import fisherz
            cg = pc(df.values, alpha=alpha, indep_test=fisherz)
            adj_matrix = cg.G.graph
            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    if adj_matrix[i, j] == -1 and adj_matrix[j, i] == 1:
                        edges.append(CausalEdge(source=nodes[i], target=nodes[j], edge_type="directed"))
                    elif adj_matrix[i, j] == -1 and adj_matrix[j, i] == -1 and i < j:
                        edges.append(CausalEdge(source=nodes[i], target=nodes[j], edge_type="undirected"))
        elif algorithm == "ges":
            from causallearn.search.ScoreBased.GES import ges
            record = ges(df.values)
            adj_matrix = record["G"].graph
            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    if adj_matrix[i, j] == -1 and adj_matrix[j, i] == 1:
                        edges.append(CausalEdge(source=nodes[i], target=nodes[j], edge_type="directed"))
        elif algorithm in ["notears", "lingam"]:
            from causallearn.search.FCMBased import lingam
            model = lingam.ICALiNGAM() if algorithm == "lingam" else lingam.DirectLiNGAM()
            model.fit(df.values)
            adj_matrix = model.adjacency_matrix_
            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    if abs(adj_matrix[i, j]) > threshold:
                        edges.append(CausalEdge(
                            source=nodes[j], target=nodes[i], edge_type="directed",
                            confidence=float(abs(adj_matrix[i, j])),
                        ))

        return CausalDAG(
            nodes=nodes,
            edges=edges,
            discovery_method=f"{algorithm.upper()} (legacy)",
            treatment_variable=treatment_var,
            outcome_variable=outcome_var,
        )
    except ImportError as e:
        logger.warning(f"import_failed_{algorithm}", error=str(e))
        return None
    except Exception as e:
        logger.warning(f"algorithm_failed_{algorithm}", error=str(e))
        return None


def check_path(dag: CausalDAG, source: str, target: str) -> bool:
    """BFS check for a directed path source -> target inside the DAG."""
    visited: set[str] = set()
    queue = [source]
    while queue:
        node = queue.pop(0)
        if node == target:
            return True
        if node in visited:
            continue
        visited.add(node)
        for edge in dag.edges:
            if edge.edge_type == "directed" and edge.source == node:
                queue.append(edge.target)
    return False


def assess_dag_quality(edges: int, n_nodes: int) -> str:
    """One-line edge-density label for downstream warnings."""
    if edges == 0:
        return "empty"
    if edges < 2:
        return "poor"
    if edges < n_nodes * 0.5:
        return "sparse"
    return "good"


def create_simple_dag(
    treatment_var: str | None,
    outcome_var: str | None,
    profile,
) -> CausalDAG:
    """Fallback DAG: confounders -> treatment, confounders -> outcome, treatment -> outcome."""
    nodes: list[str] = []
    edges: list[CausalEdge] = []

    if treatment_var:
        nodes.append(treatment_var)
    if outcome_var and outcome_var not in nodes:
        nodes.append(outcome_var)

    if profile and profile.potential_confounders:
        for conf in profile.potential_confounders[:5]:
            if conf not in nodes:
                nodes.append(conf)
                if treatment_var:
                    edges.append(CausalEdge(source=conf, target=treatment_var, edge_type="directed"))
                if outcome_var:
                    edges.append(CausalEdge(source=conf, target=outcome_var, edge_type="directed"))

    if treatment_var and outcome_var:
        edges.append(CausalEdge(source=treatment_var, target=outcome_var, edge_type="directed"))

    return CausalDAG(
        nodes=nodes,
        edges=edges,
        discovery_method="Simple DAG (fallback)",
        treatment_variable=treatment_var,
        outcome_variable=outcome_var,
    )


def auto_finalize(
    discovered_graphs: dict[str, CausalDAG],
    treatment_var: str | None,
    outcome_var: str | None,
) -> dict[str, Any]:
    """Pick the best graph from what's been run, identify confounders heuristically."""
    preferred_order = ["pc", "ges", "lingam", "notears"]
    chosen_alg = None
    for alg in preferred_order:
        if alg in discovered_graphs:
            chosen_alg = alg
            break
    if not chosen_alg and discovered_graphs:
        chosen_alg = list(discovered_graphs.keys())[0]

    confounders: list[str] = []
    if chosen_alg and chosen_alg in discovered_graphs:
        dag = discovered_graphs[chosen_alg]
        if treatment_var and outcome_var:
            t_parents = {e.source for e in dag.edges if e.edge_type == "directed" and e.target == treatment_var}
            o_parents = {e.source for e in dag.edges if e.edge_type == "directed" and e.target == outcome_var}
            confounders = list(t_parents & o_parents)

    return {
        "chosen_algorithm": chosen_alg or "simple",
        "confounders": confounders,
        "mediators": [],
        "interpretation": "Causal structure discovered through automated analysis.",
        "confidence": "medium" if chosen_alg else "low",
    }


def initial_observation_text(state) -> str:
    treatment, outcome = state.get_primary_pair()
    treatment = treatment or "use tools to identify"
    outcome = outcome or "use tools to identify"
    if state.data_profile:
        n_samples = state.data_profile.n_samples
        n_features = state.data_profile.n_features
    else:
        n_samples = state.dataset_info.n_samples or "unknown"
        n_features = state.dataset_info.n_features or "unknown"

    return (
        f"Job: {state.job_id}\n"
        f"Treatment: {treatment} | Outcome: {outcome}\n"
        f"Dataset: {n_samples} samples, {n_features} features\n"
        f"Use get_data_characteristics to inspect the data before choosing algorithms."
    )
