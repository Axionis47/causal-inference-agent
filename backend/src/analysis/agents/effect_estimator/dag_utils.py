"""DAG utilities: backdoor adjustment set + variable role classification.

Both are pure functions over `CausalDAG`. They live here instead of
on the agent class so they can be unit-tested independently and so
the covariate-selection helper does not need an agent reference.
"""

from __future__ import annotations

from src.analysis.agents.base import CausalDAG


def compute_adjustment_set(
    dag: CausalDAG,
    treatment: str,
    outcome: str,
) -> list[str] | None:
    """Backdoor adjustment set: ancestors that are not descendants of treatment.

    Returns None if the DAG does not contain both treatment and outcome.
    """
    import networkx as nx

    G = nx.DiGraph()
    for edge in dag.edges:
        if edge.edge_type == "directed":
            G.add_edge(edge.source, edge.target)
        else:
            # Undirected/bidirected edges: add both directions conservatively
            G.add_edge(edge.source, edge.target)
            G.add_edge(edge.target, edge.source)

    if treatment not in G.nodes or outcome not in G.nodes:
        return None

    descendants_of_t = nx.descendants(G, treatment)
    descendants_of_t.add(treatment)
    descendants_of_t.add(outcome)

    ancestors_of_y = nx.ancestors(G, outcome)
    ancestors_of_t = nx.ancestors(G, treatment)
    relevant_ancestors = ancestors_of_y | ancestors_of_t

    adjustment_set = [
        node for node in relevant_ancestors
        if node not in descendants_of_t
    ]

    return sorted(adjustment_set) if adjustment_set else None


def classify_causal_role(
    dag: CausalDAG,
    variable: str,
    treatment: str,
    outcome: str,
) -> str:
    """Classify variable relative to (treatment, outcome).

    Returns one of: "confounder", "potential_mediator",
    "potential_collider", "unknown".
    """
    import networkx as nx

    G = nx.DiGraph()
    for edge in dag.edges:
        if edge.edge_type == "directed":
            G.add_edge(edge.source, edge.target)

    if variable not in G.nodes:
        return "unknown"

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

    return "unknown"
