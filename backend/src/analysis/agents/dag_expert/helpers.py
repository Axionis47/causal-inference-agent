"""Pure helpers for the dag_expert agent.

Domain pattern lookup, DAG fusion, adjustment-set computation. None of
these touch agent state directly — they take their inputs explicitly so
the logic can be unit-tested independently of the ReAct loop.
"""

from src.analysis.agents.causal_discovery.output import CausalDAG, CausalEdge

# Map domain -> typical causal patterns. Used by analyze_domain to hint at
# what variable roles to look for. Anything not in the map gets the generic
# pre-treatment / treatment / outcome story.
TYPICAL_PATTERNS: dict[str, list[str]] = {
    "economics": [
        "Demographics -> Treatment selection -> Economic outcome",
        "Education -> Employment -> Earnings",
        "Policy/Program -> Behavior change -> Outcome",
    ],
    "healthcare": [
        "Risk factors -> Treatment -> Health outcome",
        "Demographics -> Disease risk -> Treatment -> Survival",
        "Lifestyle -> Biomarkers -> Disease",
    ],
    "education": [
        "Background -> Educational intervention -> Academic outcome",
        "Prior achievement -> Program participation -> Future achievement",
    ],
}

_DEFAULT_PATTERNS = [
    "Pre-treatment covariates -> Treatment -> Outcome",
    "Confounders affect both treatment and outcome",
]

# Confidence-string -> numeric used when building CausalEdges from
# domain reasoning.
CONFIDENCE_MAP = {"high": 0.9, "medium": 0.7, "low": 0.5}


def patterns_for_domain(domain: str | None) -> list[str]:
    """Return the typical causal patterns for a domain string."""
    if domain and domain in TYPICAL_PATTERNS:
        return TYPICAL_PATTERNS[domain]
    return _DEFAULT_PATTERNS


def generate_interpretation(
    edges: list[CausalEdge],
    conflicts: list[dict],
    edge_sources: dict,
) -> str:
    """Build the markdown interpretation string attached to the refined DAG."""
    lines = ["## Validated Causal DAG\n"]

    lines.append("### Edge Sources:")
    domain_count = sum(1 for v in edge_sources.values() if v == "domain")
    data_count = sum(1 for v in edge_sources.values() if v == "data")
    lines.append(f"- Domain-derived: {domain_count} edges")
    lines.append(f"- Data-derived: {data_count} edges")

    if conflicts:
        lines.append(f"\n### Conflicts Resolved: {len(conflicts)}")
        for c in conflicts[:3]:
            lines.append(f"- Domain said {c['domain']}, Data said {c['data']}")

    lines.append("\n### Key Assumptions:")
    lines.append("- Demographics are pre-treatment (cannot be caused by treatment)")
    lines.append("- Temporal ordering respected where identified")
    lines.append("- Unobserved confounders may exist (interpret with caution)")

    lines.append("\n### Recommendations:")
    lines.append("- Use sensitivity analysis to test robustness")
    lines.append("- Consider multiple adjustment strategies")

    return "\n".join(lines)


def fuse_edges(
    domain_edges: list[dict],
    data_edges: list[dict],
    forbidden_edges: list[tuple[str, str, str]],
    conflict_resolution: str,
    treatment_var: str | None,
    outcome_var: str | None,
    variable_roles: dict[str, str],
) -> tuple[CausalDAG, list[dict], dict]:
    """Fuse domain and data edges into a single CausalDAG.

    Returns (dag, conflicts, edge_sources). edge_sources maps each (s,t)
    in the final DAG to "domain" or "data" so the caller can summarise.
    """
    forbidden_pairs = {(s, t) for s, t, _ in forbidden_edges}

    all_nodes: set[str] = set()
    for edge in domain_edges + data_edges:
        all_nodes.add(edge["source"])
        all_nodes.add(edge["target"])
    if treatment_var:
        all_nodes.add(treatment_var)
    if outcome_var:
        all_nodes.add(outcome_var)

    domain_edge_set = {(e["source"], e["target"]) for e in domain_edges}
    data_edge_set = {(e["source"], e["target"]) for e in data_edges}

    conflicts: list[dict] = []
    for s, t in domain_edge_set:
        if (t, s) in data_edge_set:
            conflicts.append({
                "domain": f"{s} -> {t}",
                "data": f"{t} -> {s}",
                "resolution": conflict_resolution,
            })

    final_edges: list[CausalEdge] = []
    edge_sources: dict[tuple[str, str], str] = {}
    processed: set[tuple[str, str]] = set()

    for edge in domain_edges:
        key = (edge["source"], edge["target"])
        reverse_key = (edge["target"], edge["source"])

        if key in forbidden_pairs or key in processed:
            continue

        if reverse_key in data_edge_set:
            if conflict_resolution == "data_priority":
                key = reverse_key
            elif conflict_resolution == "consensus_only":
                continue

        final_edges.append(CausalEdge(
            source=key[0],
            target=key[1],
            edge_type="directed",
            confidence=CONFIDENCE_MAP.get(edge.get("confidence", "medium"), 0.7),
        ))
        edge_sources[key] = "domain"
        processed.add(key)

    for edge in data_edges:
        key = (edge["source"], edge["target"])
        reverse_key = (edge["target"], edge["source"])

        if key in processed or reverse_key in processed:
            continue
        if key in forbidden_pairs:
            continue

        final_edges.append(CausalEdge(
            source=edge["source"],
            target=edge["target"],
            edge_type=edge.get("edge_type", "directed"),
            confidence=edge.get("confidence", 0.6),
        ))
        edge_sources[key] = "data"
        processed.add(key)

    dag = CausalDAG(
        nodes=list(all_nodes),
        edges=final_edges,
        discovery_method=f"domain_expert_fusion ({conflict_resolution})",
        treatment_variable=treatment_var,
        outcome_variable=outcome_var,
        interpretation=generate_interpretation(final_edges, conflicts, edge_sources),
        forbidden_edges=[
            {"source": s, "target": t, "reason": r}
            for s, t, r in forbidden_edges
        ] if forbidden_edges else None,
        variable_roles=dict(variable_roles) if variable_roles else None,
    )
    return dag, conflicts, edge_sources


def adjustment_set_from_dag(dag: CausalDAG, treatment: str, outcome: str) -> dict:
    """Backdoor adjustment set + classification of every variable."""
    children: dict[str, set[str]] = {n: set() for n in dag.nodes}
    parents: dict[str, set[str]] = {n: set() for n in dag.nodes}

    for edge in dag.edges:
        if edge.edge_type == "directed":
            children[edge.source].add(edge.target)
            parents[edge.target].add(edge.source)

    def ancestors_of(node: str) -> set[str]:
        out: set[str] = set()
        queue = list(parents.get(node, []))
        while queue:
            p = queue.pop()
            if p not in out:
                out.add(p)
                queue.extend(parents.get(p, []))
        return out

    def descendants_of(node: str) -> set[str]:
        out: set[str] = set()
        queue = list(children.get(node, []))
        while queue:
            c = queue.pop()
            if c not in out:
                out.add(c)
                queue.extend(children.get(c, []))
        return out

    t_ancestors = ancestors_of(treatment)
    o_ancestors = ancestors_of(outcome)
    t_descendants = descendants_of(treatment)
    o_descendants = descendants_of(outcome)

    confounders = t_ancestors & o_ancestors
    mediators = t_descendants & o_ancestors
    colliders = t_descendants & o_descendants
    adjustment_set = sorted(confounders - {treatment, outcome})

    return {
        "adjustment_set": adjustment_set,
        "confounders": sorted(confounders),
        "mediators": sorted(mediators),
        "colliders": sorted(colliders),
    }


def initial_observation_text(state) -> str:
    return f"""Task: Construct a validated causal DAG for job {state.job_id}

Dataset: {state.dataset_info.name or state.dataset_info.url}
Treatment: {state.treatment_variable or "To be identified"}
Outcome: {state.outcome_variable or "To be identified"}

Your goal is to build a high-quality DAG by:
1. First, analyze the domain from metadata (analyze_domain)
2. Classify key variables into causal roles (classify_variable_role)
3. Propose edges based on domain reasoning (propose_edge)
4. Mark impossible edges as forbidden (mark_forbidden_edge)
5. Compare with data discovery results (get_discovery_edges)
6. Fuse domain and data to create final DAG (fuse_and_validate)
7. Extract the adjustment set for effect estimation (get_adjustment_set)

Start by analyzing the domain to understand the causal context."""
