"""Causal structure section renderer."""

import json

from nbformat.v4 import new_code_cell, new_markdown_cell

from src.agents.base import AnalysisState

from ._skip import render_skipped_cell


def render_causal_structure(state: AnalysisState) -> list:
    """Report causal discovery agent findings."""
    dag = state.proposed_dag

    # Defensive: even though the notebook agent currently guards this call
    # site, a future caller might not. A skipped placeholder beats an
    # AttributeError on dag.discovery_method when no DAG was produced.
    if dag is None:
        return render_skipped_cell(
            "Causal Structure",
            reason=(
                "Neither causal_discovery nor dag_expert produced a DAG "
                "for this run. The adjustment-set rationale and the graph "
                "visualization are not available."
            ),
            upstream_agent="causal_discovery",
        )

    cells = []

    # Detect if dag_expert refined the graph
    is_refined = "domain_expert_fusion" in (dag.discovery_method or "")
    agent_label = "DAG Expert agent" if is_refined else "Causal Discovery agent"

    md = "## Causal Structure\n\n"
    md += f"*{'Refined' if is_refined else 'Discovered'} by the {agent_label} using **{dag.discovery_method}**.*\n\n"
    md += "The graph below represents the discovered causal relationships.\n"
    md += "**Green** = treatment, **Red** = outcome, **Blue** = other variables.\n"
    cells.append(new_markdown_cell(md))

    # DAG visualization code
    edges_json = json.dumps([(e.source, e.target) for e in dag.edges])
    nodes_json = json.dumps(dag.nodes)

    dag_code = f'''# Causal graph from pipeline
import networkx as nx

G = nx.DiGraph()
G.add_nodes_from({nodes_json})
G.add_edges_from({edges_json})

fig, ax = plt.subplots(figsize=(12, 8))
pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

treatment_node = "{state.treatment_variable}"
outcome_node = "{state.outcome_variable}"

node_colors = []
for node in G.nodes():
    if node == treatment_node:
        node_colors.append('lightgreen')
    elif node == outcome_node:
        node_colors.append('lightcoral')
    else:
        node_colors.append('lightblue')

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.9, ax=ax)
nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20,
                       connectionstyle="arc3,rad=0.1", ax=ax)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
ax.set_title('Discovered Causal Graph')
ax.axis('off')
plt.tight_layout()
plt.show()

print(f"Nodes: {{len(G.nodes())}}, Edges: {{len(G.edges())}}")'''
    cells.append(new_code_cell(dag_code))

    # DAG interpretation
    if dag.interpretation:
        cells.append(new_markdown_cell(
            f"### Causal Graph Interpretation\n\n{dag.interpretation}\n"
        ))

    # Edge details
    if dag.edges:
        edge_md = "### Edge Details\n\n"
        edge_md += "| Source | Target | Type | Confidence |\n"
        edge_md += "|--------|--------|------|------------|\n"
        for e in dag.edges:
            conf_str = f"{e.confidence:.2f}" if isinstance(e.confidence, (int, float)) else str(e.confidence)
            edge_md += f"| {e.source} | {e.target} | {e.edge_type} | {conf_str} |\n"
        edge_md += "\n"
        cells.append(new_markdown_cell(edge_md))

    # Variable roles table (from dag_expert)
    if dag.variable_roles:
        roles_md = "### Variable Roles\n\n"
        roles_md += "*Classified by the DAG Expert using domain knowledge and metadata.*\n\n"
        roles_md += "| Variable | Role |\n"
        roles_md += "|----------|------|\n"
        for var, role in sorted(dag.variable_roles.items()):
            roles_md += f"| {var} | {role} |\n"
        roles_md += "\n"
        cells.append(new_markdown_cell(roles_md))

    # Forbidden edges (from dag_expert)
    if dag.forbidden_edges:
        forbidden_md = "### Forbidden Edges\n\n"
        forbidden_md += "*Domain constraints that prevent impossible causal directions.*\n\n"
        forbidden_md += "| Forbidden Edge | Reason |\n"
        forbidden_md += "|----------------|--------|\n"
        for fe in dag.forbidden_edges:
            src = fe.get("source", "?")
            tgt = fe.get("target", "?")
            reason = fe.get("reason", "Domain constraint")
            forbidden_md += f"| {src} → {tgt} | {reason} |\n"
        forbidden_md += "\n"
        cells.append(new_markdown_cell(forbidden_md))

    # Adjustment set (from dag_expert)
    if dag.adjustment_set is not None:
        adj_md = "### Adjustment Set (Backdoor Criterion)\n\n"
        if dag.adjustment_set:
            adj_md += (
                "The following covariates were identified by the DAG Expert for "
                "adjustment in effect estimation (blocks all backdoor paths):\n\n"
            )
            adj_md += f"**Adjust for**: {', '.join(dag.adjustment_set)}\n\n"
        else:
            adj_md += (
                "No confounders identified by the backdoor criterion — "
                "treatment and outcome have no common causes in the discovered DAG.\n\n"
            )
        cells.append(new_markdown_cell(adj_md))

    # DAG evolution: if both stages ran, surface what dag_expert changed
    # so a reader can see how domain knowledge refined the data-driven DAG.
    if state.discovered_dag is not None and state.refined_dag is not None:
        raw = state.discovered_dag
        refined = state.refined_dag
        raw_edges = {(e.source, e.target) for e in raw.edges}
        refined_edges = {(e.source, e.target) for e in refined.edges}
        added = sorted(refined_edges - raw_edges)
        removed = sorted(raw_edges - refined_edges)

        evol_md = "### DAG Evolution: Discovery to Domain-Refined\n\n"
        evol_md += (
            f"The data-driven discovery produced a graph with **{len(raw.nodes)}** nodes "
            f"and **{len(raw_edges)}** edges via **{raw.discovery_method}**. "
            f"The DAG Expert then refined it to **{len(refined.nodes)}** nodes and "
            f"**{len(refined_edges)}** edges by fusing domain constraints.\n\n"
        )
        if added:
            evol_md += "**Edges added by domain refinement:**\n\n"
            for s, t in added[:20]:
                evol_md += f"- `{s} -> {t}`\n"
            if len(added) > 20:
                evol_md += f"- ... and {len(added) - 20} more\n"
            evol_md += "\n"
        if removed:
            evol_md += "**Edges removed by domain refinement (e.g., forbidden directions):**\n\n"
            for s, t in removed[:20]:
                evol_md += f"- `{s} -> {t}`\n"
            if len(removed) > 20:
                evol_md += f"- ... and {len(removed) - 20} more\n"
            evol_md += "\n"
        if not added and not removed:
            evol_md += "_No edge changes; refinement updated metadata only (variable roles, adjustment set, forbidden edges)._\n\n"
        cells.append(new_markdown_cell(evol_md))

    return cells
