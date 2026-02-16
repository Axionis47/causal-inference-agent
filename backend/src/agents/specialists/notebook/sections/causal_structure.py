"""Causal structure section renderer."""

import json

from nbformat.v4 import new_code_cell, new_markdown_cell

from src.agents.base import AnalysisState


def render_causal_structure(state: AnalysisState) -> list:
    """Report causal discovery agent findings."""
    cells = []
    dag = state.proposed_dag

    md = "## Causal Structure\n\n"
    md += f"*Discovered by the Causal Discovery agent using **{dag.discovery_method}**.*\n\n"
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

    return cells
