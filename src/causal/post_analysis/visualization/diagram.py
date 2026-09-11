"""Lay out only supplied causal nodes/edges, preserving their epistemic status."""

from __future__ import annotations

from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import Any, Literal

import graphviz  # type: ignore[import-untyped]
import vl_convert

from causal.post_analysis.visualization.contracts import CausalDiagram, RenderedVisual, VisualSpec
from causal.post_analysis.visualization.render import (
    decorate_svg,
    digest,
    encoded,
    renderer_versions,
    write_objects,
)

_EDGES = {
    "evidenced": ("solid", "#334155"),
    "hypothesis": ("dashed", "#59677a"),
    "unknown": ("dotted", "#7b8797"),
    "disputed": ("dashed", "#ae3434"),
}


def _build_graph(diagram: CausalDiagram, title: str, direction: str) -> Any:
    """Bind supplied nodes and edges to a safe DOT graph; infer no scientific relationships."""
    graph = graphviz.Digraph(
        graph_attr={
            "rankdir": direction,
            "bgcolor": "white",
            "pad": "0.25",
            "nodesep": "0.5",
            "ranksep": "0.7",
            "label": graphviz.escape(title),
            "labelloc": "t",
            "fontname": "Noto Sans",
            "fontsize": "17",
        },
        node_attr={
            "shape": "box",
            "style": "rounded,filled",
            "fillcolor": "#f1f6fa",
            "color": "#758ca3",
            "fontname": "Noto Sans",
            "fontsize": "12",
            "margin": "0.18",
        },
        edge_attr={"arrowsize": "0.8", "penwidth": "1.5"},
    )
    # Generated graph identifiers prevent scientific IDs from becoming DOT expressions.
    ids = {node.node_id: f"n{i}" for i, node in enumerate(diagram.nodes)}
    for node in diagram.nodes:
        graph.node(
            ids[node.node_id],
            label=graphviz.escape(node.label),
            tooltip=graphviz.escape(node.node_id),
        )
    for edge in diagram.edges:
        style, color = _EDGES[edge.status]
        graph.edge(
            ids[edge.source_node],
            ids[edge.target_node],
            style=style,
            color=color,
            tooltip=graphviz.escape(f"{edge.edge_id}: {edge.status}"),
        )
    return graph


def render_dag(
    visual_id: str,
    diagram: CausalDiagram,
    output_dir: Path,
    *,
    title: str = "Causal structure",
    caption: str = "",
    qualifications: tuple[str, ...] = (),
    direction: Literal["LR", "TB"] = "LR",
) -> RenderedVisual:
    """Graphviz chooses positions only. The input's edge set, directions and states are fixed."""
    # Reuse the bounded identifier/text fields, without representing the DAG as numeric data.
    bounded = VisualSpec(
        visual_id=visual_id,
        table_id="causal_graph",
        kind="table",
        title=title,
        caption=caption,
        qualifications=qualifications,
    )
    if direction not in {"LR", "TB"}:
        raise ValueError("unsupported_diagram_direction")
    body = diagram.model_dump(mode="json")
    spec = bounded.model_dump(mode="json") | {"kind": "dag", "direction": direction}
    base: dict[str, Any] = {
        "visual_id": visual_id,
        "source": diagram.source,
        "selector": diagram.selector,
        "spec_hash": digest(spec),
        "table_hash": digest(body),
    }
    predecessors: dict[str, set[str]] = {node.node_id: set() for node in diagram.nodes}
    for edge in diagram.edges:
        predecessors[edge.target_node].add(edge.source_node)
    try:
        tuple(TopologicalSorter(predecessors).static_order())
    except CycleError:
        return RenderedVisual(**base, status="failed", error_codes=("causal_graph_has_cycle",))
    versions = renderer_versions() | {"graphviz": ".".join(map(str, graphviz.version()))}
    graph = _build_graph(diagram, title, direction)
    legend = (
        "Edge status: solid = evidenced; dashed gray = hypothesis; dotted = unknown; "
        "dashed red = disputed. Edge support does not prove causal identification."
    )
    full_caption = " ".join(part for part in (caption, *qualifications, legend) if part)
    try:
        svg = decorate_svg(graph.pipe(format="svg").decode(), full_caption)
        png = bytes(vl_convert.svg_to_png(svg, scale=2))
    except (ValueError, RuntimeError, graphviz.CalledProcessError) as error:
        return RenderedVisual(
            **base,
            status="failed",
            versions=versions,
            error_codes=(f"diagram_render_failed:{type(error).__name__}",),
            caption=full_caption,
        )
    identity = digest(
        {"spec": base["spec_hash"], "graph": base["table_hash"], "versions": versions}
    )
    objects, hashes = write_objects(
        output_dir.resolve() / visual_id / identity,
        {
            "data.json": encoded(body),
            "spec.json": encoded(spec),
            "figure.svg": svg.encode(),
            "figure.png": png,
        },
    )
    return RenderedVisual(
        **base,
        status="rendered",
        versions=versions,
        objects=objects,
        object_hashes=hashes,
        caption=full_caption,
    )
