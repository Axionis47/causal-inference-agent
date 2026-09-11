"""Deterministic source-bound visualization; scientific computation stays upstream."""

from causal.post_analysis.visualization.contracts import (
    CausalDiagram,
    Column,
    DataTable,
    DiagramEdge,
    DiagramNode,
    RenderedVisual,
    TableRowSource,
    VisualSpec,
)
from causal.post_analysis.visualization.diagram import render_dag
from causal.post_analysis.visualization.render import render_visual

__all__ = [
    "CausalDiagram",
    "Column",
    "DataTable",
    "DiagramEdge",
    "DiagramNode",
    "RenderedVisual",
    "TableRowSource",
    "VisualSpec",
    "render_dag",
    "render_visual",
]
