"""Harness-owned CausalGraphView compiler: DOT, SVG, legend, fidelity (PRD-002 §12.3; D-042)."""

from __future__ import annotations

import shutil
import subprocess
from collections import Counter
from typing import Annotated, Any, Final, Literal, Self, cast
from xml.etree import ElementTree

import graphviz  # type: ignore[import-untyped]
from graphviz.quoting import quote as dot_quote  # type: ignore[import-untyped]
from pydantic import Field, model_validator

from causal.design.contracts import _Payload, _Row
from causal.design.semantics import (
    CausalContextV1,
    ConceptStatus,
    MeasurementMapV1,
    RoleLedgerV1,
    RoleName,
)
from causal.shared.canonical import content_hash
from causal.shared.contracts import ArtifactRef, Identity, Sha256Hex
from causal.shared.envelope import EpistemicStatus

__all__ = [
    "GRAPH_VIEW_INFIDELITY",
    "LEGEND_TEXT",
    "RENDERER_UNAVAILABLE",
    "CausalGraphViewV1",
    "GraphEdgeViewV1",
    "GraphNodeViewV1",
    "GraphSpec",
    "RendererError",
    "build_graph_spec",
    "graphviz_version",
    "render_causal_graph",
]

RENDERER_UNAVAILABLE: Final = "renderer_unavailable"
GRAPH_VIEW_INFIDELITY: Final = "graph_view_infidelity"
# Edge status is carried by line style plus text, never by color (§12.3).
_EDGE_STYLE: Final[dict[EpistemicStatus, tuple[str, str]]] = {
    EpistemicStatus.EVIDENCED: ("solid", ""), EpistemicStatus.HYPOTHESIS: ("dashed", ""),
    EpistemicStatus.DISPUTED: ("dotted", ""), EpistemicStatus.UNKNOWN: ("dotted", "?")}
_SUFFIX: Final[dict[ConceptStatus, str]] = {
    ConceptStatus.OBSERVED: "(observed)", ConceptStatus.PROXY_MEASURED: "(proxy)",
    ConceptStatus.UNMEASURED: "(unmeasured)"}
_EDGE_FIELDS: Final = {"edge_id", "source_concept_id", "target_concept_id", "status"}
LEGEND_TEXT: Final = ("Legend - line style and text carry every status; color carries none.\n"
    "Edge solid: evidenced. Edge dashed: hypothesis. Edge dotted: disputed. Edge dotted and "
    "labelled ?: unknown.\nNode box: the treatment or the outcome, also named in its label.\n"
    "Node dashed outline: an unmeasured concept.\nNode label suffix (observed), (proxy), or "
    "(unmeasured): measurement status; suffix 'roles: ...': the role claims for this frame.")


class GraphNodeViewV1(_Row):
    concept_id: Identity
    label: str
    status: ConceptStatus
    roles: tuple[RoleName, ...]


class GraphEdgeViewV1(_Row):
    edge_id: Identity
    source_concept_id: Identity
    target_concept_id: Identity
    status: EpistemicStatus


class CausalGraphViewV1(_Payload):
    """One deterministic, accessible rendering of a causal graph alternative."""

    schema_version: Literal["causal-graph-view.v1"] = "causal-graph-view.v1"
    parents: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]
    nodes: Annotated[tuple[GraphNodeViewV1, ...], Field(min_length=2)]
    edges: tuple[GraphEdgeViewV1, ...]
    selected_alternative_id: Identity | None
    layout_direction: Literal["TB", "LR"]
    renderer_profile: Identity
    legend_text: str
    disclosure_text: str
    spec_hash: Sha256Hex
    svg: str
    accessible_summary: str
    node_edge_table: str
    renderer_version: Identity
    theme_version: Identity
    validator_version: Identity
    validation_status: Identity

    @model_validator(mode="after")
    def _edges_reference_declared_nodes(self) -> Self:
        declared = {node.concept_id for node in self.nodes}
        for edge in self.edges:
            unknown = sorted({edge.source_concept_id, edge.target_concept_id} - declared)
            if unknown:
                raise ValueError(
                    f"edge {edge.edge_id} references undeclared concepts {unknown}")
        return self


# What build_graph_spec returns: node rows, edge rows, the canonical spec dict, and DOT source.
GraphSpec = tuple[tuple[GraphNodeViewV1, ...], tuple[GraphEdgeViewV1, ...], dict[str, Any], str]


class RendererError(ValueError):
    """The renderer cannot produce a faithful view; `code` is a stable contract value."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


def graphviz_version() -> str | None:
    if (binary := shutil.which("dot")) is None:
        return None
    probe = subprocess.run([binary, "-V"], capture_output=True, text=True, check=False)
    text = (probe.stderr.strip() or probe.stdout.strip()).removeprefix("dot - ")
    return text.split("version ", 1)[-1].split(" (", 1)[0].strip() or "unknown"


# One labelled row per concept: measurement status, role annotations, and frame position (§12.3).
def _node_rows(context: CausalContextV1, ledger: RoleLedgerV1,
               measurements: MeasurementMapV1) -> tuple[GraphNodeViewV1, ...]:
    status = {c.concept_id: c.status for c in measurements.concepts}
    seats = {context.frame.treatment: " [treatment]", context.frame.outcome: " [outcome]"}
    rows = []
    for cid in sorted(set(context.concept_ids)):
        held = tuple(sorted({c.role for c in ledger.claims if c.concept_id == cid}))
        state = status.get(cid, ConceptStatus.UNMEASURED)
        shown = f" roles: {', '.join(role.value for role in held)}" if held else ""
        rows.append(GraphNodeViewV1(concept_id=cid, status=state, roles=held,
                                    label=f"{cid}{seats.get(cid, '')} {_SUFFIX[state]}{shown}"))
    return tuple(rows)


def build_graph_spec(context: CausalContextV1, ledger: RoleLedgerV1, measurements: MeasurementMapV1,
                     *, alternative_id: str | None = None, layout_direction: str = "TB",
                     renderer_profile: str = "design-graph-renderer.v1") -> GraphSpec:
    """The deterministic half of the view - rows, canonical spec, DOT - needing no `dot` binary."""
    nodes = _node_rows(context, ledger, measurements)
    # A named alternative's edge set fully replaces the base one (validators.py wall 5).
    replacements = {alt.alternative_id: alt.edges for alt in context.alternatives}
    if alternative_id is not None and alternative_id not in replacements:
        raise RendererError(f"unknown graph alternative {alternative_id!r}", GRAPH_VIEW_INFIDELITY)
    ranked = sorted(replacements.get(alternative_id or "", context.edges),
                    key=lambda e: (e.source_concept_id, e.target_concept_id, e.edge_id))
    edges = tuple(GraphEdgeViewV1(**e.model_dump(include=_EDGE_FIELDS)) for e in ranked)
    spec: dict[str, Any] = {
        "nodes": [n.model_dump(mode="json") for n in nodes], "layout_direction": layout_direction,
        "edges": [e.model_dump(mode="json") for e in edges], "renderer_profile": renderer_profile,
        "selected_alternative_id": alternative_id}
    digraph = graphviz.Digraph("causal_graph", graph_attr={"rankdir": layout_direction})
    boxed = {context.frame.treatment, context.frame.outcome}
    for node in nodes:
        digraph.node(graphviz.escape(node.concept_id), label=graphviz.escape(node.label),
                     **({"style": "dashed"} if node.status is ConceptStatus.UNMEASURED else {}),
                     **({"shape": "box"} if node.concept_id in boxed else {}))
    for view in edges:
        line, mark = _EDGE_STYLE[view.status]
        # Digraph.edge interprets every colon as a node:port separator, even when the
        # canonical concept ID contains a literal colon. Quote whole endpoints as node IDs.
        source, target = (dot_quote(graphviz.escape(cid)) for cid in (
            view.source_concept_id, view.target_concept_id))
        attributes = f'label="{mark}" style={line}' if mark else f"style={line}"
        digraph.body.append(f"\t{source} -> {target} [{attributes}]\n")
    return nodes, edges, spec, str(digraph.source)


# Disclosure sentence, accessible summary, and node-edge table: the non-visual view (§12.3).
def _texts(context: CausalContextV1, selected: str | None, nodes: tuple[GraphNodeViewV1, ...],
           edges: tuple[GraphEdgeViewV1, ...]) -> tuple[str, str, str]:
    labels = {alt.alternative_id: alt.label for alt in context.alternatives}
    shown = "the base graph" if selected is None else f"alternative '{labels[selected]}'"
    disclosure = (f"This view renders {shown}; {len(context.alternatives)} alternative graph "
                  "view(s) are kept and rendered separately, never merged into this diagram.")
    summary = " ".join([f"{len(nodes)} concepts and {len(edges)} causal edges.",
        *sorted(f"{e.source_concept_id} causes {e.target_concept_id} ({e.status.value})."
                for e in edges)])
    table = "\n".join(["concept | status | roles",
        *(f"{n.concept_id} | {n.status.value} | {', '.join(r.value for r in n.roles) or 'none'}"
          for n in nodes), "edge | status",
        *(f"{e.source_concept_id} -> {e.target_concept_id} | {e.status.value}" for e in edges)])
    return disclosure, summary, table


# SVG comes from the local `dot` binary only; an absent binary is a typed blocker (D-042).
def _render_svg(dot_source: str) -> str:
    if (binary := shutil.which("dot")) is None:
        raise RendererError("no local Graphviz `dot` binary; nothing is substituted for it",
                            RENDERER_UNAVAILABLE)
    done = subprocess.run([binary, "-Tsvg"], input=dot_source, capture_output=True, text=True,
                          check=False)
    if done.returncode != 0 or "<svg" not in done.stdout:
        raise RendererError(f"`dot -Tsvg` failed: {done.stderr.strip()}", RENDERER_UNAVAILABLE)
    return done.stdout


# Wall 8: every validated node, edge, and status survived rendering; nothing was invented.
def _check_fidelity(svg: str, _dot: str, table: str, spec: GraphSpec) -> None:
    nodes, edges, _, _ = spec
    try:
        root = ElementTree.fromstring(svg)
    except ElementTree.ParseError as error:
        raise RendererError("the rendered view is not valid SVG", GRAPH_VIEW_INFIDELITY) from error
    rendered: dict[str, Counter[str]] = {"node": Counter(), "edge": Counter()}
    for group in root.iter():
        kind = group.attrib.get("class", "")
        if kind in rendered and (title := group.find("{*}title")) is not None:
            rendered[kind][title.text or ""] += 1
    lost = [n.concept_id for n in nodes
            if rendered["node"][n.concept_id] != 1
            or f"{n.concept_id} | {n.status.value} |" not in table]
    pairs = {e.edge_id: f"{e.source_concept_id} -> {e.target_concept_id}" for e in edges}
    wanted_edges = Counter(f"{e.source_concept_id}->{e.target_concept_id}" for e in edges)
    # Compare the actual rendered groups, not source DOT substrings or labels: disconnected
    # canonical nodes plus invented node/port edges must never pass the fidelity gate.
    if rendered["node"] != Counter(n.concept_id for n in nodes):
        lost.append("svg_node_set")
    if rendered["edge"] != wanted_edges:
        lost.append("svg_edge_set")
    lost += [e.edge_id for e in edges if f"{pairs[e.edge_id]} | {e.status.value}" not in table]
    if lost:
        raise RendererError(f"the rendered view lost {sorted(set(lost))}", GRAPH_VIEW_INFIDELITY)


def render_causal_graph(
    *, context: CausalContextV1, ledger: RoleLedgerV1, measurement_map: MeasurementMapV1,
    parents: tuple[ArtifactRef, ...], selected_alternative_id: str | None,
    layout_direction: str = "TB", renderer_profile: str = "design-graph-renderer.v1",
    theme_version: str = "design-graph-theme.v1", validator_version: str = "design-validators.v1",
) -> CausalGraphViewV1:
    """Compile one reviewable CausalGraphView; the local Graphviz binary is required (D-042)."""
    nodes, edges, canonical, dot = spec = build_graph_spec(
        context, ledger, measurement_map, alternative_id=selected_alternative_id,
        layout_direction=layout_direction, renderer_profile=renderer_profile)
    disclosure, summary, table = _texts(context, selected_alternative_id, nodes, edges)
    svg = _render_svg(dot)
    _check_fidelity(svg, dot, table, spec)
    if (version := graphviz_version()) is None:
        raise RendererError("the local Graphviz version is unreadable", RENDERER_UNAVAILABLE)
    return CausalGraphViewV1(
        parents=parents, nodes=nodes, edges=edges, spec_hash=content_hash(canonical), svg=svg,
        selected_alternative_id=selected_alternative_id, renderer_profile=renderer_profile,
        layout_direction=cast(Literal["TB", "LR"], layout_direction), legend_text=LEGEND_TEXT,
        disclosure_text=disclosure, accessible_summary=summary, node_edge_table=table,
        renderer_version=f"graphviz-{version}", validation_status="validated",
        theme_version=theme_version, validator_version=validator_version)
