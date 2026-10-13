"""Design-qualified figures drawn entirely from frozen adjustment artifacts."""

from __future__ import annotations

from causal_agent.families.adjustment.lane.support import signature
from causal_agent.viz.spec import Edge, FigureSpec, Node, Series


def causal_graph(graph: dict, names: dict[str, str], *, design_id: str) -> FigureSpec:
    t, y = graph["treatment"], graph["outcome"]

    def role(n):
        if n == t:
            return "treatment"
        if n == y:
            return "outcome"
        if n.startswith("hidden:"):
            return "hidden"
        parents = {e["src"] for e in graph["edges"] if e["dst"] == n}
        children = {e["dst"] for e in graph["edges"] if e["src"] == n}
        if t in children and y in children:
            return "confounder"
        if t in parents and y in children:
            return "mediator"
        if y in children:
            return "driver"
        return "other"

    return FigureSpec(
        id=f"graph_{design_id}",
        kind="graph",
        title=f"Causal assumptions for {design_id}",
        nodes=[Node(id=n, label=names.get(n, n), role=role(n)) for n in graph["nodes"]],
        edges=[Edge(src=e["src"], dst=e["dst"], cites=e["cites"], status=e["status"]) for e in graph["edges"]],
        note="; ".join(f"{e['src']} -> {e['dst']}: {e['status']}" for e in graph["edges"]),
        draws_on=[f"design:{design_id}.graph"],
    )


def comparisons(estimates: list) -> list[FigureSpec]:
    groups: dict[tuple, list] = {}
    for e in estimates:
        if e.error or e.modifier is not None:
            continue
        identity = signature(e)
        groups.setdefault(identity, []).append(e)
    out = []
    for n, (identity, es) in enumerate(groups.items()):
        out.append(
            FigureSpec(
                id=f"design_comparison_{n}",
                kind="interval",
                title=f"{es[0].target.get('type')} estimates for {identity[0]}",
                x_label="Frozen design",
                y_label=es[0].target.get("scale", "effect"),
                series=[Series(name="effect", x=[e.design_id for e in es], y=[e.value for e in es], lo=[e.ci_low for e in es], hi=[e.ci_high for e in es])],
                draws_on=[e.tag + ".value" for e in es],
                note="Only identical contrasts, targets, scales and frozen analysis populations are plotted together.",
            )
        )
    return out
