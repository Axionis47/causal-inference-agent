"""The adjustment lane's own figures, pure over its artifacts: the causal graph it drew, and the balance of every column it
adjusted for before and after weighting on the propensity. Every drawn value rests on an address the run produced."""

from __future__ import annotations

from causal_agent.common.contracts.base import _slug
from causal_agent.viz.spec import Edge, FigureSpec, Mark, Node, Role, Series

ROLE_WORDS = {
    "confounder": "drives both",
    "driver": "drives the outcome",
    "treatment driver": "drives the treatment",
    "mediator": "carries the effect",
    "instrument": "pushes units in",
    "hidden": "not in the file",
    "excluded": "set aside",
}


def _role(graph: dict, node: str) -> Role:
    t, y = graph.get("treatment"), graph.get("outcome")
    edges = graph.get("edges") or []
    if node == t:
        return "treatment"
    if node == y:
        return "outcome"
    if node == "unobserved":
        return "hidden"
    to_t = any(e.get("src") == node and e.get("dst") == t for e in edges)
    to_y = any(e.get("src") == node and e.get("dst") == y for e in edges)
    from_t = any(e.get("src") == t and e.get("dst") == node for e in edges)
    if to_t and to_y:
        return "confounder"
    if from_t and to_y:
        return "mediator"
    if to_t and not to_y:
        return "instrument" if not from_t else "driver"
    if to_y:
        return "driver"
    return "other"


def causal_graph(graph: dict, names: dict[str, str] | None = None) -> FigureSpec | None:
    """The graph the lane built: one node per column with its role, one arrow per edge with the addresses it rests on, and the
    columns the lane set aside as nodes with no arrows."""
    if not graph or not graph.get("nodes"):
        return None
    names = names or {}
    nodes = [Node(id=n, label=names.get(n, n), role=_role(graph, n)) for n in graph["nodes"]]
    nodes += [
        Node(id=x["column"], label=names.get(x["column"], x["column"]), role="excluded")
        for x in graph.get("excluded") or []
        if x.get("column") not in graph["nodes"]
    ]
    edges = [Edge(src=e["src"], dst=e["dst"], cites=list(e.get("cites") or [])) for e in graph.get("edges") or []]
    cites = sorted({c for e in edges for c in e.cites})
    conf = [n.label for n in nodes if n.role == "confounder"]
    note = ("adjusted for " + ", ".join(conf) if conf else "nothing drives both the treatment and the outcome in this graph") + (
        "; a hidden factor is drawn in" if any(n.role == "hidden" for n in nodes) else ""
    )
    return FigureSpec(
        id="causal_graph", kind="graph", title="The graph the analysis drew", nodes=nodes, edges=edges, note=note, draws_on=["design.graph"] + cites
    )


def balance(facts: dict[str, dict], contrast: str, threshold: float | None = None, names: dict[str, str] | None = None) -> FigureSpec | None:
    """The standardised mean difference of every adjustment column between the arms, before and after weighting on the score."""
    if not facts:
        return None
    names = names or {}
    cols = list(facts)
    before = [facts[c].get("before") for c in cols]
    after = [facts[c].get("after") for c in cols]
    x = [names.get(c, c) for c in cols]
    marks = [Mark(kind="hline", at=float(threshold), label="soft threshold")] if threshold is not None else []
    worst = max((v for v in after if v is not None), default=0.0)
    return FigureSpec(
        id=f"balance_{contrast}",
        kind="bars",
        title="How alike the arms are on what was adjusted for",
        x_label="",
        y_label="standardised mean difference",
        series=[Series(name="before adjustment", x=x, y=before), Series(name="after weighting on the score", x=x, y=after)],
        marks=marks,
        note=f"after weighting the largest difference is {worst:.2f}",
        draws_on=[f"check:{contrast}.balance.{c}" for c in cols] + ["check_facts.balance"],
    )


def effect_by_modifier(estimates: list[dict], contrast: str, names: dict[str, str] | None = None) -> FigureSpec | None:
    """The effect within each level of each modifier the design named, with its interval, beside the primary estimate."""
    names = names or {}
    within = [e for e in estimates if e.get("contrast") == contrast and e.get("modifier") is not None and e.get("error") is None and e.get("value") is not None]
    if not within:
        return None
    prim = next(
        (e for e in estimates if e.get("contrast") == contrast and e.get("modifier") is None and not e.get("secondary") and e.get("error") is None), None
    )
    x: list[str | float] = []
    y: list[float | None] = []
    lo: list[float | None] = []
    hi: list[float | None] = []
    draws_on: list[str] = []
    if prim is not None:
        x.append("all rows")
        y.append(float(prim["value"]))
        lo.append(prim.get("ci_low"))
        hi.append(prim.get("ci_high"))
        draws_on += [f"estimate:{contrast}.value", f"estimate:{contrast}.ci"]
    for e in within:
        x.append(f"{names.get(e['modifier'], e['modifier'])} = {e['level']}")
        y.append(float(e["value"]))
        lo.append(e.get("ci_low"))
        hi.append(e.get("ci_high"))
        draws_on.append(f"estimate:{contrast}.by.{_slug(e['modifier'])}.{_slug(e['level'])}.value")
    spread = max(float(e["value"]) for e in within) - min(float(e["value"]) for e in within)
    return FigureSpec(
        id=f"effect_by_modifier_{contrast}",
        kind="interval",
        title="The effect within each level of the modifiers",
        x_label="",
        y_label="effect",
        series=[Series(name="effect", x=x, y=y, lo=lo, hi=hi)],
        marks=[Mark(kind="hline", at=0.0, label="no effect")],
        note=f"the effect ranges {spread:.3g} across the levels shown",
        draws_on=draws_on,
    )
