"""What a run said about each column, written back to the memory as drafts. A relationship between columns is a claim under
the escalation rule: the lane's reading is drafted with its reason, the person confirms or corrects it, and only then does the
next run take it as settled. The gate refuses a write over a confirmed field."""

from __future__ import annotations

from causal_agent.common.contracts import Handoff
from causal_agent.memory import ops
from causal_agent.memory.catalogue import Catalogue
from causal_agent.memory.records import Memory

SOURCE = "model:roles"
MEASURE_WHY = "another measurement of the outcome"


def _edges_into(edges: list[dict], src: str, dst: str | None) -> bool:
    return dst is not None and any(e.get("src") == src and e.get("dst") == dst for e in edges)


def absorb_relations(memory: Memory, result: dict, h: Handoff, cat: Catalogue | None = None) -> list[str]:
    """The run's reading as drafts: for every column in the pack that the graph placed, `feeds_treatment` (an edge into the
    treatment) and `moves_outcome` (an edge into the outcome); for a column excluded as a measure of the outcome,
    `measures_outcome`; and from the roles rung, `same_as`, `nested_in`, `stands_for` where it named them and `may_modify` for
    every column it placed. Returns the addresses written; a result with neither a graph nor relations writes nothing."""
    design = result.get("design") if isinstance(result, dict) else None
    graph = design.get("graph") if isinstance(design, dict) else None
    graph = graph if isinstance(graph, dict) else {}
    roles = {r.get("column"): r for r in ((result.get("relations") if isinstance(result, dict) else None) or []) if isinstance(r, dict)}
    if not graph and not roles:
        return []
    t, y = graph.get("treatment"), graph.get("outcome")
    nodes = [str(n) for n in graph.get("nodes") or []]
    edges = [e for e in graph.get("edges") or [] if isinstance(e, dict)]
    excluded = {x.get("column"): str(x.get("why") or "") for x in graph.get("excluded") or [] if isinstance(x, dict)}
    updates: list[ops.Update] = []
    for b in h.columns:
        if b.role in ("outcome", "treatment") or b.key in (t, y):
            continue
        k = b.key
        if k in nodes:
            for field, dst in (("feeds_treatment", t), ("moves_outcome", y)):
                drawn = _edges_into(edges, k, dst)
                reason = "the run's graph drew this edge" if drawn else "the run's graph drew no such edge"
                updates.append(ops.Update(address=f"{b.address}.{field}", value=drawn, status="drafted", source=SOURCE, reason=reason))
        elif k in excluded and MEASURE_WHY in excluded[k]:
            updates.append(
                ops.Update(address=f"{b.address}.measures_outcome", value=True, status="drafted", source=SOURCE, reason=f"excluded as {excluded[k]}")
            )
        r = roles.get(k)
        if r is None:
            continue
        for field, key in (("same_as", "redundant_with"), ("nested_in", "nested_in"), ("stands_for", "stands_for")):
            if r.get(key):
                updates.append(ops.Update(address=f"{b.address}.{field}", value=r[key], status="drafted", source=SOURCE, reason="the roles rung read it so"))
        mod = bool(r.get("modifier_candidate"))
        updates.append(
            ops.Update(
                address=f"{b.address}.may_modify",
                value=mod,
                status="drafted",
                source=SOURCE,
                reason="the roles rung marked it as one the effect could differ by" if mod else "the roles rung saw no reason the effect would differ by it",
            )
        )
    if not updates:
        return []
    rejected = ops.apply(memory, updates, cat)
    refused = {r.split(": ", 1)[0] for r in rejected}  # the gate's line starts with the address it refused
    return [u.address for u in updates if u.address not in refused]
