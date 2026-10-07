"""How a lane's graph is built and compiled. Built from the lane's declaration (`rungs.yaml`, ADR 0011): one node per rung,
its destinations the rung's exits and the stop, a conditional edge where a route picks or fans out to a worker, a plain edge
where one exit follows, then the builder's tail (feasibility, figures, assemble). Compiled as a node inside the desk's graphs,
or standalone for tests and the command line. The retry policy every judgement node shares lives here too."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from causal_agent.common.llm import RETRY as RETRY  # the policy every judgement node shares
from causal_agent.lane.rungs import STOP, TAIL, LadderSpec


def build(spec: LadderSpec, nodes: Any, state: type, *, input_schema: type | None = None) -> StateGraph:
    """The graph the declaration describes, over the lane's node functions (`nodes.<rung>`, `nodes.<route>`, and the tail's
    `feasibility`, `figures`, `assemble`). A rung that routes by Command declares its destinations, so what Studio draws and
    what the structure test reads is the declaration and nothing else."""
    b: StateGraph = StateGraph(state, input_schema=input_schema) if input_schema is not None else StateGraph(state)
    for r in spec.rungs:
        fn = getattr(nodes, r.name)
        targets = spec.exits_of(r) + ([STOP] if r.stop else [])
        kwargs: dict[str, Any] = {"retry_policy": RETRY} if r.judgement else {}
        by_command = r.route is None and (len(targets) > 1 or spec.rung(targets[0]).worker if targets[0] in spec.names else len(targets) > 1)
        if by_command:
            kwargs["destinations"] = tuple(targets)
        b.add_node(r.name, fn, **kwargs)
    for n in (STOP, *TAIL):
        b.add_node(n, getattr(nodes, n))
    b.add_edge(START, spec.names[0])
    for r in spec.rungs:
        targets = spec.exits_of(r) + ([STOP] if r.stop else [])
        if r.route is not None:
            b.add_conditional_edges(r.name, getattr(nodes, r.route), targets)
        elif len(targets) == 1 and not (targets[0] in spec.names and spec.rung(targets[0]).worker):
            b.add_edge(r.name, targets[0])
    b.add_edge(STOP, TAIL[0])
    b.add_edge(TAIL[0], TAIL[1])
    b.add_edge(TAIL[1], END)
    return b


def compile_subgraph(build: Callable[[], StateGraph], name: str) -> Any:
    """As a node inside the desk's graphs: no checkpointer of its own, no interrupts; named, so its trace root reads as the lane."""
    return build().compile(checkpointer=False, name=name)


def compile_local(build: Callable[[], StateGraph], name: str) -> Any:
    """Standalone, for tests and the command line: an in-memory checkpointer, a thread_id per run."""
    return build().compile(checkpointer=InMemorySaver(), name=name)
