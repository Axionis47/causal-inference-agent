"""The router graph. Nodes are constant; workers scale with columns and families.

    START ─ load_pack ─(prefilter × N, wide only)─ frame ─(test_family × F)─ decide ─ gate ─ handoff ─ specialist:<family> ─ END
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from causal_agent.router import nodes as N
from causal_agent.router.state import Context, RouterState
from causal_agent.specialists import SPECIALISTS

_retry = RetryPolicy(max_attempts=3, initial_interval=1.0)


def build() -> StateGraph:
    b = StateGraph(RouterState, context_schema=Context)
    b.add_node("load_pack", N.load_pack)
    b.add_node("prefilter", N.prefilter, retry_policy=_retry)
    b.add_node("frame", N.frame, retry_policy=_retry)
    b.add_node("test_family", N.test_family, retry_policy=_retry)
    b.add_node("decide", N.decide, retry_policy=_retry)
    b.add_node("gate", N.gate)
    b.add_node("handoff", N.handoff)
    for name, sub in SPECIALISTS.items():
        b.add_node(f"specialist_{name}", sub)

    b.add_edge(START, "load_pack")
    b.add_conditional_edges("load_pack", N.fan_out_prefilter, ["prefilter", "frame"])
    b.add_edge("prefilter", "frame")
    b.add_conditional_edges("frame", N.fan_out_families, ["test_family"])
    b.add_edge("test_family", "decide")
    b.add_edge("decide", "gate")
    # gate returns Command(goto=decide | handoff | END)
    b.add_conditional_edges("handoff", N.route_specialist, [f"specialist_{n}" for n in SPECIALISTS] + [END])
    for name in SPECIALISTS:
        b.add_edge(f"specialist_{name}", END)
    return b


# For `langgraph dev`: the platform injects its own checkpointer.
graph = build().compile()


def compile_local():
    """For tests and scripts: in-memory checkpointer; pass a thread_id per run."""
    return build().compile(checkpointer=InMemorySaver())
