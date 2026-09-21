"""The routing graph: from a memory and a question to a hand-off and a lane. Nodes are constant; the prefilter workers scale
with columns on a wide table.

    START ─ load ─ mine ─(prefilter × N, wide only)─ frame ─ fit ─ decide ─ gate ─ handoff ─ specialist:<family> ─ END

`fit` is code over the memory; `decide` is a judgement only when more than one family stands. This graph is what the old
router became; it goes into the one desk graph at stage 4.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from causal_agent.common.llm import RETRY as _retry
from causal_agent.desk.nodes import decide as D
from causal_agent.desk.nodes import frame as F
from causal_agent.desk.state import Context, RouteState
from causal_agent.families import registry as R


def build() -> StateGraph:
    b = StateGraph(RouteState, context_schema=Context)
    b.add_node("load", F.load)
    b.add_node("mine", F.mine, retry_policy=_retry)
    b.add_node("prefilter", F.prefilter, retry_policy=_retry)
    b.add_node("frame", F.frame, retry_policy=_retry)
    b.add_node("fit", D.fit)
    b.add_node("decide", D.decide, retry_policy=_retry)
    b.add_node("gate", D.gate)
    b.add_node("handoff", D.handoff)
    lanes = R.lanes()
    for name, sub in lanes.items():
        b.add_node(f"specialist_{name}", sub)
    b.add_edge(START, "load")
    b.add_edge("load", "mine")
    b.add_conditional_edges("mine", F.fan_out_prefilter, ["prefilter", "frame"])
    b.add_edge("prefilter", "frame")
    b.add_edge("frame", "fit")
    b.add_edge("fit", "decide")
    b.add_edge("decide", "gate")
    # gate returns Command(goto=decide | handoff | END)
    b.add_conditional_edges("handoff", D.route_specialist, [f"specialist_{n}" for n in lanes] + [END])
    for name in lanes:
        b.add_edge(f"specialist_{name}", END)
    return b


graph = build().compile()  # for `langgraph dev`: the platform injects its own checkpointer


def compile_local():
    """For tests and scripts: in-memory checkpointer; pass a thread_id per run."""
    return build().compile(checkpointer=InMemorySaver())
