"""The interview graph: a loop with one interrupt.

START ─ load ─ extract ─ check ─ probe ─ status ─ respond ─ listen ─┐
                 ▲                                                   │ (a user turn)
                 └───────────────────────────────────────────────────┘
listen ─ (run, when ready) ─▶ write_pack ─ END
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from causal_agent.intake.interview import nodes as N
from causal_agent.intake.interview.state import InterviewState

_retry = RetryPolicy(max_attempts=3, initial_interval=1.0)


def build() -> StateGraph:
    b = StateGraph(InterviewState)
    b.add_node("load", N.load)
    b.add_node("extract", N.extract, retry_policy=_retry)
    b.add_node("check", N.check)
    b.add_node("probe", N.probe)
    b.add_node("status", N.status)
    b.add_node("respond", N.respond, retry_policy=_retry)
    b.add_node("listen", N.listen)
    b.add_node("write_pack", N.write_pack)
    b.add_edge(START, "load")
    b.add_edge("load", "extract")
    # extract → extract (repair) | check, via Command
    b.add_edge("check", "probe")
    b.add_edge("probe", "status")
    b.add_edge("status", "respond")
    # respond → respond (repair) | listen | write_pack (a requested run, now ready); listen → extract | check (run on drafts) | write_pack | END, via Command
    b.add_edge("write_pack", END)
    return b


# For `langgraph dev`: the platform injects its own checkpointer, which the interrupt needs.
graph = build().compile()


# The checkpointer round-trips the interview's contracts; the serializer is told they are ours.
_serde = JsonPlusSerializer(allowed_msgpack_modules=[
    ("causal_agent.memory.claims", "ClaimTable"), ("causal_agent.memory.claims", "Claim"),
    ("causal_agent.memory.claims", "ProbeResult"), ("causal_agent.memory.claims", "Status"),
    ("causal_agent.memory.claims", "Reply"), ("causal_agent.memory.claims", "Question"),
    ("causal_agent.common.contracts", "Thought"),
])


def compile_local():
    """For the CLI and tests: in-memory checkpointer; pass a thread_id per conversation."""
    return build().compile(checkpointer=InMemorySaver(serde=_serde))
