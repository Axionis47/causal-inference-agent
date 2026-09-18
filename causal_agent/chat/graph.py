"""The desk graph.

START ─ interview ─ run ─ brief ─ talk ─(interrupt)─ turn ─┬─ answer ──────────────── talk
                                                          ├─ revise ─ interview ─ run ─ brief ─ talk
                                                          ├─ requestion ─ run ─ brief ─ talk
                                                          └─ done ─ END
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from causal_agent.chat import nodes as N
from causal_agent.chat.state import ChatState
from causal_agent.intake.interview.graph import build as build_interview

_retry = RetryPolicy(max_attempts=3, initial_interval=1.0)


def build() -> StateGraph:
    b = StateGraph(ChatState)
    b.add_node("interview", build_interview().compile())  # no checkpointer of its own: the interrupt inside uses the parent's
    b.add_node("run", N.run)
    b.add_node("brief", N.brief)
    b.add_node("talk", N.talk)
    b.add_node("turn", N.turn, retry_policy=_retry)
    b.add_node("answer", N.answer)
    b.add_node("revise", N.revise)
    b.add_node("requestion", N.requestion)
    b.add_edge(START, "interview")
    b.add_conditional_edges("interview", N.after_interview, ["run", END])
    b.add_edge("run", "brief")
    b.add_edge("brief", "talk")
    # talk → turn | END; turn → answer | revise | requestion | END; revise → interview | talk, via Command
    b.add_edge("answer", "talk")
    b.add_edge("requestion", "run")
    return b


graph = build().compile()

_serde = JsonPlusSerializer(allowed_msgpack_modules=[
    ("causal_agent.intake.interview.contracts", "ClaimTable"), ("causal_agent.intake.interview.contracts", "Claim"),
    ("causal_agent.intake.interview.contracts", "ProbeResult"), ("causal_agent.intake.interview.contracts", "Status"),
    ("causal_agent.intake.interview.contracts", "Reply"), ("causal_agent.intake.interview.contracts", "Question"),
    ("causal_agent.intake.interview.contracts", "ClaimUpdate"), ("causal_agent.intake.interview.contracts", "FieldValue"),
    ("causal_agent.chat.contracts", "RunRecord"), ("causal_agent.chat.contracts", "AfterReply"),
    ("causal_agent.chat.contracts", "NumberStated"), ("causal_agent.chat.contracts", "Exchange"),
    ("causal_agent.common.contracts", "Thought"),
])


serde = _serde  # the server builds its own checkpointer with it


def compile_local():
    """For the CLI and tests: in-memory checkpointer; one thread per conversation."""
    return build().compile(checkpointer=InMemorySaver(serde=_serde))


def compile_with(checkpointer):
    """For the server: any checkpointer built with `serde`; the graph is otherwise the same."""
    return build().compile(checkpointer=checkpointer)
