"""The desk: one conversation from a CSV to a run and back.

    START ─ load ─ ask_question ─(interrupt)─ mine ─ read_question ─┬─ (invalid) ─ ask_question
                                                                   └─ check ─ probe_fit ─ ask ─(convince, when ready)─ listen ─(interrupt)─┬─ infer ─ check …
                                                                                                                   └─ (run, ready) ─ fit ─ decide ─ gate ─ handoff ─ run ─ brief ─ talk ─(interrupt)─ turn ─┬─ answer ─ talk
                                                                                                                                                                                                           ├─ revise ─ check …
                                                                                                                                                                                                           ├─ requestion ─ read_question …
                                                                                                                                                                                                           └─ done ─ END

The first thing asked is the causal question, validated against the file. Then one question per turn until nothing a
surviving family needs is vague. The routing is code over the memory, one judgement only when more than one family stands.
The lane runs in its own process on the pack. After the run the chat is free: answer, revise, requestion, done."""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from causal_agent.desk.nodes import after as A
from causal_agent.desk.nodes import decide as D
from causal_agent.desk.nodes import frame as F
from causal_agent.desk.nodes import journey as J
from causal_agent.desk.state import Context, DeskState

_retry = RetryPolicy(max_attempts=3, initial_interval=1.0)


def build() -> StateGraph:
    b = StateGraph(DeskState, context_schema=Context)
    b.add_node("load", J.load)
    b.add_node("ask_question", J.ask_question)
    b.add_node("mine", F.mine, retry_policy=_retry)
    b.add_node("read_question", J.read_question, retry_policy=_retry)
    b.add_node("check", J.check)
    b.add_node("probe_fit", J.probe_fit)
    b.add_node("ask", J.ask)
    b.add_node("convince", J.convince, retry_policy=_retry)
    b.add_node("listen", J.listen)
    b.add_node("infer", J.infer, retry_policy=_retry)
    b.add_node("fit", D.fit)
    b.add_node("decide", D.decide, retry_policy=_retry)
    b.add_node("gate", J.gate)
    b.add_node("handoff", J.handoff)
    b.add_node("run", J.run)
    b.add_node("brief", A.brief)
    b.add_node("talk", A.talk)
    b.add_node("turn", A.turn, retry_policy=_retry)
    b.add_node("answer", A.answer)
    b.add_node("revise", A.revise)
    b.add_node("requestion", A.requestion)
    b.add_edge(START, "load")
    b.add_edge("load", "ask_question")
    # ask_question → mine | END, via Command
    b.add_edge("mine", "read_question")
    # read_question → ask_question | check, via Command
    b.add_edge("check", "probe_fit")
    b.add_edge("probe_fit", "ask")
    # ask → listen | convince | fit; listen → infer | fit | handoff | check | END; infer → infer | check, via Command
    b.add_edge("convince", "listen")
    b.add_edge("fit", "decide")
    b.add_edge("decide", "gate")
    # gate → decide | handoff, via Command
    b.add_edge("handoff", "run")
    b.add_edge("run", "brief")
    b.add_edge("brief", "talk")
    # talk → turn | END; turn → answer | revise | requestion | END; revise → check | talk, via Command
    b.add_edge("answer", "talk")
    b.add_edge("requestion", "read_question")
    return b


graph = build().compile()  # for `langgraph dev`: the platform injects its own checkpointer

_CONTRACTS = [
    ("causal_agent.common.contracts", n) for n in (
        "Thought", "Cited", "Candidate", "Scope", "QuestionFrame", "PrefilterVote", "NeedCheck", "FamilyVerdict", "Rejection", "FamilyDecision",
        "Handoff", "ColumnBrief", "ColumnFacts", "Provenance", "Belief", "Said", "Probe", "AdjustmentDesign", "DidDesign", "RdDesign",
    )
] + [
    ("causal_agent.desk.contracts", n) for n in ("Ask", "FieldUpdate", "Inference", "RunRecord", "NumberStated", "AfterReply", "Exchange", "Finding")
] + [
    ("causal_agent.memory.claims", n) for n in ("ProbeResult", "Status")
] + [
    ("causal_agent.memory.ops", "Open"),
]

serde = JsonPlusSerializer(allowed_msgpack_modules=_CONTRACTS)


def compile_local():
    """For the CLI and tests: in-memory checkpointer; one thread per conversation."""
    return build().compile(checkpointer=InMemorySaver(serde=serde))


def compile_with(checkpointer):
    """For the server: any checkpointer built with `serde`; the graph is otherwise the same."""
    return build().compile(checkpointer=checkpointer)
