"""The adjustment-lane subgraph on DoWhy. Nodes are constant; workers scale with columns and contrasts.

    load ─ case ─ contrast ─(relate × N)─ merge_graph ─ verify_graph ─ identify ─ check_design
         ─(assess, only when flagged)─ pick_estimator ─ freeze_design ─(analyse × C)─ after_analyse
         ─(interpret × C)─ figures ─ assemble ─ END
    any typed stop, or an ask back ─────────────────────────────▶ feasibility ─ figures ─ assemble

verify_graph reruns only the relate workers it rejected (3 tries). assess may send a revision delta back
to merge_graph (3 times). after_analyse may re-pick the estimator once on a fit failure. Nothing loops after
an estimate exists.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from causal_agent.specialists.dowhy import nodes as N
from causal_agent.specialists.dowhy.state import SpecialistState

_retry = RetryPolicy(max_attempts=3, initial_interval=1.0)


def build() -> StateGraph:
    b = StateGraph(SpecialistState)
    b.add_node("load", N.load)
    b.add_node("case", N.case)
    b.add_node("contrast", N.contrast, retry_policy=_retry)
    b.add_node("relate", N.relate, retry_policy=_retry)
    b.add_node("merge_graph", N.merge_graph)
    b.add_node("verify_graph", N.verify_graph)
    b.add_node("identify", N.identify)
    b.add_node("check_design", N.check_design)
    b.add_node("assess", N.assess, retry_policy=_retry)
    b.add_node("pick_estimator", N.pick_estimator, retry_policy=_retry)
    b.add_node("freeze_design", N.freeze_design)
    b.add_node("analyse", N.analyse)
    b.add_node("after_analyse", N.after_analyse)
    b.add_node("interpret", N.interpret, retry_policy=_retry)
    b.add_node("feasibility", N.feasibility)
    b.add_node("figures", N.figures)
    b.add_node("assemble", N.assemble)

    b.add_edge(START, "load")
    # load returns Command(goto=case | feasibility)
    b.add_edge("case", "contrast")
    b.add_conditional_edges("contrast", N.fan_out_relate, ["relate", "merge_graph", "feasibility"])
    b.add_edge("relate", "merge_graph")
    b.add_edge("merge_graph", "verify_graph")
    # verify_graph returns Command(goto=identify | [Send relate...] | feasibility)
    # identify returns Command(goto=check_design | feasibility)
    b.add_conditional_edges("check_design", N.after_checks, ["assess", "pick_estimator"])
    # assess returns Command(goto=pick_estimator | merge_graph | feasibility)
    # pick_estimator returns Command(goto=freeze_design | feasibility)
    b.add_conditional_edges("freeze_design", N.fan_out_analyse, ["analyse"])
    b.add_edge("analyse", "after_analyse")
    # after_analyse returns Command(goto=[Send interpret...] | pick_estimator | figures | feasibility)
    b.add_edge("interpret", "figures")
    b.add_edge("feasibility", "figures")
    b.add_edge("figures", "assemble")
    b.add_edge("assemble", END)
    return b


def compile_subgraph():
    """As a node inside the desk's graphs: no checkpointer of its own, no interrupts."""
    return build().compile(checkpointer=False)


def compile_local():
    """Standalone, for tests and the CLI: in-memory checkpointer, thread_id per run."""
    return build().compile(checkpointer=InMemorySaver())


graph = build().compile()
