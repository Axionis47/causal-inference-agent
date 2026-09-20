"""The diff-in-diff subgraph on pyfixest. Nodes are constant; workers scale with columns and placebos.

    load ─ case ─ groups ─ periods ─ shape_table ─(relate × N)─ merge_controls ─ verify ─ check_design
         ─(assess, only when flagged)─ pick_estimator ─ freeze_design ─ estimate ─(placebo × K)─ interpret ─ figures ─ assemble
    any typed stop, or an ask back ───────────────────────────────────────────▶ feasibility ─ figures ─ assemble

verify reruns only the relate workers it rejected (3 tries). assess may send a control delta back to
merge_controls (3 times). estimate may re-pick once on a fit failure. Nothing loops after an estimate exists.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from causal_agent.families.diff_in_diff.lane import nodes as N
from causal_agent.families.diff_in_diff.lane.state import SpecialistState

_retry = RetryPolicy(max_attempts=3, initial_interval=1.0)


def build() -> StateGraph:
    b = StateGraph(SpecialistState)
    b.add_node("load", N.load)
    b.add_node("case", N.case)
    b.add_node("groups", N.groups, retry_policy=_retry)
    b.add_node("periods", N.periods, retry_policy=_retry)
    b.add_node("shape_table", N.shape_table)
    b.add_node("relate", N.relate, retry_policy=_retry)
    b.add_node("merge_controls", N.merge_controls)
    b.add_node("verify", N.verify)
    b.add_node("check_design", N.check_design)
    b.add_node("assess", N.assess, retry_policy=_retry)
    b.add_node("pick_estimator", N.pick_estimator, retry_policy=_retry)
    b.add_node("freeze_design", N.freeze_design)
    b.add_node("estimate", N.estimate)
    b.add_node("placebo", N.placebo)
    b.add_node("interpret", N.interpret, retry_policy=_retry)
    b.add_node("feasibility", N.feasibility)
    b.add_node("figures", N.figures)
    b.add_node("assemble", N.assemble)

    b.add_edge(START, "load")
    # load returns Command(goto=case | feasibility)
    b.add_edge("case", "groups")
    b.add_conditional_edges("groups", N.after_groups, ["periods", "feasibility"])
    b.add_conditional_edges("periods", N.after_periods, ["shape_table", "feasibility"])
    # shape_table returns Command(goto=[Send relate...] | merge_controls | feasibility)
    b.add_edge("relate", "merge_controls")
    b.add_edge("merge_controls", "verify")
    # verify returns Command(goto=check_design | [Send relate...] | feasibility)
    b.add_conditional_edges("check_design", N.after_checks, ["assess", "pick_estimator"])
    # assess returns Command(goto=pick_estimator | merge_controls | feasibility)
    # pick_estimator returns Command(goto=freeze_design | feasibility)
    b.add_edge("freeze_design", "estimate")
    # estimate returns Command(goto=[Send placebo...] | interpret | pick_estimator | feasibility)
    b.add_edge("placebo", "interpret")
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
