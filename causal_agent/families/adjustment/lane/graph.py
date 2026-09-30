"""The adjustment-lane subgraph on DoWhy. Nodes are constant; workers scale with contrasts.

    load ─ case ─ pair ─ mechanism ─ time ─ roles ─ post_roles ─ merge_graph ─ verify_graph ─ road ─ heterogeneity ─ threats
         ─ check_design ─(assess, only when flagged)─ pick_estimator ─ freeze_design ─(analyse × C)─ after_analyse
         ─(interpret × C)─ figures ─ assemble ─ END
    any typed stop, or an ask back ─────────────────────────────▶ feasibility ─ figures ─ assemble

pair, mechanism, time, roles, post_roles, road, heterogeneity and threats are the rungs of the ladder: code where the pack
settles the rung, a bounded episode where it does not, each gated up to three tries inside the episode. assess may send a revision delta back to
merge_graph (3 times). after_analyse may re-pick the estimator once on a fit failure. Nothing loops after an estimate exists.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from causal_agent.families.adjustment.lane import nodes as N
from causal_agent.families.adjustment.lane.state import SpecialistState
from causal_agent.lane import graph as G
from causal_agent.lane.graph import RETRY as _retry


def build() -> StateGraph:
    b = StateGraph(SpecialistState)
    b.add_node("load", N.load)
    b.add_node("case", N.case)
    b.add_node("pair", N.pair, retry_policy=_retry)
    b.add_node("mechanism", N.mechanism, retry_policy=_retry)
    b.add_node("time", N.timing)
    b.add_node("roles", N.roles, retry_policy=_retry)
    b.add_node("post_roles", N.post_roles, retry_policy=_retry)
    b.add_node("merge_graph", N.merge_graph)
    b.add_node("verify_graph", N.verify_graph)
    b.add_node("road", N.road, retry_policy=_retry)
    b.add_node("heterogeneity", N.heterogeneity, retry_policy=_retry)
    b.add_node("threats", N.threats)
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
    b.add_edge("case", "pair")
    # pair returns Command(goto=mechanism | feasibility); mechanism returns Command(goto=time | feasibility)
    b.add_edge("time", "roles")
    # roles returns Command(goto=post_roles | feasibility); post_roles returns Command(goto=merge_graph | feasibility)
    b.add_edge("merge_graph", "verify_graph")
    # verify_graph returns Command(goto=road | feasibility); road returns Command(goto=heterogeneity | feasibility)
    # heterogeneity returns Command(goto=threats | feasibility)
    b.add_edge("threats", "check_design")
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
    return G.compile_subgraph(build)


def compile_local():
    """Standalone, for tests and the command line: an in-memory checkpointer, a thread_id per run."""
    return G.compile_local(build)


graph = build().compile()
