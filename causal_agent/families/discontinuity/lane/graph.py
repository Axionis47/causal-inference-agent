"""The discontinuity subgraph on rdrobust and rddensity. Nodes are constant; workers scale with columns and placebos.

    load ─ case ─ score ─ shape_table ─(relate × N)─ merge_covariates ─ verify ─ check_design
         ─(assess, only when flagged)─ pick_estimator ─ freeze_design ─ estimate ─(placebo × K)─ interpret ─ figures ─ assemble
    any typed stop, or an ask back ───────────────────────────────────────────▶ feasibility ─ figures ─ assemble

verify reruns only the relate workers it rejected (3 tries). estimate may re-pick once on a fit failure.
Nothing loops after an estimate exists.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from causal_agent.families.discontinuity.lane import nodes as N
from causal_agent.families.discontinuity.lane.state import SpecialistState
from causal_agent.lane import graph as G
from causal_agent.lane.graph import RETRY as _retry


def build() -> StateGraph:
    b = StateGraph(SpecialistState)
    b.add_node("load", N.load)
    b.add_node("case", N.case)
    b.add_node("score", N.score, retry_policy=_retry)
    b.add_node("shape_table", N.shape_table)
    b.add_node("relate", N.relate, retry_policy=_retry)
    b.add_node("merge_covariates", N.merge_covariates)
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
    b.add_edge("case", "score")
    # score returns Command(goto=shape_table | feasibility)
    # shape_table returns Command(goto=[Send relate...] | merge_covariates | feasibility)
    b.add_edge("relate", "merge_covariates")
    b.add_edge("merge_covariates", "verify")
    # verify returns Command(goto=check_design | [Send relate...] | feasibility)
    b.add_conditional_edges("check_design", N.after_checks, ["assess", "pick_estimator"])
    # assess returns Command(goto=pick_estimator | feasibility)
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
    return G.compile_subgraph(build)


def compile_local():
    """Standalone, for tests and the command line: an in-memory checkpointer, a thread_id per run."""
    return G.compile_local(build)


graph = build().compile()
