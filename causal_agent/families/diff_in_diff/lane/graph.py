"""The diff_in_diff subgraph on pyfixest, built from knowledge/rungs.yaml: the declaration is the graph. Nodes are constant;
workers scale with the tasks a fan-out sends. docs/ladders/diff_in_diff.md is the ladder as declared, written by `make ladder`."""

from __future__ import annotations

from langgraph.graph import StateGraph

from causal_agent.families.diff_in_diff.lane import nodes as N
from causal_agent.families.diff_in_diff.lane.knowledge import SPEC
from causal_agent.families.diff_in_diff.lane.state import SpecialistState
from causal_agent.lane import graph as G
from causal_agent.lane.state import LaneInput


def build() -> StateGraph:
    return G.build(SPEC, N, SpecialistState, input_schema=LaneInput)


def compile_subgraph():
    """As a node inside the desk's graphs: no checkpointer of its own, no interrupts."""
    return G.compile_subgraph(build, "lane:diff_in_diff")


def compile_local():
    """Standalone, for tests and the command line: an in-memory checkpointer, a thread_id per run."""
    return G.compile_local(build, "lane:diff_in_diff")


graph = build().compile(name="lane:diff_in_diff")
