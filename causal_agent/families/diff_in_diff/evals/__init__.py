"""The diff-in-diff family's evals: its cases and stored hand-offs here, its summariser and its own evaluators beside
them; the runner is causal_agent.evals."""

from __future__ import annotations

from pathlib import Path

from causal_agent.evals.spec import EvalSpec
from causal_agent.families.diff_in_diff.evals.evaluators import ALL
from causal_agent.families.diff_in_diff.evals.summarise import summarise


def _lane():
    from causal_agent.families.diff_in_diff.lane.graph import compile_local

    return compile_local()


SPEC = EvalSpec(
    family="diff_in_diff",
    dataset="causal-did-v0",
    description="Diff-in-diff lane on pyfixest: Card and Krueger through the router, cigar and marketing forced",
    prefix="did",
    cases_dir=Path(__file__).parent,
    lane_graph=_lane,
    summarise=summarise,
    evaluators=ALL,
    skip_keys=("report", "design", "controls"),
)
