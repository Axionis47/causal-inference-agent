"""The discontinuity family's evals: its cases and stored hand-offs here, its summariser and its own evaluators beside
them; the runner is causal_agent.evals."""

from __future__ import annotations

from pathlib import Path

from causal_agent.evals.spec import EvalSpec
from causal_agent.families.discontinuity.evals.evaluators import ALL
from causal_agent.families.discontinuity.evals.summarise import summarise


def _lane():
    from causal_agent.families.discontinuity.lane.graph import compile_local

    return compile_local()


SPEC = EvalSpec(
    family="discontinuity",
    dataset="causal-rd-v0",
    description="Discontinuity lane on rdrobust and rddensity: five real datasets through the router, Card and Krueger and students forced",
    prefix="rd",
    cases_dir=Path(__file__).parent,
    lane_graph=_lane,
    summarise=summarise,
    evaluators=ALL,
    skip_keys=("report", "design", "covariates"),
)
