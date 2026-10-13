"""The adjustment family's evals: its cases and stored hand-offs here, its summariser and its own evaluators beside
them; the runner is causal_agent.evals."""

from __future__ import annotations

from pathlib import Path

from causal_agent.evals.spec import EvalSpec
from causal_agent.families.adjustment.evals.evaluators import ALL
from causal_agent.families.adjustment.evals.summarise import summarise


def _lane():
    from causal_agent.families.adjustment.lane.graph import compile_local

    return compile_local()


SPEC = EvalSpec(
    family="adjustment",
    dataset="causal-adjustment-endpoints",
    description="Adjustment endpoint trajectories: real Gemini judgements over known study protocols, plus routed fixtures and explicit deficiency",
    prefix="dowhy",
    cases_dir=Path(__file__).parent,
    lane_graph=_lane,
    summarise=summarise,
    evaluators=ALL,
    skip_keys=("report", "designs", "graph"),
)
