"""The families that have evals, by name. An explicit list, like the registry."""

from __future__ import annotations

from causal_agent.evals.spec import EvalSpec
from causal_agent.families.adjustment.evals import SPEC as ADJUSTMENT
from causal_agent.families.diff_in_diff.evals import SPEC as DIFF_IN_DIFF
from causal_agent.families.discontinuity.evals import SPEC as DISCONTINUITY

SPECS: dict[str, EvalSpec] = {s.family: s for s in (ADJUSTMENT, DIFF_IN_DIFF, DISCONTINUITY)}


def spec(family: str) -> EvalSpec:
    if family not in SPECS:
        raise SystemExit(f"no evals for {family!r}; families with evals: {', '.join(SPECS)}")
    return SPECS[family]
