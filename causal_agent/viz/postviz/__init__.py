"""Fallback figures over explicit recorded designs."""

from __future__ import annotations

from causal_agent.common.contracts import RunRecord
from causal_agent.viz.postviz.common import effect_plot
from causal_agent.viz.spec import FigureSpec


def figures(rec: RunRecord) -> list[FigureSpec]:
    result = rec.specialist_result
    return [
        f
        for design_id in result.get("designs") or {}
        if (f := effect_plot(result.get("estimates") or [], result.get("falsifications") or [], design_id)) is not None
    ]
