"""Figures from design-qualified DiD estimates."""

from __future__ import annotations

from causal_agent.common.contracts import Estimate
from causal_agent.viz.spec import FigureSpec, Series


def design_comparison(estimates: list[Estimate]) -> FigureSpec | None:
    good = [e for e in estimates if e.error is None and e.value is not None]
    if not good:
        return None
    return FigureSpec(
        id="did_design_comparison",
        kind="interval",
        title="Effects by frozen design and target",
        x_label="design / target",
        y_label="outcome effect",
        series=[
            Series(
                name="effect",
                x=[e.design_id + ":" + str(e.target.get("component")) for e in good],
                y=[e.value for e in good],
                lo=[e.ci_low for e in good],
                hi=[e.ci_high for e in good],
            )
        ],
        note="Different target populations or weights must be read separately.",
        draws_on=[e.tag + ".value" for e in good],
    )
