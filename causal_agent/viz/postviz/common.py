"""Pure, design-qualified figures from recorded execution artifacts."""

from __future__ import annotations

from causal_agent.common.contracts import Estimate, Falsification
from causal_agent.viz.spec import FigureSpec, Mark, Series


def effect_plot(estimates: list[dict], falsifications: list[dict], design_id: str) -> FigureSpec | None:
    est = next((Estimate.model_validate(e) for e in estimates if e.get("design_id") == design_id and not e.get("secondary") and not e.get("modifier")), None)
    if est is None or est.error or est.value is None:
        return None
    fs = [
        Falsification.model_validate(f)
        for f in falsifications
        if f.get("design_id") == design_id and f.get("contrast") == est.contrast and f.get("new_effect") is not None
    ]
    x: list[str | float] = [est.method]
    x.extend(f.refuter for f in fs)
    values: list[float | None] = [est.value]
    values.extend(f.new_effect for f in fs)
    passed, failed = sum(f.passed is True for f in fs), sum(f.passed is False for f in fs)
    return FigureSpec(
        id=f"effect_{design_id}_{est.contrast}",
        kind="interval",
        title=f"{est.endpoint_id}: estimate and falsifications",
        series=[Series(name="effect", x=x, y=values, lo=[est.ci_low] + [None] * len(fs), hi=[est.ci_high] + [None] * len(fs))],
        marks=[Mark(kind="hline", at=0.0, label="zero")],
        y_label="effect",
        note=f"{passed} falsifications passed; {failed} failed" if fs else "No applicable falsification was run.",
        draws_on=[est.tag + ".value", est.tag + ".ci"] + [f.address + ".new_effect" for f in fs],
    )
