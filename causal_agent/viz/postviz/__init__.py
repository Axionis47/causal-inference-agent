"""The figures a run leaves behind, drawn from its artifacts alone: the estimate against its falsifications and sensitivity
ranges (every lane), the effect by period around the change (diff-in-diff), the estimate across bandwidths (discontinuity).
Pure over the run record; no lane state, no model. Every drawn value has an address the chat can cite."""

from __future__ import annotations

from causal_agent.desk.contracts import RunRecord
from causal_agent.viz.spec import FigureSpec, Mark, Series


def _primary(rec: RunRecord) -> dict | None:
    ests = (rec.specialist_result or {}).get("estimates") or rec.artifacts.get("estimates") or []
    return next((e for e in ests if not e.get("secondary") and e.get("error") is None and e.get("value") is not None), None)


def effect_and_refutations(rec: RunRecord) -> FigureSpec | None:
    """The estimate with its interval beside every falsification's new effect and every sensitivity range, on one axis."""
    est = _primary(rec)
    if est is None:
        return None
    contrast = est.get("contrast") or "all"
    refs = [r for r in ((rec.specialist_result or {}).get("refutations") or rec.artifacts.get("refutations") or []) if r.get("contrast") in (contrast, None)]
    x: list[str | float] = [str(est.get("method") or "estimate")]
    y: list[float | None] = [float(est["value"])]
    lo: list[float | None] = [est.get("ci_low")]
    hi: list[float | None] = [est.get("ci_high")]
    for r in refs:
        name = str(r.get("refuter") or r.get("name") or "check")
        if r.get("kind") == "sensitivity" and r.get("range_low") is not None and r.get("range_high") is not None:
            x.append(name)
            y.append((float(r["range_low"]) + float(r["range_high"])) / 2)
            lo.append(float(r["range_low"]))
            hi.append(float(r["range_high"]))
        elif r.get("new_effect") is not None:
            x.append(name + ("" if r.get("passed") is None else (" ✓" if r.get("passed") else " ✗")))
            y.append(float(r["new_effect"]))
            lo.append(None)
            hi.append(None)
    prefix = "refute" if rec.specialist == "dowhy" else "placebo"
    failed = [r.get("refuter") for r in refs if r.get("passed") is False]
    return FigureSpec(
        id=f"effect_{contrast}", kind="interval", title=f"The estimate and what was thrown at it ({contrast})", x_label="", y_label="effect",
        series=[Series(name="effect", x=x, y=y, lo=lo, hi=hi)], marks=[Mark(kind="hline", at=0.0, label="no effect")],
        note=("every falsification passed" if refs and not failed else f"failed: {', '.join(map(str, failed))}" if failed else "no falsification ran"),
        draws_on=[f"estimate:{contrast}.value", f"estimate:{contrast}.ci"] + [f"{prefix}:{contrast}.{r.get('refuter')}.new_effect" for r in refs if r.get("new_effect") is not None],
    )


def dynamic_effects(rec: RunRecord) -> FigureSpec | None:
    """Diff-in-diff: the effect by period relative to the change, with the change marked; leads near zero are the parallel-paths check."""
    dyn = (rec.specialist_result or {}).get("dynamic") or rec.artifacts.get("dynamic") or {}
    if not dyn:
        return None
    keys = sorted(dyn, key=lambda k: int(k))
    vals = [dyn[k] for k in keys]
    return FigureSpec(
        id="dynamic_effects", kind="interval", title="The effect by period relative to the change", x_label="periods from the change", y_label="effect",
        series=[Series(name="effect", x=[float(int(k)) for k in keys], y=[float(v[0]) for v in vals], lo=[float(v[1]) for v in vals], hi=[float(v[2]) for v in vals])],
        marks=[Mark(kind="vline", at=-0.5, label="the change"), Mark(kind="hline", at=0.0, label="no effect")],
        note="before the change the effect should sit at zero; after it, the estimate", draws_on=["design.dynamic"],
    )


def figures(rec: RunRecord) -> list[FigureSpec]:
    """Every post-run figure the artifacts allow, in the order the chat shows them."""
    out = [f for f in (effect_and_refutations(rec), dynamic_effects(rec)) if f is not None]
    return out
