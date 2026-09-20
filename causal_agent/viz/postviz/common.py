"""Post-run figures every lane can draw, pure over plain dicts: the estimate against its falsifications and sensitivity
ranges, and the effect by period relative to a change. A lane's figures node calls these on its own artifacts; the desk's
fallback (`postviz.figures`) calls them on a run record."""

from __future__ import annotations

from causal_agent.viz.spec import FigureSpec, Mark, Series


def primary(estimates: list[dict]) -> dict | None:
    return next((e for e in estimates if not e.get("secondary") and e.get("error") is None and e.get("value") is not None), None)


def effect_and_refutations(estimates: list[dict], refutations: list[dict], prefix: str) -> FigureSpec | None:
    """The estimate with its interval beside every falsification's new effect and every sensitivity range, on one axis.
    `prefix` is the lane's refutation address prefix: refute for dowhy, placebo for the others."""
    est = primary(estimates)
    if est is None:
        return None
    contrast = est.get("contrast") or "all"
    refs = [r for r in refutations if r.get("contrast") in (contrast, None)]
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
    failed = [r.get("refuter") for r in refs if r.get("passed") is False]
    return FigureSpec(
        id=f"effect_{contrast}", kind="interval", title=f"The estimate and what was thrown at it ({contrast})", x_label="", y_label="effect",
        series=[Series(name="effect", x=x, y=y, lo=lo, hi=hi)], marks=[Mark(kind="hline", at=0.0, label="no effect")],
        note=("every falsification passed" if refs and not failed else f"failed: {', '.join(map(str, failed))}" if failed else "no falsification ran"),
        draws_on=[f"estimate:{contrast}.value", f"estimate:{contrast}.ci"] + [f"{prefix}:{contrast}.{r.get('refuter')}.new_effect" for r in refs if r.get("new_effect") is not None],
    )


def event_study(dynamic: dict, contrast: str | None = None, draws_on: list[str] | None = None) -> FigureSpec | None:
    """The effect by period relative to the change, with the change marked; leads near zero are the parallel-paths check."""
    if not dynamic:
        return None
    keys = sorted(dynamic, key=lambda k: int(k))
    vals = [dynamic[k] for k in keys]
    return FigureSpec(
        id="dynamic_effects" if contrast is None else f"event_study_{contrast}", kind="interval", title="The effect by period relative to the change",
        x_label="periods from the change", y_label="effect",
        series=[Series(name="effect", x=[float(int(k)) for k in keys], y=[float(v[0]) for v in vals], lo=[float(v[1]) for v in vals], hi=[float(v[2]) for v in vals])],
        marks=[Mark(kind="vline", at=-0.5, label="the change"), Mark(kind="hline", at=0.0, label="no effect")],
        note="before the change the effect should sit at zero; after it, the estimate", draws_on=list(draws_on or ["design.dynamic"]),
    )
