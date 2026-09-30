"""The figures a lane leaves behind, checked. Every spec goes through `check_spec` against the addresses this run can
cite; a figure that draws on an address the run did not produce is dropped with a Decline, never shown."""

from __future__ import annotations

import json
from pathlib import Path

from causal_agent.common.addresses import norm_address
from causal_agent.common.contracts import Decline, Estimate, Handoff
from causal_agent.viz.spec import FigureSpec


def check_spec(spec: FigureSpec, ok: set[str]) -> list[str]:
    """The problems with a figure, by code: a graph needs nodes and arrows between them; any other kind needs values; and
    every address the figure draws on must resolve in `ok`. Empty means the figure stands."""
    problems = []
    if spec.kind == "graph":
        ids = {n.id for n in spec.nodes}
        if not spec.nodes:
            problems.append("a graph with no nodes")
        loose = [f"{e.src} -> {e.dst}" for e in spec.edges if e.src not in ids or e.dst not in ids]
        if loose:
            problems.append("arrows between nodes the figure does not have: " + ", ".join(loose))
    elif not spec.series or all(not s.x for s in spec.series):
        problems.append("no series to draw")
    elif all(y is None for s in spec.series for y in s.y):
        problems.append("every value is empty")
    okn = {norm_address(a) for a in ok}
    bad = [a for a in spec.draws_on if norm_address(a) not in okn]
    if bad:
        problems.append("draws on addresses that do not resolve: " + ", ".join(bad))
    return problems


def ok_addresses(h: Handoff, state: dict, prefix: str = "refute") -> set[str]:
    """What a figure of this run may draw on: the pack, the design, every check, estimate, refutation and decline."""
    ok = set(h.addresses())
    ok.update(
        {
            "design",
            "design.graph",
            "design.dynamic",
            "design.bandwidth",
            "design.periods",
            "design.controls",
            "design.covariates",
            "design.estimand",
            "design.assumption",
            "design.estimand.adjustment_set",
        }
    )
    d = state.get("design")
    if d is not None:
        dd = d.model_dump() if hasattr(d, "model_dump") else dict(d)
        ok.update(f"design.{k}" for k in dd)
    for c in state.get("checks") or []:
        addr = c.address if hasattr(c, "address") else f"check:{c.get('contrast')}.{c.get('name')}"
        ok.update({addr, addr + ".value", addr + ".threshold"})
    for k in state.get("check_facts") or {}:
        ok.add(f"check_facts.{k}")
    for e in state.get("estimates") or []:
        est = e if isinstance(e, Estimate) else Estimate.model_validate(e)
        ok.update({f"{est.tag}.value", f"{est.tag}.ci", f"{est.tag}.n", f"{est.tag}.p", f"estimate:{est.contrast}.{est.method}.value"})
    for r in state.get("refutations") or []:
        r = r.model_dump() if hasattr(r, "model_dump") else r
        tag = f"{prefix}:{r.get('contrast')}.{r.get('refuter')}"
        ok.update({tag, f"{tag}.new_effect", f"{tag}.p_value", f"{tag}.passed", f"{tag}.detail"})
    for d in state.get("declines") or []:
        ok.add(d.address if hasattr(d, "address") else str(d.get("address")))
    return ok


def write(run_dir: str | Path | None, specs: list[FigureSpec | None], ok: set[str], stage: str = "figures") -> tuple[list[FigureSpec], list[Decline]]:
    """The specs that pass their check, written to figures.json; the ones that do not, as declines."""
    kept, declines = [], []
    for spec in specs:
        if spec is None:
            continue
        problems = check_spec(spec, ok)
        if problems:
            declines.append(Decline(stage=stage, kind="declined", about=spec.address, check="figure.check", reason="; ".join(problems)))
            continue
        kept.append(spec)
    if run_dir:
        Path(run_dir, "figures.json").write_text(json.dumps([s.model_dump() for s in kept], indent=2, default=str))
    return kept, declines
