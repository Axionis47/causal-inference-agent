"""The figures a lane leaves behind, checked. Every spec goes through `check_spec` against the addresses this run can
cite; a figure that draws on an address the run did not produce is dropped with a Decline, never shown."""

from __future__ import annotations

import json
from pathlib import Path

from causal_agent.common import trace
from causal_agent.common.addresses import norm_address
from causal_agent.common.contracts import CheckResult, Decline, Estimate, Falsification, Handoff, Sensitivity
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


def ok_addresses(h: Handoff, state: dict) -> set[str]:
    """Addresses produced by this snapshot, ledger, designs and their own execution records."""
    ok = set(h.addresses())

    def tree(prefix, value):
        value = value.model_dump() if hasattr(value, "model_dump") else value
        ok.add(prefix)
        if isinstance(value, dict):
            for k, v in value.items():
                tree(f"{prefix}.{k}", v)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                tree(f"{prefix}.{i}", v)

    for ident, design in (state.get("designs") or {}).items():
        tree(f"design:{ident}", design)
    ledger = state.get("ledger")
    if ledger:
        raw = ledger.model_dump() if hasattr(ledger, "model_dump") else ledger
        for ident, entry in raw.get("entries", {}).items():
            tree(ident, entry)
        for change in raw.get("history", []):
            tree(f"ledger:revision.{change['revision']}", change)
    for c in state.get("checks") or []:
        check = c if isinstance(c, CheckResult) else CheckResult.model_validate(c)
        tree(check.address, check)
    for k, v in (state.get("check_facts") or {}).items():
        tree(f"check_facts.{k}", v)
    for e in state.get("estimates") or []:
        est = e if isinstance(e, Estimate) else Estimate.model_validate(e)
        tree(est.tag, est)
        ok.update({est.tag + ".ci", est.tag + ".n", est.tag + ".p"})
    for field, cls in (("falsifications", Falsification), ("sensitivities", Sensitivity)):
        for value in state.get(field) or []:
            item = value if isinstance(value, cls) else cls.model_validate(value)
            tree(item.address, item)
    for d in state.get("declines") or []:
        ok.add(d.address if hasattr(d, "address") else str(d.get("address")))
    for episode in (state.get("episodes") or {}).values():
        for fact in episode.facts:
            ok.add(fact.address)
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
        trace.write(Path(run_dir, "figures.json"), json.dumps([s.model_dump() for s in kept], indent=2, default=str))
    return kept, declines
