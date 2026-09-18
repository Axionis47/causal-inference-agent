"""The table: every family against every claim kind, from the catalogue's family needs and the claim table.
Code only. The flag is a cell count."""

from __future__ import annotations

from causal_agent.intake.interview.contracts import Cell, ClaimTable, ProbeResult, Status
from causal_agent.intake.knowledge import Catalogue


def _measured_keys(table: ClaimTable) -> list[str]:
    return [c.key for c in table.of_kind("measured")]


def _cell(kind: str, fam, table: ClaimTable) -> Cell:
    if kind not in fam.requires:
        return "not_needed"
    claims = table.of_kind(kind) if kind == "measured" else [table.get(kind)] if table.get(kind) else []
    if not claims or any(c is None for c in claims):
        return "unknown"
    if any(c.status in {"empty", "refuted"} for c in claims):
        return "unknown"
    # a drafted value already says whether the family fits; the person confirms it the same turn
    for path, allowed in fam.fits.items():
        k, _, field = path.partition(".")
        if k != kind:
            continue
        v = table.value(kind, field)
        if v is None:
            return "unknown"
        if v not in allowed and str(v).lower() not in {str(a).lower() for a in allowed}:
            return "does_not_fit"
    return "fits"


def compute(cat: Catalogue, table: ClaimTable, probes: list[ProbeResult]) -> Status:
    grid: dict[str, dict[str, Cell]] = {}
    struck: dict[str, str] = {}
    failed = {p.family: p for p in probes if p.passed is False}
    for name, fam in cat.families.items():
        grid[name] = {kind: _cell(kind, fam, table) for kind in cat.kinds}
        bad = [k for k, v in grid[name].items() if v == "does_not_fit"]
        if bad:
            struck[name] = f"{bad[0]} does not fit"
        elif name in failed:
            struck[name] = f"{failed[name].name}: {failed[name].detail}"
    surviving = [f for f in cat.families if f not in struck]
    kinds_required: set[str] = set()
    for f in surviving:
        kinds_required.update(cat.families[f].requires)
    if not surviving:  # nothing fits yet: keep the family-independent claims required so the interview can continue
        kinds_required = {k for k, spec in cat.kinds.items() if not spec.uncheckable}
    a = table.get("assignment")
    assignment_known = bool(a and a.settled() and a.fields.get("kind"))
    required: list[str] = []
    for kind in cat.ordered():
        if kind.name not in kinds_required:
            continue
        if kind.uncheckable and not assignment_known:  # asked last, once the family set is known
            continue
        if kind.per_column:
            required.extend(_measured_keys(table))
        else:
            required.append(kind.name)
    settled = [k for k in required if (c := table.get(k)) and c.settled()]
    open_ = [k for k in required if k not in settled]
    contradictions = [c.key for c in table.claims.values() if c.status == "contradiction"]
    ready = not open_ and bool(surviving)
    return Status(table=grid, surviving=surviving, struck=struck, required=required, settled=settled, open=open_, ready=ready, contradictions=contradictions)
