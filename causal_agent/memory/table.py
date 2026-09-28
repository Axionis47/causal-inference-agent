"""The table: every family against every claim kind, from the families' needs and the claim table.
Code only. The flag is a cell count. Beside each cell, the address that decided it, so the matrix can say why."""

from __future__ import annotations

from collections.abc import Mapping

from causal_agent.memory.catalogue import Catalogue, FamilyNeeds
from causal_agent.memory.claims import Cell, ClaimTable, ProbeResult, Status


def _measured_keys(table: ClaimTable) -> list[str]:
    return [c.key for c in table.of_kind("measured")]


def _cell(kind: str, fam, table: ClaimTable) -> tuple[Cell, str | None]:
    """One cell and the address that decided it: the claim that is empty or refuted, the `fits` field that is unset, failed,
    or passed, or the claims themselves when the kind has no `fits` field."""
    if kind not in fam.requires:
        return "not_needed", None
    found = table.of_kind(kind) if kind == "measured" else [table.get(kind)]
    claims = [c for c in found if c is not None]
    if not claims:
        return "unknown", f"claim:{kind}"
    vague = next((c for c in claims if c.status in {"empty", "refuted"}), None)
    if vague is not None:
        return "unknown", vague.key if kind == "measured" else f"claim:{vague.key}"
    # a drafted value already says whether the family fits; the person confirms it the same turn
    decided: str | None = None
    for path, allowed in fam.fits.items():
        k, _, field = path.partition(".")
        if k != kind:
            continue
        v = table.value(kind, field)
        if v is None:
            return "unknown", f"claim:{path}"
        if v not in allowed and str(v).lower() not in {str(a).lower() for a in allowed}:
            return "does_not_fit", f"claim:{path}"
        decided = f"claim:{path}"
    return "fits", decided or ", ".join(c.key if kind == "measured" else f"claim:{c.key}" for c in claims)


def compute(cat: Catalogue, needs: Mapping[str, FamilyNeeds], table: ClaimTable, probes: list[ProbeResult]) -> Status:
    grid: dict[str, dict[str, Cell]] = {}
    set_by: dict[str, dict[str, str | None]] = {}
    struck: dict[str, str] = {}
    failed = {p.family: p for p in probes if p.passed is False}
    for name, fam in needs.items():
        cells = {kind: _cell(kind, fam, table) for kind in cat.kinds}
        grid[name] = {kind: cell for kind, (cell, _) in cells.items()}
        set_by[name] = {kind: by for kind, (_, by) in cells.items()}
        bad = [k for k, v in grid[name].items() if v == "does_not_fit"]
        if bad:
            struck[name] = f"{bad[0]} does not fit"
        elif name in failed:
            struck[name] = f"{failed[name].name}: {failed[name].detail}"
    surviving = [f for f in needs if f not in struck]
    kinds_required: set[str] = set()
    for f in surviving:
        kinds_required.update(needs[f].requires)
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
    return Status(
        table=grid,
        set_by=set_by,
        surviving=surviving,
        struck=struck,
        required=required,
        settled=settled,
        open=open_,
        ready=ready,
        contradictions=contradictions,
    )
