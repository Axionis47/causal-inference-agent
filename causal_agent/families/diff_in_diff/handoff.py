"""How the desk fills the diff-in-diff block, by code: the unit and time columns, the treated group, the periods before the
change, the controls that may enter, and the beliefs as they stand. Never a constraint."""

from __future__ import annotations

from typing import Literal

from causal_agent.common.addresses import key
from causal_agent.families.base import BlockInputs
from causal_agent.families.diff_in_diff.design import DidDesign


def design_block(i: BlockInputs) -> DidDesign:
    a, ch, g, briefs, beliefs, entry, treatment = i.claims["assignment"], i.claims["change"], i.claims["grain"], i.briefs, i.beliefs, i.entry, i.treatment
    time = ch.get("date_column") or entry.get("time")
    tkey = key(time) if time else None
    units = [c for c in (g.get("key_columns") or []) if key(c) != tkey] if g.get("panel") is True else []
    unit = units[0] if units else (entry.get("entity") or [None])[0]
    tb = next((b for b in briefs if tkey and b.key == tkey), None)
    period_kind: Literal["date", "integer"] | None = ("date" if tb.facts.kind == "datetime" else "integer") if tb else None
    treated_group = (
        {"column": a["treatment_column"], "level": a["treated_level"]}
        if a.get("treatment_column") and a.get("treated_level") is not None
        else ({"column": treatment, "level": a["treated_level"]} if treatment and a.get("treated_level") is not None else {})
    )
    pre = next((p for p in i.probes if p.family == "diff_in_diff" and p.name == "pre_periods"), None)
    controls = [
        b.key
        for b in briefs
        if b.role == "candidate" and b.moved_by_change is not True and (b.when == "before" or b.facts.varies_over in ("entity", "time", "both"))
    ]
    return DidDesign(
        unit=key(unit) if unit else None,
        time=tkey,
        period_kind=period_kind,
        change_period=str(ch["period_value"]) if ch.get("period_value") is not None else None,
        treated_group=treated_group,
        pre_periods=int(pre.value) if pre and pre.value is not None else None,
        controls_allowed=controls,
        cluster_level=key(a["level_column"]) if a.get("level_column") else (key(unit) if unit else None),
        trend_belief=beliefs.get("trend_continues"),
        spillover=beliefs.get("spillover"),
    )
