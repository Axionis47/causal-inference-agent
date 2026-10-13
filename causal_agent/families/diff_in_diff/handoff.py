"""How the desk fills the diff-in-diff block, by code: the unit and time columns, the treated group, the periods before the
change, the controls that may enter, and the beliefs as they stand. Never a constraint."""

from __future__ import annotations

from typing import Literal, cast

from causal_agent.common.addresses import key
from causal_agent.families.base import BlockInputs
from causal_agent.families.diff_in_diff.design import DidDesign


def design_block(i: BlockInputs) -> DidDesign:
    a, ch, g, briefs, entry, treatment = i.claims["assignment"], i.claims["change"], i.claims["grain"], i.briefs, i.entry, i.treatment
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
    if ch.get("adoption_column"):
        treated_group = {**treated_group, "cohort_column": key(ch["adoption_column"])}
    target = i.claims.get("target", {})
    before = g.get("before_column") or next((b.key for b in briefs if b.when == "before" and b.measures_outcome is True), None)
    aggregation = target.get("aggregation") or (
        i.scope.target if i.scope.target in {"horizon", "horizon_profile", "cohort", "cohort_profile", "cohort_horizon"} else "pooled"
    )
    return DidDesign(
        unit=key(unit) if unit else None,
        before_column=before if not tkey else None,
        after_column=g.get("after_column") or next((b.key for b in briefs if b.role == "outcome"), None) if not tkey else None,
        aggregation=cast(Literal["pooled", "horizon", "horizon_profile", "cohort", "cohort_profile", "cohort_horizon"], aggregation),
        target_weighting=target.get("weighting"),
        horizons=target.get("horizons") or [],
        cohorts=[str(c) for c in (target.get("cohorts") or [])],
        prefix_scope_accepted=target.get("prefix_scope_accepted") is True,
        adoption_representation="cohort_column"
        if ch.get("adoption_column")
        else "indicator"
        if any(b.key == key(treatment or "") and b.facts.switch and b.facts.switch.get("entities_that_switch", 0) for b in briefs)
        else "label",
        time=tkey,
        period_kind=period_kind,
        change_period=str(ch["period_value"]) if ch.get("period_value") is not None else None,
        treated_group=treated_group,
        cluster_level=key(a["level_column"]) if a.get("level_column") else (key(unit) if unit else None),
    )
