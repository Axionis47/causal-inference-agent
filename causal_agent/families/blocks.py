"""How the desk fills each family's design block, by code, from the briefs, the claims, the beliefs and the probes. Never a
constraint: an empty field means the desk could not say, and the lane's own judgement fills what it can."""

from __future__ import annotations

from typing import Literal

from causal_agent.common.addresses import key
from causal_agent.common.contracts import AdjustmentDesign, DidDesign, RdDesign
from causal_agent.families.registry import BlockInputs


def adjustment(i: BlockInputs) -> AdjustmentDesign:
    a, briefs, beliefs = i.claims["assignment"], i.briefs, i.beliefs
    dep = [key(c) for c in a.get("depends_on") or []]
    before = [b.key for b in briefs if b.role not in ("outcome", "treatment") and b.when == "before" and b.moved_by_change is not True]
    forbidden = [b.key for b in briefs if b.role not in ("outcome", "treatment", "depends_on") and (b.when in ("after", "at") or b.moved_by_change is True)]
    excl, med, unob = beliefs.get("exclusion"), beliefs.get("mediator"), beliefs.get("unobserved")
    instrument = key(excl.column) if excl and excl.known() and excl.value and excl.column else None
    mediator = key(med.column) if med and med.known() and med.value and med.column else None
    kind = a.get("kind")
    voluntary = True if kind == "own_choice" else False if kind in ("cutoff_rule", "date_by_others", "lottery") else (True if a.get("movable") else None)
    return AdjustmentDesign(
        adjustment_candidates=list(dict.fromkeys(dep + before)),
        forbidden=list(dict.fromkeys(forbidden)),
        instrument=instrument,
        mediator=mediator,
        unobserved_confounding=unob.value if unob and unob.known() else None,
        voluntary_uptake=voluntary,
        target_units=i.scope.target,
        contrast=i.scope.contrast,
    )


def diff_in_diff(i: BlockInputs) -> DidDesign:
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


def discontinuity(i: BlockInputs) -> RdDesign:
    a, samp, briefs, beliefs, entry, treatment = i.claims["assignment"], i.claims["sampling"], i.briefs, i.beliefs, i.entry, i.treatment
    score = key(a["score_column"]) if a.get("score_column") else None
    sb = next((b for b in briefs if score and b.key == score), None)
    takeup = (
        {"column": treatment, "level": a.get("treated_level")}
        if treatment and (not score or key(treatment) != score) and a.get("treated_level") is not None
        else None
    )
    covs = [b.key for b in briefs if b.role == "candidate" and b.when == "before" and b.moved_by_change is not True]
    cluster = a.get("level_column") or (entry.get("entity") or [None])[0]
    return RdDesign(
        score=score,
        cutoff=float(a["cutoff"]) if a.get("cutoff") is not None else None,
        treated_side=a.get("treated_side"),
        cutoff_value_treated=a.get("cutoff_value_treated"),
        score_fixed_before=(sb.when == "before") if sb and sb.when != "unknown" else None,
        movable=a.get("movable"),
        takeup=takeup,
        covariates_allowed=covs,
        cluster=key(cluster) if cluster else None,
        sampled_by_side=samp.get("how") == "by_side" or bool(entry.get("sampled_by_side", False)),
        cutoff_only=beliefs.get("cutoff_only"),
    )
