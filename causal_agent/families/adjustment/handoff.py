"""How the desk fills the adjustment block, by code: the columns the offer depended on and every before-column as candidates,
the after-columns as forbidden, the beliefs as they stand. Never a constraint."""

from __future__ import annotations

from causal_agent.common.addresses import key
from causal_agent.families.adjustment.design import AdjustmentDesign
from causal_agent.families.base import BlockInputs


def design_block(i: BlockInputs) -> AdjustmentDesign:
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
