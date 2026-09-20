"""How the desk fills the discontinuity block, by code: the score and cutoff, the treated side, whether the score was fixed
before the decision, the take-up column when the rule is not the change, and the covariates that may enter. Never a constraint."""

from __future__ import annotations

from causal_agent.common.addresses import key
from causal_agent.families.base import BlockInputs
from causal_agent.families.discontinuity.design import RdDesign


def design_block(i: BlockInputs) -> RdDesign:
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
