"""Deterministic lead selection from admitted targets and pre-effect evidence."""

from __future__ import annotations

from causal_agent.common.contracts import Handoff
from causal_agent.families.discontinuity.lane.contracts import Design
from causal_agent.families.discontinuity.lane.knowledge import estimator
from causal_agent.lane.ledger import Ledger


def select_lead(handoff: Handoff, designs: dict[str, Design], ledger: Ledger) -> dict:
    locality = handoff.fact("claim:target.locality")
    if locality is None or locality.status != "confirmed" or locality.value not in {"cutoff", "window"}:
        raise ValueError("lead selection requires the admitted, sourced target locality")
    compliance = handoff.fact("claim:assignment.compliance")
    if compliance is None or handoff.admission is None:
        raise ValueError("lead selection requires admitted compliance and endpoint obligations")
    sharp = compliance.value == "sharp"
    comparison = handoff.fact("claim:target.comparison")
    if not sharp and (comparison is None or comparison.status != "confirmed" or comparison.value not in {"receipt", "eligibility"}):
        raise ValueError("fuzzy lead selection requires the admitted, sourced target comparison")
    if sharp:
        requested_contrast = "assignment"
    else:
        if comparison is None:
            raise ValueError("fuzzy lead selection requires target comparison")
        requested_contrast = comparison.value
    ranking: list[dict] = []
    for design in designs.values():
        assumption_address = design.identification_certificate["assumption_entry"]
        assumption = ledger.entries.get(assumption_address)
        admission = next(item for item in handoff.admission.endpoints if item.endpoint_id == design.endpoint_id)
        missing_obligations = sum(item.status not in {"met", "assumed", "inapplicable"} for item in admission.obligations)
        assumption_penalty = int(assumption is None or assumption.value is not True or assumption.status in {"contested", "rejected"}) + missing_obligations
        flags = design.checks.flags
        support_penalty = [sum(check.level == "hard" for check in flags), sum(check.level == "soft" for check in flags)]
        contrast_penalty = int(design.target["contrast"] != requested_contrast)
        locality_penalty = int(design.target["locality"] != locality.value)
        rank = [contrast_penalty, locality_penalty, assumption_penalty, *support_penalty, estimator(design.estimator).rank, design.id]
        ranking.append(
            {
                "design_id": design.id,
                "endpoint_id": design.endpoint_id,
                "rank": rank,
                "target_aligned": contrast_penalty == locality_penalty == 0,
                "target": design.target,
                "assumption_support": {
                    "entry": assumption_address,
                    "status": assumption.status if assumption else "missing",
                    "admitted": admission.admitted,
                    "missing_obligations": missing_obligations,
                    "evidence": [e.model_dump() for e in assumption.evidence] if assumption else [],
                },
                "support_flags": [check.model_dump() for check in flags],
                "catalogue_rank": estimator(design.estimator).rank,
                "cites": [
                    f"design:{design.id}.target",
                    f"design:{design.id}.support",
                    assumption_address,
                    *[check.address for check in flags],
                ],
            }
        )
    ranking.sort(key=lambda record: tuple(record["rank"]))
    aligned = [record for record in ranking if record["target_aligned"]]
    chosen = aligned[0]["design_id"] if aligned else None
    return {
        "policy": "Exact admitted comparison and locality, assumption support, hard/soft support flags, declared estimator catalogue rank, stable design ID.",
        "stage": "freeze_designs",
        "effect_blind": True,
        "chosen_id": chosen,
        "requested": {
            "locality": locality.value,
            "comparison": "receipt (equivalent to assignment under sharp compliance)" if sharp else comparison.value,
            "scope": handoff.scope.model_dump(),
        },
        "requested_target_identified": bool(aligned),
        "reason": "The first aligned frozen design wins the recorded ranking; different targets remain companions."
        if aligned
        else "No frozen design identifies the requested comparison and locality. Surviving different-target designs remain companions, with no replacement lead.",
        "ranking": ranking,
        "cites": ["claim:target.locality", "claim:assignment.compliance", *([] if sharp else ["claim:target.comparison"])],
    }
