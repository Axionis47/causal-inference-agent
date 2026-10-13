"""Append-only trajectory records, including the desk's admitted obligations."""

from causal_agent.families.diff_in_diff.lane.state import SpecialistState


def event(state: SpecialistState, stage: str, **details) -> dict:
    h = state["handoff"]
    return {
        "stage": stage,
        "context_id": h.context_id,
        "ledger_revision": state["ledger"].revision if state.get("ledger") else None,
        "endpoint_obligations": [e.model_dump() for e in h.admission.endpoints] if h.admission else [],
        **details,
    }


def revisions(state: SpecialistState, stage: str, candidates) -> list[dict]:
    prior = {c.id: c for c in state.get("candidates", [])}
    return [event(state, stage, candidate_id=c.id, before=prior[c.id].model_dump() if c.id in prior else None, after=c.model_dump()) for c in candidates]
