"""One application path for DiD-specific assumptions and evidence."""

from __future__ import annotations

from causal_agent.common.contracts import Handoff
from causal_agent.families.diff_in_diff.lane.contracts import Reading
from causal_agent.lane.episode import EpisodeLog
from causal_agent.lane.ledger import Entry, Ledger, apply


def entries(reading: Reading) -> list[Entry]:
    return [
        Entry(id=f"rel:{op.kind}.{op.subject}", kind=op.kind, src=op.subject, value=op.value, status=op.status, evidence=op.evidence, reason=op.reason)
        for op in reading.operations
    ]


def commit(ledger: Ledger, reading: Reading, stage: str, resolves, *, handoff: Handoff, log: EpisodeLog):
    proposed = entries(reading)
    for entry in proposed:
        evidence = []
        for reference in entry.evidence:
            address = reference.address.removeprefix("rel:context.")
            field = handoff.fact(address)
            computed = field is not None and (field.source == "data" or (field.source or "").startswith("code:"))
            computed |= address.startswith(("probe:", "check:")) or log.resolve(reference.address)
            if computed:
                category = (
                    "diagnostic" if address.startswith("check:") else "measurement_timing" if address.endswith((".when", ".moved_by_change")) else "association"
                )
                reference = reference.model_copy(update={"source": "data", "category": category})
            evidence.append(reference)
        entry.evidence = evidence
    return apply(ledger, proposed, by=stage, resolves=resolves)


def controls(ledger: Ledger) -> tuple[list[str], list[str]]:
    supported, rivals = [], []
    for entry in ledger.entries.values():
        if entry.kind != "control" or entry.value is not True:
            continue
        if entry.status in {"confirmed", "supported"}:
            supported.append(str(entry.src))
        elif entry.status == "proposed":
            rivals.append(str(entry.src))
    return sorted(supported), sorted(rivals)
