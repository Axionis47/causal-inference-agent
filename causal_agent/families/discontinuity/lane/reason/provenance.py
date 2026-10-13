"""Evidence classification preserves study accounts versus measured diagnostics."""

from __future__ import annotations

from causal_agent.common.contracts import Handoff
from causal_agent.families.discontinuity.lane.contracts import Line
from causal_agent.lane.ledger import Entry, Evidence


def _account(address: str, fact) -> bool:
    causal_account = address == "claim:assignment.movable" or address.startswith(
        ("claim:cutoff_only.", "claim:local_randomisation.", "claim:exclusion.", "claim:monotonicity.", "claim:sampling.")
    )
    return causal_account and fact.status == "confirmed" and bool(fact.source) and fact.source.startswith(("user:", "doc:", "fixture:", "eval:"))


def _evidence(handoff: Handoff, address: str) -> Evidence:
    fact = handoff.fact(address)
    if fact is not None:
        if _account(address, fact):
            return Evidence(source="pack", address=address, category="study_account", note=f"confirmed account source {fact.source}")
        timing = address.startswith("col:") and address.endswith(".when")
        measured = bool(fact.source) and fact.source.startswith(("data", "code:"))
        return Evidence(
            source="data" if measured else "judgement" if (fact.source or "").startswith("model:") else "pack",
            address=address,
            category="measurement_timing" if timing else "diagnostic" if measured else "assumption",
            note=f"context field status {fact.status}; source {fact.source or 'unspecified'}",
        )
    if any(fact.source == address and _account(key, fact) for key, fact in handoff.facts.items()):
        return Evidence(source="pack", address=address, category="study_account", note="source of the confirmed causal study account shown in this view")
    return Evidence(source="data" if address.startswith(("col:", "probe:")) else "test", address=address, category="diagnostic")


def continuity_entry(handoff: Handoff, record: Line) -> Entry:
    evidence = [_evidence(handoff, address) for address in record.cites]
    supported = any(item.source in {"pack", "person"} and item.category == "study_account" for item in evidence)
    return Entry(
        id="rel:continuity",
        kind="continuity",
        value=record.clean,
        status="supported" if record.clean and supported else "proposed" if record.clean else "contested",
        reason=record.why,
        evidence=evidence,
    )
