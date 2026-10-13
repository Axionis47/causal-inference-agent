"""Source-backed relationship proposals return to memory as drafts for human review."""

from __future__ import annotations

from causal_agent.common.addresses import key
from causal_agent.common.contracts import Handoff
from causal_agent.memory import ops
from causal_agent.memory.catalogue import Catalogue
from causal_agent.memory.records import Memory


def absorb_relations(memory: Memory, result: dict, h: Handoff, cat: Catalogue | None = None) -> list[str]:
    """Write explicit ledger assertions. A missing arrow never implies a negative claim."""
    entries = (result.get("ledger") or {}).get("entries") or {}
    updates = []
    t, y = key(h.treatment or ""), key(h.outcome)
    for entry in entries.values():
        if entry.get("status") not in {"proposed", "supported"}:
            continue
        evidence = entry.get("evidence") or []
        source = [
            e
            for e in evidence
            if e.get("category") in {"study_account", "measurement_timing"}
            and e.get("source") in {"pack", "person", "data"}
            and h.resolve(e.get("address") or "")
        ]
        if not source:
            continue
        src, dst, kind = entry.get("src"), entry.get("dst"), entry.get("kind")
        if not src or h.column(src) is None or src in {t, y}:
            continue
        field = "feeds_treatment" if dst == t else "moves_outcome" if dst == y else None
        if kind not in {"edge", "no_edge"} or field is None:
            continue
        updates.append(
            ops.Update(
                address=f"col:{src}.{field}",
                value=kind == "edge",
                status="drafted",
                source="model:ledger",
                said="; ".join(e.get("note") or e["address"] for e in source),
                reason=f"{entry.get('reason') or 'explicit lane proposal'}; evidence {', '.join(e['address'] for e in source)}",
            )
        )
    rejected = ops.apply(memory, updates, cat) if updates else []
    refused = {r.split(": ", 1)[0] for r in rejected}
    return [u.address for u in updates if u.address not in refused]
