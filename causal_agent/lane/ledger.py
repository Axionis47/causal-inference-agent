"""A lane's evidence ledger. Pure writes preserve every accepted change."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    source: Literal["pack", "data", "judgement", "person", "test"]
    address: str
    category: Literal["study_account", "data_fact", "measurement_timing", "association", "assumption", "diagnostic"] = "assumption"
    note: str = ""


class Entry(BaseModel):
    id: str
    kind: str
    src: str | None = None
    dst: str | None = None
    value: Any = None
    status: Literal["proposed", "supported", "confirmed", "contested", "rejected"] = "proposed"
    evidence: list[Evidence] = Field(default_factory=list)
    by: str = ""
    reason: str = ""


class Change(BaseModel):
    revision: int
    entry_id: str
    before: Entry | None
    after: Entry
    by: str
    reason: str


class Ledger(BaseModel):
    revision: int = 0
    entries: dict[str, Entry] = Field(default_factory=dict)
    history: list[Change] = Field(default_factory=list)

    def lines(self) -> list[tuple[str, str]]:
        return [(e.id, f"{e.kind}: {e.src or ''} {e.dst or ''} {e.value!r}; {e.status}; {e.reason}") for e in self.entries.values()]

    def resolve(self, address: str) -> bool:
        return address in self.entries

    def render(self) -> str:
        return "\n".join(f"[{a}] {text}" for a, text in self.lines())


def apply(ledger: Ledger, entries: list[Entry], *, by: str, resolves=None) -> tuple[Ledger, list[str]]:
    """Apply an atomic episode update. A diagnostic cannot certify a causal relationship."""
    out = ledger.model_copy(deep=True)
    errors = []
    for entry in entries:
        old = out.entries.get(entry.id)
        if not entry.reason.strip():
            errors.append(f"{entry.id}: a change needs a reason")
        if resolves is not None:
            errors.extend(f"{entry.id}: unresolved evidence {e.address}" for e in entry.evidence if not resolves(e.address))
        causal = entry.kind in {
            "edge",
            "no_edge",
            "hidden",
            "exclusion",
            "monotonicity",
            "continuity",
            "parallel_trends",
            "selection",
            "anticipation",
            "spillover",
            "other_shock",
            "exchangeability",
            "modifies",
        }
        if entry.status == "confirmed" and not any(e.source in {"person", "pack"} and e.category == "study_account" for e in entry.evidence):
            if causal or not any(e.source == "data" and e.category in {"data_fact", "measurement_timing"} for e in entry.evidence):
                errors.append(f"{entry.id}: confirmation needs a study account or, for a noncausal fact, a data check")
        if entry.status == "supported" and not any(e.source in {"pack", "data", "person"} for e in entry.evidence):
            errors.append(f"{entry.id}: supported needs source evidence")
        source_account = any(e.source in {"pack", "person"} and e.category == "study_account" for e in entry.evidence)
        timing_absence = entry.kind == "no_edge" and any(e.source == "data" and e.category == "measurement_timing" for e in entry.evidence)
        if causal and entry.status == "supported" and not (source_account or timing_absence):
            errors.append(
                f"{entry.id}: causal support needs a study account; profiles, associations, timing or judgement assumptions cannot promote a causal claim"
            )
        if old is not None and old.status == "confirmed" and not any(e.source == "person" for e in entry.evidence):
            changed_value = (old.kind, old.src, old.dst, old.value) != (entry.kind, entry.src, entry.dst, entry.value)
            if changed_value or entry.status in {"rejected", "proposed"}:
                errors.append(f"{entry.id}: a model or statistical test cannot erase a confirmed account")
        changed = entry.model_copy(update={"by": by})
        if old != changed:
            out.revision += 1
            out.entries[entry.id] = changed
            out.history.append(Change(revision=out.revision, entry_id=entry.id, before=old, after=changed, by=by, reason=entry.reason))
    return (ledger, errors) if errors else (out, [])


def seed(handoff) -> Ledger:
    entries = []
    for address, fact in handoff.facts.items():
        if fact.value is None and fact.status == "empty":
            continue
        provenance = fact.source or ""
        if provenance.startswith(("user:", "doc:", "fixture:", "eval:")):
            evidence = Evidence(source="pack", address=address, category="study_account", note=fact.said or fact.reason or "")
        elif provenance.startswith(("data", "code:")):
            evidence = Evidence(source="data", address=address, category="data_fact", note=fact.said or fact.reason or "")
        else:
            evidence = Evidence(source="judgement", address=address, category="assumption", note=fact.said or fact.reason or "")
        status: Literal["proposed", "supported", "confirmed", "contested", "rejected"] = (
            "confirmed"
            if fact.status == "confirmed" and evidence.source in {"pack", "data"}
            else "contested"
            if fact.status in {"refuted", "contradiction"}
            else "proposed"
        )
        entries.append(
            Entry(
                id=f"rel:context.{address}",
                kind="reported_field",
                value=fact.value,
                status=status,
                evidence=[evidence],
                reason=f"context says {fact.status}; source {fact.source or 'unspecified'}",
            )
        )
    ledger, errors = apply(Ledger(), entries, by="seed", resolves=handoff.resolve)
    if errors:
        raise ValueError("; ".join(errors))
    return ledger
