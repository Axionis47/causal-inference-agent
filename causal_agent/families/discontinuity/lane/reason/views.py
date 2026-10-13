"""Decision-specific context. Resolvers admit only addresses actually rendered."""

from __future__ import annotations

import re
from dataclasses import dataclass

from causal_agent.common.addresses import norm_address
from causal_agent.common.contracts import Handoff
from causal_agent.lane.ledger import Ledger


@dataclass(frozen=True)
class View:
    text: str
    addresses: frozenset[str]

    def resolve(self, address: str) -> bool:
        return norm_address(address) in self.addresses


def build(
    h: Handoff, *, claims: tuple[str, ...] = (), columns: list[str] | None = None, lines: list[tuple[str, str]] | None = None, ledger: Ledger | None = None
) -> View:
    text = [f"QUESTION: {h.question}"]
    source_lines = {}
    for a, f in h.facts.items():
        if any(a.startswith(f"claim:{kind}.") for kind in claims):
            text.append(f"[{a}] {f.value!r}; {f.status}; source {f.source}; words {f.said or ''}; reason {f.reason or ''}")
            if f.source:
                source_lines.setdefault(f.source, []).append(f"{a}: {f.said or f.value!r}")
    for k in columns or []:
        card = h.column(k)
        if card:
            updates = {}
            provenance = dict(card.provenance)
            for a, f in h.facts.items():
                if not a.startswith(f"col:{k}."):
                    continue
                if f.source:
                    source_lines.setdefault(f.source, []).append(f"{a}: {f.said or f.value!r}")
                field = a.removeprefix(f"col:{k}.")
                if field in type(card).model_fields and field not in {"facts", "provenance"} and f.value is not None:
                    updates[field] = f.value
                    provenance[field] = f
            updates["provenance"] = provenance
            text.append(card.model_copy(update=updates).render())
    text.extend(f"[{a}] admitted study source for this view: {'; '.join(words)}" for a, words in source_lines.items())
    text.extend(f"[{a}] {v}" for a, v in lines or [])
    if ledger:
        text.extend(f"[{a}] {v}" for a, v in ledger.lines())
    rendered = "\n".join(text)
    addresses = frozenset(norm_address(a) for a in re.findall(r"\[([^\]\n]+)\]", rendered))
    return View(rendered, addresses)
