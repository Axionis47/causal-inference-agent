"""A judgement may cite exactly the addressed evidence its view displayed."""

from __future__ import annotations

import re
from dataclasses import dataclass

from causal_agent.common.contracts import Handoff
from causal_agent.lane.ledger import Ledger


@dataclass(frozen=True)
class View:
    text: str
    addresses: frozenset[str]

    def resolve(self, address: str) -> bool:
        return address in self.addresses


def build(node: str, h: Handoff, ledger: Ledger, *, material: str = "", columns: list[str] | None = None) -> View:
    parts = [f"QUESTION: {h.question}\nTARGET: {h.scope.target}", h.render_pair()]
    prefixes = {
        "mechanism": ("claim:assignment.", "claim:change.", "claim:grain.", "claim:sampling."),
        "pair": ("claim:assignment.",),
        "relate": ("claim:unobserved.", "claim:exclusion.", "claim:mediator.", "claim:monotonicity."),
        "probe": ("claim:unobserved.", "claim:exclusion.", "claim:mediator.", "claim:monotonicity."),
    }.get(node, ())
    if node in ("mechanism", "relate", "probe"):
        parts.append(h.render_story())
        if node in {"relate", "probe"}:
            parts.append(
                "SOURCE POLICY: The question and pair.target/pair.treatment/pair.outcome define the requested candidate, not a causal fact. "
                "Without a supplied supporting study account, keep T->Y proposed and conditional; cite those displayed pair addresses. "
                "Profiles/timing are diagnostics. QUESTION and TARGET are unaddressed headings and cannot be cited."
            )
        parts.extend(
            f"[{a}] value={f.value!r}; status={f.status}; source={f.source}; original={f.said!r}; reason={f.reason}"
            for a, f in h.facts.items()
            if a.startswith(prefixes)
        )
        if node == "mechanism":
            parts.append(h.render_change())
    if node in ("pair", "mechanism", "relate", "probe"):
        parts.extend(b.render() for b in h.columns if columns is None or b.key in columns)
    if node in ("relate", "probe", "pick_estimator"):
        parts.append(ledger.render())
        if node in {"relate", "probe"}:
            confirmed = [e.model_dump() for e in ledger.entries.values() if e.status == "confirmed"]
            parts.append(
                "READ-ONLY CONFIRMED LEDGER: These entries are already retained, so do not put them in entries again, even with supported status. "
                "Cite them and include their columns in considered. A confirmed modifies attribute already settles whether that baseline variable modifies the effect. "
                "It is not a mediation edge. Only unresolved or new relationships belong in entries.\n" + "\n".join(str(e) for e in confirmed)
            )
    parts.append(material)
    text = "\n\n".join(p for p in parts if p)
    addresses = frozenset(
        a
        for a in re.findall(r"\[([^\]\n]+)\]", text)
        if a.startswith(("claim:", "col:", "probe:", "check:", "design:", "estimator:", "rel:", "pair.", "scope.", "change:"))
    )
    return View(
        text
        + "\n\nCITATION FORMAT: cite the raw address only, never the sentence or brackets. Examples: claim:story.text, col:c.when.\nADDRESSES YOU MAY CITE\n"
        + "\n".join(sorted(addresses)),
        addresses,
    )
