"""A model's answer about a column against what the pack settled. A claim that contradicts a fact is rejected unless the
answer cites that address and the address is one the pack itself marks contested; a claim that departs from the last reading
needs a `Departure` naming it with the cite that changed it; the citations must resolve either way."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from causal_agent.common.addresses import norm_address
from causal_agent.common.contracts import Departure, Handoff
from causal_agent.lane.case import Case

# (claim, claim value, field, the fact values the claim contradicts)
Rule = tuple[str, bool, str, tuple]


def cites_resolve(cites: list[str], h: Handoff, also: Callable[[str], bool] | None = None) -> list[str]:
    """Errors for every cite the pack does not resolve; `also` resolves what this run itself made (an episode's facts)."""
    return [f"citation {c!r} does not resolve in the pack" for c in cites if not (h.resolve(c) or (also is not None and also(c)))]


def is_address(cite: str) -> bool:
    """Whether a cite is shaped like an address at all (a tag with a colon, or a dotted design line), not a heading copied from the material."""
    a = norm_address(cite)
    return ":" in a or a.startswith(("design.", "pair.", "time.", "dataset", "change:"))


def resolves(cite: str, h: Handoff, *, checks: Iterable[str] = (), also: Callable[[str], bool] | None = None) -> bool:
    """Whether one cite names a check of this run, a pack address, or what `also` resolves (the ladder, an episode's facts)."""
    a = norm_address(cite)
    return a in {norm_address(c) for c in checks} or h.resolve(a) or (also is not None and also(a))


def departures(
    claims: dict[str, Any], drafted: dict[str, Any], departures: list[Departure], h: Handoff, also: Callable[[str], bool] | None = None
) -> list[str]:
    """Errors for every claim that departs from the last reading (a drafted field) without a `Departure` naming it, or with one
    that cites nothing that resolves."""
    errors: list[str] = []
    for claim, value in drafted.items():
        if claim not in claims or claims[claim] == value:
            continue
        named = [d for d in departures if d.claim == claim]
        if not named:
            errors.append(
                f"{claim} = {claims[claim]!r} departs from the last reading {value!r} with no departure named; keep the reading or list "
                f"{claim} under departures with the reason and the cite that changed it"
            )
            continue
        for d in named:
            if not d.cites:
                errors.append(f"the departure for {claim} cites nothing; cite what changed the reading")
            errors += cites_resolve(d.cites, h, also)
    return errors


def contradictions(claims: dict[str, Any], column_key: str, case: Case, rules: list[Rule], cites: list[str], h: Handoff) -> list[str]:
    """Errors for every claim that contradicts a settled fact on this column without citing the contested address."""
    errors = []
    cited = {norm_address(c) for c in cites}
    for claim, claim_value, field, fact_values in rules:
        if claims.get(claim) is not claim_value:
            continue
        addr = f"col:{column_key}.{field}"
        if not case.is_fact(addr) or case.fact(addr) not in fact_values:
            continue
        if norm_address(addr) in cited and any(norm_address(a) == norm_address(addr) for a in h.contradictions):
            continue
        errors.append(
            f"{claim} = {claim_value} contradicts [{addr}] = {case.fact(addr)!r}, which the pack settled; leave the claim {not claim_value} or cite that address if the pack marks it contested"
        )
    return errors
