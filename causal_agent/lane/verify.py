"""A model's answer about a column against what the pack settled. A claim that contradicts a fact is rejected unless the
answer cites that address and the address is one the pack itself marks contested; the citations must resolve either way."""

from __future__ import annotations

from typing import Any

from causal_agent.common.addresses import norm_address
from causal_agent.common.contracts import Handoff
from causal_agent.lane.case import Case

# (claim, claim value, field, the fact values the claim contradicts)
Rule = tuple[str, bool, str, tuple]


def cites_resolve(cites: list[str], h: Handoff) -> list[str]:
    return [f"citation {c!r} does not resolve in the pack" for c in cites if not h.resolve(c)]


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
