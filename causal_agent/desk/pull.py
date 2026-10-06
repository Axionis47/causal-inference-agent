"""One pull over the shelves. Everything the chat may cite at a point in the conversation is one material: a line per item,
each with an address. This module says what shelf an address is on and gives the material its two reads: `pull` one item
by address, `shelf` everything on one shelf. The material is built by `explainer.before_material` before a run and by
`material.render` after one; nothing else assembles what is citable."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from causal_agent.common.addresses import norm_address

Shelf = Literal["understanding", "artifact", "experiment", "journal", "words", "fit", "family", "desk"]

# the address grammar, by what made the thing
_PREFIXES: list[tuple[tuple[str, ...], Shelf]] = [
    (("col:", "claim:"), "understanding"),
    (("artifact:",), "artifact"),
    (("step:",), "journal"),
    (("user:turn:", "said:"), "words"),
    (("matrix", "probe:"), "fit"),
    (("family:",), "family"),
    (("desk.",), "desk"),
]


def shelf_of(address: str) -> Shelf:
    """Which shelf an address is on. What a run left (estimate:, check:, refute:, placebo:, figure:, design., decision.,
    interpretation:, decline:, run., ladder:, primary.) is the experiment."""
    a = norm_address(address)
    for prefixes, shelf in _PREFIXES:
        if a.startswith(prefixes):
            return shelf
    return "experiment"


@dataclass(frozen=True)
class Item:
    """One pullable thing: its address, the shelf it is on, its line, and its number when it has one."""

    address: str
    shelf: Shelf
    text: str
    value: float | None = None

    @property
    def parent(self) -> str | None:
        head, dot, _ = self.address.rpartition(".")
        return head if dot else None
