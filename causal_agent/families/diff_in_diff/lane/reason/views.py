"""Stage material and its exact citation boundary."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from causal_agent.common.contracts import Handoff
from causal_agent.families.diff_in_diff.lane.reason.rivals import eligible_control
from causal_agent.families.diff_in_diff.lane.state import SpecialistState


@dataclass(frozen=True)
class View:
    text: str
    addresses: frozenset[str]

    def resolves(self, address: str) -> bool:
        return address in self.addresses


def build(state: SpecialistState, stage: str) -> View:
    h: Handoff = state["handoff"]
    chunks = [f"QUESTION: {h.question or state.get('question', '')}", f"[pair.target] requested target: {h.scope.target}"]
    prefixes = {
        "mechanism": ("claim:assignment.", "claim:change.", "claim:grain."),
        "comparison": ("claim:trend_continues.", "claim:spillover.", "claim:sampling.", "claim:missing.", "claim:assignment."),
        "controls": ("col:",),
        "probe": ("claim:", "col:"),
        "pick_estimator": (),
    }.get(stage, ())
    for address, fact in h.facts.items():
        if address.startswith(prefixes) and (stage != "controls" or any(address.startswith(f"col:{c.key}.") for c in h.columns if c.role == "candidate")):
            chunks.append(f"[{address}] {fact.value!r}; status={fact.status}; source={fact.source}; words={fact.said!r}; reason={fact.reason!r}")
    if stage in {"mechanism", "comparison", "probe"} and h.story:
        chunks.append(f"[dataset.note] {h.story}")
    if stage == "controls":
        chunks += [c.render() for c in h.columns if c.role == "candidate"]
        panel = pd.read_csv(state["panel_path"])
        chunks += [
            f"[probe:controls.{c.key}.eligible] {eligible_control(h, panel, c.key)}: confirmed safe timing and variation beyond unit/time fixed effects are required"
            for c in h.columns
            if c.role == "candidate"
        ]
    for address, text in state["ledger"].lines():
        chunks.append(f"[{address}] {text}")
    if stage in {"pick_estimator", "interpret"}:
        for c in state.get("candidates", []):
            chunks.append(f"[candidate:{c.id}] {c.model_dump()}")
    text = "\n".join(chunks)
    return View(text=text, addresses=frozenset(re.findall(r"^\[([^\[\]\n]+)\]", text, re.MULTILINE)))
