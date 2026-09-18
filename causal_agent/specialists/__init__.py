"""Specialist subgraphs, one per family, keyed by family name.

The router reaches a specialist by looking the chosen family up here. A family whose registry entry
names a built specialist gets that specialist's compiled subgraph. Every other family gets a stub that
reports "not supported yet" with the hand-off preserved. Adding a specialist: build its subgraph in its
own folder, register it in BUILT below, flip the family's status in knowledge/families.yaml.
"""

from __future__ import annotations

from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

from causal_agent.common.contracts import Handoff
from causal_agent.knowledge import load_registry


class SpecialistState(TypedDict, total=False):
    handoff: Handoff | None
    dataset: str
    specialist_result: dict | None


def _stub(family: str, specialist: str, built: bool):
    def run(state: SpecialistState) -> dict:
        h = state.get("handoff")
        if h is None:
            return {"specialist_result": {"status": "no_handoff", "family": family}}
        summary = {
            "family": h.family,
            "specialist": h.specialist,
            "outcome": h.outcome,
            "treatment": h.treatment,
            "relevant_columns": [c.column for c in h.relevant_columns],
            "chosen_assumption": h.chosen_assumption,
        }
        if built:
            return {"specialist_result": {"status": "stub", "message": f"{specialist} specialist not implemented yet; would run on this hand-off", **summary}}
        return {"specialist_result": {"status": "not_supported", "message": f"family {family} has no specialist yet ({specialist})", **summary}}

    g = StateGraph(SpecialistState)
    g.add_node("run", run)
    g.add_edge(START, "run")
    g.add_edge("run", END)
    return g.compile(checkpointer=False)


def _dowhy():
    from causal_agent.specialists.dowhy.graph import compile_subgraph

    return compile_subgraph()


def _pyfixest():
    from causal_agent.specialists.did.graph import compile_subgraph

    return compile_subgraph()


def _rdrobust():
    from causal_agent.specialists.rd.graph import compile_subgraph

    return compile_subgraph()


# specialist name (as written in families.yaml) -> factory for its compiled subgraph
BUILT = {"dowhy": _dowhy, "pyfixest": _pyfixest, "rdrobust": _rdrobust}


def build_specialists() -> dict[str, object]:
    out: dict[str, object] = {}
    for fam in load_registry():
        if fam.status == "built" and fam.specialist in BUILT:
            out[fam.name] = BUILT[fam.specialist]()
        else:
            out[fam.name] = _stub(fam.name, fam.specialist, fam.status == "built")
    return out


SPECIALISTS = build_specialists()
