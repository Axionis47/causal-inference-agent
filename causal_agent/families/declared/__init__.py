"""The families the routing knows and can rule in or out, with no lane built yet. Each has its knowledge and needs in a
yaml file here, a probe where one is computable, and the stub lane."""

from __future__ import annotations

from functools import partial
from pathlib import Path

from causal_agent.families.base import FamilyDef, load_family_yaml, stub_lane
from causal_agent.families.declared import probes

_HERE = Path(__file__).parent
ORDER = ["interrupted_series", "synthetic_control", "instrument", "root_cause"]
PROBES = {"interrupted_series": probes.interrupted_series, "synthetic_control": probes.synthetic_control, "instrument": probes.instrument}


def _one(name: str) -> FamilyDef:
    knowledge, needs = load_family_yaml(_HERE / f"{name}.yaml", name=name)
    return FamilyDef(
        name=name,
        knowledge=knowledge,
        needs=needs,
        probes=PROBES.get(name),
        lane=partial(stub_lane, name, knowledge.specialist, knowledge.status == "built"),
    )


FAMILIES: list[FamilyDef] = [_one(n) for n in ORDER]
