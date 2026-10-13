"""The adjustment lane's design space: the catalogue keys on its dimensions and serves every cell, the lane's own
graph decides each dimension, and a file that reaches no cell stops at load with the declared reason and no model call."""

from __future__ import annotations

import numpy as np
import pandas as pd

from causal_agent.families.adjustment.lane.graph import build
from causal_agent.families.adjustment.lane.knowledge import SPACE, load_estimators, load_refuters
from causal_agent.families.adjustment.tests.test_lane import FakeLLM, _run, _synthetic_handoff, _synthetic_memory
from causal_agent.lane.space import Covered, coverage


def test_every_cell_of_the_space_is_served_by_an_estimator_and_a_refuter():
    got = coverage(SPACE, load_estimators(), load_refuters())
    assert isinstance(got, Covered), got
    # 3 roads × 2 outcome kinds × 2 adjustment sets × 2 sensitivity flags; frontdoor on a binary outcome is declared out in its 4 cells
    assert got.cells == 24 and got.unsupported == 4


def test_the_catalogue_keys_on_the_space_and_the_graph_decides_every_dimension():
    keys = {k for e in [*load_estimators(), *load_refuters()] for k in e.applies_when}
    assert keys <= set(SPACE.dimensions), keys - set(SPACE.dimensions)
    nodes = set(build().compile().get_graph().nodes)
    assert {d.decided_by for d in SPACE.dimensions.values()} <= nodes


def test_a_dose_treatment_stops_at_load_with_the_declared_reason_and_no_model_call(tmp_path):
    rng = np.random.default_rng(0)
    n = 600
    dose = rng.integers(0, 30, n)
    frame = pd.DataFrame({"z": rng.normal(size=n), "treated": dose, "m": rng.normal(size=n), "y": dose + rng.normal(size=n)})
    csv = tmp_path / "synthetic.csv"
    frame.to_csv(csv, index=False)
    out = _run(FakeLLM(), _synthetic_handoff(_synthetic_memory(csv)), question="Did the programme raise y?")
    r = out["specialist_result"]
    assert r["status"] == "infeasible" and r["feasibility"]["stage"] == "load"
    assert r["feasibility"]["reason"] == SPACE.dimensions["contrasts"].unsupported["above"]
    assert "contrasts = 29" in " ".join(r["feasibility"]["facts"])
    assert not out.get("episodes"), "a stop at load spends no judgement"
