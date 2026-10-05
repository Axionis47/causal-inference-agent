"""A lane's design space, declared once: every cell is served or declared unsupported; a file outside it stops with the declared
reason; a deciding rung is offered only runnable values; a reached shape is a cell or a named way out."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from causal_agent.lane.space import Covered, Gap, Outside, Space, bounds, cell, coverage, limit, offer

SPACE = Space.model_validate(
    {
        "dimensions": {
            "estimand": {"values": ["backdoor", "iv"], "decided_by": "road"},
            "outcome": {"values": ["continuous", "binary"], "decided_by": "load", "unsupported": {"other": "no estimator fits it"}},
            "target": {"values": ["ate", "att"], "decided_by": "pair", "soft": True},
            "confounders": {"range": [0, "smallest_arm/10"], "decided_by": "road", "unsupported": {"above": "too many for the arm"}},
            "modifiers": {"range": [0, 2], "decided_by": "heterogeneity"},
        },
        "unsupported": [{"when": {"estimand": "iv", "outcome": "binary"}, "reason": "the instrument fits a number only"}],
    }
)


def _entry(**applies_when):
    return SimpleNamespace(applies_when=applies_when)


ESTIMATORS = [
    _entry(estimand="backdoor", outcome=["continuous", "binary"], target=["ate", "att"]),
    _entry(estimand="iv", outcome=["continuous"], target=["ate"]),
]
REFUTERS = [_entry(estimand=["backdoor", "iv"])]


def test_an_unknown_key_refuses_to_load():
    with pytest.raises(ValidationError):
        Space.model_validate({"dimensions": {"x": {"values": ["a"], "decided_by": "pair", "colour": "red"}}})


def test_every_cell_is_served_or_declared():
    assert coverage(SPACE, ESTIMATORS, REFUTERS) == Covered(cells=4, unsupported=1)
    # drop the instrument entry: the iv × continuous cell has nothing and nothing declares it
    assert coverage(SPACE, ESTIMATORS[:1], REFUTERS) == Gap(cell={"estimand": "iv", "outcome": "continuous"}, missing="estimator")
    # drop the refuter: the first cell has an estimator and no refuter
    assert coverage(SPACE, ESTIMATORS, []) == Gap(cell={"estimand": "backdoor", "outcome": "continuous"}, missing="refuter")


def test_a_soft_dimension_never_makes_a_gap():
    # no entry serves att on iv; att is soft, so the cell counts as covered by the ate entry
    assert isinstance(coverage(SPACE, ESTIMATORS, REFUTERS), Covered)


def test_a_file_outside_the_space_stops_with_the_declared_reason():
    assert bounds(SPACE, {"outcome": "continuous", "confounders": 3, "smallest_arm": 40}) is None
    assert bounds(SPACE, {"outcome": "other"}) == Outside(dimension="outcome", value="other", reason="no estimator fits it")
    out = bounds(SPACE, {"confounders": 5, "smallest_arm": 40})
    assert out == Outside(dimension="confounders", value=5, reason="too many for the arm")
    assert bounds(SPACE, {"modifiers": 3}) is not None and bounds(SPACE, {"modifiers": 2}) is None


def test_a_bound_that_reads_a_fact_needs_it():
    with pytest.raises(KeyError):
        bounds(SPACE, {"confounders": 1})


def test_a_deciding_rung_is_offered_only_runnable_values():
    assert offer(SPACE, ESTIMATORS, {})["estimand"] == ["backdoor", "iv"]
    # a binary outcome fixed: the instrument's entry is inconsistent, and the combination is declared out
    got = offer(SPACE, ESTIMATORS, {"outcome": "binary"})
    assert got["estimand"] == ["backdoor"] and got["target"] == ["ate", "att"] and got["modifiers"] == [0.0, 2.0]
    assert "confounders" not in got, "a bound that reads a fact not yet fixed is not offered"
    assert offer(SPACE, ESTIMATORS, {"smallest_arm": 40})["confounders"] == [0.0, 4.0]
    # the road fixed to the instrument: only the target it serves is offered
    assert offer(SPACE, ESTIMATORS, {"estimand": "iv"})["target"] == ["ate"]
    assert limit(SPACE, "modifiers") == 2.0 and limit(SPACE, "confounders", {"smallest_arm": 50}) == 5.0


def test_a_reached_shape_is_a_cell_or_a_named_way_out():
    facts = {"estimand": "backdoor", "outcome": "binary", "target": "att", "confounders": 2, "modifiers": 1, "smallest_arm": 40}
    assert cell(SPACE, facts) == {"estimand": "backdoor", "outcome": "binary", "target": "att", "confounders": 2, "modifiers": 1}
    out = cell(SPACE, {**facts, "estimand": "iv"})
    assert isinstance(out, Outside) and out.reason == "the instrument fits a number only"
    out = cell(SPACE, {**facts, "confounders": 9})
    assert isinstance(out, Outside) and out.dimension == "confounders"
