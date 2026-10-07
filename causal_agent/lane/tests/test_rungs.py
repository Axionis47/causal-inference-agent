"""A lane's ladder, declared once: the declaration refuses what the graph could not run, the edges it describes are the graph's,
and the end states of a lane are derived from it with the space, the checks and the beliefs, never listed."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from causal_agent.lane import rungs as RG
from causal_agent.lane.ladder import HARD_THREATS
from causal_agent.lane.space import Space

RUNGS = [
    {"name": "load", "decider": {"code": "always"}, "stops_on": ["intake"]},
    {"name": "pair", "decider": {"judgement": "always"}, "budget": 2, "tools": ["describe"]},
    {"name": "road", "decider": {"code": "one road", "ask": "none open", "judgement": "several"}, "budget": 0, "asks": ["claim:mediator.exists"]},
    {"name": "check_design", "decider": {"code": "always"}, "stop": False, "exits": ["assess", "pick"], "route": "after_checks"},
    {"name": "assess", "decider": {"judgement": "a flag stands"}, "budget": 0, "exits": ["pick", "pair"], "stops_on": ["checks", "beliefs", "threats"]},
    {"name": "pick", "decider": {"judgement": "always"}, "budget": 0, "exits": ["freeze"]},
    {"name": "freeze", "decider": {"code": "always"}, "stop": False, "exits": ["estimate"]},
    {"name": "estimate", "decider": {"code": "always"}, "stops_on": ["engine"], "exits": ["interpret"]},
    {"name": "interpret", "decider": {"judgement": "always"}, "budget": 0, "stop": False},
]
LOOPS = [{"from": "assess", "to": "pair", "max": 3, "why": "a revision"}]


def _spec(rungs=RUNGS, loops=LOOPS, goal="freeze") -> RG.LadderSpec:
    return RG.LadderSpec.model_validate({"goal": goal, "loops": loops, "rungs": rungs})


def test_the_declaration_refuses_what_the_graph_could_not_run():
    with pytest.raises(ValidationError, match="colour"):
        _spec([{"name": "a", "decider": {"code": "x"}, "colour": "red"}], [], "a")
    with pytest.raises(ValidationError, match="budget"):
        _spec([{"name": "a", "decider": {"judgement": "x"}}], [], "a")
    with pytest.raises(ValidationError, match="only a judgement"):
        _spec([{"name": "a", "decider": {"code": "x"}, "budget": 2}], [], "a")
    with pytest.raises(ValidationError, match="not declared causes"):
        _spec([{"name": "a", "decider": {"code": "x"}, "stops_on": ["weather"]}], [], "a")
    with pytest.raises(ValidationError, match="not a rung"):
        _spec([{"name": "a", "decider": {"code": "x"}, "exits": ["b"]}], [], "a")
    with pytest.raises(ValidationError, match="loops declared"):  # an edge back down that no loop declares
        _spec([{"name": "a", "decider": {"code": "x"}}, {"name": "b", "decider": {"code": "x"}, "exits": ["a"]}], [], "b")
    with pytest.raises(ValidationError, match="loops declared"):  # a loop declared that is not an edge
        _spec([{"name": "a", "decider": {"code": "x"}}, {"name": "b", "decider": {"code": "x"}}], [{"from": "b", "to": "a", "max": 1, "why": "w"}], "b")
    with pytest.raises(ValidationError, match="goal"):
        _spec([{"name": "a", "decider": {"code": "x"}}], [], "z")
    with pytest.raises(ValidationError, match="builder's"):
        _spec([{"name": "a", "decider": {"code": "x"}, "exits": ["feasibility"]}], [], "a")


def test_the_edges_are_the_exits_the_stops_and_the_builders_tail():
    spec = _spec()
    assert spec.exits_of(spec.rung("load")) == ["pair"], "the next rung when not said"
    assert spec.exits_of(spec.rung("interpret")) == ["figures"], "the tail after the last rung"
    edges = spec.edges()
    assert {("__start__", "load"), ("load", "pair"), ("load", "feasibility"), ("pair", "road"), ("pair", "feasibility")} <= edges
    assert ("check_design", "feasibility") not in edges and ("freeze", "feasibility") not in edges, "a rung that never stops has no stop edge"
    assert {("assess", "pair"), ("assess", "pick"), ("assess", "feasibility")} <= edges
    assert {("feasibility", "figures"), ("figures", "assemble"), ("assemble", "__end__"), ("interpret", "figures")} <= edges
    assert spec.budget("pair") == 2
    with pytest.raises(KeyError):
        spec.budget("load")


SPACE = Space.model_validate(
    {
        "dimensions": {
            "estimand": {"values": ["backdoor", "iv"], "decided_by": "road"},
            "outcome": {
                "values": ["continuous", "binary"],
                "decided_by": "load",
                "read_from": "col:<outcome>.profile.kind",
                "unsupported": {"other": "no estimator fits it"},
            },
            "contrasts": {"range": [1, 11], "decided_by": "pair", "unsupported": {"above": "a dose needs another lane"}},
        },
        "unsupported": [{"when": {"estimand": "iv", "outcome": "binary"}, "reason": "the instrument fits a number only"}],
    }
)
ESTIMATORS = [
    SimpleNamespace(name="ols", applies_when={"estimand": "backdoor", "outcome": ["continuous", "binary"]}),
    SimpleNamespace(name="iv", applies_when={"estimand": "iv", "outcome": ["continuous"]}),
]
REFUTERS = [SimpleNamespace(name="placebo", applies_when={"estimand": ["backdoor", "iv"]})]
CHECKS = {
    "overlap": {"share": {"soft": 0.95, "hard": 0.8}, "note": "words"},
    "balance": {"smd": {"soft": 0.1}},
    "in_words": {"overlap": "whether both arms are found"},
}
BELIEFS = {
    "beliefs": {
        "trend": {
            "value_field": "believed",
            "by_status": {
                "confirmed_false": {"level": "stop", "caveat": "the person says the groups moved apart"},
                "empty": {"level": "soft", "ask": {"question": "Would they have kept moving together?"}},
            },
            "with_check": {
                "pre_trends": [
                    {"belief": "confirmed_true", "check": "hard", "then": "ask", "question": "Is there a reason?"},
                    {"belief": "confirmed_false", "check": "hard", "then": "stop", "reason": "the test and the person agree"},
                ]
            },
        }
    },
    "unknowns": {"claim:change.period_value": {"level": "hard", "ask": {"question": "Which period?"}}},
}


def test_the_end_states_are_derived_from_the_declarations():
    e = RG.ends(_spec(), space=SPACE, estimators=ESTIMATORS, refuters=REFUTERS, checks=CHECKS, beliefs=BELIEFS)
    # answers: the served cells, with what runs them; the declared-out cell is not an answer
    assert [(a.point, a.estimators, a.refuters) for a in e.answers] == [
        ({"estimand": "backdoor", "outcome": "continuous"}, ["ols"], ["placebo"]),
        ({"estimand": "backdoor", "outcome": "binary"}, ["ols"], ["placebo"]),
        ({"estimand": "iv", "outcome": "continuous"}, ["iv"], ["placebo"]),
    ]
    # questions: the rung's own, then what the belief rules ask at the rung that reads them
    assert [(a.rung, a.address) for a in e.asks] == [
        ("road", "claim:mediator.exists"),
        ("assess", "claim:trend.believed"),
        ("assess", "claim:trend.believed"),
        ("assess", "claim:change.period_value"),
    ]
    assert e.asks[1].question == "Would they have kept moving together?" and e.asks[2].question == "Is there a reason?"
    stops = {(s.rung, s.cause): s for s in e.stops}
    assert stops[("load", "intake")].fix == "a hand-off whose file and columns exist"
    assert stops[("load", "space:outcome")].fix == "a file inside the space", "a dimension read from the file stops at load"
    assert stops[("load", "space:outcome=other")].reason == "no estimator fits it"
    assert stops[("pair", "space:contrasts=above")].reason == "a dose needs another lane" and stops[("pair", "gate")].reason == RG.GATE[0]
    assert stops[("road", "space:estimand=iv+outcome=binary")].reason == "the instrument fits a number only", (
        "a combination stops at the rung deciding its last dimension"
    )
    assert ("road", "space:estimand") not in stops, "a dimension offered only runnable values is never outside"
    assert stops[("assess", "check:overlap.share")].reason == "whether both arms are found; share past 0.8" and ("assess", "check:balance.smd") not in stops
    assert stops[("assess", "belief:trend.confirmed_false")].fix == "a different answer at [claim:trend.believed]"
    assert stops[("assess", "belief:trend+check:pre_trends")].reason == "the test and the person agree"
    assert {c for r, c in stops if r == "assess" and c.startswith("threat:")} == {f"threat:{n}" for n in HARD_THREATS}
    assert stops[("estimate", "engine")].reason == "the estimator failed to fit"
    assert not any(r == "interpret" for r, _ in stops), "a judgement that never stops raises no gate stop"
    order = _spec().names
    assert [s.rung for s in e.stops] == [s.rung for s in sorted(e.stops, key=lambda s: order.index(s.rung))], "in ladder order"


def test_a_lane_without_a_space_has_no_answers_yet():
    e = RG.ends(_spec(), checks=CHECKS, beliefs={})
    assert e.answers == [] and not any(s.cause.startswith("space:") for s in e.stops)
    assert {s.cause for s in e.stops if s.rung == "assess"} == {"check:overlap.share", *(f"threat:{n}" for n in HARD_THREATS), "gate"}
