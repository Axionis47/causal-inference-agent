"""Every family with evals has a whole spec: cases that load, stored hand-offs that parse, a lane, a summariser, evaluators."""

from __future__ import annotations

from causal_agent.common.contracts import Handoff
from causal_agent.evals import dataset
from causal_agent.evals import evaluators as EV
from causal_agent.evals.families import SPECS, spec


def test_every_spec_is_whole_and_its_cases_load():
    assert set(SPECS) == {"adjustment", "diff_in_diff", "discontinuity"}
    for s in SPECS.values():
        cases = dataset.load_cases(s)
        assert cases and all({"id", "dataset", "question", "expected"} <= set(c) for c in cases)
        for c in cases:
            if c.get("handoff"):
                h = Handoff.model_validate({k: v for k, v in c["handoff_json"].items() if k != "question"})
                assert h.family == s.family
        assert callable(s.summarise) and callable(s.lane_graph) and all(callable(e) for e in s.evaluators)
        assert set(EV.SHARED) <= set(s.evaluators)
        assert s.summarise({}).get("status") is None


def test_the_shared_evaluators_read_both_shapes_of_expectation():
    run = {"outputs": {"status": "done", "effect": 2.0, "effects": {"a_vs_b": -1.0}, "stage": None, "hard_flags": ["x"], "placebos": ["p"]}}
    assert EV.status_match(run, {"outputs": {"status_in": ["done", "stopped"]}})["score"] == 1
    assert EV.sign_match(run, {"outputs": {"effect_sign": "positive"}})["score"] == 1
    assert EV.sign_match(run, {"outputs": {"effect_sign": {"a_vs_b": "negative"}}})["score"] == 1
    assert EV.sign_match(run, {"outputs": {"effect_sign": {"a_vs_b": "positive"}}})["score"] == 0
    assert (
        EV.stopped_at(
            {"outputs": {"status": "stopped", "stage": "load", "hard_flags": ["x"]}}, {"outputs": {"stopped_at": "load", "hard_flags_contain": ["x"]}}
        )["score"]
        == 1
    )
    assert EV.stopped_at(run, {"outputs": {"stopped_at": "load", "status_in": ["done"]}})["score"] == 1
    assert EV.placebo_ran(run, {"outputs": {"placebo_ran_if_done": ["p"]}})["score"] == 1


def test_an_unknown_family_says_which_have_evals():
    import pytest

    with pytest.raises(SystemExit, match="families with evals"):
        spec("root_cause")
