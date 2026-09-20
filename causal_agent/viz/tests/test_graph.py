"""The viz subgraph with a fake model. The memory is held in this process."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from causal_agent.common.llm import set_llm
from causal_agent.memory import store
from causal_agent.memory.records import Memory
from causal_agent.viz import graph as G
from causal_agent.viz.spec import Point


class FakeLLM:
    def __init__(self, function="discontinuity.density", cites=("claim:assignment.cutoff",)):
        self.function, self.cites, self.calls = function, list(cites), []

    def with_structured_output(self, schema, include_raw=False):
        fake = self

        class R:
            def invoke(self_, messages):
                fake.calls.append(schema.__name__)
                raw = AIMessage(
                    content=[{"type": "thinking", "thinking": "t"}, "{}"], usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
                )
                return {"raw": raw, "parsed": G.Choice(function=fake.function, why="the point is about bunching", cites=fake.cites), "parsing_error": None}

        return R()


@pytest.fixture(autouse=True)
def _held(monkeypatch):
    held: dict[str, Memory] = {}

    def memory_for(name, root=None):
        if name not in held:
            held[name] = store.migrate(name, write=False)
        return held[name]

    monkeypatch.setattr(store, "memory_for", memory_for)
    monkeypatch.setattr(store, "save", lambda m, root=None: None)
    yield
    set_llm(None)


def test_one_candidate_is_chosen_by_code_and_drawn():
    fake = FakeLLM()
    set_llm(fake)
    f = G.make(
        Point(family="adjustment", claim="both arms exist at every lunch level", about=["claim:assignment.depends_on"]), "students3", outcome="math score"
    )
    assert f.made and f.function == "adjustment.overlap" and fake.calls == []
    assert f.probe.address == "probe:adjustment.overlap" and f.spec.series[0].x[0].startswith("lunch = ")
    assert "figure:overlap_lunch_parental_level_of_education" in f.spec.addresses()


def test_two_candidates_go_to_the_model():
    fake = FakeLLM()
    set_llm(fake)
    f = G.make(Point(family="discontinuity", claim="rows do not bunch just above the line"), "senate3", outcome="vote")
    assert fake.calls == ["Choice"] and f.made and f.function == "discontinuity.density" and f.spec.marks[0].label == "cutoff"
    fake = FakeLLM(function="none")
    set_llm(fake)
    f = G.make(Point(family="discontinuity", claim="something no figure shows"), "senate3", outcome="vote")
    assert not f.made and "bunching" in f.why


def test_no_candidate_is_a_refusal_with_why():
    f = G.make(Point(family="diff_in_diff", claim="the groups moved together before"), "students3", outcome="math score")
    assert not f.made and "diff_in_diff" in f.why and "settled" in f.why


def test_a_bad_pick_is_retried_then_refused():
    fake = FakeLLM(function="discontinuity.density", cites=["col:nope.note"])
    set_llm(fake)
    f = G.make(Point(family="discontinuity", claim="rows do not bunch"), "senate3", outcome="vote")
    assert fake.calls.count("Choice") == G.PICK_ATTEMPTS and not f.made and "did not resolve" in f.why


def test_check_refuses_a_figure_that_draws_on_nothing(monkeypatch):
    def bogus(memory, df, state, th):
        fig = G._render_overlap(memory, df, state, th)
        fig.spec.draws_on = ["claim:nope.field"]
        return fig

    monkeypatch.setitem(G.FUNCTIONS, "adjustment.overlap", bogus)
    f = G.make(Point(family="adjustment", claim="overlap"), "students3", outcome="math score")
    assert not f.made and "do not resolve" in f.why and f.function == "adjustment.overlap"
