"""A structured call returns the record or raises: a model that answers with nothing is one error the caller's retry sees,
never a None the caller dereferences."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from causal_agent.common.llm import set_llm, structured


class Answer(BaseModel):
    text: str


class _Fake:
    def __init__(self, out):
        self.out = out

    def with_structured_output(self, schema, include_raw=False):
        out = self.out

        class R:
            def invoke(self_, messages):
                return out

        return R()


@pytest.fixture(autouse=True)
def _restore():
    yield
    set_llm(None)


def test_a_record_comes_back_as_the_record():
    set_llm(_Fake({"raw": None, "parsed": Answer(text="yes"), "parsing_error": None}))
    parsed, thought = structured(Answer, "s", "u", node="t")
    assert parsed == Answer(text="yes") and thought.node == "t"
    set_llm(_Fake(Answer(text="direct")))  # a fake that returns the object itself
    assert structured(Answer, "s", "u", node="t")[0].text == "direct"


def test_no_record_raises_and_names_the_node():
    set_llm(_Fake({"raw": None, "parsed": None, "parsing_error": None}))
    with pytest.raises(ValueError, match="t: the model returned no structured answer"):
        structured(Answer, "s", "u", node="t")
    set_llm(_Fake({"raw": None, "parsed": None, "parsing_error": "bad json"}))
    with pytest.raises(ValueError, match="failed to parse: bad json"):
        structured(Answer, "s", "u", node="t")
    set_llm(_Fake(None))
    with pytest.raises(ValueError, match="no structured answer"):
        structured(Answer, "s", "u", node="t")
