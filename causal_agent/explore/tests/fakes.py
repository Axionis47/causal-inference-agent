"""A scripted model for the exploration agent: each entry of `looks` is one round of looking, a list of scripts to run one
step at a time (an empty list is done at once); `answers` are the records, in order. Every call is a structured call, as it is
with the real model; the prompts are kept by schema name."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from causal_agent.explore.contracts import Exploration
from causal_agent.explore.inspection import ExploreAction
from causal_agent.explore.tools import RunCodeStep


class FakeLLM:
    def __init__(self, looks: list[list[str]], answers: list[Exploration]):
        self.rounds = [list(r) for r in looks]
        self.answers = list(answers)
        self.humans: dict[str, list[str]] = {}
        self.systems: dict[str, list[str]] = {}
        self.current: list[str] | None = None

    def with_structured_output(self, schema, include_raw=False):
        fake = self

        class R:
            def invoke(self_, messages):
                fake.systems.setdefault(schema.__name__, []).append(messages[0][1])
                human = messages[-1][1]
                if isinstance(human, list):
                    human = "\n".join(p.get("text", "[image]") for p in human)
                fake.humans.setdefault(schema.__name__, []).append(human)
                if schema in (RunCodeStep, ExploreAction):
                    if fake.current is None:
                        fake.current = fake.rounds.pop(0) if fake.rounds else []
                    if fake.current:
                        item = fake.current.pop(0)
                        parsed = (
                            item
                            if isinstance(item, ExploreAction)
                            else (ExploreAction(action="run_code", code=item) if schema is ExploreAction else RunCodeStep(code=item))
                        )
                    else:
                        fake.current = None
                        parsed = schema(done=True)
                else:
                    parsed = fake.answers.pop(0)
                return {
                    "raw": AIMessage(content=[{"type": "thinking", "thinking": "looking" if schema in (RunCodeStep, ExploreAction) else "answering"}]),
                    "parsed": parsed,
                    "parsing_error": None,
                }

        return R()
