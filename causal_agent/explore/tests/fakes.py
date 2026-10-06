"""A scripted model for the exploration agent: each entry of `looks` is one round of looking, a list of scripts to run with
run_code (an empty list stops looking); `answers` are the records, in order."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from causal_agent.explore.contracts import Exploration


class FakeLLM:
    def __init__(self, looks: list[list[str]], answers: list[Exploration]):
        self.looks = list(looks)
        self.answers = list(answers)
        self.bound: list[str] = []
        self.tool_texts: list[str] = []
        self.humans: list[str] = []

    def bind_tools(self, tools):
        self.bound = [t.name for t in tools]
        return self

    def invoke(self, messages):
        for m in messages:
            if getattr(m, "type", "") == "tool":
                self.tool_texts.append(str(m.content))
        scripts = self.looks.pop(0) if self.looks else []
        return AIMessage(
            content=[{"type": "thinking", "thinking": "looking"}] if scripts else "",
            tool_calls=[{"name": "run_code", "args": {"code": code}, "id": f"c{i}"} for i, code in enumerate(scripts)],
        )

    def with_structured_output(self, schema, include_raw=False):
        fake = self

        class R:
            def invoke(self_, messages):
                fake.humans.append(messages[-1][1])
                return {"raw": AIMessage(content=""), "parsed": fake.answers.pop(0), "parsing_error": None}

        return R()
