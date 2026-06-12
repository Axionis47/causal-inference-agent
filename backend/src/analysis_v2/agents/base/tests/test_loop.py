"""The ReAct loop driver against a scripted fake client."""
from __future__ import annotations

import pandas as pd
import pytest

from src.analysis_v2.agents.base import AgentCtx, LoopTool, react_loop
from src.analysis_v2.core import AnalysisRunState, AgentRun, AnalysisStage


class FakeLLM:
    """Plays back scripted turns; messages are plain dicts."""

    def __init__(self, turns: list[dict]):
        self.turns = list(turns)
        self.seen_messages: list[list] = []

    def user_message(self, text):
        return {"role": "user", "content": text}

    def tool_results_message(self, results):
        return {
            "role": "tool_result",
            "content": [{"name": c["name"], "output": o} for c, o in results],
        }

    async def chat_with_tools(self, messages, system_instruction, tools):
        self.seen_messages.append(list(messages))
        turn = self.turns.pop(0)
        return {
            "text": turn.get("text"),
            "tool_calls": turn.get("tool_calls", []),
            "assistant_message": {"role": "assistant", "content": turn.get("text") or ""},
            "usage": turn.get("usage", {"input_tokens": 10, "output_tokens": 5}),
        }


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path))
    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    run = AnalysisRunState(job_id="job-loop", causal_question="q?")
    run.add_agent_run(AgentRun(agent="tester", stage=AnalysisStage.S1_INTAKE_PARSED))
    events: list[tuple] = []
    yield AgentCtx(
        job_id="job-loop", run=run, frame=pd.DataFrame({"a": [1]}),
        emit=lambda kind, data: events.append((kind, data)),
    ), events
    settings_mod.get_settings.cache_clear()


def _tools(calls_log):
    def peek(column: str) -> str:
        calls_log.append(column)
        return f"column {column}: numeric, no nulls"

    def boom() -> str:
        raise RuntimeError("backend exploded")

    schema = {"type": "object", "properties": {"column": {"type": "string"}}}
    return [
        LoopTool("peek", "Inspect a column", schema, peek),
        LoopTool("boom", "Always fails", {"type": "object", "properties": {}}, boom),
    ]


async def test_loop_threads_tool_results_and_returns_the_final_text(ctx):
    context, events = ctx
    calls_log: list[str] = []
    llm = FakeLLM(
        [
            {"text": "let me look", "tool_calls": [{"id": "1", "name": "peek", "args": {"column": "age"}}]},
            {"text": "the answer: age is clean"},
        ]
    )
    result = await react_loop(
        llm, prompt="inspect", system_instruction="sys",
        tools=_tools(calls_log), ctx=context,
    )

    assert result.text == "the answer: age is clean"
    assert calls_log == ["age"]
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "peek"
    assert result.tool_calls[0].ok is True
    assert result.tokens.total == 30  # two turns x (10 + 5)
    # the tool result was threaded back into the conversation
    final_messages = llm.seen_messages[-1]
    assert any(m.get("role") == "tool_result" for m in final_messages)
    # live visibility: current_step set and the event emitted
    assert context.run.agent_runs[-1].current_step.startswith("peek(")
    assert ("analysis_tool_called", {"tool": "peek", "ok": True, "headline": context.run.agent_runs[-1].current_step}) in events


async def test_a_tool_crash_becomes_an_observation_not_an_exception(ctx):
    context, _ = ctx
    llm = FakeLLM(
        [
            {"tool_calls": [{"id": "1", "name": "boom", "args": {}}]},
            {"text": "could not verify; backend failing"},
        ]
    )
    result = await react_loop(
        llm, prompt="p", system_instruction="s", tools=_tools([]), ctx=context,
    )
    assert result.text == "could not verify; backend failing"
    record = result.tool_calls[0]
    assert record.ok is False
    assert "exploded" in record.error
    assert result.transcript[0]["ok"] is False


async def test_the_budget_stops_execution_and_demands_a_final_answer(ctx):
    context, _ = ctx
    call = {"id": "1", "name": "peek", "args": {"column": "x"}}
    llm = FakeLLM(
        [
            {"tool_calls": [call, call, call]},
            {"text": "final answer under budget"},
        ]
    )
    calls_log: list[str] = []
    result = await react_loop(
        llm, prompt="p", system_instruction="s",
        tools=_tools(calls_log), ctx=context, max_tool_calls=2,
    )
    assert result.text == "final answer under budget"
    assert len(calls_log) == 2  # third call never executed
    assert len(result.tool_calls) == 2
    # the model was told to wrap up
    final_messages = llm.seen_messages[-1]
    assert any("budget exhausted" in str(m.get("content", "")).lower() for m in final_messages)


async def test_turn_limit_marks_the_result_exhausted(ctx):
    context, _ = ctx
    call = {"id": "1", "name": "peek", "args": {"column": "x"}}
    llm = FakeLLM([{"tool_calls": [call]}, {"tool_calls": [call]}])
    result = await react_loop(
        llm, prompt="p", system_instruction="s",
        tools=_tools([]), ctx=context, max_turns=2,
    )
    assert result.exhausted is True
    assert result.text is None
