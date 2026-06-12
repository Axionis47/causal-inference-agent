"""The local ReAct loop an agent runs inside execute().

The driver owns what no agent may improvise: the tool-call budget, the
ToolCallRecord trail, token accounting, the live current_step on the
agent's run record, and the analysis_tool_called event. Agents supply
the prompt and the tools; the spine still sees one AgentResult.

Every provider message stays opaque to this driver: the client builds
user messages and tool-result messages in its own shape.
"""
from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src.analysis_v2.core import TokenUsage, ToolCallRecord
from src.analysis_v2.persistence import save_run

from .context import AgentCtx

BUDGET_NOTE = (
    "Tool budget exhausted. Write your final answer now from what you have; "
    "state what you could not verify."
)


@dataclass
class LoopTool:
    """One callable the model may invoke; handler returns a string."""

    name: str
    description: str
    parameters: dict  # JSON schema for the arguments
    handler: Callable[..., Any]  # sync or async; kwargs from the model


@dataclass
class LoopResult:
    text: str | None
    transcript: list[dict] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    tokens: TokenUsage = field(default_factory=TokenUsage)
    exhausted: bool = False


async def _run_handler(tool: LoopTool, args: dict) -> str:
    raw = tool.handler(**args)
    if inspect.isawaitable(raw):
        raw = await raw
    return str(raw)


async def _live_step(ctx: AgentCtx, call: dict, ok: bool) -> None:
    """Surface the call on the run record and the event stream."""
    args = call.get("args") or {}
    brief = ", ".join(f"{k}={str(v)[:24]}" for k, v in list(args.items())[:3])
    step = f"{call['name']}({brief})"[:200]
    if ctx.run.agent_runs:
        ctx.run.agent_runs[-1].current_step = step
    await save_run(ctx.run)
    ctx.emit("analysis_tool_called", {"tool": call["name"], "ok": ok, "headline": step})


async def react_loop(
    llm: Any,
    *,
    prompt: str,
    system_instruction: str,
    tools: list[LoopTool],
    ctx: AgentCtx,
    max_tool_calls: int = 24,
    max_turns: int = 10,
) -> LoopResult:
    """Drive think -> act -> observe until the model answers in text or
    the budget runs out. Returns the final text plus the full evidence
    trail; raises nothing for tool failures (they become observations)."""
    by_name = {t.name: t for t in tools}
    specs = [
        {"name": t.name, "description": t.description, "parameters": t.parameters}
        for t in tools
    ]
    messages: list[Any] = [llm.user_message(prompt)]
    result = LoopResult(text=None)
    used = 0

    for _turn in range(max_turns):
        response = await llm.chat_with_tools(messages, system_instruction, specs)
        usage = response.get("usage") or {}
        result.tokens = result.tokens.add(
            TokenUsage(
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
            )
        )
        calls = response.get("tool_calls") or []
        if not calls:
            result.text = response.get("text")
            return result

        messages.append(response["assistant_message"])
        budget_hit = False
        for call in calls:
            args = call.get("args") or {}
            if used >= max_tool_calls:
                # every requested call still needs a result message
                output, ok, error = "tool budget exhausted", False, "budget exhausted"
                budget_hit = True
            else:
                used += 1
                tool = by_name.get(call["name"])
                started = time.monotonic()
                if tool is None:
                    output, ok, error = f"unknown tool {call['name']}", False, "unknown tool"
                else:
                    try:
                        output, ok, error = await _run_handler(tool, args), True, None
                    except Exception as exc:  # the failure becomes an observation
                        output, ok, error = f"tool error: {exc}", False, str(exc)[:500]
                result.tool_calls.append(
                    ToolCallRecord(
                        name=call["name"],
                        args_summary=json.dumps(args, default=str)[:500],
                        ok=ok,
                        error=error,
                        duration_ms=int((time.monotonic() - started) * 1000),
                    )
                )
                result.transcript.append(
                    {
                        "thought": response.get("text"),
                        "tool": call["name"],
                        "args": args,
                        "output": output[:4000],
                        "ok": ok,
                    }
                )
                await _live_step(ctx, call, ok)
            messages.append(llm.tool_result_message(call, output))
        if budget_hit:
            messages.append(llm.user_message(BUDGET_NOTE))

    result.exhausted = True
    return result
