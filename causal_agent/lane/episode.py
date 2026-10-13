"""A judgement as a bounded episode. The model reads its material, may ask code for facts about the data through a fixed set of
read-only tools, sees every fact it asked for as an addressed line, and answers in one typed record that code gates. The budget,
the tool set and the shape of the answer are fixed by the caller; the leash is short and visible, and every fact looked at is
logged with its address, `probe:<node>.<n>`, so the record, the report and the chat after can cite it.

Looking and answering are two calls. The model looks first, in a conversation with the tools bound; then the facts it gathered
are rendered under FACTS YOU ASKED FOR and the record is asked for with the same structured call every judgement uses. A refused
answer goes back with its errors, the log persists, and the model may look again while budget remains. Every try is kept: the
text the model was shown, the answer it gave, and what the gate said, so a watcher can read why a rung went the way it did.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any, Protocol, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from causal_agent.common import llm as LLM
from causal_agent.common import trace
from causal_agent.common.addresses import norm_address
from causal_agent.common.contracts import Thought
from causal_agent.lane.nodes import rejected
from causal_agent.lane.tools import Refused, Result

T = TypeVar("T", bound=BaseModel)
MAX_TRIES = 3


class ToolSet(Protocol):
    """What an episode needs of its tools: their names, their schemas for the model, and one call by name. The lane's tools
    and the exploration agent's code tool both have this shape. A tool set that also has `step_schema()` is looked at one
    step at a time as a structured record instead of a function call: the record names the one tool's arguments and whether
    the model is done. A model that mangles a function call carrying a script (Gemini does) still answers a schema cleanly."""

    def names(self) -> tuple[str, ...]: ...

    def schemas(self) -> list[Any]: ...

    def call(self, name: str, args: dict[str, Any]) -> Result: ...


TOOL_RULE = (
    "Before you answer you may look at the data with the tools offered, at most {budget} calls in all. Each result comes back "
    "with an address [probe:{node}.<n>]; cite it as you would a pack address. A tool that would join the outcome with the "
    "treatment is refused, and the refusal says why. When you have seen enough, stop calling tools."
)
STEP_RULE = (
    "Before you answer you act one step at a time, at most {budget} steps in all: return the next step as the record asked for, "
    "or say done when you have what was asked for. Each step's result comes back with an address [probe:{node}.<n>] under "
    "WHAT YOU RAN SO FAR; cite it as you would a pack address. A step that is refused says why; read it and act again."
)


class Fact(BaseModel):
    """One thing the model asked code for, with the address it may cite."""

    node: str
    visit: str = "manual"
    n: int
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    text: str
    value: float | None = None

    @property
    def address(self) -> str:
        return f"probe:{self.node}.{self.visit}.{self.n}"

    def render(self) -> str:
        return f"[{self.address}] {self.tool}({_args(self.args)}): {self.text}"


class Refusal(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str


class Try(BaseModel):
    """One try of a judgement: the text the model was shown, the answer as it gave it, and the gate's errors, empty when it passed."""

    n: int
    user: str
    answer: dict[str, Any]
    errors: list[str] = Field(default_factory=list)


class EpisodeLog(BaseModel):
    """What one episode looked at and did: the system text once, the facts with their addresses, the refusals, the calls spent,
    and every try with what the model was shown, what it answered, and what the gate said."""

    node: str
    visit: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    notes: str = ""
    system: str = ""
    commit_system: str = ""
    facts: list[Fact] = Field(default_factory=list)
    refusals: list[Refusal] = Field(default_factory=list)
    calls: int = 0
    tries: list[Try] = Field(default_factory=list)

    def find(self, address: str) -> Fact | None:
        a = norm_address(address)
        return next((f for f in self.facts if norm_address(f.address) == a), None)

    def resolve(self, address: str) -> bool:
        return self.find(address) is not None

    def render(self) -> str:
        return ("FACTS YOU ASKED FOR\n" + "\n".join(f.render() for f in self.facts)) if self.facts else ""


def _args(a: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in a.items())


def _material(user: str, log: EpisodeLog, errors: list[str]) -> str:
    facts = log.render()
    return user + (f"\n\n{facts}\n" if facts else "") + rejected(errors)


def _run(tools: ToolSet, name: str, args: dict[str, Any], budget: int, log: EpisodeLog) -> str:
    """One tool call: the addressed fact the model reads back, or the refusal; a tool span either way."""
    with trace.span(name, run_type="tool", inputs=args) as s:
        if log.calls >= budget:
            text = f"refused: the budget of {budget} calls is spent; answer from what you have"
            s.end(outputs={"refused": text})
            return text
        log.calls += 1
        try:
            r = tools.call(name, args)
        except Refused as e:
            log.refusals.append(Refusal(tool=name, args=args, reason=str(e)))
            s.end(outputs={"refused": str(e)})
            return str(e)
        brief = getattr(tools, "brief_args", None)  # a tool set may say its arguments shorter than they are (a script by its length)
        f = Fact(node=log.node, visit=log.visit, n=len(log.facts) + 1, tool=name, args=brief(name, args) if brief else args, text=r.text, value=r.value)
        log.facts.append(f)
        s.end(outputs={"address": f.address, "text": f.text, "value": f.value})
        return f.render()


def _look(system: str, user: str, tools: ToolSet, budget: int, log: EpisodeLog, errors: list[str]) -> list[Thought]:
    """The tool phase: the model calls tools until it stops or the budget is spent; every result is appended as a tool message."""
    with trace.span("look", inputs={"budget": budget, "spent": log.calls, "tools": list(tools.names())}) as s:
        thoughts = _look_in(system, user, tools, budget, log, errors)
        s.end(
            outputs={
                "spent": log.calls,
                "facts": [f.render() for f in log.facts],
                "refusals": [r.reason for r in log.refusals],
                "thoughts": [t.text for t in thoughts if t.text],
            }
        )
        return thoughts


def _look_in(system: str, user: str, tools: ToolSet, budget: int, log: EpisodeLog, errors: list[str]) -> list[Thought]:
    step_schema = getattr(tools, "step_schema", None)
    if step_schema is not None:
        return _look_by_record(system, user, tools, step_schema(), budget, log, errors)
    bound = LLM.current().bind_tools(tools.schemas())
    messages: list[Any] = [
        SystemMessage(content=system + "\n\n" + TOOL_RULE.format(budget=budget, node=f"{log.node}.{log.visit}")),
        HumanMessage(content=_material(user, log, errors)),
    ]
    thoughts: list[Thought] = []
    while log.calls < budget:
        ai = bound.invoke(messages)
        text, tt, ot = LLM.extract_thoughts(ai)
        if text:
            thoughts.append(Thought(node=f"{log.node}:look", text=text, thinking_tokens=tt, output_tokens=ot))
        calls = list(getattr(ai, "tool_calls", None) or [])
        if not calls:
            log.notes = text
            break
        messages.append(ai)
        for i, call in enumerate(calls):
            content = _run(tools, str(call.get("name")), dict(call.get("args") or {}), budget, log)
            messages.append(ToolMessage(content=content, tool_call_id=str(call.get("id") or f"{log.calls}.{i}")))
    return thoughts


def _look_by_record(system: str, user: str, tools: ToolSet, schema: type[BaseModel], budget: int, log: EpisodeLog, errors: list[str]) -> list[Thought]:
    """The look phase as records: each step is one structured call answering the tool set's step schema (its one tool's
    arguments, and `done`); the step runs and its result is shown under WHAT YOU RAN SO FAR for the next."""
    name = tools.names()[0]
    thoughts: list[Thought] = []
    seen: list[str] = []
    while log.calls < budget:
        shown = _material(user, log, errors) + ("\n\nWHAT YOU RAN SO FAR\n" + "\n\n".join(seen) if seen else "")
        record, th = LLM.structured(schema, system + "\n\n" + STEP_RULE.format(budget=budget, node=log.node), shown, node=f"{log.node}:look")
        thoughts.append(th)
        if getattr(record, "done", False):
            break
        args = record.model_dump(mode="json", exclude={"done"})
        seen.append(_run(tools, name, args, budget, log))
    return thoughts


def run_episode(
    schema: type[T],
    system: str,
    user: str,
    *,
    tools: ToolSet | None,
    budget: int,
    gate: Callable[[T, EpisodeLog], list[str]],
    node: str,
    tries: int = MAX_TRIES,
    notes: bool = False,
    commit_system: str | None = None,
) -> tuple[T | None, EpisodeLog, list[Thought], list[str]]:
    """Run one episode. Returns the record the gate accepted (or None after `tries` refusals), the log, the thoughts, and the
    errors of the last refusal. The episode is one span, each try a span under it, each tool call and gate a span under that."""
    log = EpisodeLog(node=node, system=system, commit_system=commit_system or system)
    thoughts: list[Thought] = []
    errors: list[str] = []
    with trace.span(
        f"episode:{node}",
        inputs={
            "schema": schema,
            "system": system,
            "commit_system": log.commit_system,
            "user": user,
            "budget": budget,
            "tools": list(tools.names()) if tools else [],
            "tries": tries,
        },
    ) as ep:
        for n in range(1, tries + 1):
            with trace.span(f"try {n}", inputs={"errors": list(errors)}) as t:
                if tools is not None and log.calls < budget:
                    try:
                        thoughts += _look(system, user, tools, budget, log, errors)
                    except ValueError as exc:
                        errors = trace.gate([f"structured tool step could not be parsed: {exc}"], name="gate", attempt=n)
                        shown = _material(user, log, errors)
                        log.tries.append(Try(n=n, user=shown, answer={"parsing_error": str(exc)}, errors=list(errors)))
                        t.end(outputs={"errors": list(errors), "parsing_error": str(exc)})
                        continue
                shown = _material(user, log, errors)
                if notes and log.notes:
                    shown += "\n\nWORKING NOTES (proposals, not evidence)\n" + log.notes
                try:
                    record, th = LLM.structured(schema, log.commit_system, shown, node=node)
                except ValueError as exc:
                    errors = trace.gate([f"structured record could not be parsed: {exc}"], name="gate", attempt=n)
                    log.tries.append(Try(n=n, user=shown, answer={"parsing_error": str(exc)}, errors=list(errors)))
                    t.end(outputs={"errors": list(errors), "parsing_error": str(exc)})
                    continue
                thoughts.append(th)
                answer = record.model_dump(mode="json")  # as the model gave it, before a gate may normalise a field
                errors = trace.gate(gate(record, log))
                log.tries.append(Try(n=n, user=shown, answer=answer, errors=list(errors)))
                t.end(outputs={"answer": answer, "errors": list(errors)})
            if not errors:
                ep.end(
                    outputs={
                        "record": answer,
                        "tries": n,
                        "calls": log.calls,
                        "facts": [f.render() for f in log.facts],
                        "refusals": [r.reason for r in log.refusals],
                    }
                )
                return record, log, thoughts, []
        ep.end(
            outputs={
                "record": None,
                "tries": tries,
                "calls": log.calls,
                "facts": [f.render() for f in log.facts],
                "refusals": [r.reason for r in log.refusals],
                "errors": list(errors),
            },
            error="no record passed the gate in " + str(tries) + " tries: " + "; ".join(errors[:3]),
        )
    return None, log, thoughts, errors
