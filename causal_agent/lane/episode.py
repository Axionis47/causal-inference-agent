"""A judgement as a bounded episode. The model reads its material, may ask code for facts about the data through a fixed set of
read-only tools, sees every fact it asked for as an addressed line, and answers in one typed record that code gates. The budget,
the tool set and the shape of the answer are fixed by the caller; the leash is short and visible, and every fact looked at is
logged with its address, `probe:<node>.<n>`, so the record, the report and the chat after can cite it.

Looking and answering are two calls. The model looks first, in a conversation with the tools bound; then the facts it gathered
are rendered under FACTS YOU ASKED FOR and the record is asked for with the same structured call every judgement uses. A refused
answer goes back with its errors, the log persists, and the model may look again while budget remains.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from causal_agent.common import llm as LLM
from causal_agent.common.addresses import norm_address
from causal_agent.common.contracts import Thought
from causal_agent.lane.nodes import rejected
from causal_agent.lane.tools import Refused, Tools

T = TypeVar("T", bound=BaseModel)
MAX_TRIES = 3

TOOL_RULE = (
    "Before you answer you may look at the data with the tools offered, at most {budget} calls in all. Each result comes back "
    "with an address [probe:{node}.<n>]; cite it as you would a pack address. A tool that would join the outcome with the "
    "treatment is refused, and the refusal says why. When you have seen enough, stop calling tools."
)


class Fact(BaseModel):
    """One thing the model asked code for, with the address it may cite."""

    node: str
    n: int
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    text: str
    value: float | None = None

    @property
    def address(self) -> str:
        return f"probe:{self.node}.{self.n}"

    def render(self) -> str:
        return f"[{self.address}] {self.tool}({_args(self.args)}): {self.text}"


class Refusal(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str


class EpisodeLog(BaseModel):
    """What one episode looked at: the facts with their addresses, the refusals, the calls spent, the tries it took."""

    node: str
    facts: list[Fact] = Field(default_factory=list)
    refusals: list[Refusal] = Field(default_factory=list)
    calls: int = 0
    tries: int = 0

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


def _run(tools: Tools, name: str, args: dict[str, Any], budget: int, log: EpisodeLog) -> str:
    """One tool call: the addressed fact the model reads back, or the refusal."""
    if log.calls >= budget:
        return f"refused: the budget of {budget} calls is spent; answer from what you have"
    log.calls += 1
    try:
        r = tools.call(name, args)
    except Refused as e:
        log.refusals.append(Refusal(tool=name, args=args, reason=str(e)))
        return str(e)
    f = Fact(node=log.node, n=len(log.facts) + 1, tool=name, args=args, text=r.text, value=r.value)
    log.facts.append(f)
    return f.render()


def _look(system: str, user: str, tools: Tools, budget: int, log: EpisodeLog, errors: list[str]) -> list[Thought]:
    """The tool phase: the model calls tools until it stops or the budget is spent; every result is appended as a tool message."""
    bound = LLM.current().bind_tools(tools.schemas())
    messages: list[Any] = [
        SystemMessage(content=system + "\n\n" + TOOL_RULE.format(budget=budget, node=log.node)),
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
            break
        messages.append(ai)
        for i, call in enumerate(calls):
            content = _run(tools, str(call.get("name")), dict(call.get("args") or {}), budget, log)
            messages.append(ToolMessage(content=content, tool_call_id=str(call.get("id") or f"{log.calls}.{i}")))
    return thoughts


def run_episode(
    schema: type[T],
    system: str,
    user: str,
    *,
    tools: Tools | None,
    budget: int,
    gate: Callable[[T, EpisodeLog], list[str]],
    node: str,
    tries: int = MAX_TRIES,
) -> tuple[T | None, EpisodeLog, list[Thought], list[str]]:
    """Run one episode. Returns the record the gate accepted (or None after `tries` refusals), the log, the thoughts, and the
    errors of the last refusal."""
    log = EpisodeLog(node=node)
    thoughts: list[Thought] = []
    errors: list[str] = []
    for _ in range(tries):
        log.tries += 1
        if tools is not None and log.calls < budget:
            thoughts += _look(system, user, tools, budget, log, errors)
        record, th = LLM.structured(schema, system, _material(user, log, errors), node=node)
        thoughts.append(th)
        errors = gate(record, log)
        if not errors:
            return record, log, thoughts, []
    return None, log, thoughts, errors
