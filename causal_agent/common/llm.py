"""Gemini on Vertex AI. One place to build the model, so tests can swap it.

structured() returns the parsed contract and, separately, the model's thought summary
for debugging. Thoughts are never gated, cited, or read by another node.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from typing import Any, Literal, TypeVar

from dotenv import load_dotenv
from langgraph.types import RetryPolicy
from pydantic import BaseModel

from causal_agent.common import config, decider, trace
from causal_agent.common.contracts import Thought

load_dotenv()

T = TypeVar("T", bound=BaseModel)

# light: the per-turn reading of the person's words, which a smaller model can do; heavy: everything that designs, decides
# or explains. One model serves both until GEMINI_MODEL_LIGHT names another.
Tier = Literal["light", "heavy"]

# the retry policy for a graph node that calls the model: three tries, a second apart at first
RETRY = RetryPolicy(max_attempts=3, initial_interval=1.0)


def _gcloud_project() -> str | None:
    try:
        out = subprocess.run(["gcloud", "config", "get-value", "project"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def model_name(tier: Tier = "heavy") -> str:
    m = config.get().model
    return m.light if tier == "light" else m.name


@lru_cache(maxsize=2)
def get_llm(tier: Tier = "heavy"):
    from langchain_google_vertexai import ChatVertexAI

    m = config.get().model
    kwargs: dict[str, Any] = dict(model=model_name(tier), project=m.project or _gcloud_project(), location=m.location, temperature=0, max_retries=2)
    if m.include_thoughts:
        kwargs["include_thoughts"] = True
        kwargs["thinking_budget"] = m.thinking_budget
    return ChatVertexAI(**kwargs)


_override: Any = None


def set_llm(llm: Any) -> None:
    """Tests inject a fake here. Pass None to restore the real model."""
    global _override
    _override = llm


def current(tier: Tier = "heavy") -> Any:
    """The model in use: the test's fake when one is set, whatever the tier, else Gemini at that tier."""
    return _override or get_llm(tier)


# ------------------------------------------------------------------ the decision model


_decider_override: Any = None


@lru_cache(maxsize=1)
def get_decider() -> decider.Decider:
    return decider.build()


def set_decider(d: Any) -> None:
    """Tests inject a fake here: anything with `predict(state, questions) -> dict` and a `name`. None restores the real one."""
    global _decider_override
    _decider_override = d


def current_decider() -> Any:
    return _decider_override or get_decider()


def decide(state: dict[str, Any] | str, questions: dict[str, Any], *, node: str) -> decider.Decisions:
    """One set of typed decisions about a text, no words: the state and the questions in, each answer with its confidence
    out, as a span named `node`. The one place a decision model is called."""
    d = current_decider()
    with trace.span(node, inputs={"state": state, "questions": questions, "model": d.name}) as s:
        out = decider.parse(d.predict(state, questions), model=d.name)
        s.end(outputs={"answers": {k: a.model_dump() for k, a in out.answers.items()}})
        return out


def extract_thoughts(raw: Any) -> tuple[str, int | None, int | None]:
    """Pull thought text and token counts out of an AIMessage, tolerating several shapes."""
    text_parts: list[str] = []
    content = getattr(raw, "content", None)
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in {"thinking", "thought", "reasoning"} or part.get("thought") is True:
                    t = part.get("thinking") or part.get("text") or part.get("reasoning") or ""
                    if t:
                        text_parts.append(str(t))
    extra = getattr(raw, "additional_kwargs", {}) or {}
    for key in ("thoughts", "thinking", "reasoning"):
        if isinstance(extra.get(key), str):
            text_parts.append(extra[key])
    usage = getattr(raw, "usage_metadata", None) or {}
    details = usage.get("output_token_details", {}) if isinstance(usage, dict) else {}
    thinking_tokens = details.get("reasoning") if isinstance(details, dict) else None
    output_tokens = usage.get("output_tokens") if isinstance(usage, dict) else None
    return "\n".join(text_parts), thinking_tokens, output_tokens


def structured(schema: type[T], system: str, user: str | list[dict[str, Any]], *, node: str, tier: Tier = "heavy") -> tuple[T, Thought]:
    """One judgement: the prompt in, one typed record and the model's thoughts out, as a span named `node` in the trace.
    `tier` picks the model; the span records which one answered."""
    with trace.span(node, inputs={"schema": schema, "system": system, "user": user, "tier": tier, "model": model_name(tier)}) as s:
        parsed, thought = _structured(schema, system, user, node=node, tier=tier)
        s.end(
            outputs={
                "record": trace.record(parsed),
                "thoughts": thought.text,
                "thinking_tokens": thought.thinking_tokens,
                "output_tokens": thought.output_tokens,
            }
        )
        return parsed, thought


def _structured(schema: type[T], system: str, user: str | list[dict[str, Any]], *, node: str, tier: Tier) -> tuple[T, Thought]:
    runnable = current(tier).with_structured_output(schema, include_raw=True)
    out = runnable.invoke([("system", system), ("human", user)])
    if isinstance(out, dict) and "parsed" in out:
        parsed, raw = out["parsed"], out.get("raw")
        if parsed is None:  # the model answered with no record, parseable or not: the caller's retry sees one error, never a None
            why = out.get("parsing_error")
            raise ValueError(f"{node}: structured output failed to parse: {why}" if why else f"{node}: the model returned no structured answer")
    else:  # a fake that returns the object directly
        parsed, raw = out, None
    if parsed is None:
        raise ValueError(f"{node}: the model returned no structured answer")
    if isinstance(parsed, dict):
        parsed = schema.model_validate(parsed)
    text, tt, ot = extract_thoughts(raw) if raw is not None else ("", None, None)
    return parsed, Thought(node=node, text=text, thinking_tokens=tt, output_tokens=ot)
