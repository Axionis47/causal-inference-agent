"""Gemini on Vertex AI. One place to build the model, so tests can swap it.

structured() returns the parsed contract and, separately, the model's thought summary
for debugging. Thoughts are never gated, cited, or read by another node.
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from typing import Any, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel

from causal_agent.common.contracts import Thought

load_dotenv()

T = TypeVar("T", bound=BaseModel)


def _gcloud_project() -> str | None:
    try:
        out = subprocess.run(["gcloud", "config", "get-value", "project"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def _flag(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_llm():
    from langchain_google_vertexai import ChatVertexAI

    kwargs: dict[str, Any] = dict(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        project=os.getenv("VERTEX_PROJECT") or _gcloud_project(),
        location=os.getenv("VERTEX_LOCATION", "us-central1"),
        temperature=0,
        max_retries=2,
    )
    if _flag("INCLUDE_THOUGHTS", True):
        kwargs["include_thoughts"] = True
        kwargs["thinking_budget"] = int(os.getenv("THINKING_BUDGET", "1024"))
    return ChatVertexAI(**kwargs)


_override: Any = None


def set_llm(llm: Any) -> None:
    """Tests inject a fake here. Pass None to restore the real model."""
    global _override
    _override = llm


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


def structured(schema: type[T], system: str, user: str, *, node: str) -> tuple[T, Thought]:
    llm = _override or get_llm()
    runnable = llm.with_structured_output(schema, include_raw=True)
    out = runnable.invoke([("system", system), ("human", user)])
    if isinstance(out, dict) and "parsed" in out:
        parsed, raw = out["parsed"], out.get("raw")
        if out.get("parsing_error") and parsed is None:
            raise ValueError(f"{node}: structured output failed to parse: {out['parsing_error']}")
    else:  # a fake that returns the object directly
        parsed, raw = out, None
    if isinstance(parsed, dict):
        parsed = schema.model_validate(parsed)
    text, tt, ot = extract_thoughts(raw) if raw is not None else ("", None, None)
    return parsed, Thought(node=node, text=text, thinking_tokens=tt, output_tokens=ot)
