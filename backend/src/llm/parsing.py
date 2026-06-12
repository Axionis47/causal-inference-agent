"""Shared response-text cleanup for the LLM clients."""
from __future__ import annotations


def strip_fences(text: str) -> str:
    """Drop markdown code fences around a JSON (or any) payload."""
    out = text.strip()
    if out.startswith("```json"):
        out = out[7:]
    if out.startswith("```"):
        out = out[3:]
    if out.endswith("```"):
        out = out[:-3]
    return out.strip()
