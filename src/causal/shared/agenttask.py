"""Response schema for one model task (SYSTEM-CONTRACT §5.2; decision D-063; T-019 seed)."""

from __future__ import annotations

from copy import deepcopy
from functools import cache
from typing import Any

from pydantic import BaseModel

from causal.shared.envelope import AgentTaskResultV1

__all__ = ["result_schema"]


@cache
def result_schema(draft: type[BaseModel]) -> dict[str, object]:
    """`AgentTaskResultV1`'s JSON schema with `payload` replaced by the draft's (memoized)."""
    result: dict[str, Any] = deepcopy(AgentTaskResultV1.model_json_schema())
    payload: dict[str, Any] = deepcopy(draft.model_json_schema())
    defs: dict[str, Any] = result.setdefault("$defs", {})
    for name, definition in payload.pop("$defs", {}).items():
        if defs.setdefault(name, definition) != definition:
            raise ValueError(f"{draft.__name__} redefines $defs entry {name!r}")
    result["properties"]["payload"] = payload
    return result
