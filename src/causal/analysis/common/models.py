"""Small dependency-free (apart from Pydantic) boundary primitives."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

Scalar = str | int | float | bool | None
Category = Literal[
    "missing_context", "missing_data", "unsupported_capability",
    "contradictory_configuration", "incompatible_data",
]
Applicability = Literal["applicable", "inapplicable", "unresolved"]


class Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class Issue(Model):
    category: Category
    field: str
    finding: str
    requirement: str
    explanation: str
    resolutions: tuple[str, ...] = ()


FeedbackIssue = Issue
