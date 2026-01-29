"""Shared strict record primitives. Domain-specific fields stay with their owners."""

from pydantic import BaseModel, ConfigDict, Field


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)


class ReplyNumber(Contract):
    source: str = Field(
        description="Exact key from numeric_evidence supporting this specific numerical statement."
    )
    value: float = Field(
        allow_inf_nan=False, description="The number exactly as written in the reply, including rounding."
    )


