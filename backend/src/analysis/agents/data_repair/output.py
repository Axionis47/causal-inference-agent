"""Typed output for the data_repair agent.

Each repair the agent applies is captured as a RepairRecord. Today the
agent still writes plain dicts to `state.data_repairs` (the notebook
section reads them via `.get()`); the typed model is defined here so
the S19 state-slimming pass can flip the field over without inventing
the shape from scratch.
"""

from typing import Literal

from pydantic import BaseModel, Field


class RepairRecord(BaseModel):
    """One repair action applied by the agent."""

    type: Literal["missing", "outliers", "collinearity"]
    strategy: str
    columns: list[str] = Field(default_factory=list)
    before: int | None = None
    after: int | None = None
    rows_dropped: int | None = None


class DataRepairs(BaseModel):
    """Aggregate of all repair actions for one analysis run."""

    records: list[RepairRecord] = Field(default_factory=list)
    quality_assessment: str = ""
    cautions: list[str] = Field(default_factory=list)
    original_shape: tuple[int, int] | None = None
    final_shape: tuple[int, int] | None = None
