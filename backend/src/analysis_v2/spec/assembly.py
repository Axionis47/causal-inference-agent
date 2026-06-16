"""Data-assembly contracts (S0A): the recipe to build the analyzable frame.

A Kaggle bundle can ship several files. The assembly planner proposes a
deterministic recipe, an AssemblyPlan, to build one analyzable frame from the
bundle: a base file, optional same-schema shards concatenated onto it, and
optional joins onto sibling lookup tables. The plan is executed by
`runner.assembly.assemble_frame` and replayed verbatim in the generated
notebook, so an agentic (model-selected) assembly stays reproducible. The plan
carries no reshape: wide->long stays in the DiD lane this pass.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AssemblyJoin(BaseModel):
    """One horizontal merge onto the running frame. `how` excludes right/outer
    on purpose: a NaN-padding join is anticonservative if the key is wrong."""

    right_file: str = Field(min_length=1, max_length=255)
    on: list[str] = Field(min_length=1)
    how: Literal["left", "inner"] = "left"


class AssemblyToolCall(BaseModel):
    """One assembly step, recorded for replay. `kind` is read/join/concat/finish;
    `status` is 'ok', or 'rejected' when a constraint blocked the call."""

    name: str = Field(min_length=1, max_length=120)
    kind: str = Field(default="", max_length=20)
    args: dict[str, Any] = Field(default_factory=dict)
    status: str = Field(default="", max_length=20)
    rejected: str | None = Field(default=None, max_length=300)


class AssemblyToolTrace(BaseModel):
    """The ordered assembly steps behind a run, so the agentic planner's
    selection is frozen for audit even though the run is not deterministic."""

    calls: list[AssemblyToolCall] = Field(default_factory=list)
    exhausted: bool = False


class AssemblyPlan(BaseModel):
    """A deterministic recipe to build the analyzable frame from the bundle.
    Executable in the loader and replayable as the same call in the notebook."""

    base_file: str = Field(min_length=1, max_length=255)
    concat_files: list[str] = Field(default_factory=list)
    joins: list[AssemblyJoin] = Field(default_factory=list)
    drop_columns: list[str] = Field(default_factory=list)
    rationale: str = Field(default="", max_length=1000)
    # Transport for the planner's tool trace from execute() to commit();
    # excluded from serialization so it never leaks into the notebook config
    # and has one on-disk home, the run.assembly_tool_trace slot.
    tool_trace: AssemblyToolTrace | None = Field(default=None, exclude=True)

    @property
    def is_trivial(self) -> bool:
        """True when the plan is the single-winner default: nothing assembled."""
        return not self.concat_files and not self.joins and not self.drop_columns

    @classmethod
    def single_file(cls, winner: str) -> AssemblyPlan:
        """Today's behavior, made explicit: load one file, assemble nothing."""
        return cls(base_file=winner, rationale="single file; no assembly")
