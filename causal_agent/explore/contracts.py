"""What crosses into the exploration agent and what comes back. The request is everything it may read: the table (projected
before the freeze), what is known as text, the column names, and the pair it must not join. The record is the model's answer,
flat so any structured-output backend accepts it. The trail is the whole of what happened, for the caller to keep."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from causal_agent.common.contracts import Thought

Dropped = Literal["outcome", "treatment"]


class ExploreRequest(BaseModel):
    dataset: str
    ask: str = Field(description="what to look at, in the person's words")
    context: str = Field(description="what is known about the data, as text with addresses; after a run, what the run found too")
    columns: dict[str, str] = Field(description="key -> the file's own column name, for the columns in the table sent")
    csv: Path = Field(description="the table the sandbox reads: the projection before the freeze, the file after")
    out_dir: Path = Field(description="the only place the agent writes: the artifact's folder")
    outcome: str | None = Field(default=None, description="the outcome column, by the file's own name")
    treatment: str | None = Field(default=None, description="the column that records who got the change, by the file's own name")
    dropped: Dropped | None = Field(default=None, description="which of the pair was taken out of the table sent; None after the freeze")
    frozen: bool = Field(default=False, description="a design has run: the outcome may be seen beside the treatment")
    design: int | None = Field(default=None, description="the design run the ask is about; None before any run")
    memory_version: int = 0
    data_version: int = 0
    budget: int = Field(default=0, description="scripts the agent may run; 0 takes the budget checks.yaml declares")

    def dropped_name(self) -> str | None:
        if self.dropped == "outcome":
            return self.outcome
        if self.dropped == "treatment":
            return self.treatment
        return None


class Exploration(BaseModel):
    """The agent's answer: which step's files are the product, and what it stands behind."""

    summary: str = Field(
        description="two or three sentences in the person's words on what was found; every number in them is in that step's facts.json or in what you were told"
    )
    step: int = Field(description="the run_code call whose files are the answer, counted from 1")
    facts: list[str] = Field(default_factory=list, description="the names in that step's facts.json you stand behind")
    figure: bool = Field(default=False, description="that step's figure.png is the picture asked for")
    table: str | None = Field(default=None, description="the CSV or JSON file that step wrote as the table of facts, by name, or null")


class Step(BaseModel):
    """One script that ran: what it printed, what it wrote, and the numbers it left in facts.json."""

    n: int
    ok: bool
    stdout: str
    stderr: str
    files: list[str]
    facts: dict[str, float] | None = None
    kind: str


class Trail(BaseModel):
    """Everything one exploration did, for the caller: the record or why there is none, the thoughts, the steps, the calls."""

    record: Exploration | None
    why: str = ""
    thoughts: list[Thought] = Field(default_factory=list)
    steps: list[Step] = Field(default_factory=list)
    calls: int = 0
    refusals: list[str] = Field(default_factory=list)
    sandbox: str = "subprocess"
