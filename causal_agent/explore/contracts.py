"""Whole-data exploration and separate row-preserving transformation contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from causal_agent.common.contracts import Thought

Dropped = Literal["outcome"]


class DatasetInput(BaseModel):
    """Resolved by the caller, never by the model. The entire version is supplied."""

    model_config = ConfigDict(extra="forbid")
    id: str
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ArtifactInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExploreRequest(BaseModel):
    """Private A2A execution envelope. Paths are materialized by the trusted caller."""

    model_config = ConfigDict(extra="forbid")
    trace_headers: dict[str, str] = Field(default_factory=dict)
    question: str = Field(min_length=1)
    dataset_version: DatasetInput
    context: str
    artifact_refs: list[ArtifactInput] = Field(default_factory=list)
    out_dir: Path
    request_id: str
    attempt_id: str
    conversation_id: str


class TransformRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trace_headers: dict[str, str] = Field(default_factory=dict)
    dataset: str
    ask: str = Field(description="what to look at, in the person's words")
    context: str = Field(description="what is known about the data, as text with addresses; after a run, what the run found too")
    columns: dict[str, str] = Field(description="key -> the file's own column name, for the columns in the table sent")
    csv: Path = Field(description="the table the sandbox reads: the projection before the freeze, the file after")
    out_dir: Path = Field(description="the only place the agent writes: the artifact's folder")
    outcome: str | None = Field(default=None, description="the outcome column, by the file's own name")
    treatment: str | None = Field(default=None, description="the column that records who got the change, by the file's own name")
    dropped: Dropped | None = Field(default=None, description="the outcome held back from the table before a frozen run")
    frozen: bool = Field(default=False, description="a design has run: the outcome may be seen beside the treatment")
    design: int | None = Field(default=None, description="the design run the ask is about; None before any run")
    memory_version: int = 0
    data_version: int = 0
    budget: int = Field(default=0, description="scripts the agent may run; 0 takes the budget checks.yaml declares")
    product: Literal["table"] = "table"
    rows: int = Field(default=0, description="for a table: how many rows the table sent has, so a row added can be told from one kept")
    keep: list[str] = Field(default_factory=list, description="for a table: columns the new table must carry, by the file's own names (the pair)")

    def dropped_name(self) -> str | None:
        if self.dropped == "outcome":
            return self.outcome
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
    status: Literal["completed", "needs_input"] = "completed"
    method: str = Field(default="", description="What was calculated and why it answers the question")
    findings: list[str] = Field(
        default_factory=list,
        description="Every observation MUST contain exact bracket citations to selected facts, e.g. 'The count is 20 [count]'. Use separate brackets [first] [second], never [first, second]. [figure] and [table] are valid only when those products are selected.",
    )
    limitations: list[str] = Field(default_factory=list)
    needs: list[str] = Field(default_factory=list, description="Specific missing data or clarification for the chatbot")
    tags: list[str] = Field(default_factory=list)


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
    sandbox: str = "not_run"
