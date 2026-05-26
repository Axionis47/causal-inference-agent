"""Analysis job entry contract.

JobInput is what the orchestrator receives once download is complete.
It carries the download_id (the seam to the download subsystem),
the user's variable picks and optional context, the orchestrator
mode, and the budget. Nothing else is needed to start a run.

UserContext fields are optional. None / empty means the user did not
provide that hint; the agents fall back to inference.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

BudgetTier = Literal["quick", "standard", "deep"]
OrchestratorMode = Literal["standard", "react"]


class BudgetSpec(BaseModel):
    """Per-job cost caps. Tripped budgets never crash the job; the
    affected agent's slot is marked partial and the orchestrator
    finalises with what it has.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_agent_loops: int = Field(default=20, ge=1, le=200)
    max_tool_calls_per_agent: int = Field(default=30, ge=1, le=500)
    max_total_tool_calls: int = Field(default=200, ge=1, le=5000)
    max_llm_tokens: int = Field(default=500_000, ge=1000)
    wall_clock_seconds: int = Field(default=3000, ge=30, le=14_400)


_TIER_PRESETS: dict[str, BudgetSpec] = {
    "quick": BudgetSpec(
        max_agent_loops=8,
        max_tool_calls_per_agent=12,
        max_total_tool_calls=80,
        max_llm_tokens=150_000,
        wall_clock_seconds=600,
    ),
    "standard": BudgetSpec(),
    "deep": BudgetSpec(
        max_agent_loops=40,
        max_tool_calls_per_agent=60,
        max_total_tool_calls=500,
        max_llm_tokens=1_500_000,
        wall_clock_seconds=5400,
    ),
}


def budget_for_tier(tier: BudgetTier) -> BudgetSpec:
    return _TIER_PRESETS[tier]


class UserContext(BaseModel):
    """Optional knowledge the user supplied at submit time.

    Every field is optional. Empty values mean the user gave no hint
    and the agents will infer or ignore.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    notes: str | None = None
    hypothesis: str | None = None
    known_confounders: list[str] = Field(default_factory=list)
    immutable_vars: list[str] = Field(default_factory=list)
    skip_stages: list[str] = Field(default_factory=list)


class JobInput(BaseModel):
    """The entry contract for an analysis run.

    Built at submit time from the API request plus the DownloadRecord
    referenced by download_id. The orchestrator does not look at any
    other state to start a run.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: str = Field(min_length=1)
    download_id: str = Field(min_length=1)
    treatment_variable: str | None = None
    outcome_variable: str | None = None
    orchestrator_mode: OrchestratorMode = "standard"
    budget: BudgetSpec = Field(default_factory=BudgetSpec)
    user_context: UserContext = Field(default_factory=UserContext)
