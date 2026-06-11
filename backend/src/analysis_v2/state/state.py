"""The job state for the download / data-review slice.

This is the durable, reloadable checkpoint described in
`docs/input-slice/01-files-and-flow.md`. It carries the dataset, the
deterministic profile, and the human-locked inputs (treatment / outcome / time /
ignored columns), trimmed to what this slice owns. Analysis outputs are not here:
the analysis slice reads this as its start payload and writes its own state after
launch.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from src.domain.approval import ApprovalDecision, HumanApproval
from src.domain.relational import RelationalProfile

from .dataset import DatasetInfo
from .profile import DataProfile
from .status import JobStatus

SCHEMA_VERSION = 1


class StaleParkedState(ValueError):
    """A parked state could not be loaded under the current schema version."""


class AnalysisState(BaseModel):
    """Shared state for the download / data-review slice; the confirmed record
    is the start-analysis payload (`docs/analysis-slice/00-start-analysis-payload.md`).
    """

    schema_version: int = SCHEMA_VERSION

    job_id: str
    dataset_info: DatasetInfo
    download_id: str | None = None
    status: JobStatus = JobStatus.PENDING
    # Recorded so the choice round-trips through persistence; the analysis slice
    # reads it to pick its runner.
    orchestrator_mode: Literal["standard", "react"] = "standard"

    # The causal question the analyst is asking, in plain language. This is the
    # intake's primary input now: the human states what they want to know
    # ("Does job training increase income?"), not which columns are treatment
    # and outcome. Required at submit; may be refined at the data-review gate.
    causal_question: str | None = None

    # Derived-later slots. Treatment and outcome are no longer chosen by the
    # human; the analysis derives them from the question + data after run. Kept
    # nullable so the analysis tail can fill them. `ignored_columns` is the
    # deterministically-proposed, human-confirmed drop set (e.g. "Unnamed: 0").
    treatment_variable: str | None = None
    outcome_variable: str | None = None
    ignored_columns: list[str] = Field(default_factory=list)

    # Raw Kaggle metadata blob, the deterministic profile, and the on-disk frame.
    raw_metadata: dict[str, Any] | None = None
    data_profile: DataProfile | None = None
    dataframe_path: str | None = None

    # Multi-file display only (the single used file is what analysis loads).
    file_profiles: dict[str, DataProfile] = Field(default_factory=dict)
    relational_profile: RelationalProfile | None = None

    # The data-review gate decision.
    human_approval: HumanApproval | None = None

    progress_percentage: int = 0
    # Live progress buffer for the SSE stream (download + profile events).
    sse_events: list[dict] = Field(default_factory=list)
    error_message: str | None = None
    error_agent: str | None = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def push_sse_event(self, event_type: str, data: dict) -> None:
        """Append a live event for the SSE stream; FIFO-capped at 100."""
        self.sse_events.append(
            {
                "event_type": event_type,
                "data": data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        if len(self.sse_events) > 100:
            self.sse_events = self.sse_events[-100:]

    def mark_failed(self, message: str, agent: str | None = None) -> None:
        self.status = JobStatus.FAILED
        self.error_message = message
        self.error_agent = agent
        self.completed_at = datetime.now(timezone.utc)
        self.touch()

    def mark_cancelled(self) -> None:
        self.status = JobStatus.CANCELLED
        self.completed_at = datetime.now(timezone.utc)
        self.touch()

    def is_approved(self) -> bool:
        """True once the human approved the dataset at the data-review gate."""
        return (
            self.human_approval is not None
            and self.human_approval.decision == ApprovalDecision.APPROVED
        )

    @classmethod
    def load_parked(cls, payload: dict) -> "AnalysisState":
        """Deserialize a parked state, refusing an incompatible schema version."""
        stored = payload.get("schema_version", SCHEMA_VERSION)
        if stored != SCHEMA_VERSION:
            raise StaleParkedState(
                f"parked state schema v{stored} is incompatible with the current "
                f"v{SCHEMA_VERSION}; resubmit the job"
            )
        try:
            return cls.model_validate(payload)
        except ValidationError as exc:
            raise StaleParkedState(
                f"parked state could not be loaded under schema v{SCHEMA_VERSION}; "
                "resubmit the job"
            ) from exc
