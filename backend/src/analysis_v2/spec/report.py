"""Report and notebook contracts (S10/S11).

The notebook is mandatory and only delivered after it executes top to
bottom from a clean state. Verification gets max 3 repair attempts and may
only fix reproducibility, never conclusions or estimates.
"""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class NotebookBuildResult(BaseModel):
    notebook_artifact_id: str = Field(min_length=1)
    report_artifact_id: str = Field(min_length=1)
    dashboard_artifact_id: str | None = None
    sections: list[str] = Field(min_length=1)
    referenced_artifact_ids: list[str] = Field(default_factory=list)


class NotebookStatus(StrEnum):
    VERIFIED_RUNNING = "verified_running"
    FAILED = "failed"


class NotebookVerificationResult(BaseModel):
    notebook_status: NotebookStatus
    executed_all_cells: bool
    errors: list[str] = Field(default_factory=list)
    attempts: int = Field(ge=1, le=3)
    repairs: list[str] = Field(default_factory=list)
    verified_notebook_artifact_id: str | None = None
    execution_log_artifact_id: str | None = None
    html_preview_artifact_id: str | None = None

    @model_validator(mode="after")
    def _coherent(self) -> "NotebookVerificationResult":
        if self.notebook_status == NotebookStatus.VERIFIED_RUNNING:
            if not self.executed_all_cells or self.errors:
                raise ValueError("verified_running requires all cells executed, no errors")
            if self.verified_notebook_artifact_id is None:
                raise ValueError("verified_running requires the verified notebook artifact")
        if self.notebook_status == NotebookStatus.FAILED and not self.errors:
            raise ValueError("failed verification must carry the errors")
        return self
