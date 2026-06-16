"""Report and notebook contracts (S10/S11).

The notebook is mandatory and only delivered after it executes top to
bottom from a clean state. Verification gets max 3 repair attempts and may
only fix reproducibility, never conclusions or estimates.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class ReportToolCall(BaseModel):
    """One report-building step, recorded for replay. In the deterministic
    path it is a section that rendered; in the agentic path it is a model
    tool call with its arguments. `kind` discriminates the cell kind
    (read / write_section / render_table / render_figure / finish);
    `status` carries the outcome, or 'rejected' when a constraint blocked it.
    The model's prose is frozen into the assembled notebook cells, not here:
    this trace records WHICH sections and figures were chosen and in WHAT
    order, enough to rebuild the same cell sequence."""

    name: str = Field(min_length=1, max_length=120)
    section_id: str = Field(default="", max_length=60)
    kind: str = Field(default="", max_length=30)
    args: dict[str, Any] = Field(default_factory=dict)
    status: str = Field(default="", max_length=20)
    rejected: str | None = Field(default=None, max_length=300)


class ReportToolTrace(BaseModel):
    """The ordered report-building steps behind a run. Freezes the selection
    and ordering so an agentic (model-driven) report stays reproducible even
    though the run that produced it is not deterministic."""

    calls: list[ReportToolCall] = Field(default_factory=list)
    exhausted: bool = False


class NotebookBuildResult(BaseModel):
    notebook_artifact_id: str = Field(min_length=1)
    report_artifact_id: str = Field(min_length=1)
    dashboard_artifact_id: str | None = None
    sections: list[str] = Field(min_length=1)
    referenced_artifact_ids: list[str] = Field(default_factory=list)
    # Transport for the run's report tool trace from execute() to commit();
    # excluded from serialization so the trace has one on-disk home, the
    # run.report_tool_trace slot.
    report_tool_trace: ReportToolTrace | None = Field(default=None, exclude=True)


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
