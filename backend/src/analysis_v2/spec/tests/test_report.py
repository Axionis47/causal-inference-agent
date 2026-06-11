"""Notebook verification: the download gate's invariants."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.analysis_v2.spec import (
    NotebookBuildResult,
    NotebookStatus,
    NotebookVerificationResult,
)


def test_verified_running_requires_clean_execution_and_the_artifact():
    with pytest.raises(ValidationError):
        NotebookVerificationResult(
            notebook_status=NotebookStatus.VERIFIED_RUNNING,
            executed_all_cells=False,
            attempts=1,
        )
    with pytest.raises(ValidationError):
        NotebookVerificationResult(
            notebook_status=NotebookStatus.VERIFIED_RUNNING,
            executed_all_cells=True,
            errors=["NameError in cell 7"],
            attempts=2,
        )
    with pytest.raises(ValidationError):
        NotebookVerificationResult(
            notebook_status=NotebookStatus.VERIFIED_RUNNING,
            executed_all_cells=True,
            attempts=1,
        )
    ok = NotebookVerificationResult(
        notebook_status=NotebookStatus.VERIFIED_RUNNING,
        executed_all_cells=True,
        attempts=2,
        repairs=["fixed dataset path to the job analysis dir"],
        verified_notebook_artifact_id="notebook/verified",
    )
    assert ok.attempts == 2


def test_failed_verification_must_carry_the_errors():
    with pytest.raises(ValidationError):
        NotebookVerificationResult(
            notebook_status=NotebookStatus.FAILED,
            executed_all_cells=False,
            attempts=3,
        )
    ok = NotebookVerificationResult(
        notebook_status=NotebookStatus.FAILED,
        executed_all_cells=False,
        errors=["ModuleNotFoundError: econml"],
        attempts=3,
    )
    assert ok.errors


def test_attempts_are_capped_at_three():
    with pytest.raises(ValidationError):
        NotebookVerificationResult(
            notebook_status=NotebookStatus.FAILED,
            executed_all_cells=False,
            errors=["still failing"],
            attempts=4,
        )


def test_notebook_build_result_requires_sections_and_artifacts():
    with pytest.raises(ValidationError):
        NotebookBuildResult(
            notebook_artifact_id="notebook/built",
            report_artifact_id="report/final",
            sections=[],
        )
    ok = NotebookBuildResult(
        notebook_artifact_id="notebook/built",
        report_artifact_id="report/final",
        sections=["load", "profile", "question", "spec"],
        referenced_artifact_ids=["eda/outcome_hist"],
    )
    assert "profile" in ok.sections
