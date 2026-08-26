"""Execution receipts and the tri-agreement mutation gate (T-015 §1.4; PRD-003 §17.6)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from causal.preparation.contracts import (
    DiagnosticStatus,
    FrameStage,
    PreparationDiagnosticV1,
)
from causal.shared.canonical import content_hash
from causal.shared.contracts import ArtifactRef
from causal.shared.receipts import (
    OUTPUT_ARTIFACT_MISMATCH,
    OUTPUT_HASH_MISMATCH,
    POSTCONDITION_FAILED,
    RECEIPT_NOT_TERMINAL,
    ROW_SET_NOT_INVARIANT,
    ExecutionReceiptV1,
    FrameShapeV1,
    ReceiptStatus,
    tri_agreement,
)

HASH = "a" * 64
OTHER_HASH = "b" * 64
ROW_SET = "c" * 64
DRIFTED_ROW_SET = "d" * 64
NOW = datetime(2026, 8, 26, 12, 0, 0, tzinfo=UTC)
OUTPUT = ArtifactRef(artifact_id="art-out", content_hash=HASH)


def receipt(**overrides: Any) -> ExecutionReceiptV1:
    base: dict[str, Any] = {
        "stage_run_id": "run-1", "plan_artifact_id": "art-plan", "plan_item_id": "pi-1",
        "operation_id": "numeric_median_with_indicator", "operation_version": "v1",
        "implementation_version": "v1",
        "input_ref": ArtifactRef(artifact_id="art-in", content_hash=OTHER_HASH),
        "output_ref": OUTPUT, "parameters_hash": HASH,
        "shape_before": FrameShapeV1(row_count=100, column_count=4),
        "shape_after": FrameShapeV1(row_count=100, column_count=6),
        "row_set_hash_before": ROW_SET, "row_set_hash_after": ROW_SET,
        "examined_count": 100, "changed_count": 7, "derived_count": 1, "imputed_count": 7,
        "warning_codes": (), "error_codes": (), "attempt_id": "att-1",
        "idempotency_key": "idem-1", "status": ReceiptStatus.SUCCEEDED,
        "started_at_utc": NOW, "finished_at_utc": NOW + timedelta(seconds=2),
    }
    return ExecutionReceiptV1(**{**base, **overrides})


def postcondition(**overrides: Any) -> PreparationDiagnosticV1:
    base: dict[str, Any] = {
        "diagnostic_id": "row_set_invariance", "diagnostic_version": "v1",
        "frame_stage": FrameStage.PREPARED, "inputs": (OUTPUT,), "columns_read": (),
        "total_rows": 100, "used_rows": 100, "unused_reason_counts": {},
        "row_set_hash": ROW_SET, "values": {"changed_cells": 7}, "warnings": (),
        "status": DiagnosticStatus.PASS, "implementation_version": "v1",
    }
    return PreparationDiagnosticV1(**{**base, **overrides})


class TestReceipt:
    def test_roundtrip_and_replay_stable_hash(self) -> None:
        built = receipt()
        again = ExecutionReceiptV1.model_validate(built.model_dump())
        assert again == built
        assert content_hash(again.canonical_payload()) == content_hash(built.canonical_payload())

    def test_a_succeeded_receipt_cannot_carry_error_codes(self) -> None:
        with pytest.raises(ValidationError, match="error_codes is non-empty"):
            receipt(error_codes=("operation_failed",))

    def test_a_failed_receipt_must_carry_an_error_code(self) -> None:
        with pytest.raises(ValidationError, match="error_codes is non-empty"):
            receipt(status=ReceiptStatus.FAILED)

    def test_the_clock_cannot_run_backwards(self) -> None:
        with pytest.raises(ValidationError, match="cannot precede started_at_utc"):
            receipt(finished_at_utc=NOW - timedelta(seconds=1))


class TestTriAgreement:
    def test_all_three_agreeing_passes(self) -> None:
        assert tri_agreement(receipt(), OUTPUT, postcondition()) == ()

    def test_a_non_terminal_receipt_is_reported(self) -> None:
        failed = receipt(status=ReceiptStatus.FAILED, error_codes=("operation_failed",))
        assert tri_agreement(failed, OUTPUT, postcondition()) == (RECEIPT_NOT_TERMINAL,)

    def test_a_different_output_artifact_is_reported(self) -> None:
        other = ArtifactRef(artifact_id="art-other", content_hash=HASH)
        assert tri_agreement(receipt(), other, postcondition()) == (OUTPUT_ARTIFACT_MISMATCH,)

    def test_a_reopened_hash_mismatch_is_reported(self) -> None:
        drifted = ArtifactRef(artifact_id="art-out", content_hash=OTHER_HASH)
        assert tri_agreement(receipt(), drifted, postcondition()) == (OUTPUT_HASH_MISMATCH,)

    def test_a_failing_postcondition_is_reported(self) -> None:
        failing = postcondition(status=DiagnosticStatus.FAIL, warnings=("schema_mismatch",))
        assert tri_agreement(receipt(), OUTPUT, failing) == (POSTCONDITION_FAILED,)

    def test_a_receipt_that_moved_the_row_set_is_reported(self) -> None:
        moved = receipt(row_set_hash_after=DRIFTED_ROW_SET)
        assert tri_agreement(moved, OUTPUT, postcondition()) == (ROW_SET_NOT_INVARIANT,)

    def test_a_postcondition_on_a_different_row_set_is_reported(self) -> None:
        elsewhere = postcondition(row_set_hash=DRIFTED_ROW_SET)
        assert tri_agreement(receipt(), OUTPUT, elsewhere) == (ROW_SET_NOT_INVARIANT,)

    def test_a_postcondition_without_a_row_set_is_accepted(self) -> None:
        assert tri_agreement(receipt(), OUTPUT, postcondition(row_set_hash=None)) == ()

    def test_every_failure_is_reported_together(self) -> None:
        broken = receipt(
            status=ReceiptStatus.FAILED, error_codes=("operation_failed",),
            row_set_hash_after=DRIFTED_ROW_SET,
        )
        other = ArtifactRef(artifact_id="art-other", content_hash=OTHER_HASH)
        assert tri_agreement(broken, other, postcondition(status=DiagnosticStatus.WARN)) == (
            RECEIPT_NOT_TERMINAL, OUTPUT_ARTIFACT_MISMATCH, POSTCONDITION_FAILED,
            ROW_SET_NOT_INVARIANT,
        )
