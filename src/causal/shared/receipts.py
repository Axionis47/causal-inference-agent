"""Execution receipts and the tri-agreement mutation gate (PRD-003 §17.6, §24; D-057)."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Final, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from causal.shared.contracts import ArtifactRef, Identity, Sha256Hex, UtcTimestamp

__all__ = [
    "OUTPUT_ARTIFACT_MISMATCH", "OUTPUT_HASH_MISMATCH", "POSTCONDITION_FAILED",
    "RECEIPT_NOT_TERMINAL", "ROW_SET_NOT_INVARIANT", "ExecutionReceiptV1", "FrameShapeV1",
    "OutputArtifactLike", "PostconditionLike", "ReceiptStatus", "tri_agreement",
]

RECEIPT_NOT_TERMINAL: Final = "receipt_not_terminal"
OUTPUT_ARTIFACT_MISMATCH: Final = "output_artifact_mismatch"
OUTPUT_HASH_MISMATCH: Final = "output_hash_mismatch"
POSTCONDITION_FAILED: Final = "postcondition_failed"
ROW_SET_NOT_INVARIANT: Final = "row_set_not_invariant"
# The one postcondition status that clears the gate (PRD-003 §15).
POSTCONDITION_PASS: Final = "pass"

_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True)
_NonNegInt = Annotated[int, Field(ge=0)]


class ReceiptStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class FrameShapeV1(BaseModel):
    """One frame's shape, before or after an operation."""

    model_config = _MODEL_CONFIG

    row_count: _NonNegInt
    column_count: _NonNegInt


class ExecutionReceiptV1(BaseModel):
    """The immutable proof one registered operation ran (PRD-003 §17.6)."""

    model_config = _MODEL_CONFIG

    stage_run_id: Identity
    plan_artifact_id: Identity
    plan_item_id: Identity
    operation_id: Identity
    operation_version: Identity
    implementation_version: Identity
    input_ref: ArtifactRef
    output_ref: ArtifactRef
    parameters_hash: Sha256Hex
    shape_before: FrameShapeV1
    shape_after: FrameShapeV1
    # Recorded, not asserted here: `tri_agreement` is the gate that must reject a drift.
    row_set_hash_before: Sha256Hex
    row_set_hash_after: Sha256Hex
    examined_count: _NonNegInt
    changed_count: _NonNegInt
    derived_count: _NonNegInt
    imputed_count: _NonNegInt
    warning_codes: tuple[Identity, ...]
    error_codes: tuple[Identity, ...]
    attempt_id: Identity
    idempotency_key: Identity
    status: ReceiptStatus
    started_at_utc: UtcTimestamp
    finished_at_utc: UtcTimestamp

    @model_validator(mode="after")
    def _terminal_status_matches_errors_and_clock(self) -> Self:
        if self.finished_at_utc < self.started_at_utc:
            raise ValueError("finished_at_utc cannot precede started_at_utc")
        if bool(self.error_codes) is (self.status is ReceiptStatus.SUCCEEDED):
            raise ValueError("error_codes is non-empty if and only if status is failed")
        return self

    def canonical_payload(self) -> dict[str, Any]:
        """Canonical-ready dict; `content_hash(self.canonical_payload())` is replay-stable."""
        return self.model_dump(mode="json")


class OutputArtifactLike(Protocol):
    """Any reopened output artifact: an envelope, a ref, or a PRD-004 equivalent."""

    @property
    def artifact_id(self) -> str: ...

    @property
    def content_hash(self) -> str: ...


class PostconditionLike(Protocol):
    """Any diagnostic-shaped postcondition report (`PreparationDiagnosticV1` in PRD-003)."""

    @property
    def status(self) -> str: ...

    @property
    def row_set_hash(self) -> str | None: ...


def tri_agreement(
    receipt: ExecutionReceiptV1,
    output_envelope: OutputArtifactLike,
    postcondition: PostconditionLike,
) -> tuple[str, ...]:
    """Stable error codes for the §17.6 gate; empty means receipt, output, and check agree."""
    codes: list[str] = []
    if receipt.status is not ReceiptStatus.SUCCEEDED:
        codes.append(RECEIPT_NOT_TERMINAL)
    if receipt.output_ref.artifact_id != output_envelope.artifact_id:
        codes.append(OUTPUT_ARTIFACT_MISMATCH)
    elif receipt.output_ref.content_hash != output_envelope.content_hash:
        codes.append(OUTPUT_HASH_MISMATCH)
    if postcondition.status != POSTCONDITION_PASS:
        codes.append(POSTCONDITION_FAILED)
    observed = postcondition.row_set_hash
    if receipt.row_set_hash_before != receipt.row_set_hash_after or (
        observed is not None and observed != receipt.row_set_hash_after
    ):
        codes.append(ROW_SET_NOT_INVARIANT)
    return tuple(codes)
