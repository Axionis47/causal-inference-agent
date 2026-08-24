"""Shared cross-stage contract models (SYSTEM-CONTRACT §2, §3; decisions D-004..D-007)."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    StringConstraints,
)

__all__ = [
    "ArtifactEnvelopeV1",
    "ArtifactRef",
    "HandoffManifestV1",
    "SensitivityClass",
]


class SensitivityClass(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    SECRET_REFERENCE = "secret_reference"


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(None):
        raise ValueError("timestamp must be timezone-aware UTC")
    return value


def _serialize_utc(value: datetime) -> str:
    # RFC 3339 UTC with exactly six fractional digits and a Z suffix (D-004).
    return value.strftime("%Y-%m-%dT%H:%M:%S") + f".{value.microsecond:06d}Z"


def _reject_signed_url(value: str) -> str:
    if "?" in value or "&" in value or "X-Amz-" in value:
        raise ValueError("payload_locator must not be a signed URL")
    return value


Identity = Annotated[str, StringConstraints(min_length=1, max_length=200)]
Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
UtcTimestamp = Annotated[
    datetime,
    AfterValidator(_require_utc),
    PlainSerializer(_serialize_utc, return_type=str, when_used="json"),
]
PayloadLocator = Annotated[
    str,
    StringConstraints(min_length=1, max_length=1024),
    AfterValidator(_reject_signed_url),
]

_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True)


class ArtifactRef(BaseModel):
    """One ordered (artifact ID, content hash) lineage pair."""

    model_config = _MODEL_CONFIG

    artifact_id: Identity
    content_hash: Sha256Hex


class ArtifactEnvelopeV1(BaseModel):
    """The immutable envelope every stored artifact uses (SYSTEM-CONTRACT §3)."""

    model_config = _MODEL_CONFIG

    artifact_id: Identity
    artifact_type: Identity
    schema_version: Identity
    content_hash: Sha256Hex
    analysis_id: Identity
    stage_run_id: Identity
    producer_component: Identity
    producer_version: Identity
    parent_artifacts: tuple[ArtifactRef, ...]
    sensitivity_class: SensitivityClass
    created_at_utc: UtcTimestamp
    payload_locator: PayloadLocator

    def canonical_payload(self) -> dict[str, Any]:
        """Canonical-ready dict; `content_hash(self.canonical_payload())` is replay-stable."""
        return self.model_dump(mode="json")


class HandoffManifestV1(BaseModel):
    """One cross-stage transfer record (SYSTEM-CONTRACT §3)."""

    model_config = _MODEL_CONFIG

    handoff_id: Identity
    schema_version: Identity
    analysis_id: Identity
    producing_stage_run_id: Identity
    receiving_stage_run_id: Identity
    entries: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]
    originating_outcome: Identity
    approval_ids: tuple[Identity, ...]
    registry_version: Identity
    compatibility_version: Identity
    receiver_validation_result: Identity | None
    receiver_error_codes: tuple[Identity, ...]
    created_at_utc: UtcTimestamp
    accepted_at_utc: UtcTimestamp | None

    def canonical_payload(self) -> dict[str, Any]:
        """Canonical-ready dict; `content_hash(self.canonical_payload())` is replay-stable."""
        return self.model_dump(mode="json")
