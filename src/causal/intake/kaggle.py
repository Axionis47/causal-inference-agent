"""Kaggle capture layer behind a client Protocol (PRD-001 §5; T-008; D-034).

The Protocol carries no credential parameter anywhere: a live adapter
authenticates itself from the runtime secret source at construction time.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final, Protocol

__all__ = ["CaptureResult", "KaggleClientProtocol", "KaggleError", "capture"]

FETCH_FAILED: Final = "fetch_failed"
VERSION_UNRESOLVED: Final = "version_unresolved"


class KaggleError(ValueError):
    """A provider operation failed. `code` is a stable contract value."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


class KaggleClientProtocol(Protocol):
    """The provider operations intake uses (D-034). Responses are raw JSON dicts."""

    def dataset_status(self, owner: str, slug: str) -> dict[str, object]: ...

    def dataset_metadata(self, owner: str, slug: str) -> dict[str, object]: ...

    def dataset_files(self, owner: str, slug: str) -> dict[str, object]: ...

    def download_archive(self, owner: str, slug: str, version: str) -> bytes: ...


@dataclass(frozen=True)
class CaptureResult:
    payload: dict[str, object]
    archive_bytes: bytes


def _resolve_version(
    status_response: dict[str, object], metadata_response: dict[str, object]
) -> str:
    for response in (status_response, metadata_response):
        value = response.get("currentVersionNumber")
        if value is not None and str(value).strip():
            return str(value)
    raise KaggleError("provider returned no resolvable version", VERSION_UNRESOLVED)


def capture(client: KaggleClientProtocol, kaggle_ref: str) -> CaptureResult:
    """Freeze one provider snapshot into the kaggle-capture.v1 payload (§5.1–§5.4)."""
    owner, slug = kaggle_ref.split("/")
    try:
        status_response = client.dataset_status(owner, slug)
        metadata_response = client.dataset_metadata(owner, slug)
        files_response = client.dataset_files(owner, slug)
    except KaggleError:
        raise
    except Exception as error:
        raise KaggleError(f"provider fetch failed: {error}", FETCH_FAILED) from error
    version = _resolve_version(status_response, metadata_response)
    try:
        archive_bytes = client.download_archive(owner, slug, version)
    except Exception as error:
        raise KaggleError(f"archive download failed: {error}", FETCH_FAILED) from error
    payload: dict[str, object] = {
        "schema_version": "kaggle-capture.v1", "kaggle_ref": kaggle_ref,
        "dataset": {
            "provider": "kaggle", "owner": owner, "slug": slug, "version": version,
            "status": str(status_response.get("status", "unknown")),
            "dataset_id": f"kaggle:{owner}/{slug}@{version}"},
        "status_response": status_response,
        "metadata_response": metadata_response,
        "files_response": files_response,
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
    }
    return CaptureResult(payload=payload, archive_bytes=archive_bytes)
