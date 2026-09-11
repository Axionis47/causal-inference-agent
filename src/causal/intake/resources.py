"""Archive inventory and deterministic resource processing, without persistence."""

from __future__ import annotations

import hashlib
import io
import posixpath
import zipfile
from dataclasses import dataclass, field
from typing import Final

from causal.intake.archive import EXTRACTABLE_CLASSES, ArchiveAdmission, ArchiveSafety
from causal.intake.profiler import profile_table

PROFILER_VERSION: Final = "profiler.v1"


@dataclass(frozen=True)
class SourceResource:
    name: str
    classification: str
    reason: str | None
    sha256: str
    byte_size: int
    media_type: str
    data: bytes | None = field(repr=False)


@dataclass(frozen=True)
class ResourceInventory:
    entries: tuple[SourceResource, ...]

    def manifest_payload(self, dataset_id: str) -> dict[str, object]:
        return {
            "schema_version": "source-manifest.v1", "dataset_id": dataset_id,
            "resources": [
                {"logical_name": entry.name, "classification": entry.classification,
                 "sha256": entry.sha256, "byte_size": entry.byte_size,
                 "media_type": entry.media_type, "reason": entry.reason}
                for entry in self.entries],
        }


@dataclass(frozen=True)
class ResourceResult:
    resource: SourceResource
    status: str
    reason: str | None
    profile: dict[str, object] | None = None
    document: str | None = None


def inventory_archive(
    archive_bytes: bytes, admission: ArchiveAdmission, safety: ArchiveSafety,
) -> ResourceInventory:
    """Keep every discovered member; refused archives never yield extracted bytes."""
    if not admission.decisions:
        return ResourceInventory(())
    errors: dict[str, str] = {}
    extracted = safety.extract_admitted(
        archive_bytes, admission,
        lambda name, error: errors.__setitem__(
            name, f"archive member unreadable: {type(error).__name__}")) if admission.safe else {}
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        sizes = {info.filename: info.file_size for info in archive.infolist()}
    archive_sha = hashlib.sha256(archive_bytes).hexdigest()
    entries: list[SourceResource] = []
    for decision in admission.decisions:
        data = extracted.get(decision.name)
        entries.append(SourceResource(
            decision.name, decision.classification, errors.get(decision.name, decision.reason),
            hashlib.sha256(data).hexdigest() if data is not None else archive_sha,
            len(data) if data is not None else sizes.get(decision.name, 0),
            posixpath.splitext(decision.name)[1].lstrip(".").lower(), data))
    return ResourceInventory(tuple(entries))


def process_resource(resource: SourceResource) -> ResourceResult:
    """Return one terminal parse status and any measured profile or document text."""
    kind, data = resource.classification, resource.data
    if kind not in EXTRACTABLE_CLASSES:
        status = kind if kind in ("unreadable", "unsafe") else "excluded"
        return ResourceResult(resource, status, resource.reason)
    if data is None:
        return ResourceResult(resource, "failed", resource.reason or "resource bytes unavailable")
    if kind == "table":
        try:
            profile = profile_table(data, resource.media_type, PROFILER_VERSION)
        except Exception as error:  # noqa: BLE001 -- any parse failure is terminal
            return ResourceResult(resource, "failed", f"profiling failed: {error}")
        return ResourceResult(resource, "parsed", None, profile=profile)
    if resource.media_type in ("yaml", "yml"):
        return ResourceResult(resource, "excluded", "yaml parsing deferred (PRD-001 §5.5)")
    try:
        document = data.decode("utf-8")
    except UnicodeDecodeError:
        return ResourceResult(resource, "failed", "not valid utf-8")
    return ResourceResult(resource, "parsed", None, document=document)
