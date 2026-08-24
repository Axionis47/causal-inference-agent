"""Archive safety admission and resource classification (PRD-001 §5.5; D-026)."""

from __future__ import annotations

import io
import posixpath
import stat
import zipfile
from dataclasses import dataclass
from typing import Final

__all__ = ["ArchiveAdmission", "ArchiveSafety", "ResourceDecision"]

TABLE_EXTENSIONS: Final = frozenset({".csv", ".tsv", ".parquet"})
DOCUMENT_EXTENSIONS: Final = frozenset({".txt", ".md", ".markdown"})
METADATA_EXTENSIONS: Final = frozenset({".json", ".yaml", ".yml"})
UNREADABLE_EXTENSIONS: Final = frozenset(
    {".xls", ".xlsx", ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}
)
WITHHELD_EXTENSIONS: Final = frozenset({".ipynb", ".py", ".r", ".rmd", ".jl", ".js", ".sh"})
UNSAFE_EXTENSIONS: Final = frozenset(
    {".exe", ".dll", ".so", ".dylib", ".bat", ".cmd", ".com", ".scr", ".pkl", ".pickle",
     ".joblib", ".xlsm", ".xlsb", ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar"}
)

EXTRACTABLE_CLASSES: Final = frozenset({"table", "document", "metadata"})


@dataclass(frozen=True)
class ResourceDecision:
    name: str
    classification: str  # table|document|metadata|unreadable|withheld|unsafe
    reason: str | None


@dataclass(frozen=True)
class ArchiveAdmission:
    safe: bool
    decisions: tuple[ResourceDecision, ...]
    refusal_reason: str | None


def _classify_extension(name: str) -> tuple[str, str | None]:
    extension = posixpath.splitext(name)[1].lower()
    if extension in UNSAFE_EXTENSIONS:
        return "unsafe", f"unsafe extension {extension}"
    if extension in TABLE_EXTENSIONS:
        return "table", None
    if extension in DOCUMENT_EXTENSIONS:
        return "document", None
    if extension in METADATA_EXTENSIONS:
        return "metadata", None
    if extension in WITHHELD_EXTENSIONS:
        return "withheld", "notebooks and scripts are withheld before estimation"
    if extension in UNREADABLE_EXTENSIONS:
        return "unreadable", f"no parser for {extension} in v0"
    return "unreadable", f"unrecognized extension {extension or '(none)'}"


class ArchiveSafety:
    """Admits one zip archive; unsafe bytes never extract (PRD-001 §12 refusal)."""

    def __init__(
        self,
        max_entries: int = 10_000,
        max_file_bytes: int = 512 * 1024 * 1024,
        max_total_bytes: int = 2 * 1024 * 1024 * 1024,
        max_ratio: int = 200,
        ratio_floor_bytes: int = 1024 * 1024,
    ) -> None:
        self._max_entries = max_entries
        self._max_file_bytes = max_file_bytes
        self._max_total_bytes = max_total_bytes
        self._max_ratio = max_ratio
        self._ratio_floor = ratio_floor_bytes

    def _entry_safety_reason(self, info: zipfile.ZipInfo) -> str | None:
        name = info.filename
        if name.startswith("/") or posixpath.isabs(name):
            return "absolute path"
        if ".." in name.split("/"):
            return "path traversal"
        if stat.S_ISLNK(info.external_attr >> 16):
            return "symbolic link"
        if info.flag_bits & 0x1:
            return "encrypted entry"
        if info.file_size > self._max_file_bytes:
            return f"file exceeds {self._max_file_bytes} bytes"
        compressed = max(info.compress_size, 1)
        if info.compress_size >= self._ratio_floor and info.file_size // compressed > self._max_ratio:
            return "compression ratio exceeds bomb limit"
        return None

    def admit(self, archive_bytes: bytes) -> ArchiveAdmission:
        try:
            archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
            infos = [info for info in archive.infolist() if not info.is_dir()]
        except zipfile.BadZipFile:
            return ArchiveAdmission(False, (), "corrupt or unsupported archive")
        if len(infos) > self._max_entries:
            return ArchiveAdmission(False, (), f"more than {self._max_entries} entries")
        if sum(info.file_size for info in infos) > self._max_total_bytes:
            return ArchiveAdmission(False, (), "total uncompressed size exceeds limit")
        decisions: list[ResourceDecision] = []
        for info in infos:
            safety_reason = self._entry_safety_reason(info)
            if safety_reason is not None:
                decisions.append(ResourceDecision(info.filename, "unsafe", safety_reason))
                continue
            classification, reason = _classify_extension(info.filename)
            decisions.append(ResourceDecision(info.filename, classification, reason))
        unsafe = [decision for decision in decisions if decision.classification == "unsafe"]
        refusal = f"{len(unsafe)} unsafe entries" if unsafe else None
        return ArchiveAdmission(not unsafe, tuple(decisions), refusal)

    def extract_admitted(
        self, archive_bytes: bytes, admission: ArchiveAdmission
    ) -> dict[str, bytes]:
        """Bytes for extractable classes only; never called on an unsafe archive."""
        if not admission.safe:
            raise ValueError("refusing to extract from an archive that failed admission")
        archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
        extractable = {
            decision.name
            for decision in admission.decisions
            if decision.classification in EXTRACTABLE_CLASSES
        }
        return {name: archive.read(name) for name in sorted(extractable)}
