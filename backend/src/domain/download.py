"""Domain types for the download subsystem.

These are the contracts the rest of the system reads. Analysis stages
consume a DownloadRecord; they do not call Kaggle. The download module
is the only owner of Kaggle credentials and the only writer of these
types.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class DownloadStatus(str, Enum):
    QUEUED = "queued"
    AUTHENTICATING = "authenticating"
    FETCHING_METADATA = "fetching_metadata"
    DOWNLOADING_FILES = "downloading_files"
    UNZIPPING = "unzipping"
    COMPLETE = "complete"
    FAILED = "failed"


class FileEntry(BaseModel):
    """One file from the downloaded archive, after unzip."""

    name: str
    relative_path: str
    size_bytes: int
    format: str  # csv, parquet, json, txt, tsv, xlsx, xls, other
    sha256: str


class KaggleMetadata(BaseModel):
    """Full Kaggle metadata blob, normalized into typed fields."""

    owner: str
    slug: str
    title: str | None = None
    subtitle: str | None = None
    description: str | None = None
    column_descriptions: dict[str, str] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    license_name: str | None = None
    version: str | None = None
    last_updated: datetime | None = None
    total_bytes: int | None = None
    url: str


class KaggleProfile(BaseModel):
    """A profile owning Kaggle credentials. profile_id is 'default' today;
    multi-user lands later with the auth layer.
    """

    model_config = ConfigDict(json_schema_extra={"never_returns_key": True})

    profile_id: str
    kaggle_username: str
    has_key: bool
    validated_at: datetime | None = None
    last_used_at: datetime | None = None


class DownloadRequest(BaseModel):
    """Kicks off a download for a given Kaggle URL on behalf of a profile."""

    profile_id: str = "default"
    kaggle_url: str


class DownloadRecord(BaseModel):
    """The persisted state of one download. Analysis reads from this."""

    download_id: str
    profile_id: str
    kaggle_url: str
    dataset_ref: str  # "owner/slug"
    status: DownloadStatus
    storage_path: str | None = None  # data/datasets/{profile_id}/{owner}__{slug}/{version}/
    files: list[FileEntry] = Field(default_factory=list)
    metadata: KaggleMetadata | None = None
    error: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


class DownloadEvent(BaseModel):
    """One progress event streamed to clients via SSE."""

    download_id: str
    event_type: str  # e.g. "auth.started", "metadata.ready", "file.downloaded"
    timestamp: datetime
    data: dict = Field(default_factory=dict)
