"""The dataset under review: the downloaded file bundle plus the two context
channels (the source's own metadata, and the analyst's prose).
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class FileEntry(BaseModel):
    name: str
    size_bytes: int
    format: str  # csv / parquet / json / ...
    used: bool = False  # the single file the analysis loads


class DatasetInfo(BaseModel):
    url: str
    name: str | None = None
    local_path: str | None = None
    n_samples: int | None = None
    n_features: int | None = None
    files: list[FileEntry] = Field(default_factory=list)

    # Source context channel: Kaggle's own metadata. Informative, not
    # authoritative, and often empty (see docs/input-slice contract).
    kaggle_description: str | None = None
    kaggle_subtitle: str | None = None
    kaggle_column_descriptions: dict[str, str] = Field(default_factory=dict)
    kaggle_tags: list[str] = Field(default_factory=list)
    kaggle_keywords: list[str] = Field(default_factory=list)
    kaggle_domain: str | None = None
    metadata_quality: str = "unknown"  # high / medium / low / unknown

    # Analyst context channel: the prose the user typed. Authoritative intent,
    # may be empty. Kept distinct from the source channel above, never merged.
    user_provided_context: str | None = None
