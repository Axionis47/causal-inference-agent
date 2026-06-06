"""Job-scoped dataset manifest: the record of what one job persisted.

Bytes live on disk at {local_storage_path}/{job_id}/raw/; this manifest is
the typed record of that bundle (one entry per file) plus the full, lossless
Kaggle metadata dict. Downstream stages read the manifest instead of
re-reading Kaggle or guessing at paths.

Field conventions mirror src/domain/download.py (FileEntry, DownloadRecord),
which are scoped to the sealed download module (profile_id / download_id).
The two converge when the download seam is wired; until then this is the
job-scoped record the pipeline writes.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ManifestFile(BaseModel):
    """One file from the downloaded bundle, after unzip."""

    name: str
    relative_path: str  # relative to the job dir, e.g. "raw/controls.csv"
    size_bytes: int
    format: str  # csv, parquet, json, txt, tsv, xlsx, xls, other
    sha256: str
    n_rows: int | None = None
    n_columns: int | None = None
    columns: list[str] = Field(default_factory=list)
    used: bool = False  # the file loaded into the working dataframe


class DatasetManifest(BaseModel):
    """The persisted record of one job's dataset. Downstream reads this."""

    job_id: str
    kaggle_url: str
    dataset_ref: str | None = None  # "owner/slug"
    raw_dir: str  # absolute path to the raw/ bundle directory
    files: list[ManifestFile] = Field(default_factory=list)
    winner: str | None = None  # ManifestFile.name loaded for analysis
    kaggle_metadata: dict = Field(default_factory=dict)  # full extractor dict
