"""Read-only profile contracts. Suggestions are never approved conversions."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Status = Literal["complete", "partial", "failed"]
Dtype = Literal["integer", "decimal", "boolean", "date", "datetime", "text", "mixed", "unknown"]


class Issue(BaseModel):
    id: str
    check: str
    code: str
    reason: str
    column_positions: list[int] = Field(default_factory=list)
    affected_count: int | None = None
    examples: list[str] = Field(default_factory=list)
    examples_truncated: bool = False


class Check(BaseModel):
    name: str
    status: Literal["computed", "unavailable", "failed"]
    reason: str | None = None


class Frequency(BaseModel):
    value: str
    count: int
    truncated: bool = False


class NumericSummary(BaseModel):
    min: float
    max: float
    mean: float
    p01: float
    p25: float
    p50: float
    p75: float
    p99: float


class ColumnResult(BaseModel):
    position: int
    name: str
    key: str
    status: Status
    empty_count: int | None = None
    whitespace_only_count: int | None = None
    distinct_count: int | None = None
    top_values: list[Frequency] = Field(default_factory=list)
    suspected_sentinels: list[Frequency] = Field(default_factory=list)
    sentinel_values_truncated: bool = False
    suggested_dtype: Dtype = "unknown"
    numeric_parse_count: int | None = None
    numeric_exception_count: int | None = None
    numeric: NumericSummary | None = None
    checks: list[Check] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)


class DatasetResult(BaseModel):
    rows: int | None = None
    columns: int | None = None
    duplicate_rows: int | None = None
    candidate_keys: list[list[int]] = Field(default_factory=list)
    key_search_limit: str = (
        "All single columns; pairs among at most 12 non-single keys by descending cardinality, then source position. No larger combinations."
    )
    checks: list[Check] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)


class ComputedProfile(BaseModel):
    status: Status
    dataset: DatasetResult
    columns: list[ColumnResult] = Field(default_factory=list)


class ColumnSummary(BaseModel):
    column_id: str
    position: int
    name: str
    key: str
    status: Status
    suggested_dtype: Dtype
    issue_count: int


class SavedProfile(BaseModel):
    profile_id: str
    dataset_id: str
    version_id: str
    source_checksum: str
    profiler_version: str
    settings_fingerprint: str
    created_at: datetime
    updated_at: datetime
    status: Status
    dataset: DatasetResult
    columns: list[ColumnSummary]


class ProfileView(BaseModel):
    status: Literal["not_profiled", "complete", "partial", "failed"]
    profile: SavedProfile | None = None


class SavedColumn(BaseModel):
    profile_id: str
    version_id: str
    column_id: str
    source_checksum: str
    result: ColumnResult
