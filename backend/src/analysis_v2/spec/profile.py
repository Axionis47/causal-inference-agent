"""Analysis-side dataset profile summary (S2).

Richer than the input slice's gate profile: adds quality findings the
design and EDA stages key off (duplicates, constants, id-like columns,
high-missingness flags). Heavy detail lives in artifacts; this summary is
the typed slice other agents read.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ColumnProfile(BaseModel):
    name: str
    dtype: str  # pandas dtype string
    semantic_type: str  # binary | ordinal | numeric | datetime | categorical | text | id_like
    missing_count: int = Field(ge=0)
    missing_fraction: float = Field(ge=0.0, le=1.0)
    n_unique: int = Field(ge=0)
    is_constant: bool = False
    sample_values: list[str] = Field(default_factory=list, max_length=5)


class ProfileSummary(BaseModel):
    n_rows: int = Field(ge=0)
    n_columns: int = Field(ge=0)
    columns: list[ColumnProfile] = Field(default_factory=list)
    duplicate_row_count: int = Field(default=0, ge=0)
    constant_columns: list[str] = Field(default_factory=list)
    id_like_columns: list[str] = Field(default_factory=list)
    high_missingness_columns: list[str] = Field(default_factory=list)
    likely_time_columns: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def column(self, name: str) -> ColumnProfile | None:
        for c in self.columns:
            if c.name == name:
                return c
        return None

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]
