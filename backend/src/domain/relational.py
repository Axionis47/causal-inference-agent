"""Relational structure of a multi-file dataset bundle.

Written by dataset_inspector after it profiles every candidate file, read at
the data-review gate and in the notebook audit trail. This is a DESCRIPTION of
how the files in a bundle relate, not an assembled table: the pipeline still
analyses exactly one file. It carries only cheap, checkable facts (row/column
counts, candidate keys by full-file uniqueness, and same-schema grouping), so a
wrong value fails loudly rather than silently inflating the analysis. It does
NOT carry a join plan, foreign keys, or cardinality; those are the fragile,
anticonservative-if-wrong parts and are deferred to a later, confirmed stage.

Lives in domain/ rather than the inspector's output.py because state.py holds a
slot of this type and the dataset_inspector package imports AnalysisState, so a
type defined inside that package would close an import cycle.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Coarse on purpose. Distinguishing a star schema from an entity-transaction
# bundle from a panel needs semantic meaning the deterministic pass does not
# have, and a confident wrong guess is worse than none. We classify only what
# falls out cheaply from schema identity and file count.
ShapeHint = Literal["single", "same_schema_shards", "multiple_files", "unrelated"]


class FileRelational(BaseModel):
    """Cheap, checkable facts about one candidate file in the bundle."""

    file: str
    n_rows: int
    n_cols: int
    # Columns that are unique over the FULL file with no nulls: candidate
    # primary keys. Empty when the file was not re-read (e.g. in tests).
    key_candidates: list[str] = Field(default_factory=list)


class RelationalProfile(BaseModel):
    """How the files in a bundle relate. Descriptive only, nothing assembled."""

    shape_hint: ShapeHint
    files: list[FileRelational] = Field(default_factory=list)
    # Groups of two or more files that share an identical schema (column names
    # and dtypes); the one shape we can detect reliably and cheaply.
    same_schema_groups: list[list[str]] = Field(default_factory=list)
    note: str = (
        "One file feeds the analysis; nothing was assembled. This report "
        "describes the bundle's structure only."
    )
