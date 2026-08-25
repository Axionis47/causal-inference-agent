"""Read-only catalog, product, and CSV access shared by stage harnesses (SC §14.1; D-049)."""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import polars as pl
from psycopg import Connection

from causal.shared.contracts import ArtifactEnvelopeV1, ArtifactRef

__all__ = [
    "BytesFrameSource", "CatalogReader", "CsvObjectFrameSource", "FrameSource", "ObjectReader",
    "ProductsReader", "SqlCatalogReader",
]


class CatalogReader(Protocol):
    """The narrow intake-catalogue read surface a stage entry gate needs."""

    def resources(self, dataset_id: str) -> tuple[dict[str, Any], ...]: ...

    def view_rows(self, view_name: str, dataset_id: str) -> tuple[dict[str, Any], ...]: ...


class ProductsReader(Protocol):
    """Committed-artifact envelope lookup; `ProductStore` satisfies it structurally."""

    def load_envelope(self, artifact_id: str) -> ArtifactEnvelopeV1: ...


class SqlCatalogReader:
    """`CatalogReader` over the intake `catalog` schema: resources plus a closed view set."""

    def __init__(self, conn: Connection[Any], *, views: tuple[str, ...],
                 error: Callable[[str], Exception]) -> None:
        self._conn = conn
        self._views = views
        self._error = error

    def _rows(self, sql: str, dataset_id: str) -> tuple[dict[str, Any], ...]:
        cursor = self._conn.execute(sql, (dataset_id,))
        names = [column.name for column in cursor.description or ()]
        return tuple(dict(zip(names, row, strict=True)) for row in cursor.fetchall())

    def resources(self, dataset_id: str) -> tuple[dict[str, Any], ...]:
        return self._rows(
            "SELECT logical_name, kind, object_key, object_sha256, media_type, parse_status"
            " FROM catalog.resources WHERE dataset_id = %s ORDER BY logical_name", dataset_id)

    def view_rows(self, view_name: str, dataset_id: str) -> tuple[dict[str, Any], ...]:
        if view_name not in self._views:
            raise self._error(view_name)
        # view_name is interpolated only from the constructor's closed view vocabulary.
        return self._rows(
            f"SELECT * FROM catalog.{view_name} WHERE dataset_id = %s"
            " ORDER BY table_name NULLS FIRST, column_name NULLS FIRST, field_or_slot_name",
            dataset_id)


class FrameSource(Protocol):
    """The committed CSV a diagnostic reads; nothing here may write a table (PRD-002 §4.2)."""

    def csv_ref(self) -> ArtifactRef: ...

    def frame(self) -> pl.DataFrame: ...


class ObjectReader(Protocol):
    """The one `ObjectStore` method a frame source needs."""

    def get(self, locator: str) -> bytes: ...


@dataclass(frozen=True)
class BytesFrameSource:
    """A frame source over already-held CSV bytes."""

    name: str
    data: bytes
    ref: ArtifactRef

    def csv_ref(self) -> ArtifactRef:
        return self.ref

    def frame(self) -> pl.DataFrame:
        return pl.read_csv(io.BytesIO(self.data))


@dataclass(frozen=True)
class CsvObjectFrameSource:
    """A frame source over the committed CSV object (D-019 locator)."""

    objects: ObjectReader
    locator: str
    ref: ArtifactRef

    def csv_ref(self) -> ArtifactRef:
        return self.ref

    def frame(self) -> pl.DataFrame:
        return pl.read_csv(io.BytesIO(self.objects.get(self.locator)))
