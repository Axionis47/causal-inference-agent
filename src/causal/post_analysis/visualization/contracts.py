"""Scientific tables and bounded display choices; no executable chart payloads."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from causal.shared.contracts import ArtifactRef

Scalar = str | int | float | bool | None
Name = Annotated[str, Field(min_length=1, max_length=160)]
VisualId = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class Column(_Model):
    name: Name
    label: Name
    kind: Literal["quantitative", "nominal", "ordinal", "temporal"]
    quantity: Name
    units: Name | None = None
    role: Literal["value", "dimension", "series", "lower", "upper"] = "value"


class TableRowSource(_Model):
    source: ArtifactRef
    selector: str

    @model_validator(mode="after")
    def source_pointer(self) -> Self:
        if self.selector and not self.selector.startswith("/"):
            raise ValueError("selector must be a JSON Pointer into the source artifact")
        return self


class DataTable(_Model):
    table_id: Name
    source: ArtifactRef
    selector: str
    columns: Annotated[tuple[Column, ...], Field(min_length=1, max_length=40)]
    rows: Annotated[tuple[tuple[Scalar, ...], ...], Field(max_length=10000)]
    row_sources: tuple[TableRowSource, ...] = ()

    @model_validator(mode="after")
    def coherent_table(self) -> Self:
        if self.selector and not self.selector.startswith("/"):
            raise ValueError("selector must be a JSON Pointer into the source artifact")
        if len({column.name for column in self.columns}) != len(self.columns):
            raise ValueError("table column names must be unique")
        if self.row_sources and len(self.row_sources) != len(self.rows):
            raise ValueError("row_sources must bind every supplied row")
        for row in self.rows:
            if len(row) != len(self.columns):
                raise ValueError("every row must match the complete column schema")
            for column, value in zip(self.columns, row, strict=True):
                if (
                    column.kind == "quantitative"
                    and value is not None
                    and (isinstance(value, bool) or not isinstance(value, int | float))
                ):
                    raise ValueError("quantitative fields require supplied numerical values")
        return self


class VisualSpec(_Model):
    visual_id: VisualId
    table_id: Name
    kind: Literal["point", "line", "bar", "table"]
    x: Name | None = None
    y: Name | None = None
    series: Name | None = None
    interval_lower: Name | None = None
    interval_upper: Name | None = None
    interval_axis: Literal["x", "y"] = "y"
    title: Annotated[str, Field(min_length=1, max_length=200)]
    caption: Annotated[str, Field(max_length=2000)] = ""
    qualifications: tuple[Annotated[str, Field(min_length=1, max_length=1000)], ...] = ()

    @model_validator(mode="after")
    def complete_encoding(self) -> Self:
        if self.kind != "table" and (self.x is None or self.y is None):
            raise ValueError("charts require explicit x and y field bindings")
        if (self.interval_lower is None) != (self.interval_upper is None):
            raise ValueError("uncertainty requires both supplied interval fields")
        if self.kind == "table" and any((self.x, self.y, self.series, self.interval_lower)):
            raise ValueError("table displays use all supplied columns without chart encodings")
        return self


class RenderedVisual(_Model):
    visual_id: VisualId
    status: Literal["rendered", "unavailable", "failed"]
    spec_hash: str
    table_hash: str
    source: ArtifactRef
    selector: str
    objects: dict[str, str] = {}
    object_hashes: dict[str, str] = {}
    error_codes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    caption: str = ""
    versions: dict[str, str] = {}


class DiagramNode(_Model):
    node_id: Name
    label: Annotated[str, Field(min_length=1, max_length=100)]


class DiagramEdge(_Model):
    edge_id: Name
    source_node: Name
    target_node: Name
    status: Literal["evidenced", "hypothesis", "unknown", "disputed"]
    evidence_ids: tuple[str, ...] = ()
    contrary_evidence_ids: tuple[str, ...] = ()


class CausalDiagram(_Model):
    source: ArtifactRef
    selector: str
    nodes: Annotated[tuple[DiagramNode, ...], Field(min_length=1, max_length=40)]
    edges: Annotated[tuple[DiagramEdge, ...], Field(max_length=120)]

    @model_validator(mode="after")
    def known_endpoints(self) -> Self:
        known = {node.node_id for node in self.nodes}
        if len(known) != len(self.nodes) or len({e.edge_id for e in self.edges}) != len(self.edges):
            raise ValueError("diagram node and edge identifiers must be unique")
        if any(e.source_node not in known or e.target_node not in known for e in self.edges):
            raise ValueError("every edge endpoint must name a supplied node")
        if self.selector and not self.selector.startswith("/"):
            raise ValueError("selector must be a JSON Pointer")
        return self
