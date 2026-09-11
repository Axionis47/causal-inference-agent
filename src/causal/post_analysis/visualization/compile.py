"""Compile a bounded visual choice using supplied fields, without statistical transforms."""

from __future__ import annotations

import textwrap
from collections.abc import Mapping
from typing import Any

from causal.post_analysis.visualization.contracts import Column, DataTable, VisualSpec

COMPILER_VERSION = "post-analysis-visual-compiler.v1"


def validate_visual(spec: VisualSpec, table: DataTable) -> tuple[str, ...]:
    columns = {column.name: column for column in table.columns}
    selected = (spec.x, spec.y, spec.series, spec.interval_lower, spec.interval_upper)
    codes = [f"unknown_field:{name}" for name in selected if name and name not in columns]
    if spec.table_id != table.table_id:
        codes.append("table_identity_mismatch")
    if spec.kind == "table":
        return tuple(
            codes
            + (["table_display_capacity"] if len(table.rows) > 50 or len(table.columns) > 8 else [])
        )
    if codes:
        return tuple(codes)
    x, y = columns[str(spec.x)], columns[str(spec.y)]
    if x.role in {"lower", "upper"} or y.role in {"lower", "upper"}:
        codes.append("interval_bound_is_not_a_point_estimate")
    if "quantitative" not in {x.kind, y.kind}:
        codes.append("chart_requires_quantitative_axis")
    for column in (x, y):
        if column.kind == "quantitative" and not column.units:
            codes.append(f"quantity_units_unavailable:{column.name}")
        index = next(i for i, c in enumerate(table.columns) if c.name == column.name)
        if column.kind in {"nominal", "ordinal"} and len({row[index] for row in table.rows}) > 32:
            codes.append(f"axis_display_capacity:{column.name}")
    if spec.kind == "line" and (
        x.kind not in {"quantitative", "temporal", "ordinal"} or y.kind != "quantitative"
    ):
        codes.append("line_requires_ordered_x_and_quantitative_y")
    if spec.kind == "bar" and not (
        (x.kind in {"nominal", "ordinal"} and y.kind == "quantitative")
        or (y.kind in {"nominal", "ordinal"} and x.kind == "quantitative")
    ):
        codes.append("bar_requires_category_and_quantity")
    if spec.series:
        index = next(i for i, c in enumerate(table.columns) if c.name == spec.series)
        if columns[spec.series].kind not in {"nominal", "ordinal"}:
            codes.append("series_must_be_categorical")
        if len({row[index] for row in table.rows}) > 12:
            codes.append("series_display_capacity")
    codes += _interval_codes(spec, table, columns)
    if spec.kind in {"line", "bar"}:
        coordinate = spec.x if spec.kind == "line" or x.kind != "quantitative" else spec.y
        indices = [i for i, c in enumerate(table.columns) if c.name in {coordinate, spec.series}]
        keys = [tuple(row[i] for i in indices) for row in table.rows]
        if len(keys) != len(set(keys)):
            codes.append("ambiguous_coordinate_requires_explicit_series")
    return tuple(codes)


def _interval_codes(spec: VisualSpec, table: DataTable, columns: Mapping[str, Column]) -> list[str]:
    if not spec.interval_lower:
        return []
    target = columns[str(spec.x if spec.interval_axis == "x" else spec.y)]
    low, high = columns[spec.interval_lower], columns[str(spec.interval_upper)]
    if (
        target.kind != "quantitative"
        or low.role != "lower"
        or high.role != "upper"
        or any(
            c.kind != "quantitative" or (c.quantity, c.units) != (target.quantity, target.units)
            for c in (low, high)
        )
    ):
        return ["interval_quantity_mismatch"]
    names = [c.name for c in table.columns]
    lo, hi = names.index(low.name), names.index(high.name)
    for row in table.rows:
        a, b = row[lo], row[hi]
        if (a is None) != (b is None):
            return ["incomplete_interval"]
        if isinstance(a, int | float) and isinstance(b, int | float) and a > b:
            return ["reversed_interval"]
    return []


def _channel(column: Column, key: str, *, bar: bool = False) -> dict[str, Any]:
    title = f"{column.label} ({column.units})" if column.units else column.label
    channel: dict[str, Any] = {"field": key, "type": column.kind, "title": title}
    if column.kind == "quantitative":
        channel["scale"] = {"zero": bar, "nice": True}
    elif column.kind in {"nominal", "ordinal"}:
        channel["sort"] = None
    return channel


def _encodings(
    spec: VisualSpec, table: DataTable, keys: Mapping[str, str], columns: Mapping[str, Column]
) -> dict[str, Any]:
    """Bind axes, intervals, groups and accessibility to literal supplied columns."""
    encodings: dict[str, Any] = {
        axis: _channel(columns[str(name)], keys[str(name)], bar=spec.kind == "bar")
        for axis, name in (("x", spec.x), ("y", spec.y))
    }
    encodings["tooltip"] = [
        {k: v for k, v in _channel(c, keys[c.name]).items() if k in {"field", "type", "title"}}
        for c in table.columns
    ]
    # Explicitly disable bar stacking: stacking would silently calculate a new total.
    if spec.kind == "bar":
        axis = "x" if columns[str(spec.x)].kind == "quantitative" else "y"
        encodings[axis]["stack"] = None
    if spec.series:
        channel = {
            "field": keys[spec.series],
            "type": "nominal",
            "title": columns[spec.series].label,
        }
        encodings["color"] = channel
        other = (
            "shape"
            if spec.kind == "point"
            else "strokeDash"
            if spec.kind == "line"
            else "yOffset"
            if columns[str(spec.x)].kind == "quantitative"
            else "xOffset"
        )
        encodings[other] = channel
    if spec.kind == "line":
        encodings["order"] = {"field": "row_order", "type": "quantitative"}
    return encodings


def compile_visual(spec: VisualSpec, table: DataTable) -> dict[str, Any]:
    """Use generated field keys so literal source column names cannot become expressions."""
    if codes := validate_visual(spec, table):
        raise ValueError(", ".join(codes))
    if spec.kind == "table":
        raise ValueError("table displays do not use a chart document")
    keys = {c.name: f"c{i}" for i, c in enumerate(table.columns)}
    columns = {c.name: c for c in table.columns}
    rows = [
        {**{f"c{i}": value for i, value in enumerate(row)}, "row_order": order}
        for order, row in enumerate(table.rows)
    ]
    encodings = _encodings(spec, table, keys, columns)
    layers: list[dict[str, Any]] = []
    if spec.interval_lower:
        axis = spec.interval_axis
        interval = {
            k: v
            for k, v in encodings.items()
            if k not in {axis, "shape", "strokeDash", "order", "tooltip"}
        }
        interval[axis] = {**encodings[axis], "field": keys[spec.interval_lower]}
        interval[f"{axis}2"] = {"field": keys[str(spec.interval_upper)]}
        layers.append({"mark": {"type": "rule", "strokeWidth": 1.6}, "encoding": interval})
    mark: dict[str, Any] = {
        "type": spec.kind,
        "invalid": "break-paths-show-domains",
        "color": "#24557a",
        "tooltip": True,
    }
    if spec.kind == "point":
        mark.update(filled=True, size=75)
    layers.append({"mark": mark, "encoding": encodings})
    y_column = columns[str(spec.y)]
    y_index = next(i for i, column in enumerate(table.columns) if column.name == spec.y)
    height = (
        max(340, len({row[y_index] for row in table.rows}) * 25)
        if y_column.kind in {"nominal", "ordinal"}
        else 340
    )
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
        "data": {"values": rows},
        "layer": layers,
        "width": 680,
        "height": height,
        "title": {"text": textwrap.wrap(spec.title, 70), "anchor": "start", "fontSize": 17},
        "background": "white",
        "padding": 20,
        "config": {
            "font": "Noto Sans",
            "axis": {
                "labelFontSize": 12,
                "titleFontSize": 13,
                "labelLimit": 0,
                "titleLimit": 0,
                "labelOverlap": False,
            },
            "legend": {
                "labelFontSize": 12,
                "titleFontSize": 13,
                "labelLimit": 0,
                "orient": "bottom",
            },
            "view": {"stroke": "#d8e0e8"},
        },
    }
