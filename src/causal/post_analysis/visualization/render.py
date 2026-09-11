"""Render immutable source-bound charts and tables, with matching SVG, PNG and values."""

from __future__ import annotations

import hashlib
import json
import textwrap
from collections.abc import Mapping
from html import escape
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal
from xml.etree import ElementTree as ET

import vl_convert

from causal.post_analysis.visualization.compile import (
    COMPILER_VERSION,
    compile_visual,
    validate_visual,
)
from causal.post_analysis.visualization.contracts import DataTable, RenderedVisual, VisualSpec

RENDERER_VERSION = "post-analysis-renderer.v1"
FONT_FILE = Path(__file__).resolve().parents[4] / "assets/fonts/NotoSans[wdth,wght].ttf"


def encoded(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(encoded(value)).hexdigest()


def renderer_versions() -> dict[str, str]:
    font_hash = hashlib.sha256(FONT_FILE.read_bytes()).hexdigest()
    vl_convert.register_font_directory(str(FONT_FILE.parent))
    return {
        "renderer": RENDERER_VERSION,
        "compiler": COMPILER_VERSION,
        "vl_convert": version("vl-convert-python"),
        "vega": vl_convert.get_vega_version(),
        "font_sha256": font_hash,
        "font_family": "Noto Sans",
    }


def write_objects(
    directory: Path, files: Mapping[str, bytes]
) -> tuple[dict[str, str], dict[str, str]]:
    """Replay matching bytes; a changed file can never overwrite an immutable output."""
    directory.mkdir(parents=True, exist_ok=True)
    objects, hashes = {}, {}
    for name, payload in files.items():
        path = directory / name
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError("immutable_visual_output_changed")
        else:
            with path.open("xb") as target:
                target.write(payload)
        objects[name], hashes[name] = str(path.resolve()), hashlib.sha256(payload).hexdigest()
    return objects, hashes


def _caption(spec: VisualSpec, warnings: tuple[str, ...]) -> str:
    parts = [spec.caption, *spec.qualifications]
    if "missing_coordinates" in warnings:
        parts.append(
            "Rows with missing coordinates remain in the accessible data; no value is imputed."
        )
    if "missing_intervals" in warnings:
        parts.append("Some rows have no supplied interval; their uncertainty is unavailable.")
    return " ".join(part for part in parts if part)


def decorate_svg(svg: str, caption: str) -> str:
    """Keep the same legible caption and qualifications inside both output formats."""
    if not caption:
        return svg
    root = ET.fromstring(svg)
    if "viewBox" in root.attrib:
        _, _, width, height = map(float, root.attrib["viewBox"].split())
    else:
        width, height = (
            float(root.attrib[name].removesuffix("px").removesuffix("pt"))
            for name in ("width", "height")
        )
    lines = textwrap.wrap(caption, max(30, int((width - 40) / 7)), break_long_words=True)
    new_height = height + 22 + 18 * len(lines)
    root.set("width", f"{width:g}")
    root.set("height", f"{new_height:g}")
    root.set("viewBox", f"0 0 {width:g} {new_height:g}")
    ns = "{http://www.w3.org/2000/svg}"
    ET.SubElement(
        root,
        f"{ns}rect",
        x="0",
        y=f"{height:g}",
        width=f"{width:g}",
        height=f"{new_height - height:g}",
        fill="white",
    )
    for index, line in enumerate(lines):
        element = ET.SubElement(
            root,
            f"{ns}text",
            x="20",
            y=str(height + 22 + index * 18),
            fill="#334155",
            attrib={"font-family": "Noto Sans", "font-size": "12"},
        )
        element.text = line
    return ET.tostring(root, encoding="unicode")


def _table_svg(spec: VisualSpec, table: DataTable) -> str:
    # Wide source tables wrap into column groups; the first column anchors each row.
    groups = [list(range(min(4, len(table.columns))))]
    groups += [
        [0, *range(start, min(start + 3, len(table.columns)))]
        for start in range(4, len(table.columns), 3)
    ]
    cell_width = 176
    width = cell_width * max(map(len, groups)) + 40
    elements = [f'<rect width="{width}" height="100%" fill="white"/>']
    y = 28
    for line in textwrap.wrap(spec.title, max(20, int((width - 40) / 11))):
        elements.append(
            f'<text x="20" y="{y}" font-size="17" font-weight="600">{escape(line)}</text>'
        )
        y += 22
    y += 18
    for group in groups:
        header = [
            f"{table.columns[i].label} ({table.columns[i].units})"
            if table.columns[i].units
            else table.columns[i].label
            for i in group
        ]
        values = [
            header,
            *[
                ["Unavailable" if row[i] is None else str(row[i]) for i in group]
                for row in table.rows
            ],
        ]
        for row_index, row in enumerate(values):
            wrapped = [textwrap.wrap(cell, 18, break_long_words=True) or [""] for cell in row]
            height = max(len(cell) for cell in wrapped) * 18 + 14
            color = "#e8eff5" if row_index == 0 else "#f7fafc" if row_index % 2 else "white"
            elements.append(
                f'<rect x="20" y="{y - 15}" width="{cell_width * len(group)}" '
                f'height="{height}" fill="{color}"/>'
            )
            for column_index, cell in enumerate(wrapped):
                for line_index, line in enumerate(cell):
                    elements.append(
                        f'<text x="{30 + column_index * cell_width}" '
                        f'y="{y + line_index * 18}" font-size="12">{escape(line)}</text>'
                    )
            y += height
        y += 24
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{y + 20}" '
        f'font-family="Noto Sans" fill="#1e293b">{"".join(elements)}</svg>'
    )


def _missing(spec: VisualSpec, table: DataTable) -> tuple[str, ...]:
    names = [column.name for column in table.columns]
    selected = [names.index(name) for name in (spec.x, spec.y) if name]
    warnings = (
        ["missing_coordinates"]
        if any(any(row[i] is None for i in selected) for row in table.rows)
        else []
    )
    if spec.interval_lower and any(
        row[names.index(spec.interval_lower)] is None for row in table.rows
    ):
        warnings.append("missing_intervals")
    return tuple(warnings)


def render_visual(
    spec: VisualSpec, tables: Mapping[str, DataTable], output_dir: Path
) -> RenderedVisual:
    """The caller seals tables from verified evidence; this function never creates source values."""
    if spec.table_id not in tables:
        raise ValueError(f"unknown_table:{spec.table_id}")
    table = tables[spec.table_id]
    spec_body, table_body = spec.model_dump(mode="json"), table.model_dump(mode="json")
    base: dict[str, Any] = {
        "visual_id": spec.visual_id,
        "spec_hash": digest(spec_body),
        "table_hash": digest(table_body),
        "source": table.source,
        "selector": table.selector,
    }
    if codes := validate_visual(spec, table):
        return RenderedVisual(**base, status="failed", error_codes=codes, caption=spec.caption)
    warnings = _missing(spec, table)
    caption = _caption(spec, warnings)
    columns = [column.name for column in table.columns]
    axes = [columns.index(name) for name in (spec.x, spec.y) if name]
    status: Literal["rendered", "unavailable"] = (
        "rendered"
        if table.rows and any(all(row[i] is not None for i in axes) for row in table.rows)
        else "unavailable"
    )
    versions = renderer_versions()
    files = {"data.json": encoded(table_body), "spec.json": encoded(spec_body)}
    if status == "rendered":
        try:
            if spec.kind == "table":
                svg = _table_svg(spec, table)
            else:
                document = compile_visual(spec, table)
                files["chart.json"] = encoded(document)
                svg = str(vl_convert.vegalite_to_svg(document, allowed_base_urls=[]))
            svg = decorate_svg(svg, caption)
            png = bytes(vl_convert.svg_to_png(svg, scale=2))
            if png[:8] != b"\x89PNG\r\n\x1a\n":
                raise ValueError("invalid_png_output")
            files.update({"figure.svg": svg.encode(), "figure.png": png})
        except (ValueError, RuntimeError) as error:
            return RenderedVisual(
                **base,
                status="failed",
                error_codes=(f"render_failed:{type(error).__name__}",),
                warnings=warnings,
                caption=caption,
                versions=versions,
            )
    identity = digest(
        {"spec": base["spec_hash"], "table": base["table_hash"], "versions": versions}
    )
    objects, hashes = write_objects(output_dir.resolve() / spec.visual_id / identity, files)
    return RenderedVisual(
        **base,
        status=status,
        objects=objects,
        object_hashes=hashes,
        warnings=warnings,
        caption=caption,
        versions=versions,
    )
