# §13 rendering: the vendored font or a blocker, one SVG and one PNG from the same document at
# the one pinned display profile, the §14 accessible description checks and frozen-value table,
# the gate-4 render validator, and the §13.1 renderer fingerprint. No figure loads a remote
# font, asset, or datum, and no branch here substitutes a system or synthetic face.

from __future__ import annotations

import hashlib
import platform
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final

import altair as alt
import vl_convert

from causal.presentation import compile as compiler
from causal.presentation import contracts as pc

RENDERER_VERSION: Final = "presentation-renderer.v1"
FONT_BLOCKER: Final = "vendored_font_unavailable"
# The gate-4 code family (§16 gate 4); every one of them names what failed.
ILLEGIBLE, CLIPPED = "text_below_minimum_size", "text_clipped"
PARITY_BROKEN, DESCRIPTION_INACCURATE = "svg_png_parity_broken", "description_inaccurate"
TABLE_MISMATCH: Final = "accessible_table_mismatch"
# §13: the theme's smallest permitted text, and the family the vendored file registers under.
MIN_FONT_SIZE, FONT_FAMILY = 11.0, "Noto Sans"
LAYOUT_CODES: Final = (CLIPPED, ILLEGIBLE)
# §14: the frozen columns of every accessible table, copied and never recomputed.
TABLE_COLUMNS: Final = ("panel_id", "visual_evidence_id", "series_id", "category", "x_value",
                        "y_value", "interval_lower", "interval_upper", "denominator",
                        "disclosure_status")
_CANVAS = re.compile(r'width="([0-9.]+)"\s+height="([0-9.]+)"')
_FONT_SIZE = re.compile(r'font-size(?:="|:\s*)([0-9.]+)')


def register_font(catalog: pc.VisualizationCatalogV1, root: Path) -> str:
    # §13: the vendored file, hashed, or an immediate blocker. There is no fallback branch.
    path = root / catalog.font_file_path
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise pc.PresentationError(f"vendored font {path} is unreadable", FONT_BLOCKER) from error
    if digest != catalog.font_sha256:
        raise pc.PresentationError(f"vendored font {path} hashes {digest}", FONT_BLOCKER)
    vl_convert.register_font_directory(str(path.parent))
    return digest


def themed(document: Mapping[str, Any], catalog: pc.VisualizationCatalogV1) -> dict[str, Any]:
    # The one pinned theme and the one vendored family, applied to the document both outputs
    # are drawn from. Every size here is at or above the legible minimum.
    profile = catalog.display_profile
    text = {"font": FONT_FAMILY, "labelFontSize": 12, "titleFontSize": 13}
    return dict(document) | {
        "padding": profile.outer_padding, "autosize": {"type": "pad", "contains": "padding"},
        "background": "#ffffff",
        "config": {"font": FONT_FAMILY, "axis": text, "legend": text, "header": text,
                   "title": {"font": FONT_FAMILY, "fontSize": 16, "fontWeight": 600},
                   "view": {"stroke": "#c8ccd2"}, "range": {"category": [
                       "#1b3a6b", "#8c4a00", "#1f6f54", "#6b2d5c", "#4a4a4a", "#7a5c00"]}}}


def fingerprint(catalog: pc.VisualizationCatalogV1) -> dict[str, str]:
    # §13.1: everything a byte-identical replay depends on. Two renders are required to match
    # byte for byte only when every value below is identical.
    return {"python": platform.python_version(), "altair": alt.__version__,
            "vl_convert": version("vl-convert-python"), "vega_lite_schema": alt.SCHEMA_VERSION,
            "os": platform.system(), "os_release": platform.release(),
            "arch": platform.machine(), "theme": catalog.theme_id,
            "theme_version": catalog.theme_version, "theme_hash": catalog.theme_hash,
            "font": catalog.font_id, "font_sha256": catalog.font_sha256,
            "display_profile": catalog.display_profile.display_profile_version,
            "catalog": catalog.catalog_version, "compiler": compiler.COMPILER_VERSION,
            "renderer": RENDERER_VERSION}


@dataclass(frozen=True)
class RenderOutput:
    # The committed record and the two byte streams behind it, kept together for gate 4.
    artifact: pc.RenderArtifactV1
    svg: str
    png: bytes


def render(spec: pc.FigureSpecV1, document: Mapping[str, Any],
           catalog: pc.VisualizationCatalogV1, out_dir: Path,
           lineage: Mapping[str, Any]) -> RenderOutput:
    # §13: one SVG and one PNG, both from the same themed document, at the one pinned profile.
    profile = catalog.display_profile
    ready = themed(document, catalog)
    svg = vl_convert.vegalite_to_svg(ready)
    # §13: the canvas is exactly the pinned logical width. Measured slack becomes outer padding;
    # a figure with no slack left overflows it and gate 4 asks for a layout revision instead.
    slack = int(profile.logical_width - _canvas(svg)[0])
    if slack > 0:
        pad = profile.outer_padding
        ready = dict(ready) | {"padding": {"left": pad + slack, "right": pad, "top": pad,
                                           "bottom": pad}}
        svg = vl_convert.vegalite_to_svg(ready)
    png = vl_convert.vegalite_to_png(ready, scale=profile.png_width / profile.logical_width)
    objects, hashes = {}, {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, blob in (("svg", svg.encode("utf-8")), ("png", png)):
        target = out_dir / f"{spec.figure_id}.{name}"
        target.write_bytes(blob)
        objects[name], hashes[name] = str(target), hashlib.sha256(blob).hexdigest()
    return RenderOutput(pc.RenderArtifactV1(
        parents=tuple(lineage["parents"]),
        versions=dict(lineage["versions"]) | {"renderer": RENDERER_VERSION},
        figure_id=spec.figure_id, spec_hash=spec.spec_hash(), objects=objects,
        object_hashes=hashes, logical_height=spec.logical_height,
        renderer_fingerprint=fingerprint(catalog)), svg, png)


def accessible_table(spec: pc.FigureSpecV1, entry: pc.FigureEntryV1,
                     figure_data: Mapping[str, Mapping[str, Any]], limit: int) -> dict[str, Any]:
    # §14: the same evidence the figure draws, as values. Every cell is copied from the frozen
    # figure data, and a suppressed or missing cell keeps its own state rather than being blanked.
    rows = []
    for panel, group in zip(spec.panels, entry.panel_groups, strict=True):
        for name in group:
            payload = figure_data[name]
            for point in payload.get("points") or ():
                rows.append({"panel_id": panel.panel_id, "visual_evidence_id": name,
                             "disclosure_status": payload.get("disclosure_status"),
                             **{key: point.get(key) for key in TABLE_COLUMNS[2:9]}})
    return {"figure_id": spec.figure_id, "spec_hash": spec.spec_hash(),
            "columns": list(TABLE_COLUMNS), "rows": rows[:limit], "truncated": len(rows) > limit}


def description_codes(spec: pc.FigureSpecV1) -> tuple[str, ...]:
    # Gate 4: the drafted description must name every unit the figure draws, each reference it
    # carries, its uncertainty when an interval is encoded, and each qualification it states.
    text = spec.text["accessible_description"].lower()
    wanted = {axis.unit_id for panel in spec.panels for axis in panel.axes.values()}
    wanted |= {name for panel in spec.panels
               for name in panel.encodings.get("reference_lines", ())}
    wanted |= set(spec.qualification_ids)
    if any("uncertainty" in panel.encodings for panel in spec.panels):
        wanted.add("interval")
    return tuple(f"{DESCRIPTION_INACCURATE}:{name}" for name in sorted(wanted)
                 if name.lower() not in text and name.lower().replace("_", " ") not in text)


def _canvas(svg: str) -> tuple[float, float]:
    found = _CANVAS.search(svg)
    return (float(found.group(1)), float(found.group(2))) if found else (0.0, 0.0)


def _png_width(png: bytes) -> int:
    # The IHDR width of a PNG byte stream; the header is fixed, so this reads no image library.
    return int.from_bytes(png[16:20], "big") if png[:8] == b"\x89PNG\r\n\x1a\n" else 0


def table_codes(table: Mapping[str, Any], spec: pc.FigureSpecV1) -> tuple[str, ...]:
    # §14: the table belongs to this specification and describes the panels it actually drew.
    panels = {panel.panel_id for panel in spec.panels}
    bad = (table.get("spec_hash") != spec.spec_hash() or table.get("truncated")
           or {str(row["panel_id"]) for row in table.get("rows") or ()} - panels)
    return (f"{TABLE_MISMATCH}:{spec.figure_id}",) if bad else ()


def render_codes(spec: pc.FigureSpecV1, output: RenderOutput, template: pc.TemplateV1,
                 catalog: pc.VisualizationCatalogV1,
                 table: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    # Gate 4 over one rendered figure: legible text, a canvas the pinned profile still holds,
    # an SVG and a PNG that agree with each other and with the one specification, and a
    # description that matches what was drawn.
    profile, bounds = catalog.display_profile, template.bounds
    sizes = [float(found) for found in _FONT_SIZE.findall(output.svg)]
    width, height = _canvas(output.svg)
    codes = [f"{ILLEGIBLE}:{min(sizes)}" for _ in (0,) if sizes and min(sizes) < MIN_FONT_SIZE]
    if width > profile.logical_width or height > bounds.max_logical_height + profile.outer_padding:
        codes.append(f"{CLIPPED}:{width:.0f}x{height:.0f}")
    if not bounds.min_logical_height <= spec.logical_height <= bounds.max_logical_height:
        codes.append(f"{CLIPPED}:logical_height:{spec.logical_height}")
    if (output.artifact.spec_hash != spec.spec_hash()
            or set(output.artifact.objects) != {"svg", "png"}
            or _png_width(output.png) != profile.png_width):
        codes.append(f"{PARITY_BROKEN}:{spec.figure_id}")
    codes += description_codes(spec)
    return tuple(codes) + (table_codes(table, spec) if table is not None else ())


def render_status(codes: Sequence[str]) -> pc.PresentationOutcomeStatus:
    # §13: a figure that cannot stay legible inside its bounds asks for a layout revision; it
    # is never silently shrunk, clipped, refonted, or stripped of evidence.
    if any(code.startswith(LAYOUT_CODES) for code in codes):
        return "needs_layout_revision"
    return "failed" if codes else "complete"
