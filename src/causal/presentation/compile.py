# §13 compilation: one accepted FigurePlan, its frozen figure data, and the immutable catalog
# become one declarative FigureSpec per figure, plus the one Vega-Lite document the renderer
# draws both outputs from. §11's honesty rules are mechanical here: no path in this module
# writes a transform, a second y-axis, a remote data url, or a bar baseline above zero.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final, Literal

import altair as alt

from causal.presentation import contracts as pc

AxisScale = Literal["linear", "log", "band", "time"]

COMPILER_VERSION: Final = "presentation-compiler.v1"
UNKNOWN_TEMPLATE, MISSING_FIGURE_DATA = "unknown_template", "missing_figure_data"
# §11.2: the scale each enumerated catalog axis kind declares, and the Vega-Lite type it draws.
SCALES: Final[dict[str, AxisScale]] = {"quantitative": "linear", "nominal": "band",
                                       "temporal": "time"}
VEGA_TYPES: Final = {"linear": "quantitative", "log": "quantitative", "band": "nominal",
                     "time": "temporal"}
# §11.3: a point with no value is missing or suppressed, and the two never merge into one state.
VALUE, MISSING, SUPPRESSED = "value", "missing", "suppressed"
# Vega-Lite's composite error marks carry an implicit aggregate, so V1 draws the frozen interval
# fields directly instead (§11.3: no transform beyond what the metadata permits).
DRAWN_MARKS: Final = {"errorband": "area", "errorbar": "rule"}
# §11.2: where each registered reference line sits. The rest are read from the frozen data.
FIXED_REFERENCES: Final = {"null_effect": 0.0, "zero": 0.0, "balance_threshold": 0.1}
# §14: the non-color channel each mark family carries a group distinction on. A bar separates
# its groups by position, which is already a non-color distinction.
NON_COLOR: Final = {"point": "shape", "line": "strokeDash", "rule": "strokeDash"}
NON_COLOR_CHANNELS: Final = {"point": alt.Shape, "line": alt.StrokeDash, "rule": alt.StrokeDash}
PANEL_SPACING, TITLE_BAND, AXIS_GUTTER = 24, 90, 150


def panel_width(profile: pc.DisplayProfileV1) -> int:
    # §13: the drawing area inside the one pinned outer width, leaving the gutter the theme's
    # axis labels need. A figure whose labels still overflow asks for a layout revision.
    return profile.logical_width - 2 * profile.outer_padding - AXIS_GUTTER


def _axis_kinds(template: pc.TemplateV1) -> dict[str, str]:
    return {name.rsplit("_", 1)[1]: name.rsplit("_", 1)[0] for name in template.choices["axes"]}


def _rows(points: Sequence[Mapping[str, Any]], template: pc.TemplateV1,
          measure: str) -> list[dict[str, Any]]:
    # The frozen fields the template maps, renamed to its declared roles. Nothing is computed,
    # rescaled, binned, or filled in, and an absent measure keeps its own distinct state.
    mapping, drawn = template.field_mappings, []
    for row in points:
        cell = {role: row.get(field) for role, field in mapping.items()}
        cell["state"] = VALUE if row.get(mapping[measure]) is not None else (
            SUPPRESSED if row.get("denominator") is not None else MISSING)
        drawn.append(cell)
    return drawn


def _reference(name: str, points: Sequence[Mapping[str, Any]]) -> float | None:
    # A fixed reference sits where §11.2 fixes it; a data-defined one (a cutoff, a treatment
    # start, a support limit, a denominator) is read from the frozen points, never chosen.
    if name in FIXED_REFERENCES:
        return FIXED_REFERENCES[name]
    if name == "denominator":
        found = [float(row["denominator"]) for row in points if row.get("denominator") is not None]
        return max(found) if found else None
    marked = [row for row in points if str(row.get("category")) == name]
    value = marked[0].get("x_value") if marked else None
    return float(value) if isinstance(value, int | float) else None


def _numbers(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> list[float]:
    return [float(row[key]) for row in rows for key in keys if isinstance(row.get(key), int | float)]


def _domain(rows: Sequence[Mapping[str, Any]], role: str, mark: str,
            references: Sequence[float | None], *, intervals: bool) -> tuple[float, float] | None:
    # §11.2: the domain includes every visible mark, the full uncertainty interval, and every
    # mandatory reference. A bar encoding magnitude starts at zero.
    values = _numbers(rows, (role, *(("low", "high") if intervals else ())))
    values += [row for row in references if row is not None]
    if not values:
        return None
    low, high = min(values), max(values)
    if mark == "bar":
        low = min(0.0, low)
    return (low, high) if high > low else (low, low + 1.0)


def _axis(role: str, kind: str, payload: Mapping[str, Any], evidence_id: str,
          domain: tuple[float, float] | None) -> pc.AxisSpecV1:
    # §11.2: quantity, unit, scale, and a mechanically computed domain, all from frozen fields.
    unit = str((payload.get("units") or {}).get(role) or role)
    label = str((payload.get("labels") or {}).get(role) or f"{evidence_id} ({unit})")
    return pc.AxisSpecV1(quantity_id=f"{evidence_id}:{role}", unit_id=unit, title=label,
                         scale=SCALES[kind], domain=domain)


def _panel(entry: pc.FigureEntryV1, index: int, group: Sequence[str], template: pc.TemplateV1,
           profile: pc.MethodProfileV1,
           figure_data: Mapping[str, Mapping[str, Any]]
           ) -> tuple[pc.PanelSpecV1, list[dict[str, Any]], dict[str, float | None]]:
    # One panel answering the evidence questions of one permitted group (§11.1).
    kinds = _axis_kinds(template)
    measure = "y" if kinds.get("y") == "quantitative" else "x"
    mark = entry.choices.get("marks", template.choices["marks"][0])
    points = [row for name in group for row in (figure_data[name].get("points") or ())]
    rows = _rows(points, template, measure)
    names = tuple(sorted({entry.choices.get("reference_lines", "")} | {
        name for evidence_id in group
        for name in profile.mandatory_encodings.get(evidence_id, ())
        if name in template.choices["reference_lines"]} - {""}))
    references = {name: _reference(name, points) for name in names}
    head = figure_data[group[0]]
    axes = {role: _axis(role, kind, head, group[0],
                        _domain(rows, role, mark, tuple(references.values()),
                                intervals=role == measure) if kind == "quantitative" else None)
            for role, kind in kinds.items()}
    encodings: dict[str, tuple[str, ...]] = {
        "series": tuple(sorted({str(row.get("series_id")) for row in points})) or ("(none)",),
        "fields": tuple(sorted(set(template.field_mappings.values()))),
        "states": tuple(sorted({str(row["state"]) for row in rows})) or (MISSING,)}
    if any(row.get("low") is not None for row in rows):
        encodings["uncertainty"] = ("low", "high")
    if names:
        encodings["reference_lines"] = names
    if entry.annotation_ids:
        encodings["annotations"] = entry.annotation_ids
    panel = pc.PanelSpecV1(panel_id=f"{entry.figure_id}:{index}", question="+".join(group),
                           mark=mark, axes=axes, encodings=encodings)
    return panel, rows, references


def _height(template: pc.TemplateV1, panels: int) -> int:
    bounds = template.bounds
    return min(max(bounds.min_logical_height, TITLE_BAND + 160 * panels), bounds.max_logical_height)


def compile_figures(plan: pc.FigurePlanV1, catalog: pc.VisualizationCatalogV1,
                    profile: pc.MethodProfileV1, figure_data: Mapping[str, Mapping[str, Any]],
                    lineage: Mapping[str, Any]) -> tuple[pc.FigureSpecV1, ...]:
    # One committed FigureSpec per accepted figure. The compiler selects no template, creates
    # no value, and performs no statistical transform (§13).
    templates = {row.template_id: row for row in catalog.templates}
    specs = []
    for entry in plan.figures:
        template = templates.get(entry.template_id)
        if template is None:
            raise pc.PresentationError(f"{entry.template_id} is unregistered", UNKNOWN_TEMPLATE)
        if any(name not in figure_data for name in entry.visual_evidence_ids):
            raise pc.PresentationError(f"{entry.figure_id} has no frozen data",
                                       MISSING_FIGURE_DATA)
        panels = tuple(_panel(entry, index, group, template, profile, figure_data)[0]
                       for index, group in enumerate(entry.panel_groups))
        specs.append(pc.FigureSpecV1(
            parents=tuple(lineage["parents"]), figure_id=entry.figure_id,
            versions=dict(lineage["versions"]) | {"compiler": COMPILER_VERSION},
            template_id=entry.template_id, panels=panels, text=dict(entry.text),
            qualification_ids=entry.qualification_ids,
            logical_height=_height(template, len(panels))))
    return tuple(specs)


def _channel(role: str, axis: pc.AxisSpecV1) -> Any:
    options: dict[str, Any] = {"title": axis.title, "type": VEGA_TYPES[axis.scale]}
    if axis.domain is not None:
        options["scale"] = alt.Scale(domain=list(axis.domain), zero=axis.domain[0] == 0.0)
    return (alt.X if role == "x" else alt.Y)(role, **options)


def _panel_chart(panel: pc.PanelSpecV1, rows: Sequence[Mapping[str, Any]],
                 references: Mapping[str, float | None], template: pc.TemplateV1,
                 width: int, height: int) -> Any:
    # Layered marks over one inline table: the drawn mark, its frozen uncertainty rule, and the
    # required references. `alt.Data(values=...)` is the only data source there is here.
    measure = "y" if VEGA_TYPES[panel.axes["y"].scale] == "quantitative" else "x"
    encode: dict[str, Any] = {role: _channel(role, axis) for role, axis in panel.axes.items()}
    if "group" in template.field_mappings:
        encode["color"] = alt.Color("group", type="nominal")
        if panel.mark in NON_COLOR:
            encode[NON_COLOR[panel.mark]] = NON_COLOR_CHANNELS[panel.mark]("group", type="nominal")
    quantitative = alt.X if measure == "x" else alt.Y
    base = alt.Chart(alt.Data(values=list(rows)))  # type: ignore[no-untyped-call]
    layers = [getattr(base, f"mark_{DRAWN_MARKS.get(panel.mark, panel.mark)}")().encode(**encode)]
    if "uncertainty" in panel.encodings:
        span = {measure: quantitative("low", type="quantitative",
                                      title=panel.axes[measure].title),
                f"{measure}2": (alt.X2 if measure == "x" else alt.Y2)("high")}
        layers.append(base.mark_rule().encode(**{
            role: channel for role, channel in encode.items()
            if role not in (measure, "shape", "strokeDash")}, **span))
    for name, value in sorted(references.items()):
        if value is not None:
            layers.append(alt.Chart(alt.Data(  # type: ignore[no-untyped-call]
                values=[{"reference": value, "name": name}]))
                          .mark_rule(strokeDash=[4, 4])
                          .encode(**{measure: quantitative("reference", type="quantitative")}))
    return alt.layer(*layers).properties(width=width, height=height)


def figure_document(spec: pc.FigureSpecV1, entry: pc.FigureEntryV1, template: pc.TemplateV1,
                    profile: pc.MethodProfileV1, figure_data: Mapping[str, Mapping[str, Any]],
                    width: int) -> dict[str, Any]:
    # The one Vega-Lite document both the SVG and the PNG are drawn from, built from the same
    # frozen fields the specification names. Panels stack; no scale resolves onto a second axis.
    charts = []
    height = max(120, (spec.logical_height - TITLE_BAND) // len(spec.panels))
    for index, group in enumerate(entry.panel_groups):
        panel, rows, references = _panel(entry, index, group, template, profile, figure_data)
        charts.append(_panel_chart(panel, rows, references, template, width, height))
    chart = charts[0] if len(charts) == 1 else alt.vconcat(*charts, spacing=PANEL_SPACING)
    return dict(chart.properties(title=spec.text["title"]).to_dict())
