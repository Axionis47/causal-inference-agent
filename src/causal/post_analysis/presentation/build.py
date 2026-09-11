"""Deterministic final pages; reviewed previews and the delivered HTML share exact bytes."""

from __future__ import annotations

import base64
import hashlib
import json
import struct
from html import escape
from pathlib import Path
from typing import Any

import vl_convert

from causal.post_analysis.contracts import ReportDraft
from causal.post_analysis.visualization.contracts import RenderedVisual
from causal.post_analysis.visualization.render import digest, renderer_versions, write_objects
from causal.shared.canonical import content_hash

PAGE_WIDTH, PAGE_HEIGHT, MARGIN, MAX_PAGES = 850, 1100, 48, 24
_CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN
_BOTTOM = PAGE_HEIGHT - 60
BUILD_VERSION = "post-analysis-pages.v1"


def _width(text: str, size: int) -> float:
    # Conservative advances keep untrusted text inside its box without a font dependency.
    return size * sum(
        0.38 if char in " ilI.,:;'|!" else 1.05 if char in "WM@#%" or ord(char) > 127 else 0.74
        for char in text
    )


def _wrap(text: str, size: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        line = ""
        for word in paragraph.split():
            while _width(word, size) > _CONTENT_WIDTH:
                if line:
                    lines.append(line)
                    line = ""
                cut = 1
                while cut < len(word) and _width(word[: cut + 1], size) <= _CONTENT_WIDTH:
                    cut += 1
                lines.append(word[:cut])
                word = word[cut:]
            candidate = f"{line} {word}".strip()
            if _width(candidate, size) > _CONTENT_WIDTH:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    return lines


class _Pages:
    def __init__(self) -> None:
        self.pages: list[list[str]] = [[]]
        self.y = float(MARGIN)

    def room(self, height: float) -> None:
        if self.y + height <= _BOTTOM:
            return
        if len(self.pages) == MAX_PAGES:
            raise ValueError("report_page_capacity_exceeded")
        self.pages.append([])
        self.y = float(MARGIN)

    def text(
        self,
        text: str,
        *,
        size: int = 15,
        color: str = "#24354a",
        bold: bool = False,
        gap: int = 12,
    ) -> None:
        leading = size * 1.5
        for line in _wrap(text, size):
            self.room(leading)
            self.y += leading
            self.pages[-1].append(
                f'<text x="{MARGIN}" y="{self.y:g}" font-size="{size}" '
                f'fill="{color}" font-weight="{"600" if bold else "400"}">{escape(line)}</text>'
            )
        self.y += gap

    def visual(self, png: bytes, visual_id: str) -> None:
        if png[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"invalid_visual_png:{visual_id}")
        width, height = struct.unpack(">II", png[16:24])
        # Chart PNGs are 2x their logical SVG. Never squash or clip a reviewed chart.
        logical_width, logical_height = width / 2, height / 2
        scale = min(1.0, _CONTENT_WIDTH / logical_width, (_BOTTOM - MARGIN) / logical_height)
        if scale < 0.75:
            raise ValueError(f"visual_exceeds_page_capacity:{visual_id}")
        display_width, display_height = logical_width * scale, logical_height * scale
        self.room(display_height + 14)
        data = base64.b64encode(png).decode()
        x = MARGIN + (_CONTENT_WIDTH - display_width) / 2
        self.pages[-1].append(
            f'<image x="{x:g}" y="{self.y:g}" width="{display_width:g}" '
            f'height="{display_height:g}" href="data:image/png;base64,{data}" '
            f'aria-label="{escape(visual_id, quote=True)}"/>'
        )
        self.y += display_height + 14

    def documents(self) -> list[str]:
        total = len(self.pages)
        return [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{PAGE_WIDTH}" '
            f'height="{PAGE_HEIGHT}" viewBox="0 0 {PAGE_WIDTH} {PAGE_HEIGHT}" '
            f'font-family="Noto Sans" role="img" aria-label="Report page {index} of {total}">'
            '<rect width="100%" height="100%" fill="white"/>'
            f'{"".join(page)}<line x1="48" x2="802" y1="1060" y2="1060" stroke="#d9e2eb"/>'
            f'<text x="48" y="1082" fill="#596d82" font-size="11">Causal analysis</text>'
            f'<text x="802" y="1082" text-anchor="end" fill="#596d82" font-size="11">'
            f"{index} / {total}</text></svg>"
            for index, page in enumerate(self.pages, 1)
        ]


def _verified(visual: RenderedVisual, key: str) -> bytes:
    if key not in visual.objects or key not in visual.object_hashes:
        raise ValueError(f"missing_visual_object:{visual.visual_id}:{key}")
    raw = Path(visual.objects[key]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != visual.object_hashes[key]:
        raise ValueError(f"visual_object_hash_mismatch:{visual.visual_id}:{key}")
    return raw


def _accessible(visual: RenderedVisual) -> str:
    data = json.loads(_verified(visual, "data.json"))
    heading = f"<h3>{escape(visual.visual_id)}</h3>"
    source = f"<p>Source: {escape(visual.source.artifact_id)} {escape(visual.selector)}; "
    source += f"SHA-256 {visual.source.content_hash}</p>"
    if "columns" in data and "rows" in data:
        headers = [
            str(column["label"]) + (f" ({column['units']})" if column.get("units") else "")
            for column in data["columns"]
        ]
        rows = data["rows"]
        if data.get("row_sources"):
            headers += ["Source artifact", "Source SHA-256", "Source selector"]
            rows = [
                row
                + [
                    binding["source"]["artifact_id"],
                    binding["source"]["content_hash"],
                    binding["selector"],
                ]
                for row, binding in zip(rows, data["row_sources"], strict=True)
            ]
    elif "nodes" in data and "edges" in data:
        headers = [
            "Record",
            "ID",
            "Label / source",
            "Target",
            "Status",
            "Evidence",
            "Contrary evidence",
        ]
        rows = [["Node", node["node_id"], node["label"], "", "", "", ""] for node in data["nodes"]]
        rows += [
            [
                "Edge",
                edge["edge_id"],
                edge["source_node"],
                edge["target_node"],
                edge["status"],
                ", ".join(edge["evidence_ids"]),
                ", ".join(edge["contrary_evidence_ids"]),
            ]
            for edge in data["edges"]
        ]
    else:
        raise ValueError(f"unknown_accessible_data:{visual.visual_id}")

    def cells(row: list[Any]) -> str:
        return "".join(
            f"<td>{escape('Unavailable' if cell is None else str(cell))}</td>" for cell in row
        )

    return (
        heading
        + source
        + "<table><thead><tr>"
        + "".join(f'<th scope="col">{escape(h)}</th>' for h in headers)
        + "</tr></thead><tbody>"
        + "".join(f"<tr>{cells(row)}</tr>" for row in rows)
        + "</tbody></table>"
    )


def _compose_pages(
    draft: ReportDraft, visuals: dict[str, RenderedVisual]
) -> tuple[list[str], dict[str, RenderedVisual]]:
    """Lay out every statement, citation and selected visual before exporting any page."""
    pages = _Pages()
    pages.text(draft.title, size=28, bold=True, color="#173653", gap=22)
    selected: dict[str, RenderedVisual] = {}
    for section in draft.sections:
        pages.room(100)
        pages.text(section.title, size=20, bold=True, color="#173653", gap=14)
        for statement in section.statements:
            pages.text(statement.text)
            citations = "; ".join(
                citation.evidence_id + citation.selector for citation in statement.citations
            )
            pages.text("Sources: " + citations, size=11, color="#596d82", gap=18)
        for visual_id in section.visual_ids:
            if visual_id not in visuals or visuals[visual_id].visual_id != visual_id:
                raise ValueError(f"unknown_report_visual:{visual_id}")
            visual = visuals[visual_id]
            selected[visual_id] = visual
            if visual.status == "rendered":
                # Verify all participating files, not merely the raster selected for display.
                for key in visual.objects:
                    _verified(visual, key)
                pages.visual(_verified(visual, "figure.png"), visual_id)
            else:
                pages.text(
                    f"{visual_id}: {visual.status}. {visual.caption} "
                    + "; ".join(visual.error_codes),
                    size=13,
                )
            pages.text(
                f"Visual source: {visual.source.artifact_id}{visual.selector}",
                size=11,
                color="#596d82",
                gap=18,
            )
    if draft.coverage:
        pages.room(100)
        pages.text("Evidence coverage", size=20, bold=True, color="#173653")
        for evidence_id, explanation in draft.coverage.items():
            pages.text(f"{evidence_id}: {explanation}", size=13)
    documents = pages.documents()
    return documents, selected


def build_report(
    draft: ReportDraft, visuals: dict[str, RenderedVisual], output_dir: Path
) -> dict[str, Any]:
    """Copy every authored statement/citation and selected visual into an immutable paginated export."""
    versions = renderer_versions() | {"report_builder": BUILD_VERSION}
    documents, selected = _compose_pages(draft, visuals)
    files: dict[str, bytes] = {}
    previews: list[str] = []
    for index, svg in enumerate(documents, 1):
        key = f"page_{index:03d}"
        files[f"{key}_svg"] = svg.encode()
        files[f"{key}_png"] = bytes(vl_convert.svg_to_png(svg, scale=1.5))
        previews.append(f"{key}_png")
    accessible = "".join(
        _accessible(visual) for visual in selected.values() if "data.json" in visual.objects
    )
    html = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" '
        f'content="width=device-width,initial-scale=1"><title>{escape(draft.title)}</title>'
        "<style>body{margin:0;background:#e9eef3;font:16px system-ui;color:#24354a}"
        ".page{max-width:850px;margin:24px auto;box-shadow:0 4px 24px #23364a1a}"
        ".page>svg{width:100%;height:auto;display:block}aside{max-width:1050px;margin:40px auto;"
        "padding:32px;background:white;overflow:auto}table{border-collapse:collapse;width:100%;"
        "margin:20px 0}th,td{text-align:left;border:1px solid #d9e2eb;padding:8px;vertical-align:top}"
        "th{background:#edf3f8}@media print{body{background:white}.page{margin:0;box-shadow:none;"
        "break-after:page}aside{break-before:page}}@page{size:850px 1100px;margin:0}</style><main>"
        + "".join(f'<section class="page">{svg}</section>' for svg in documents)
        + '</main><aside aria-label="Accessible figure data"><h2>Accessible figure data</h2>'
        + accessible
        + "</aside></html>"
    )
    files["html"] = html.encode()
    draft_hash = content_hash(draft.model_dump(mode="json"))
    identity = digest(
        {
            "draft": draft_hash,
            "visuals": {key: visual.model_dump(mode="json") for key, visual in selected.items()},
            "versions": versions,
        }
    )
    filenames = {
        key: "report.html" if key == "html" else key.rsplit("_", 1)[0] + "." + key.rsplit("_", 1)[1]
        for key in files
    }
    objects, hashes = write_objects(
        output_dir.resolve() / identity, {filenames[key]: raw for key, raw in files.items()}
    )
    return {
        "objects": {key: objects[name] for key, name in filenames.items()},
        "object_hashes": {key: hashes[name] for key, name in filenames.items()},
        "preview_keys": previews,
        "draft_hash": draft_hash,
        "versions": versions,
    }
