"""The pictures the documentation shows, drawn from the checked-in records so they stay true to the code.

    uv run python -m causal_agent.evals.docfigs            # writes docs/diagrams/*.svg from docs/demo/
    make docfigs

Four pictures, each in a light and a dark file:

- flow: the whole desk, one card per stage, coloured code or judgement, a gate on every model call.
- lane-swimlane: the adjustment lane's stages in two bands, code above and model below, with its repair loops.
- matrix-diff: the matrix before and after the belief answers of one interview, read from `matrix.json` and the journal;
  the changed cells are outlined.
- judgement-relate: one relate call opened up, from a real run: the prompt the model got, rendered from the stored pack by the
  lane's own helpers; the answer as the frozen graph records it; the gate's checks run again on that answer.

Plain SVG strings, no plotting library, so a regeneration is byte for byte the same. The visual language is in
`docs/diagrams/README.md`: blue is code, amber is a judgement, grey is a store, a diamond is a gate, a dashed arrow is a loop.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, cast
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "docs" / "demo"
OUT = ROOT / "docs" / "diagrams"

THEMES: dict[str, dict[str, str]] = {
    "light": {
        "bg": "#ffffff",
        "ink": "#111827",
        "muted": "#6b7280",
        "hair": "#d1d5db",
        "code_fill": "#e0ecff",
        "code_stroke": "#3b82f6",
        "code_ink": "#1e3a8a",
        "judge_fill": "#fff1cc",
        "judge_stroke": "#f59e0b",
        "judge_ink": "#78350f",
        "store_fill": "#f1f5f9",
        "store_stroke": "#94a3b8",
        "store_ink": "#334155",
        "band_code": "#f5f8ff",
        "band_model": "#fffaf0",
        "fits": "#bbf7d0",
        "fits_ink": "#14532d",
        "does_not_fit": "#fecaca",
        "does_not_fit_ink": "#7f1d1d",
        "unknown": "#f3f4f6",
        "unknown_ink": "#6b7280",
        "not_needed": "#ffffff",
        "changed": "#f59e0b",
        "ok": "#15803d",
        "ok_fill": "#dcfce7",
    },
    "dark": {
        "bg": "#0d1117",
        "ink": "#e6edf3",
        "muted": "#9da7b3",
        "hair": "#30363d",
        "code_fill": "#172554",
        "code_stroke": "#60a5fa",
        "code_ink": "#dbeafe",
        "judge_fill": "#3b2a0a",
        "judge_stroke": "#f59e0b",
        "judge_ink": "#fde68a",
        "store_fill": "#161b22",
        "store_stroke": "#6e7681",
        "store_ink": "#c9d1d9",
        "band_code": "#0f1a2e",
        "band_model": "#1f1808",
        "fits": "#14532d",
        "fits_ink": "#bbf7d0",
        "does_not_fit": "#7f1d1d",
        "does_not_fit_ink": "#fecaca",
        "unknown": "#21262d",
        "unknown_ink": "#9da7b3",
        "not_needed": "#0d1117",
        "changed": "#fbbf24",
        "ok": "#3fb950",
        "ok_fill": "#12261a",
    },
}

SANS = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


# ------------------------------------------------------------------ primitives


def _svg(width: int, height: int, body: list[str], th: dict[str, str]) -> str:
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'font-family="{SANS}" font-size="13" fill="{th["ink"]}">\n'
        f'<rect width="{width}" height="{height}" fill="{th["bg"]}"/>\n'
        "<defs>"
        f'<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{th["muted"]}"/></marker>'
        "</defs>\n"
    )
    return head + "\n".join(body) + "\n</svg>\n"


def _t(
    x: float,
    y: float,
    s: str,
    th: dict[str, str],
    size: float = 13,
    *,
    anchor: str = "start",
    color: str | None = None,
    weight: str = "normal",
    mono: bool = False,
    italic: bool = False,
) -> str:
    fill = color or th["ink"]
    font = f' font-family="{MONO}"' if mono else ""
    style = ' font-style="italic"' if italic else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{fill}"{font}{style}>{escape(s)}</text>'


def _card(x: float, y: float, w: float, h: float, kind: str, th: dict[str, str], head: str, sub: str | None = None, *, r: int = 10, size: float = 14) -> str:
    out = [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" fill="{th[f"{kind}_fill"]}" stroke="{th[f"{kind}_stroke"]}" stroke-width="1.5"/>'
    ]
    cx = x + w / 2
    if sub:
        out.append(_t(cx, y + h / 2 - 3, head, th, size, anchor="middle", color=th[f"{kind}_ink"], weight="600"))
        out.append(_t(cx, y + h / 2 + 14, sub, th, size - 2.5, anchor="middle", color=th["muted"]))
    else:
        out.append(_t(cx, y + h / 2 + size / 3, head, th, size, anchor="middle", color=th[f"{kind}_ink"], weight="600"))
    return "\n".join(out)


def _gate(cx: float, cy: float, th: dict[str, str], r: float = 8) -> str:
    return (
        f'<path d="M {cx} {cy - r} L {cx + r} {cy} L {cx} {cy + r} L {cx - r} {cy} z" fill="{th["bg"]}" stroke="{th["ink"]}" stroke-width="1.6">'
        "<title>gate: code checks the answer</title></path>"
    )


def _line(x1: float, y1: float, x2: float, y2: float, th: dict[str, str], *, arrow: bool = True, dashed: bool = False, width: float = 1.5) -> str:
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    mk = ' marker-end="url(#arrow)"' if arrow else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{th["muted"]}" stroke-width="{width}"{mk}{dash}/>'


def _path(d: str, th: dict[str, str], *, dashed: bool = False) -> str:
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    return f'<path d="{d}" fill="none" stroke="{th["muted"]}" stroke-width="1.5" marker-end="url(#arrow)"{dash}/>'


def _legend(x: float, y: float, th: dict[str, str], items: list[tuple[str, str]]) -> str:
    out, cx = [], x
    for kind, label in items:
        if kind == "gate":
            out.append(_gate(cx + 8, y - 5, th, 7))
        elif kind == "loop":
            out.append(_line(cx, y - 5, cx + 22, y - 5, th, dashed=True))
        else:
            out.append(
                f'<rect x="{cx}" y="{y - 13}" width="16" height="16" rx="4" fill="{th[f"{kind}_fill"]}" stroke="{th[f"{kind}_stroke"]}" stroke-width="1.5"/>'
            )
        out.append(_t(cx + 26, y, label, th, 12, color=th["muted"]))
        cx += 26 + 6.6 * len(label) + 28
    return "\n".join(out)


def _title(body: list[str], th: dict[str, str], title: str, sub: str, x: int = 32) -> None:
    body.append(_t(x, 40, title, th, 21, weight="600"))
    body.append(_t(x, 62, sub, th, 13, color=th["muted"]))


def _wrap(s: str, width: int) -> list[str]:
    words, lines, cur = s.split(" "), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        lines.append(cur)
    return lines


def _cut(s: str, width: int) -> str:
    return s if len(s) <= width else s[: width - 1].rstrip() + "…"


# ------------------------------------------------------------------ 1. the flow


FLOW_STAGES: list[tuple[str, str, str, list[str]]] = [
    ("code", "The question", "one CSV, one causal question", []),
    ("code", "The interview", "one question per turn", ["Reader", "Explainer", "Drawer"]),
    ("store", "The memory", "fields with a source; the matrix", []),
    ("code", "The design brief", "one family stands", ["Designer"]),
    ("code", "The lane", "one run, its own process", ["the rungs", "assess", "pick", "interpret"]),
    ("code", "The chat after", "every number cited", ["Explainer", "Drawer"]),
]


def flow(th: dict[str, str]) -> str:
    cw, ch, gap, x0, y0 = 178, 78, 34, 32, 96
    width = x0 * 2 + 6 * cw + 5 * gap
    body: list[str] = []
    _title(
        body,
        th,
        "One CSV and one question in. One run of one family out.",
        "Blue is code. Every amber pill is one model call, and every one of them answers through a gate.",
    )
    y_rail = y0 + ch + 74
    body.append(_line(x0, y_rail - 30, width - x0, y_rail - 30, th, arrow=False, dashed=True, width=1))
    body.append(_t(x0, y_rail - 38, "JUDGEMENTS", th, 10.5, color=th["muted"], weight="600"))
    for i, (kind, head, sub, judges) in enumerate(FLOW_STAGES):
        x = x0 + i * (cw + gap)
        body.append(_card(x, y0, cw, ch, kind, th, head, sub))
        if i < len(FLOW_STAGES) - 1:
            body.append(_line(x + cw + 2, y0 + ch / 2, x + cw + gap - 2, y0 + ch / 2, th))
        if judges:
            cx = x + cw / 2
            body.append(_line(cx, y0 + ch, cx, y_rail - 4, th, arrow=False))
            body.append(_gate(cx, y0 + ch + 22, th))
            pw = 128
            for j, name in enumerate(judges):
                body.append(_card(cx - pw / 2, y_rail + j * 32, pw, 26, "judge", th, name, r=13, size=12.5))
    body.append(
        _t(
            x0,
            y_rail + 5 * 32 + 26,
            "A judgement is one closed question with a typed answer. Code reads the answer, checks its cites and its numbers, and decides the next step.",
            th,
            12,
            color=th["muted"],
        )
    )
    body.append(_legend(x0, y_rail + 5 * 32 + 56, th, [("code", "code"), ("judge", "judgement"), ("store", "store"), ("gate", "gate")]))
    return _svg(width, y_rail + 5 * 32 + 76, body, th)


# ------------------------------------------------------------------ 2. the lane as a swimlane


LANE_STAGES: list[tuple[str, str]] = [
    ("load", "code"),
    ("case", "code"),
    ("pair", "code"),
    ("mechanism", "judge"),
    ("time", "code"),
    ("roles", "judge"),
    ("post roles", "judge"),
    ("graph", "code"),
    ("road", "judge"),
    ("modifiers", "judge"),
    ("threats", "code"),
    ("checks", "code"),
    ("assess", "judge"),
    ("pick", "judge"),
    ("freeze", "code"),
    ("analyse", "code"),
    ("interpret", "judge"),
    ("figures", "code"),
]
LANE_QUESTION = {
    "mechanism": "what set the treatment",
    "roles": "every pre-treatment column, together",
    "post roles": "every later column, together",
    "road": "which open road",
    "modifiers": "where the effect could differ",
    "assess": "proceed, revise or stop",
    "pick": "one estimator, from a list",
    "interpret": "the answer, every number cited",
}
LANE_LOOPS: list[tuple[str, str, str]] = [
    ("assess", "graph", "revise: the graph changes, the road runs again"),
    ("analyse", "pick", "a failed fit: the next estimator"),
]


def swimlane(th: dict[str, str]) -> str:
    nw, nh, gap, x0 = 72, 46, 12, 96
    width = x0 + len(LANE_STAGES) * (nw + gap) + 20
    y_code, y_model = 210, 340
    pos = {name: x0 + i * (nw + gap) for i, (name, _) in enumerate(LANE_STAGES)}
    body: list[str] = []
    _title(
        body,
        th,
        "The adjustment lane: fourteen stages, five judgements, three loops",
        "Code decides every route. The model answers five closed questions, each through a gate. A loop runs only on a failure fact.",
    )
    body.append(f'<rect x="20" y="{y_code - 22}" width="{width - 40}" height="{nh + 44}" rx="8" fill="{th["band_code"]}"/>')
    body.append(f'<rect x="20" y="{y_model - 22}" width="{width - 40}" height="{nh + 74}" rx="8" fill="{th["band_model"]}"/>')
    body.append(_t(34, y_code + nh / 2 + 4, "CODE", th, 11, color=th["code_stroke"], weight="700"))
    body.append(_t(34, y_model + nh / 2 + 4, "MODEL", th, 11, color=th["judge_stroke"], weight="700"))
    for i, (name, kind) in enumerate(LANE_STAGES):
        x = pos[name]
        body.append(_card(x, y_code, nw, nh, "code", th, name, size=12.5, r=8))
        if i < len(LANE_STAGES) - 1:
            body.append(_line(x + nw + 1, y_code + nh / 2, x + nw + gap - 1, y_code + nh / 2, th))
        if kind == "judge":
            cx = x + nw / 2
            body.append(_card(x - 6, y_model, nw + 12, nh, "judge", th, name, size=12.5, r=8))
            body.append(_line(cx - 11, y_code + nh, cx - 11, y_model - 1, th))
            body.append(_line(cx + 11, y_model, cx + 11, y_code + nh + 1, th))
            body.append(_gate(cx + 11, y_code + nh + 28, th, 7))
            for j, piece in enumerate(_wrap(LANE_QUESTION[name], 15)):
                body.append(_t(cx, y_model + nh + 18 + j * 14, piece, th, 11, anchor="middle", color=th["muted"]))
    for k, (src, dst, label) in enumerate(LANE_LOOPS):
        xs, xd = pos[src] + nw / 2, pos[dst] + nw / 2
        top = y_code - 44 - k * 34
        apex = 0.25 * (y_code - 22) + 0.75 * top
        body.append(_path(f"M {xs} {y_code - 22} C {xs} {top} {xd} {top} {xd} {y_code - 23}", th, dashed=True))
        body.append(_t((xs + xd) / 2, apex - 8, label, th, 11, anchor="middle", color=th["muted"]))
    body.append(
        _legend(
            32,
            y_model + nh + 90,
            th,
            [
                ("code", "code: a fact computed"),
                ("judge", "judgement: one model call"),
                ("gate", "gate: the answer checked"),
                ("loop", "loop, bounded in code"),
            ],
        )
    )
    return _svg(width, y_model + nh + 112, body, th)


# ------------------------------------------------------------------ 3. the matrix, before and after


_CHANGE = re.compile(r"([a-z_]+)\.([a-z_]+): (\w+) -> (\w+)")
GLYPH = {"fits": "✓", "does_not_fit": "×", "unknown": "?", "not_needed": ""}


def _before_grid(matrix: dict[str, Any], journal: list[dict[str, Any]], since: int) -> dict[str, dict[str, str]]:
    """The grid as it stood at memory version `since`: today's cells, with every `fit` step after `since` undone in reverse."""
    grid = {fam: {kind: c["value"] for kind, c in row.items()} for fam, row in matrix["cells"].items()}
    steps = [s for s in journal if s["kind"] == "fit" and s["memory_version"] > since]
    for s in reversed(steps):
        for fam, kind, before, _after in _CHANGE.findall(s["note"]):
            if fam in grid and kind in grid[fam]:
                grid[fam][kind] = before
    return grid


def _verdict(grid: dict[str, dict[str, str]]) -> tuple[dict[str, str], list[str], bool]:
    struck = {}
    for fam, row in grid.items():
        bad = [k for k, v in row.items() if v == "does_not_fit"]
        if bad:
            struck[fam] = f"{bad[0]} does not fit"
    surviving = [f for f in grid if f not in struck]
    ready = bool(surviving) and not any(v == "unknown" for f in surviving for v in grid[f].values())
    return struck, surviving, ready


def matrix_diff(matrix_path: Path, journal_path: Path, since: int, th: dict[str, str]) -> str:
    matrix = json.loads(matrix_path.read_text())
    journal = [json.loads(line) for line in journal_path.read_text().splitlines() if line.strip()]
    after = {fam: {kind: c["value"] for kind, c in row.items()} for fam, row in matrix["cells"].items()}
    before = _before_grid(matrix, journal, since)
    families = list(after)
    kinds = list(next(iter(after.values())))
    changed = {(f, k) for f in families for k in kinds if before[f][k] != after[f][k]}
    fits = [s for s in journal if s["kind"] == "fit" and s["memory_version"] > since]
    moved_at = {s["memory_version"] for s in fits}
    claims = [s for s in journal if s["kind"] == "claim" and s["memory_version"] in moved_at]

    cell, label_w, top = 34, 168, 172
    grid_w, grid_h = len(families) * cell, len(kinds) * cell
    gap = 200
    x_left = 32 + label_w
    x_right = x_left + grid_w + gap + label_w
    width = x_right + grid_w + 32
    body: list[str] = []
    _title(
        body,
        th,
        f"The matrix before and after {len(claims)} answers",
        "Families across, claim kinds down. The first cell that does not fit strikes a family. Ready: nothing a surviving family still needs is unknown.",
    )

    def grid_at(x0: int, grid: dict[str, dict[str, str]], title: str, outline: bool) -> None:
        struck, surviving, ready = _verdict(grid)
        body.append(_t(x0 - label_w, 96, title, th, 15, weight="600"))
        for j, fam in enumerate(families):
            cx = x0 + j * cell + cell / 2
            color = th["muted"] if fam in struck else th["ink"]
            body.append(f'<text transform="translate({cx + 5:.1f},{top - 8}) rotate(-42)" font-size="12" fill="{color}">{escape(fam.replace("_", " "))}</text>')
        for i, kind in enumerate(kinds):
            y = top + i * cell
            body.append(_t(x0 - 10, y + cell / 2 + 4, kind.replace("_", " "), th, 12, anchor="end", color=th["muted"]))
            for j, fam in enumerate(families):
                v = grid[fam][kind]
                x = x0 + j * cell
                body.append(
                    f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="5" fill="{th[v]}" stroke="{th["bg"]}" stroke-width="3">'
                    f"<title>{escape(f'{fam}.{kind}: {v}')}</title></rect>"
                )
                if v == "not_needed":
                    body.append(f'<circle cx="{x + cell / 2}" cy="{y + cell / 2}" r="1.5" fill="{th["hair"]}"/>')
                else:
                    body.append(_t(x + cell / 2, y + cell / 2 + 5, GLYPH[v], th, 15, anchor="middle", color=th[f"{v}_ink"], weight="700"))
                if outline and (fam, kind) in changed:
                    body.append(
                        f'<rect x="{x + 1.5}" y="{y + 1.5}" width="{cell - 3}" height="{cell - 3}" rx="5" fill="none" stroke="{th["changed"]}" stroke-width="2.5"/>'
                    )
        yv = top + grid_h + 22
        for j, fam in enumerate(families):
            cx = x0 + j * cell + cell / 2
            if fam in struck:
                body.append(_t(cx, yv, "×", th, 14, anchor="middle", color=th["does_not_fit_ink"], weight="700"))
            else:
                body.append(f'<circle cx="{cx}" cy="{yv - 4}" r="5" fill="{th["fits"]}" stroke="{th["fits_ink"]}"/>')
        body.append(_t(x0 - 10, yv, "in play", th, 12, anchor="end", color=th["muted"]))
        verdict = f"{'ready' if ready else 'not ready'} · in play: {', '.join(f.replace('_', ' ') for f in surviving)}"
        pill_w = 7.2 * len(verdict) + 24
        fill, ink = (th["ok_fill"], th["ok"]) if ready else (th["unknown"], th["unknown_ink"])
        body.append(f'<rect x="{x0 - label_w}" y="{yv + 18}" width="{pill_w:.0f}" height="26" rx="13" fill="{fill}" stroke="{ink}"/>')
        body.append(_t(x0 - label_w + 12, yv + 35, verdict, th, 12, color=ink, weight="600"))

    grid_at(x_left, before, f"Before · v{since}", False)
    grid_at(x_right, after, f"After · v{matrix['memory_version']}, {len(claims)} answers later", True)

    # the answers between the grids, one line each, from the journal's claim steps
    mx = x_left + grid_w + 24
    my = top + 20
    body.append(_line(mx, top + grid_h / 2, x_right - label_w - 8, top + grid_h / 2, th, width=2))
    body.append(_t(mx, my - 40, "THE PERSON ANSWERED", th, 10.5, color=th["muted"], weight="600"))
    for k, s in enumerate(claims):
        turn = next((r for r in s["read"] if r.startswith("user:turn:")), "").replace("user:turn:", "turn ")
        body.append(_t(mx, my + k * 34 - 16, turn, th, 11, color=th["muted"]))
        body.append(_t(mx, my + k * 34, _cut(s["note"], 26), th, 11.5, mono=True))
    # what those answers moved
    ly = top + grid_h + 96
    body.append(_t(32, ly, "WHAT MOVED", th, 10.5, color=th["muted"], weight="600"))
    moves = [f"{fam}.{kind}: {b} → {a}" for s in fits for fam, kind, b, a in _CHANGE.findall(s["note"])]
    cols = 2
    per = -(-len(moves) // cols)
    for k, m in enumerate(moves):
        col, row = divmod(k, per)
        body.append(_t(32 + col * 400, ly + 18 + row * 16, m, th, 11.5, mono=True, color=th["ink"]))
    y_leg = ly + 18 + per * 16 + 22
    lx = 32.0
    for v, label in [("fits", "fits"), ("does_not_fit", "does not fit"), ("unknown", "unknown"), ("not_needed", "not needed by this family")]:
        body.append(f'<rect x="{lx:.1f}" y="{y_leg - 13}" width="16" height="16" rx="4" fill="{th[v]}" stroke="{th["hair"]}"/>')
        body.append(_t(lx + 24, y_leg, label, th, 12, color=th["muted"]))
        lx += 24 + 6.6 * len(label) + 28
    body.append(f'<rect x="{lx:.1f}" y="{y_leg - 13}" width="16" height="16" rx="4" fill="none" stroke="{th["changed"]}" stroke-width="2.5"/>')
    body.append(_t(lx + 24, y_leg, "changed by these answers", th, 12, color=th["muted"]))
    body.append(
        _t(
            32,
            y_leg + 22,
            f"Read from {matrix_path.relative_to(ROOT)} and {journal_path.relative_to(ROOT)}. The chat cites a cell as matrix:<family>.<kind>.",
            th,
            11,
            color=th["muted"],
        )
    )
    return _svg(width, y_leg + 40, body, th)


# ------------------------------------------------------------------ 4. one judgement, opened up


def _relation_from_design(design: dict[str, Any], column: str) -> tuple[dict[str, bool], list[str], str]:
    """The four claims for one column as the frozen graph records them, the cites on its edges, and how the graph placed it."""
    g = design["graph"]
    t, y = g["treatment"], g["outcome"]
    edges = [e for e in g["edges"] if e["src"] == column]
    excluded = next((x for x in g["excluded"] if x["column"] == column), None)
    claims = {
        "affects_treatment": any(e["dst"] == t for e in edges),
        "affects_outcome": any(e["dst"] == y for e in edges),
        "affected_by_treatment": bool(excluded and "after" in excluded["why"]),
        "is_outcome_measure": bool(excluded and "measurement" in excluded["why"]),
    }
    cites = sorted({c for e in edges for c in e["cites"]})
    if excluded:
        placed = f"excluded: {excluded['why']}"
    elif claims["affects_treatment"] and claims["affects_outcome"]:
        placed = f"a confounder: {column} → {t} and {column} → {y}"
    elif claims["affects_outcome"]:
        placed = f"an outcome driver: {column} → {y}"
    else:
        placed = "kept out of the graph"
    return claims, cites, placed


KEEP_WHOLE = {"QUESTION", "THE COLUMNS TO PLACE (1)"}
KEEP_HEAD = 2


def _abridge(prompt: str) -> list[tuple[str, str]]:
    """The prompt as (kind, line) pairs: a header, a kept line, or an ellipsis saying how many lines were left out."""
    out: list[tuple[str, str]] = []
    header: str | None = None
    buf: list[str] = []

    def flush() -> None:
        lines = [line for line in buf if line.strip()]
        if header is None:
            out.extend(("line", line) for line in lines)
            return
        out.append(("header", header))
        if header in KEEP_WHOLE or header.startswith("SETTLED BY") or header.startswith("THE LAST READING") or len(lines) <= KEEP_HEAD + 1:
            out.extend(("line", line) for line in lines)
        else:
            out.extend(("line", line) for line in lines[:KEEP_HEAD])
            out.append(("more", f"… {len(lines) - KEEP_HEAD} more lines"))

    for line in prompt.splitlines():
        is_header = bool(line) and line == line.upper() and line[0].isalpha()
        if is_header or line.startswith("SETTLED BY") or line.startswith("THE LAST READING"):
            flush()
            header, buf = line.split(" (")[0], []
        elif line.startswith("Answer the four questions"):
            flush()
            header, buf = None, [line]
        else:
            buf.append(line)
    flush()
    return out


def opened_judgement(run_dir: Path, column: str, th: dict[str, str]) -> str:
    import causal_agent.families.registry  # noqa: F401  (registers every family's block before a pack is read)
    from causal_agent.common.contracts import Cited, Handoff
    from causal_agent.families.adjustment.lane import reason as N
    from causal_agent.families.adjustment.lane import prompts as P
    from causal_agent.families.adjustment.lane.contracts import Relation
    from causal_agent.lane import case as C
    from causal_agent.lane import nodes as L
    from causal_agent.lane import verify as V
    from causal_agent.lane.state import LaneState

    h = Handoff.model_validate_json((run_dir / "handoff.json").read_text())
    design = json.loads((run_dir / "design.json").read_text())
    report = (run_dir / "report.md").read_text()
    state = cast(LaneState, {"handoff": h, "question": h.question, "columns": {c.key: c.name for c in h.columns}, "dataset": h.pack_name})
    case = C.weigh(h, N.load_beliefs())
    t, y = design["graph"]["treatment"], design["graph"]["outcome"]
    settled = N._settled_text(h, column, case) + N._drafted_text(h, column, case)
    prompt = P.ROLES_USER.format(
        question=h.question,
        frame=L.frame_text(state),
        treatment_card=L.card(h, t),
        outcome_card=L.card(h, y),
        count=1,
        columns=f"COLUMN {column!r}\n{L.card(h, column)}\n{settled}",
        errors="",
    )
    claims, cites, placed = _relation_from_design(design, column)
    reasons = [Cited(reason=f"as the frozen graph records it: {placed}", cites=cites)] if any(claims.values()) else []
    r = Relation.model_validate({"column": column, "reasons": reasons, **claims})

    # the gate, run again on this answer: the same checks verify_graph makes per column
    checks: list[tuple[str, list[str]]] = []
    n_true = sum(claims.values())
    checks.append(("a claim marked true carries a reason", ["claims marked true but no reasons given"] if n_true and not r.reasons else []))
    checks.append(("every reason cites an address", [e for reason in r.reasons for e in (["a reason has no citation"] if not reason.cites else [])]))
    checks.append(("every cite resolves in the pack", V.cites_resolve(cites, h)))
    checks.append(
        (
            "not both feeding the treatment and changed by it",
            ["cannot both feed the treatment and be changed by it"] if r.affects_treatment and r.affected_by_treatment else [],
        )
    )
    settled_claims, _ = N.settled_claims(h, column, case)
    rules = [rule for rule in N.CONTRADICTION_RULES if rule[0] not in settled_claims]
    checks.append(("no claim contradicts a fact the pack settled", V.contradictions(dict(claims), column, case, rules, cites, h)))
    drafted, _ = N.drafted_claims(h, column, case)
    checks.append(("no unexplained departure from the last reading", V.departures(dict(claims), drafted, r.departures, h)))
    failed = sum(1 for _, e in checks if e)

    m = re.search(rf"\[relate:{re.escape(column)}\] (.+)", report)
    thought = (m.group(1).strip() if m else "")[:460]

    lw, rw, pad, lh = 660, 540, 32, 17
    lines = _abridge(prompt)
    left_h = 70 + len(lines) * lh + 20
    width = lw + rw + pad * 3
    body: list[str] = []
    _title(
        body,
        th,
        "One judgement, opened up",
        f"roles · the column '{column}' · from the run under {run_dir.relative_to(ROOT)}. One column of the roles rung; code checks what comes back.",
    )

    def panel(x: float, y: float, w: float, hgt: float, kind: str, n: str, head: str) -> None:
        body.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{hgt}" rx="12" fill="{th[f"{kind}_fill"]}" stroke="{th[f"{kind}_stroke"]}" stroke-width="1.5"/>'
        )
        body.append(f'<circle cx="{x + 24}" cy="{y + 24}" r="12" fill="{th[f"{kind}_stroke"]}"/>')
        body.append(_t(x + 24, y + 28.5, n, th, 13, anchor="middle", color=th["bg"], weight="700"))
        body.append(_t(x + 46, y + 29, head, th, 14, weight="600", color=th[f"{kind}_ink"]))

    # 1 · given
    y0 = 90
    panel(pad, y0, lw, left_h, "judge", "1", "What the model was given")
    body.append(
        _t(
            pad + 46,
            y0 + 48,
            "the user prompt, rendered again from the stored pack by the lane's own helpers; long sections cut to their first lines",
            th,
            11,
            color=th["muted"],
        )
    )
    yy = y0 + 76
    for kind, line in lines:
        if kind == "header":
            body.append(_t(pad + 20, yy, line, th, 11, weight="700", color=th["judge_ink"]))
        elif kind == "more":
            body.append(_t(pad + 20, yy, line, th, 11, color=th["muted"], italic=True))
        else:
            body.append(_t(pad + 20, yy, _cut(line, 86), th, 11, mono=True))
        yy += lh

    # 2 · answered
    rx = pad * 2 + lw
    ah = 214
    panel(rx, y0, rw, ah, "code", "2", "What came back")
    body.append(_t(rx + 46, y0 + 48, "a Relation, as structured output; here as the frozen graph records it", th, 11, color=th["muted"]))
    yy = y0 + 82
    for name, v in claims.items():
        fill, ink, word = (th["fits"], th["fits_ink"], "yes") if v else (th["unknown"], th["unknown_ink"], "no")
        body.append(_t(rx + 20, yy, name, th, 12.5, mono=True))
        body.append(f'<rect x="{rx + 250}" y="{yy - 13}" width="40" height="19" rx="9.5" fill="{fill}" stroke="{ink}"/>')
        body.append(_t(rx + 270, yy + 1, word, th, 11.5, anchor="middle", color=ink, weight="600"))
        yy += 24
    body.append(_t(rx + 20, yy + 2, f"cites  {', '.join(cites) or 'none'}", th, 12, mono=True, color=th["code_ink"]))
    body.append(_t(rx + 20, yy + 20, _cut(f"placed in the graph as {placed}", 78), th, 11.5, color=th["muted"]))

    # 3 · gated
    gy = y0 + ah + 20
    gh = 76 + 24 * len(checks) + 30
    panel(rx, gy, rw, gh, "store", "3", "The gate, run again on that answer")
    body.append(
        _t(
            rx + 46,
            gy + 48,
            "the checks verify_graph makes for every column; a failure is re-asked with the errors",
            th,
            11,
            color=th["muted"],
        )
    )
    yy = gy + 80
    for label, errs in checks:
        fill, ink, mark = (th["does_not_fit"], th["does_not_fit_ink"], "×") if errs else (th["ok_fill"], th["ok"], "✓")
        body.append(f'<circle cx="{rx + 30}" cy="{yy - 4}" r="9" fill="{fill}" stroke="{ink}"/>')
        body.append(_t(rx + 30, yy, mark, th, 12, anchor="middle", color=ink, weight="700"))
        body.append(_t(rx + 50, yy, label, th, 12.5))
        yy += 24
    verdict = f"rejected: {failed} checks failed" if failed else "passed: the graph takes it"
    fill, ink = (th["does_not_fit"], th["does_not_fit_ink"]) if failed else (th["ok_fill"], th["ok"])
    body.append(f'<rect x="{rx + 20}" y="{yy - 6}" width="{7 * len(verdict) + 24:.0f}" height="26" rx="13" fill="{fill}" stroke="{ink}"/>')
    body.append(_t(rx + 32, yy + 11, verdict, th, 12, color=ink, weight="600"))

    # the model's own words
    ty = gy + gh + 26
    if thought:
        pieces = _wrap(thought + " …", 82)
        body.append(_t(rx, ty, "THE MODEL'S OWN SUMMARY OF ITS REASONING", th, 10.5, color=th["muted"], weight="600"))
        body.append(_t(rx, ty + 14, "kept in report.md as debug only; never read by code", th, 10.5, color=th["muted"]))
        for k, piece in enumerate(pieces):
            body.append(_t(rx, ty + 36 + k * 15, piece, th, 11.5, color=th["muted"], italic=True))
        ty += 36 + 15 * len(pieces)
    return _svg(width, max(y0 + left_h, ty) + 30, body, th)


# ------------------------------------------------------------------ 5. the desk graph, one band per phase


Node = tuple[str, str, str]  # name, kind (code | judge), an optional flag: "pause" for an interrupt

DESK_BANDS: list[tuple[str, str, list[Node], list[tuple[int, int, str]]]] = [
    (
        "The question first",
        "asked before anything about the data; one judgement reads it, code validates it against the file",
        [("load", "code", ""), ("ask_question", "code", "pause"), ("mine", "judge", ""), ("read_question", "judge", ""), ("check", "code", "")],
        [(3, 1, "invalid: asked again with the test it failed")],
    ),
    (
        "The interview, one turn at a time",
        "the data checks, the matrix, the next thing to ask; the Reader over each answer; a picture or an answer before the next ask",
        [
            ("check", "code", ""),
            ("probe_fit", "code", ""),
            ("ask", "code", ""),
            ("listen", "code", "pause"),
            ("infer", "judge", ""),
            ("explore", "judge", ""),
            ("explain", "judge", ""),
        ],
        [(6, 0, "every turn returns to the checks"), (4, 0, "")],
    ),
    (
        "The routing and the run, by code",
        "the matrix, one judgement only when several families stand, the gate, the brief, the pack, the lane in its own process",
        [
            ("fit", "code", ""),
            ("decide", "judge", ""),
            ("gate", "code", ""),
            ("design", "judge", ""),
            ("handoff", "code", ""),
            ("run", "code", ""),
            ("brief", "code", ""),
        ],
        [(2, 1, "retry, three at most"), (5, 3, "")],
    ),
    (
        "The chat after",
        "one routed reply per message: answer, revise, what if, a new question, a picture, done",
        [
            ("talk", "code", "pause"),
            ("turn", "judge", ""),
            ("answer", "code", ""),
            ("revise", "code", ""),
            ("what_if", "code", ""),
            ("requestion", "code", ""),
            ("explore_after", "judge", ""),
        ],
        [],
    ),
]
DESK_CROSS = [
    "listen · run  →  fit",
    "run · the lane asks back  →  listen",
    "revise  →  check",
    "what_if  →  fit, on a copy",
    "requestion  →  read_question",
]


def desk_graph(th: dict[str, str]) -> str:
    nw, nh, gap, x0 = 112, 44, 26, 32
    band_h, band_gap, y0 = 150, 22, 96
    width = x0 + 7 * (nw + gap) + 250
    body: list[str] = []
    _title(
        body,
        th,
        "The desk: one graph, four phases, three places it waits",
        "Blue nodes are code. Amber nodes call a model, each through a gate. A pause mark is an interrupt: the desk waits for the person.",
    )
    for b, (head, sub, nodes, loops) in enumerate(DESK_BANDS):
        by = y0 + b * (band_h + band_gap)
        band = th["band_code"] if b % 2 == 0 else th["band_model"]
        body.append(f'<rect x="20" y="{by}" width="{width - 40}" height="{band_h}" rx="10" fill="{band}"/>')
        body.append(_t(x0, by + 24, head, th, 14, weight="600"))
        body.append(_t(x0, by + 42, sub, th, 11.5, color=th["muted"]))
        ny = by + 66
        for i, (name, kind, flag) in enumerate(nodes):
            x = x0 + i * (nw + gap)
            body.append(_card(x, ny, nw, nh, kind, th, name, size=12.5, r=8))
            if flag == "pause":
                body.append(f'<rect x="{x + nw - 22}" y="{ny + 6}" width="4" height="12" rx="1" fill="{th["ink"]}"/>')
                body.append(f'<rect x="{x + nw - 14}" y="{ny + 6}" width="4" height="12" rx="1" fill="{th["ink"]}"/>')
            if kind == "judge":
                body.append(_gate(x + nw - 2, ny - 1, th, 6))
            if i < len(nodes) - 1:
                body.append(_line(x + nw + 1, ny + nh / 2, x + nw + gap - 1, ny + nh / 2, th))
        for src, dst, label in loops:
            xs, xd = x0 + src * (nw + gap) + nw / 2, x0 + dst * (nw + gap) + nw / 2
            top = ny + nh + 30
            apex = 0.25 * (ny + nh) + 0.75 * top
            body.append(_path(f"M {xs} {ny + nh} C {xs} {top} {xd} {top} {xd} {ny + nh + 1}", th, dashed=True))
            if label:
                body.append(_t((xs + xd) / 2, apex + 13, label, th, 10.5, anchor="middle", color=th["muted"]))
    cx = x0 + 7 * (nw + gap) + 10
    body.append(_t(cx, y0 + 24, "BETWEEN THE PHASES", th, 10.5, color=th["muted"], weight="600"))
    for k, line in enumerate(DESK_CROSS):
        body.append(_t(cx, y0 + 46 + k * 18, line, th, 11.5, mono=True))
    body.append(_t(cx, y0 + 46 + len(DESK_CROSS) * 18 + 14, "Every other route is a Command the node", th, 11, color=th["muted"]))
    body.append(_t(cx, y0 + 46 + len(DESK_CROSS) * 18 + 28, "returns from a typed result, never a guess.", th, 11, color=th["muted"]))
    y_leg = y0 + 4 * (band_h + band_gap) + 14
    body.append(_legend(x0, y_leg, th, [("code", "code"), ("judge", "judgement"), ("gate", "gate"), ("loop", "loop or return")]))
    body.append(
        f'<rect x="{x0 + 470}" y="{y_leg - 13}" width="4" height="12" rx="1" fill="{th["ink"]}"/><rect x="{x0 + 478}" y="{y_leg - 13}" width="4" height="12" rx="1" fill="{th["ink"]}"/>'
    )
    body.append(_t(x0 + 492, y_leg, "interrupt: the desk waits for the person", th, 12, color=th["muted"]))
    return _svg(width, y_leg + 24, body, th)


# ------------------------------------------------------------------ 6. the drawing tool as a sequence


DRAW_COLS = [
    ("person", "store"),
    ("Reader or Explainer", "judge"),
    ("desk: draw node", "code"),
    ("drawing model", "judge"),
    ("sandbox", "code"),
    ("store", "store"),
]
DRAW_STEPS: list[tuple[int, int, str, str]] = [
    (0, 1, "show me math score by lunch", ""),
    (1, 2, 'Reading.draw = "math score by lunch"', "a judgement: the request read out of the message"),
    (2, 3, "DrawRequest: the ask, what is known, the columns", ""),
    (3, 2, "DrawCode: plan, script, caption, fact names", "a judgement: one script, nothing else"),
    (2, 2, "gate 1: the script parses", ""),
    (2, 4, "run code.py with DATA_CSV", "subprocess or container, 60 s, network off in the container"),
    (4, 2, "figure.png, facts.json", ""),
    (2, 2, "gate 2: a PNG, facts all numbers, every promised name present", "three tries, then a Decline and no folder"),
    (2, 5, "artifact.json, code.py, figure.png, facts.json", "viz/pre/<id>/ before a run, designs/<n>/viz/<id>/ after"),
    (2, 0, "the caption, cited as artifact:<id>, then the next question", "the chat after a run can quote artifact:<id>.<fact>"),
]


def draw_sequence(th: dict[str, str]) -> str:
    col_w, x0, y0, step_h = 196, 60, 130, 50
    xs = [x0 + i * col_w + col_w / 2 for i in range(len(DRAW_COLS))]
    height = y0 + len(DRAW_STEPS) * step_h + 70
    width = x0 * 2 + len(DRAW_COLS) * col_w
    body: list[str] = []
    _title(
        body,
        th,
        "The drawing tool, one request end to end",
        "Nothing here is a tool call or a keyword match: two judgements and two gates around one sandboxed script.",
    )
    for i, (name, kind) in enumerate(DRAW_COLS):
        body.append(_card(xs[i] - 78, 80, 156, 34, kind, th, name, size=12.5, r=17))
        body.append(f'<line x1="{xs[i]}" y1="114" x2="{xs[i]}" y2="{height - 40}" stroke="{th["hair"]}" stroke-width="1.5" stroke-dasharray="3 5"/>')
    for k, (a, b, label, note) in enumerate(DRAW_STEPS):
        y = y0 + k * step_h + 20
        if a == b:
            body.append(_gate(xs[a], y, th, 9))
            body.append(_t(xs[a] + 16, y + 4, label, th, 11.5, weight="600"))
            if note:
                body.append(_t(xs[a] + 16, y + 18, note, th, 10.5, color=th["muted"]))
            continue
        x1, x2 = xs[a], xs[b]
        d = 1 if x2 > x1 else -1
        body.append(_line(x1 + d * 6, y, x2 - d * 6, y, th, width=1.6))
        mid = (x1 + x2) / 2
        body.append(_t(mid, y - 6, label, th, 11.5, anchor="middle", weight="600"))
        if note:
            body.append(_t(mid, y + 14, note, th, 10.5, anchor="middle", color=th["muted"]))
    body.append(_legend(x0, height - 12, th, [("code", "code"), ("judge", "judgement"), ("store", "person or store"), ("gate", "gate")]))
    return _svg(width, height, body, th)


# ------------------------------------------------------------------ 7. what each design decision rests on


CLAIM_GROUPS: list[tuple[str, str]] = [
    ("assignment", "who got it, and how that was decided"),
    ("change", "what the change was, and when"),
    ("rows", "what one row is, which rows are in"),
    ("columns", "each column: what it records, when it was set"),
    ("beliefs", "what the design would bet on"),
    ("data", "a check on the file itself"),
]
_ROWS = {"grain", "sampling", "missing"}
_BELIEFS = {"unobserved", "exclusion", "spillover", "trend_continues", "cutoff_only", "mediator"}


def _group_of(address: str) -> str:
    if address.startswith("probe:"):
        return "data"
    if address.startswith("col:"):
        return "columns"
    kind = address.split(":", 1)[1].split(".", 1)[0]
    if kind == "assignment":
        return "assignment"
    if kind == "change":
        return "change"
    if kind in _ROWS:
        return "rows"
    if kind in _BELIEFS:
        return "beliefs"
    return "rows"


def decisions(th: dict[str, str]) -> str:
    from causal_agent.families import registry as R

    fams = [f for f in R.knowledge() if f.decisions]
    label_w, col_w, row_h, x0, top = 344, 84, 30, 32, 150
    width = x0 + label_w + len(CLAIM_GROUPS) * col_w + 32
    body: list[str] = []
    _title(
        body,
        th,
        "What each design decision rests on",
        "Six kinds of claim carry every decision a design makes. The interview asks for a claim only because a decision needs it.",
    )
    for j, (key, words) in enumerate(CLAIM_GROUPS):
        cx = x0 + label_w + j * col_w + col_w / 2
        body.append(_t(cx, top - 44, key, th, 12.5, anchor="middle", weight="600"))
        for k, piece in enumerate(_wrap(words, 16)[:3]):
            body.append(_t(cx, top - 28 + k * 12, piece, th, 9.5, anchor="middle", color=th["muted"]))
    y = top
    for fi, fam in enumerate(fams):
        band = th["band_code"] if fi % 2 == 0 else th["band_model"]
        h = row_h * len(fam.decisions) + 34
        body.append(f'<rect x="20" y="{y}" width="{width - 40}" height="{h}" rx="8" fill="{band}"/>')
        body.append(_t(x0, y + 22, fam.name.replace("_", " "), th, 13, weight="700"))
        body.append(_t(x0 + 8 * len(fam.name) + 10, y + 22, _cut(f"bets on: {fam.assumes}", 100), th, 11, color=th["muted"]))
        ry = y + 34
        for d in fam.decisions:
            cy = ry + row_h / 2
            body.append(_t(x0, cy + 4, _cut(d.asks, 56), th, 12))
            by_group: dict[str, list[str]] = {}
            for a in d.rests_on:
                by_group.setdefault(_group_of(a), []).append(a)
            for j, (key, _) in enumerate(CLAIM_GROUPS):
                cx = x0 + label_w + j * col_w + col_w / 2
                if key in by_group:
                    fill = th["judge_fill"] if key == "beliefs" else th["code_fill"]
                    stroke = th["judge_stroke"] if key == "beliefs" else th["code_stroke"]
                    ink = th["judge_ink"] if key == "beliefs" else th["code_ink"]
                    n = len(by_group[key])
                    body.append(
                        f'<circle cx="{cx}" cy="{cy}" r="10" fill="{fill}" stroke="{stroke}" stroke-width="1.5"><title>{escape(", ".join(by_group[key]))}</title></circle>'
                    )
                    body.append(_t(cx, cy + 4, str(n), th, 11, anchor="middle", color=ink, weight="700"))
                else:
                    body.append(f'<circle cx="{cx}" cy="{cy}" r="1.5" fill="{th["hair"]}"/>')
            ry += row_h
        y += h + 14
    body.append(
        _t(x0, y + 12, "A number is how many claims of that kind the decision rests on; amber is a belief only the person can give.", th, 11, color=th["muted"])
    )
    body.append(_t(x0, y + 28, "Read from each family's family.yaml, the same file the Designer fills and the matrix reads.", th, 11, color=th["muted"]))
    return _svg(width, y + 44, body, th)


# ------------------------------------------------------------------ all of them


def write_all(out: Path = OUT, demo: Path = DEMO) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, th in THEMES.items():
        pictures = {
            "flow": flow(th),
            "lane-swimlane": swimlane(th),
            "matrix-diff": matrix_diff(demo / "interview" / "matrix.json", demo / "interview" / "journal.jsonl", 35, th),
            "judgement-relate": opened_judgement(demo / "run", "gender", th),
            "desk-graph": desk_graph(th),
            "draw-sequence": draw_sequence(th),
            "decisions": decisions(th),
        }
        for stem, svg in pictures.items():
            p = out / f"{stem}-{name}.svg"
            p.write_text(svg)
            written.append(p)
    return written


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--demo", default=str(DEMO))
    args = ap.parse_args(argv)
    for p in write_all(Path(args.out), Path(args.demo)):
        print(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p)


if __name__ == "__main__":
    main()
