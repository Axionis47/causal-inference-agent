"""Render the current declared lane trajectories as SVG documentation."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

from causal_agent.common import config
from causal_agent.evals.ladder import built, knowledge_of


def render(family: str) -> str:
    spec = knowledge_of(family).SPEC
    width, row = 920, 54
    height = 90 + row * len(spec.rungs)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#faf9f7"/>',
        f'<text x="30" y="35" font-family="sans-serif" font-size="22">{html.escape(family)}: declared reasoning trajectory</text>',
    ]
    for index, rung in enumerate(spec.rungs):
        y = 58 + index * row
        color = "#f6dfab" if rung.judgement else "#cce4ee"
        parts.append(f'<rect x="30" y="{y}" width="260" height="38" rx="7" fill="{color}"/>')
        parts.append(f'<text x="42" y="{y + 25}" font-family="monospace" font-size="15">{html.escape(rung.name)}</text>')
        exits = ", ".join(spec.exits_of(rung)) + (", feasibility" if rung.stop else "")
        parts.append(f'<text x="310" y="{y + 25}" font-family="sans-serif" font-size="14">{html.escape(exits)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def write_all(out: Path = Path(config.ROOT) / "docs" / "diagrams") -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for family in built():
        p = out / f"{family}-trajectory.svg"
        p.write_text(render(family))
        paths.append(p)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(config.ROOT) / "docs" / "diagrams")
    args = parser.parse_args(argv)
    for path in write_all(args.out):
        print(path)


if __name__ == "__main__":
    main()
