"""The ladder tables: docs/ladders/<family>.md for every built lane, written from the lane's declarations (rungs.yaml, space.yaml,
checks.yaml, endpoints.yaml and the catalogue), never by hand. Three tables per lane: the rungs in order, the space, and the end
states the declarations allow (the cells a design can reach, the questions a rung can send back, the typed stops each rung
can raise). `make ladder` writes them; CI fails when the files differ from what the declarations say.

    uv run python -m causal_agent.evals.ladder            # write every lane's table
    uv run python -m causal_agent.evals.ladder --check    # exit 1 when a file is stale
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from causal_agent.common import config
from causal_agent.families import registry as R
from causal_agent.lane import rungs as RG
from causal_agent.lane.space import Space

OUT = Path(config.ROOT) / "docs" / "ladders"


def knowledge_of(family: str) -> ModuleType:
    """A built lane's knowledge package, by the one convention every family follows."""
    return importlib.import_module(f"causal_agent.families.{family}.lane.knowledge")


def built() -> list[str]:
    return [n for n, f in R.REGISTRY.items() if f.built and f.lane is not None]


def _cell(s: Any) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def _table(header: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _rungs_table(spec: RG.LadderSpec, space: Space | None) -> str:
    decides: dict[str, list[str]] = {}
    for name, d in (space.dimensions if space else {}).items():
        decides.setdefault(d.decided_by, []).append(name)
    rows = []
    for r in spec.rungs:
        how = "; ".join(f"{k}: {v}" for k, v in r.decider.items())
        goes = ", ".join(spec.exits_of(r)) + (" · stop" if r.stop else "")
        rows.append(
            [
                r.name + (" (worker)" if r.worker else ""),
                r.level,
                how,
                ", ".join(decides.get(r.name, [])) or "",
                goes,
                "" if r.budget is None else str(r.budget),
                ", ".join(r.asks) or "",
            ]
        )
    return _table(["rung", "level", "decided by", "fixes", "goes on to", "budget", "may ask"], rows)


def _space_table(space: Space) -> str:
    rows = []
    for name, d in space.dimensions.items():
        values = ", ".join(str(v) for v in d.values) if d.values is not None else f"{d.range[0]} to {d.range[1]}" if d.range else ""
        unsupported = "; ".join(f"{k}: {v}" for k, v in d.unsupported.items())
        rows.append([name, values + (" (soft)" if d.soft else ""), d.decided_by, d.read_from or "", unsupported])
    for u in space.unsupported:
        rows.append(["+".join(f"{k}={v}" for k, v in u.when.items()), "", "", "", u.reason])
    return _table(["dimension", "values", "decided by", "read from", "not run"], rows)


def _ends_tables(e: RG.Ends) -> str:
    parts = []
    if e.answers:
        rows = [[", ".join(f"{k}={v}" for k, v in a.point.items()), ", ".join(a.estimators), ", ".join(a.refuters)] for a in e.answers]
        parts.append("### Answers: the cells a design can reach\n\n" + _table(["cell", "estimators", "refuters"], rows))
    else:
        parts.append("### Answers\n\nThis lane declares no space yet, so its answers are not enumerated.")
    rows = [[a.rung, a.address, a.question] for a in e.asks]
    parts.append("### Questions back\n\n" + (_table(["rung", "address", "question"], rows) if rows else "(none)"))
    rows = [[s.rung, s.cause, s.reason, s.fix] for s in e.stops]
    parts.append("### Typed stops\n\n" + (_table(["rung", "cause", "reason", "what would fix it"], rows) if rows else "(none)"))
    return "\n\n".join(parts)


def render(family: str) -> str:
    k = knowledge_of(family)
    spec: RG.LadderSpec = k.SPEC
    space: Space | None = getattr(k, "SPACE", None)
    estimators = k.load_estimators() if hasattr(k, "load_estimators") else []
    refuters = k.load_refuters() if hasattr(k, "load_refuters") else []
    family_needs = R.REGISTRY[family].needs
    catalogue = family_needs.endpoints if family_needs else None
    e = RG.ends(spec, space=space, estimators=estimators, refuters=refuters, checks=k.load_checks(), endpoints=catalogue)
    loops = "\n".join(f"- {lp.src} → {lp.to}, at most {lp.max}: {lp.why}" for lp in spec.loops) or "(none)"
    head = (
        f"# The {family.replace('_', '-')} lane's ladder\n\n"
        f"Written by `make ladder` from `causal_agent/families/{family}/lane/knowledge/` (rungs.yaml, space.yaml, checks.yaml, endpoints.yaml\n"
        "and the catalogue). Do not edit; change the declarations.\n\n"
        f"The ladder climbs to `{spec.goal}`; every rung may go only where this table says, and a stop goes to `feasibility`.\n\n"
        f"Loops still in the graph:\n\n{loops}\n"
    )
    body = ["## The rungs\n\n" + _rungs_table(spec, space)]
    if space is not None:
        body.append("## The space\n\n" + _space_table(space))
    if catalogue:
        body.append(
            "## Executable endpoint catalogue\n\nValid designs are conditional on the stated assumptions. Diagnostics cannot prove those assumptions.\n\n"
            + _table(
                ["endpoint", "target", "configuration", "trajectory", "estimators / inference", "terminal failures"],
                [
                    [
                        ep.id,
                        ", ".join(ep.targets),
                        str(ep.configuration),
                        " → ".join(ep.trajectory),
                        ", ".join(ep.estimators) + " / " + ", ".join(ep.inference),
                        ", ".join(ep.terminal_failures),
                    ]
                    for ep in catalogue.endpoints
                ],
            )
        )
        for ep in catalogue.endpoints:
            body.append(
                f"### Context for `{ep.id}`\n\n"
                + _table(
                    ["requirement", "address", "accepted status", "acceptable sources", "permitted unknown", "condition", "allowed values"],
                    [
                        [r.id, r.address, ", ".join(r.allowed_statuses), ", ".join(r.allowed_sources), r.allow_unknown, str(r.when), str(r.values)]
                        for r in ep.requirements
                    ],
                )
                + "\n\nAssumptions: "
                + "; ".join(ep.assumptions)
                + ".\n\nChecks: "
                + ", ".join(ep.checks)
                + "."
            )
    counts = f"{len(e.answers)} engine configurations, {len(e.asks)} questions, {len(e.stops)} stops"
    body.append("## The end states\n\n" + f"Derived from the declarations: {counts}.\n\n" + _ends_tables(e))
    return head + "\n" + "\n\n".join(body) + "\n"


def write(out: Path = OUT) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for family in built():
        p = out / f"{family}.md"
        p.write_text(render(family))
        paths.append(p)
    return paths


def stale(out: Path = OUT) -> list[str]:
    """The families whose table on disk differs from the declarations."""
    return [f for f in built() if not (out / f"{f}.md").exists() or (out / f"{f}.md").read_text() != render(f)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="exit 1 when a table is stale instead of writing")
    args = ap.parse_args(argv)
    if args.check:
        s = stale()
        for f in s:
            print(f"stale: docs/ladders/{f}.md; run make ladder")
        return 1 if s else 0
    for p in write():
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
