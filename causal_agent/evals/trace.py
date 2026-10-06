"""The newest trace, as a tree in the terminal: every span with its kind, how long it took, whether it failed, and for a gate
or a try the errors it found. The fast way to read a real run without the browser.

    uv run python -m causal_agent.evals.trace                 # the newest root in the project, as a tree
    uv run python -m causal_agent.evals.trace --last 8        # the last eight roots, one line each
    uv run python -m causal_agent.evals.trace --trace <id>    # one trace by id
    uv run python -m causal_agent.evals.trace --thoughts      # also the model's thoughts under each judgement
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from langsmith import Client

CUT = 160


def project() -> str:
    return os.environ.get("LANGSMITH_PROJECT") or "default"


def roots(client: Client, n: int) -> list[Any]:
    return list(client.list_runs(project_name=project(), is_root=True, limit=n))


def runs_of(client: Client, trace_id: Any) -> list[Any]:
    return sorted(client.list_runs(project_name=project(), trace_id=trace_id), key=lambda r: r.dotted_order)


def _secs(r: Any) -> str:
    if r.end_time is None or r.start_time is None:
        return "…"
    return f"{(r.end_time - r.start_time).total_seconds():.1f}s"


def _one(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= CUT else text[:CUT] + "…"


def line(r: Any) -> str:
    """One run as one line: name, kind, time, tokens, the error's first line."""
    bits = [r.name, f"[{r.run_type}]", _secs(r)]
    if r.total_tokens:
        bits.append(f"{r.total_tokens}tok")
    if r.error and "GraphInterrupt" in r.error:  # the desk pausing for the person is not a failure
        bits.append("⏸ waiting for the person")
    elif r.error:
        bits.append("✗ " + _one(r.error.splitlines()[0]))
    return " ".join(bits)


def details(r: Any, thoughts: bool) -> list[str]:
    """What is worth reading under a line: a gate's errors, a try's errors, an episode's outcome, a judgement's thoughts."""
    out: list[str] = []
    o = r.outputs or {}
    if r.name.startswith("gate") or r.name.startswith("try "):
        for e in o.get("errors") or []:
            out.append("  ! " + _one(e))
    if r.name.startswith("episode:"):
        out.append(f"  tries={o.get('tries')} calls={o.get('calls')} refusals={len(o.get('refusals') or [])}")
    if r.name.startswith("artifact:"):
        out.append(f"  {o.get('bytes')} bytes")
    if thoughts and o.get("thoughts"):
        t = o["thoughts"]
        for part in t if isinstance(t, list) else [t]:
            out.append("  ~ " + _one(part))
    return out


PLUMBING = ("Runnable", "PydanticToolsParser", "JsonOutputParser")


def visible(r: Any) -> bool:
    """A run worth a line: not LangChain's own plumbing between a judgement span and its model call."""
    return not r.name.startswith(PLUMBING)


def tree(runs: list[Any], thoughts: bool) -> str:
    if not runs:
        return "(no runs)"
    depth: dict[Any, int] = {}
    lines = []
    for r in runs:
        d = depth.get(r.parent_run_id, -1) + (1 if visible(r) else 0)
        depth[r.id] = d
        if not visible(r):
            continue
        pad = "  " * d
        lines.append(pad + line(r))
        lines += [pad + x for x in details(r, thoughts)]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=0, help="list the last N roots, one line each")
    ap.add_argument("--trace", help="one trace by id")
    ap.add_argument("--thoughts", action="store_true", help="show the model's thoughts under each judgement")
    args = ap.parse_args(argv)
    client = Client()
    if args.last:
        for r in roots(client, args.last):
            print(f"{r.id}  {r.start_time:%Y-%m-%d %H:%M}  {line(r)}")
        return
    trace_id = args.trace
    if trace_id is None:
        top = roots(client, 1)
        if not top:
            raise SystemExit(f"no traces in project {project()!r}")
        trace_id = top[0].trace_id
    print(tree(runs_of(client, trace_id), args.thoughts))


if __name__ == "__main__":
    main()
