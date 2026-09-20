"""Run the diff-in-diff lane from the command line.

uv run python -m causal_agent.families.diff_in_diff.lane.run <dataset> "<question>"        # through the desk's routing graph, end to end
uv run python -m causal_agent.families.diff_in_diff.lane.run --handoff handoff.json         # the specialist alone, from a stored hand-off
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

import causal_agent.families.registry  # noqa: F401  (registers every family's block before a pack is read)
from causal_agent.common.contracts import Handoff


def run_from_handoff(handoff: Handoff, question: str) -> dict:
    from causal_agent.families.diff_in_diff.lane.graph import compile_local

    g = compile_local()
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}, "tags": [f"dataset:{handoff.pack_name}", "lane:did"], "metadata": {"dataset": handoff.pack_name}}
    final = None
    for mode, chunk in g.stream({"question": question, "handoff": handoff, "dataset": handoff.pack_name}, cfg, stream_mode=["custom", "values"]):
        if mode == "custom":
            key = next(iter(chunk))
            if key in ("report", "design", "controls"):
                continue
            print(f"· {key}: {json.dumps(chunk[key], default=str)[:400]}")
        else:
            final = chunk
    assert final is not None
    return final.get("specialist_result") or {}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("question", nargs="?")
    ap.add_argument("--handoff", help="path to a hand-off JSON; runs the specialist alone")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if args.handoff:
        raw = json.loads(Path(args.handoff).read_text())
        question = raw.pop("question", args.question or "")
        result = run_from_handoff(Handoff.model_validate(raw), question)
    else:
        if not (args.dataset and args.question):
            ap.error("give <dataset> and <question>, or --handoff")
        from causal_agent.desk.route import compile_local as route_local

        g = route_local()
        cfg = {"configurable": {"thread_id": str(uuid.uuid4())}, "tags": [f"dataset:{args.dataset}"], "metadata": {"dataset": args.dataset}}
        out = g.invoke({"question": args.question, "dataset": args.dataset}, cfg)
        result = out.get("specialist_result") or {}
    print()
    print(result.get("report", "(no report)"))
    print(f"\nrun directory: {result.get('run_dir')}")
    if args.json:
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
