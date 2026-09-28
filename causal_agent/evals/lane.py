"""One run of a family's lane from the command line.

    uv run python -m causal_agent.evals.lane <family> <dataset> "<question>"     # routed first, then the lane on the hand-off
    uv run python -m causal_agent.evals.lane <family> --handoff handoff.json      # the lane alone, from a stored hand-off

Streams progress, prints the report, and says where the run directory is.
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

import causal_agent.families.registry  # noqa: F401  (registers every family's block before a pack is read)
from causal_agent.common.contracts import Handoff
from causal_agent.desk.route import route
from causal_agent.evals.families import spec
from causal_agent.evals.spec import EvalSpec


def run_from_handoff(s: EvalSpec, handoff: Handoff, question: str) -> dict:
    g = s.lane_graph()
    cfg = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "tags": [f"dataset:{handoff.pack_name}", f"lane:{s.prefix}"],
        "metadata": {"dataset": handoff.pack_name},
    }
    final = None
    for mode, chunk in g.stream({"question": question, "handoff": handoff, "dataset": handoff.pack_name}, cfg, stream_mode=["custom", "values"]):
        if mode == "custom":
            key = next(iter(chunk))
            if key in s.skip_keys:
                continue
            print(f"· {key}: {json.dumps(chunk[key], default=str)[:400]}")
        else:
            final = chunk
    assert final is not None
    return final.get("specialist_result") or {}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("family")
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("question", nargs="?")
    ap.add_argument("--handoff", help="path to a hand-off JSON; runs the lane alone")
    ap.add_argument("--json", action="store_true", help="print the lane's result as JSON")
    args = ap.parse_args(argv)
    s = spec(args.family)

    if args.handoff:
        raw = json.loads(Path(args.handoff).read_text())
        question = raw.pop("question", args.question or "")
        result = run_from_handoff(s, Handoff.model_validate(raw), question)
    else:
        if not (args.dataset and args.question):
            ap.error("give <dataset> and <question>, or --handoff")
        r = route(args.question, args.dataset)
        if r.handoff is None or r.handoff.family != s.family:
            print(r.decision_record)
            raise SystemExit(f"routed to {r.handoff.family if r.handoff else 'no family'}, not {s.family}: nothing run")
        result = run_from_handoff(s, r.handoff, args.question)
    print()
    print(result.get("report", "(no report)"))
    print(f"\nrun directory: {result.get('run_dir')}")
    if args.json:
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
