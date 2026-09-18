"""The intake chat from the terminal.

    uv run python -m causal_agent.intake.chat <csv> --name <dataset> [--context notes.md] [--question "..."] [--auto]

Turn 0 sends the description. Each turn prints the reply, the table, and the status line, then reads a line.
`run` when READY writes the pack and hands off to the router. `status` reprints the table. `quit` ends without writing.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from langgraph.types import Command

from causal_agent.intake.interview.graph import compile_local


def _drain(g, inp, cfg):
    """Run until the next interrupt or the end; return (payload or None, final state values)."""
    payload = None
    for mode, chunk in g.stream(inp, cfg, stream_mode=["updates", "values"]):
        if mode == "updates" and "__interrupt__" in chunk:
            payload = chunk["__interrupt__"][0].value
    return payload, g.get_state(cfg).values


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--name", required=True, help="dataset name to write under data/")
    ap.add_argument("--context", default=None, help="a markdown or text file: a paragraph on the dataset and one line per column")
    ap.add_argument("--question", default=None, help="the causal question, used at hand-off")
    ap.add_argument("--auto", action="store_true", help="hand off as soon as the flag is up")
    ap.add_argument("--json", action="store_true", help="print the claims file and the hand-off as JSON at the end")
    args = ap.parse_args(argv)

    if args.context:
        text = Path(args.context).read_text()
    else:
        print("Describe the dataset: a paragraph on what it is and what changed, then one line per column. End with a blank line.")
        lines = []
        while True:
            line = sys.stdin.readline()
            if not line or not line.strip():
                break
            lines.append(line.rstrip("\n"))
        text = "\n".join(lines)

    g = compile_local()
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}, "tags": [f"dataset:{args.name}", "intake"], "metadata": {"dataset": args.name}}
    payload, values = _drain(g, {"dataset": args.name, "csv": str(Path(args.csv).resolve()), "docs": {"context": text}, "question": args.question}, cfg)

    while payload is not None:
        print()
        print(payload["text"])
        print()
        print(payload["status"])
        if payload["ready"] and args.auto:
            answer = "run"
            print("\n> run  (auto)")
        else:
            try:
                answer = input("\n> ").strip()
            except EOFError:
                answer = "quit"
        if answer.lower() == "status":
            print(payload["status"])
            continue
        payload, values = _drain(g, Command(resume=answer), cfg)

    written = values.get("written")
    if not written:
        print("\nEnded without writing the pack.")
        return
    print(f"\nWrote {written['note']}, {written['claims']}, {written['profile']}; datasets.yaml entry {written['dataset']!r}.")
    question = args.question or input("\nThe causal question for the router: ").strip()
    if not question:
        return
    from causal_agent.router.graph import compile_local as router_local

    r = router_local()
    rcfg = {"configurable": {"thread_id": str(uuid.uuid4())}, "tags": [f"dataset:{args.name}"], "metadata": {"dataset": args.name}}
    out = r.invoke({"question": question, "dataset": args.name}, rcfg)
    print()
    print(out.get("decision_record") or "no decision record (gate failed)\n" + "\n".join(out.get("gate_errors", [])))
    if args.json:
        print(json.dumps({"written": written, "handoff": out["handoff"].model_dump() if out.get("handoff") else None}, indent=2, default=str))


if __name__ == "__main__":
    main()
