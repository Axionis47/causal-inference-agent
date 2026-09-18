"""The desk from the terminal.

    uv run python -m causal_agent.chat <csv> --name <dataset> --question "..." [--context notes.md] [--auto]

Before the run: the interview. After it: ask anything, change a claim, ask a new question, or say done.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

from langgraph.types import Command

from causal_agent.chat.graph import compile_local
from causal_agent.intake.chat import _drain


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--name", required=True)
    ap.add_argument("--question", required=True, help="the causal question")
    ap.add_argument("--context", default=None, help="a paragraph on the data and the change, one line per column")
    ap.add_argument("--auto", action="store_true", help="hand off as soon as the interview is ready")
    args = ap.parse_args(argv)
    if args.context:
        text = Path(args.context).read_text()
    else:
        print("Describe the dataset, then one line per column. End with a blank line.")
        lines = []
        while True:
            line = sys.stdin.readline()
            if not line or not line.strip():
                break
            lines.append(line.rstrip("\n"))
        text = "\n".join(lines)
    g = compile_local()
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}, "tags": [f"dataset:{args.name}", "desk"], "metadata": {"dataset": args.name}}
    payload, values = _drain(g, {"dataset": args.name, "csv": str(Path(args.csv).resolve()), "docs": {"context": text}, "question": args.question}, cfg)
    while payload is not None:
        print()
        print(payload["text"])
        print()
        print(payload["status"])
        if payload.get("phase") != "after" and payload.get("ready") and args.auto:
            answer = "run"
            print("\n> run  (auto)")
        else:
            try:
                answer = input("\n> ").strip()
            except EOFError:
                answer = "quit"
        payload, values = _drain(g, Command(resume=answer), cfg)
    print("\nDone." if values.get("runs") else "\nEnded before any run.")


if __name__ == "__main__":
    main()
