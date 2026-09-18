"""Run the router once from the command line, streaming progress and printing the decision record."""

from __future__ import annotations

import argparse
import json
import uuid

from causal_agent.router.graph import compile_local


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("question")
    ap.add_argument("--json", action="store_true", help="print the hand-off and specialist result as JSON")
    ap.add_argument("--json-file", default=None, help="write the hand-off, decision, record, and specialist result as JSON to this path")
    args = ap.parse_args(argv)

    g = compile_local()
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}, "tags": [f"dataset:{args.dataset}"], "metadata": {"dataset": args.dataset}}
    final = None
    for mode, chunk in g.stream({"question": args.question, "dataset": args.dataset}, cfg, stream_mode=["custom", "values"]):
        if mode == "custom":
            key = next(iter(chunk))
            if key == "decision_record":
                continue
            print(f"· {key}: {json.dumps(chunk[key], default=str)[:400]}")
        else:
            final = chunk
    assert final is not None
    print()
    print(final.get("decision_record") or "no decision record (gate failed)\n" + "\n".join(final.get("gate_errors", [])))
    payload = {"handoff": final["handoff"].model_dump() if final.get("handoff") else None, "specialist_result": final.get("specialist_result"),
               "decision": final["decision"].model_dump() if final.get("decision") else None, "decision_record": final.get("decision_record") or "",
               "gate_errors": final.get("gate_errors") or []}
    if args.json:
        print()
        print(json.dumps({"handoff": payload["handoff"], "specialist_result": payload["specialist_result"]}, indent=2, default=str))
    if args.json_file:
        from pathlib import Path

        Path(args.json_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_file).write_text(json.dumps(payload, default=str))


if __name__ == "__main__":
    main()
