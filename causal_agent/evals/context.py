"""Real-model context acceptance. Explicit input files, no changes to originals or app storage."""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from causal_agent.common import trace
from causal_agent.memory.context.contracts import ColumnIdentity
from causal_agent.memory.context.extract import extract


def verify_result(result, source, columns):
    """Independent structural acceptance checks, not a claim of semantic correctness."""
    if not 2 <= result.model_calls <= 4 or result.assessment.corrections:
        raise ValueError("Model calls exceeded the budget or assessment corrections remain")
    if {c.column_id for c in result.content.columns} != {c.column_id for c in columns} or len(result.content.columns) != len(columns):
        raise ValueError("Not every original column is represented exactly once")
    statements = [*result.content.dataset, *result.content.concepts, *result.content.relationships]
    statements.extend(s for c in result.content.columns for s in c.statements)
    if not statements:
        raise ValueError("Nonempty acceptance source produced no statements")
    for statement in statements:
        if not statement.citations or any(source[c.start : c.end] != c.quote for c in statement.citations):
            raise ValueError("Citation does not match unchanged source bytes decoded as text")


def verify_runs(runs, trace_id, model_calls):
    ids = {str(r.id) for r in runs}
    if str(trace_id) not in ids or any(not r.end_time or r.error for r in runs):
        raise ValueError("Trace is missing, unfinished or contains errors")
    roots = [r for r in runs if r.parent_run_id is None]
    if len(roots) != 1 or str(roots[0].id) != str(trace_id) or any(r.parent_run_id and str(r.parent_run_id) not in ids for r in runs):
        raise ValueError("Trace parent hierarchy is incomplete")
    calls = [r for r in runs if r.name in {"context:fill", "context:revise", "context:assess"}]
    if len(calls) != model_calls:
        raise ValueError("Trace does not account for each application model call")
    for name in ("context:validate", "context:assessment"):
        gates = [r for r in runs if r.name == name]
        if not gates or not (gates[-1].outputs or {}).get("passed"):
            raise ValueError(f"Final gate did not pass: {name}")


def readback(trace_id, result, out):
    from langsmith import Client

    from causal_agent.evals.trace import runs_of, tree

    if not trace_id:
        raise ValueError("Live acceptance requires LANGSMITH_TRACING=true and configured credentials")
    client = Client()
    runs = []
    for attempt in range(6):
        runs = runs_of(client, trace_id)
        try:
            verify_runs(runs, trace_id, result.model_calls)
            break
        except ValueError:
            pass  # Bounded wait for the uploaded trace to become complete.
        if attempt < 5:
            time.sleep(2)
    trace.write(out.with_suffix(".trace.json"), json.dumps([r.model_dump(mode="json") for r in runs], default=str))
    trace.write(out.with_suffix(".trace.txt"), tree(runs, thoughts=True))
    verify_runs(runs, trace_id, result.model_calls)
    trace.write(
        out.with_suffix(".checks.json"),
        json.dumps(
            {"trace_id": trace_id, "runs": len(runs), "model_calls": result.model_calls, "structural_checks": "passed", "semantic_review": "required"}, indent=2
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--verify-trace",
        action="store_true",
        help="Check exact citations, unchanged inputs and the completed LangSmith trace; semantic review remains separate",
    )
    args = parser.parse_args()
    if args.verify_trace and not trace.on():
        raise ValueError("Enable tracing before starting model acceptance; no model was called")
    original = args.csv.read_bytes()
    source_bytes = args.source.read_bytes()
    source = source_bytes.decode("utf-8")
    with args.csv.open(encoding="utf-8-sig", newline="") as stream:
        headers = next(csv.reader(stream))
    columns = [ColumnIdentity(column_id=str(uuid5(NAMESPACE_URL, f"{args.csv}:{i}")), position=i, name=name, key=name) for i, name in enumerate(headers)]
    with trace.span(
        "acceptance:context",
        inputs={
            "csv": str(args.csv),
            "csv_sha256": hashlib.sha256(original).hexdigest(),
            "source": str(args.source),
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        },
    ) as span:
        trace_id = str(span.run.id) if span.run else None
        try:
            result = extract(source, columns)
            record = {"trace_id": trace_id, "source": str(args.source), "csv": str(args.csv), "result": result.model_dump(mode="json")}
            trace.write(args.out, json.dumps(record, indent=2, ensure_ascii=False), note="Real context extraction acceptance record")
            span.end(outputs={"record": str(args.out), "model_calls": result.model_calls})
            print(json.dumps({"trace_id": trace_id, "record": str(args.out), "model_calls": result.model_calls}))
        except Exception as exc:
            trace.write(args.out, json.dumps({"trace_id": trace_id, "error": str(exc)}), note="Failed context extraction acceptance")
            raise
    if args.csv.read_bytes() != original or args.source.read_bytes() != source_bytes:
        raise ValueError("Acceptance inputs were modified")
    trace.flush()
    if args.verify_trace:
        verify_result(result, source, columns)
        readback(trace_id, result, args.out)


if __name__ == "__main__":
    main()
