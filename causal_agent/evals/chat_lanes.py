"""Real-source acceptance through compact authorization references and isolated lanes."""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from causal_agent.common import trace
from causal_agent.common.contracts import Handoff
from causal_agent.desk import pipeline
from causal_agent.evals.chat import ROOT, verify_trace
from causal_agent.memory.context.snapshot import publish

CASES = [("adjustment", "gov_transfers_forced.json"), ("diff_in_diff", "card_krueger_forced.json"), ("discontinuity", "senate_context.json")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    if not trace.on():
        raise ValueError("Real acceptance requires tracing.")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = []
    with trace.span("acceptance:chat-lane-boundaries", inputs={"cases": CASES}) as span:
        trace_id = str(span.run.id) if span.run else None
        for family, filename in CASES:
            source = ROOT / "causal_agent/families" / family / "evals/handoffs" / filename
            handoff = Handoff.model_validate_json(source.read_text())
            directory = out / family
            directory.mkdir(exist_ok=True)
            reference = publish(directory, handoff, "recorded-source:" + source.stem)
            with trace.span("run", inputs={"family": family, "source": str(source), "authorization": str(reference)}):
                record = pipeline.run(reference, 1, handoff.pack_name, handoff.question, thread_id="chat-acceptance-" + family, snapshot=True)
            trace.write(directory / "record.json", record.model_dump_json(indent=2))
            report.append(
                {
                    "family": family,
                    "source": str(source),
                    "context_id": handoff.context_id,
                    "dataset_identity": handoff.dataset_identity,
                    "status": record.status,
                    "deficiency": (record.specialist_result or {}).get("context_deficiency"),
                }
            )
            trace.write(out / "results.json", json.dumps(report, indent=2))
            print(f"{family}: {record.status}", flush=True)
            if family == "adjustment":
                if record.status != "context_deficient":
                    raise ValueError("The deliberately insufficient real-source handoff must stop at intake.")
            elif record.status not in {"done", "infeasible", "unidentified", "bounded_reasoning_stop"}:
                raise ValueError(f"{family} did not finish with a recorded scientific outcome: {record.status}")
        span.end(outputs={"results": report})
    verify_trace(trace_id, out, required_node="lane:diff_in_diff")


if __name__ == "__main__":
    main()
