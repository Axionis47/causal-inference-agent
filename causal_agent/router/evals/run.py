"""Run the router against the LangSmith dataset and score it. Each output carries the model's thoughts."""

from __future__ import annotations

import uuid

from dotenv import load_dotenv
from langsmith import evaluate

from causal_agent.router.graph import compile_local
from causal_agent.router.evals.dataset import DATASET
from causal_agent.router.evals.evaluators import ALL

load_dotenv()


def _thoughts(out: dict) -> list[dict]:
    return [{"node": t.node, "text": t.text} for t in out.get("debug") or [] if t.text]


def run_router(inputs: dict) -> dict:
    g = compile_local()
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}, "tags": [f"dataset:{inputs['dataset']}"], "metadata": {"dataset": inputs["dataset"]}}
    out = g.invoke({"question": inputs["question"], "dataset": inputs["dataset"]}, cfg)
    h, d = out.get("handoff"), out.get("decision")
    return {
        "family": h.family if h else None,
        "outcome": h.outcome if h else None,
        "treatment": h.treatment if h else None,
        "admissible": d.admissible if d else [],
        "rejected": [r.family for r in d.rejected] if d else [],
        "gate_errors": out.get("gate_errors") or [],
        "decision_record": out.get("decision_record"),
        "thoughts": _thoughts(out),
    }


def main() -> None:
    results = evaluate(run_router, data=DATASET, evaluators=ALL, experiment_prefix="router", max_concurrency=2)
    print(results)


if __name__ == "__main__":
    main()
