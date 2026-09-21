"""Run a family's cases against its LangSmith dataset and score them.

A case with a stored hand-off runs the lane alone; every other case goes through the desk's routing graph end to end.
Each output carries the model's thoughts.

    uv run python -m causal_agent.evals.run <family>
"""

from __future__ import annotations

import sys
import uuid
from functools import partial

from langsmith import evaluate

import causal_agent.families.registry  # noqa: F401  (registers every family's block before a pack is read)
from causal_agent.common.contracts import Handoff
from causal_agent.evals.families import spec
from causal_agent.evals.spec import EvalSpec


def thoughts(out: dict) -> list[dict]:
    return [{"node": t.node, "text": t.text} for t in out.get("debug") or [] if t.text]


def run_case(s: EvalSpec, inputs: dict) -> dict:
    if inputs.get("handoff"):
        raw = dict(inputs["handoff"])
        raw.pop("question", None)
        h = Handoff.model_validate(raw)
        g = s.lane_graph()
        cfg = {
            "configurable": {"thread_id": str(uuid.uuid4())},
            "tags": [f"dataset:{h.pack_name}", f"lane:{s.prefix}", "forced"],
            "metadata": {"dataset": h.pack_name},
        }
        out = g.invoke({"question": inputs["question"], "handoff": h, "dataset": h.pack_name}, cfg)
        summary = s.summarise(out.get("specialist_result") or {})
        checks = out.get("checks") or []  # the lane's own check results, when the result carried none
        summary["hard_flags"] = sorted({c.name for c in checks if c.level == "hard"}) or summary.get("hard_flags") or []
        summary["thoughts"] = thoughts(out)
        return summary
    from causal_agent.desk.route import compile_local as route_local

    g = route_local()
    cfg = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "tags": [f"dataset:{inputs['dataset']}", f"lane:{s.prefix}"],
        "metadata": {"dataset": inputs["dataset"]},
    }
    out = g.invoke({"question": inputs["question"], "dataset": inputs["dataset"]}, cfg)
    summary = s.summarise(out.get("specialist_result") or {})
    h = out.get("handoff")
    summary["routed_family"] = h.family if h else None
    summary["thoughts"] = thoughts(out)
    return summary


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        raise SystemExit("usage: python -m causal_agent.evals.run <family>")
    s = spec(args[0])
    results = evaluate(partial(run_case, s), data=s.dataset, evaluators=s.evaluators, experiment_prefix=s.prefix, max_concurrency=1)
    print(results)


if __name__ == "__main__":
    main()
