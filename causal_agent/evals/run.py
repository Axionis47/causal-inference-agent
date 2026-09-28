"""Run a family's cases against its LangSmith dataset and score them.

A case with a stored hand-off runs the lane alone; every other case is routed first, then the family's lane runs on the
hand-off. Each output carries the model's thoughts.

    uv run python -m causal_agent.evals.run <family>
"""

from __future__ import annotations

import sys
import uuid
from functools import partial

from langsmith import evaluate

import causal_agent.families.registry  # noqa: F401  (registers every family's block before a pack is read)
from causal_agent.common.contracts import Handoff, Thought
from causal_agent.desk.route import route
from causal_agent.evals.families import spec
from causal_agent.evals.spec import EvalSpec


def thoughts(debug: list[Thought]) -> list[dict]:
    return [{"node": t.node, "text": t.text} for t in debug if t.text]


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
        summary["thoughts"] = thoughts(out.get("debug") or [])
        return summary
    r = route(inputs["question"], inputs["dataset"])
    routed = r.handoff
    if routed is None or routed.family != s.family:  # no family stands, or another family's: reported, not run through this lane
        return {"routed_family": routed.family if routed else None, "decision_record": r.decision_record, "thoughts": thoughts(r.debug)}
    h = routed
    g = s.lane_graph()
    cfg = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "tags": [f"dataset:{inputs['dataset']}", f"lane:{s.prefix}"],
        "metadata": {"dataset": inputs["dataset"]},
    }
    out = g.invoke({"question": inputs["question"], "handoff": h, "dataset": h.pack_name}, cfg)
    summary = s.summarise(out.get("specialist_result") or {})
    summary["routed_family"] = h.family
    summary["thoughts"] = thoughts(r.debug) + thoughts(out.get("debug") or [])
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
