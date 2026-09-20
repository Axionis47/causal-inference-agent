"""Run the discontinuity cases against the LangSmith dataset and score them.

The five real datasets go through the desk's routing graph end to end. A case with a stored hand-off runs the specialist
alone. Each output carries the model's thoughts.
"""

from __future__ import annotations

import uuid

from dotenv import load_dotenv
from langsmith import evaluate

import causal_agent.families.registry  # noqa: F401  (registers every family's block before a pack is read)
from causal_agent.common.contracts import Handoff
from causal_agent.families.discontinuity.evals.dataset import DATASET
from causal_agent.families.discontinuity.evals.evaluators import ALL

load_dotenv()


def _thoughts(out: dict) -> list[dict]:
    return [{"node": t.node, "text": t.text} for t in out.get("debug") or [] if t.text]


def _summarise(result: dict) -> dict:
    d = result.get("design") or {}
    sc = result.get("score") or {}
    shape = result.get("shape") or {}
    checks = result.get("checks") or []
    est = [e for e in result.get("estimates") or [] if e.get("error") is None and e.get("method") == d.get("estimator")]
    f = result.get("feasibility") or {}
    a = result.get("assessment")
    return {
        "status": result.get("status"),
        "stage": f.get("stage"),
        "score_column": sc.get("column"),
        "cutoff": sc.get("cutoff"),
        "treated_side": sc.get("treated_side") if sc.get("column") else None,
        "takeup_column": sc.get("takeup_column"),
        "kind": shape.get("kind"),
        "cluster": shape.get("cluster_column"),
        "estimator": d.get("estimator"),
        "estimand": d.get("estimand"),
        "effect": est[0]["value"] if est else None,
        "ci": [est[0]["ci_low"], est[0]["ci_high"]] if est else None,
        "checks_ran": sorted({c["name"] for c in checks}),
        "flags": sorted({c["name"] for c in checks if c["level"] != "pass"}),
        "hard_flags": sorted({c["name"] for c in checks if c["level"] == "hard"}),
        "assess_cites": a.get("cites") if a else None,
        "assess_action": a.get("action") if a else None,
        "placebos": sorted({r["refuter"] for r in result.get("refutations") or []}),
        "placebos_failed": sorted({r["refuter"] for r in result.get("refutations") or [] if r.get("passed") is False}),
        "secondaries": sorted({e["method"] for e in result.get("estimates") or [] if e.get("secondary") and e.get("error") is None}),
        "run_dir": result.get("run_dir"),
    }


def run_case(inputs: dict) -> dict:
    if inputs.get("handoff"):
        from causal_agent.families.discontinuity.lane.graph import compile_local

        raw = dict(inputs["handoff"])
        raw.pop("question", None)
        h = Handoff.model_validate(raw)
        g = compile_local()
        cfg = {"configurable": {"thread_id": str(uuid.uuid4())}, "tags": [f"dataset:{h.pack_name}", "lane:rd", "forced"], "metadata": {"dataset": h.pack_name}}
        out = g.invoke({"question": inputs["question"], "handoff": h, "dataset": h.pack_name}, cfg)
        summary = _summarise(out.get("specialist_result") or {})
        summary["thoughts"] = _thoughts(out)
        return summary
    from causal_agent.desk.route import compile_local as route_local

    g = route_local()
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}, "tags": [f"dataset:{inputs['dataset']}", "lane:rd"], "metadata": {"dataset": inputs["dataset"]}}
    out = g.invoke({"question": inputs["question"], "dataset": inputs["dataset"]}, cfg)
    summary = _summarise(out.get("specialist_result") or {})
    h = out.get("handoff")
    summary["routed_family"] = h.family if h else None
    summary["thoughts"] = _thoughts(out)
    return summary


def main() -> None:
    results = evaluate(run_case, data=DATASET, evaluators=ALL, experiment_prefix="rd", max_concurrency=1)
    print(results)


if __name__ == "__main__":
    main()
