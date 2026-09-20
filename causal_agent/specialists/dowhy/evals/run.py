"""Run the adjustment-lane cases against the LangSmith dataset and score them.

Positive cases go through the desk's routing graph end to end. A case with a stored hand-off runs the specialist alone.
Each output carries the model's thoughts.
"""

from __future__ import annotations

import uuid

from dotenv import load_dotenv
from langsmith import evaluate

from causal_agent.common.contracts import Handoff
from causal_agent.specialists.dowhy.evals.dataset import DATASET
from causal_agent.specialists.dowhy.evals.evaluators import ALL

load_dotenv()


def _thoughts(out: dict) -> list[dict]:
    return [{"node": t.node, "text": t.text} for t in out.get("debug") or [] if t.text]


def _summarise(result: dict) -> dict:
    d = result.get("design") or {}
    est = [e for e in result.get("estimates") or [] if not e.get("secondary") and e.get("error") is None]
    refs: dict[str, dict[str, bool | None]] = {}
    for r in result.get("refutations") or []:
        refs.setdefault(r["contrast"], {})[r["refuter"]] = r.get("passed")
    f = result.get("feasibility") or {}
    checks = (d.get("checks") or {}).get("results") or []
    return {
        "status": result.get("status"),
        "stage": f.get("stage"),
        "adjustment_set": (d.get("estimand") or {}).get("adjustment_set"),
        "excluded": [x["column"] for x in (d.get("graph") or {}).get("excluded") or []],
        "nodes": list((d.get("graph") or {}).get("nodes") or []),
        "estimator": d.get("estimator"),
        "contrasts": len(d.get("contrasts") or []) or None,
        "effects": {e["contrast"]: e["value"] for e in est},
        "refuters": refs,
        "hard_flags": sorted({c["name"] for c in checks if c["level"] == "hard"}),
        "run_dir": result.get("run_dir"),
    }


def run_case(inputs: dict) -> dict:
    if inputs.get("handoff"):
        from causal_agent.specialists.dowhy.graph import compile_local

        raw = dict(inputs["handoff"])
        raw.pop("question", None)
        h = Handoff.model_validate(raw)
        g = compile_local()
        cfg = {"configurable": {"thread_id": str(uuid.uuid4())}, "tags": [f"dataset:{h.pack_name}", "lane:dowhy", "forced"], "metadata": {"dataset": h.pack_name}}
        out = g.invoke({"question": inputs["question"], "handoff": h, "dataset": h.pack_name}, cfg)
        result = out.get("specialist_result") or {}
        checks = out.get("checks") or []
        summary = _summarise(result)
        summary["hard_flags"] = sorted({c.name for c in checks if c.level == "hard"}) or summary["hard_flags"]
        summary["thoughts"] = _thoughts(out)
        return summary
    from causal_agent.desk.route import compile_local as route_local

    g = route_local()
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}, "tags": [f"dataset:{inputs['dataset']}"], "metadata": {"dataset": inputs["dataset"]}}
    out = g.invoke({"question": inputs["question"], "dataset": inputs["dataset"]}, cfg)
    result = out.get("specialist_result") or {}
    summary = _summarise(result)
    h = out.get("handoff")
    summary["routed_family"] = h.family if h else None
    summary["thoughts"] = _thoughts(out)
    return summary


def main() -> None:
    results = evaluate(run_case, data=DATASET, evaluators=ALL, experiment_prefix="dowhy", max_concurrency=1)
    print(results)


if __name__ == "__main__":
    main()
