"""Run the diff-in-diff cases against the LangSmith dataset and score them.

Card and Krueger goes through the desk's routing graph end to end. A case with a stored hand-off runs the specialist alone.
Each output carries the model's thoughts.
"""

from __future__ import annotations

import uuid

from dotenv import load_dotenv
from langsmith import evaluate

import causal_agent.families.registry  # noqa: F401  (registers every family's block before a pack is read)
from causal_agent.common.contracts import Handoff
from causal_agent.specialists.did.evals.dataset import DATASET
from causal_agent.specialists.did.evals.evaluators import ALL

load_dotenv()


def _thoughts(out: dict) -> list[dict]:
    return [{"node": t.node, "text": t.text} for t in out.get("debug") or [] if t.text]


def _summarise(result: dict) -> dict:
    d = result.get("design") or {}
    shape = result.get("shape") or {}
    checks = result.get("checks") or []
    est = [e for e in result.get("estimates") or [] if e.get("error") is None and e.get("method") == d.get("estimator")]
    f = result.get("feasibility") or {}
    return {
        "status": result.get("status"),
        "stage": f.get("stage"),
        "shape": shape.get("kind"),
        "periods_pre": shape.get("periods_pre"),
        "estimator": d.get("estimator"),
        "controls": (result.get("controls") or {}).get("included"),
        "controls_excluded": sorted(
            {x["column"] for x in ((result.get("controls") or {}).get("excluded") or []) + ((result.get("controls") or {}).get("dropped_fixed") or [])}
        ),
        "effect": est[0]["value"] if est else None,
        "flags": sorted({c["name"] for c in checks if c["level"] != "pass"}),
        "hard_flags": sorted({c["name"] for c in checks if c["level"] == "hard"}),
        "placebos": sorted({r["refuter"] for r in result.get("refutations") or []}),
        "run_dir": result.get("run_dir"),
    }


def run_case(inputs: dict) -> dict:
    if inputs.get("handoff"):
        from causal_agent.specialists.did.graph import compile_local

        raw = dict(inputs["handoff"])
        raw.pop("question", None)
        h = Handoff.model_validate(raw)
        g = compile_local()
        cfg = {"configurable": {"thread_id": str(uuid.uuid4())}, "tags": [f"dataset:{h.pack_name}", "lane:did", "forced"], "metadata": {"dataset": h.pack_name}}
        out = g.invoke({"question": inputs["question"], "handoff": h, "dataset": h.pack_name}, cfg)
        summary = _summarise(out.get("specialist_result") or {})
        summary["thoughts"] = _thoughts(out)
        return summary
    from causal_agent.desk.route import compile_local as route_local

    g = route_local()
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}, "tags": [f"dataset:{inputs['dataset']}"], "metadata": {"dataset": inputs["dataset"]}}
    out = g.invoke({"question": inputs["question"], "dataset": inputs["dataset"]}, cfg)
    summary = _summarise(out.get("specialist_result") or {})
    h = out.get("handoff")
    summary["routed_family"] = h.family if h else None
    summary["thoughts"] = _thoughts(out)
    return summary


def main() -> None:
    results = evaluate(run_case, data=DATASET, evaluators=ALL, experiment_prefix="did", max_concurrency=1)
    print(results)


if __name__ == "__main__":
    main()
