"""Play each scripted conversation into the desk and score it. Writes packs under data/ like a real conversation."""

from __future__ import annotations

import uuid

from dotenv import load_dotenv
from langgraph.types import Command
from langsmith import evaluate

from causal_agent.chat.evals.dataset import DATASET
from causal_agent.chat.evals.evaluators import ALL
from causal_agent.chat.graph import compile_local
from causal_agent.intake.chat import _drain
from causal_agent.intake.datasets import ROOT

load_dotenv()


def run_case(inputs: dict) -> dict:
    g = compile_local()
    name = inputs["name"]
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}, "tags": [f"dataset:{name}", "desk"], "metadata": {"dataset": name}}
    payload, _ = _drain(g, {"dataset": name, "csv": str(ROOT / inputs["csv"]), "docs": {"context": inputs["context"]}, "question": inputs["question"]}, cfg)
    replies, briefs, kinds, fails = [], [], [], []
    turns = list(inputs["turns"])
    while payload is not None and turns:
        replies.append(payload["text"])
        v = g.get_state(cfg).values
        if payload.get("phase") == "after":
            if v.get("after_reply") is None and payload["text"]:
                briefs.append(payload["text"])
            if any("fell back" in e for e in v.get("after_errors") or []):
                fails.append(len(replies))
        payload, _ = _drain(g, Command(resume=turns.pop(0)), cfg)
    v = g.get_state(cfg).values
    runs = v.get("runs") or []
    return {
        "runs": len(runs),
        "families": [r.family for r in runs],
        "families_considered": sorted({f for r in runs for f in list((r.decision or {}).get("over") or {}) + ([r.family] if r.family else [])}),
        "effects": [r.effect for r in runs],
        "kinds": [e.kind for e in v.get("exchanges") or []],
        "replies": replies,
        "briefs": briefs,
        "gate_failures": fails,
        "thoughts": [{"node": t.node, "text": t.text} for t in v.get("debug") or [] if t.text],
    }


def main() -> None:
    results = evaluate(run_case, data=DATASET, evaluators=ALL, experiment_prefix="desk", max_concurrency=1)
    print(results)


if __name__ == "__main__":
    main()
