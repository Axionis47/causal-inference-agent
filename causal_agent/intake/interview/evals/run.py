"""Play each scripted conversation into the interview graph and score it. Nothing is written to data/; the run
stops at the ready flag. Each output carries the replies, the claims, the families, and the model's thoughts."""

from __future__ import annotations

import uuid
from pathlib import Path

from dotenv import load_dotenv
from langgraph.types import Command
from langsmith import evaluate

from causal_agent.intake.chat import _drain
from causal_agent.intake.datasets import ROOT
from causal_agent.intake.interview.evals.dataset import DATASET
from causal_agent.intake.interview.evals.evaluators import ALL
from causal_agent.intake.interview.graph import compile_local

load_dotenv()


def _flat(table) -> dict:
    out = {}
    for c in table.claims.values():
        for k, v in c.fields.items():
            out[f"{c.key}.{k}"] = v
        out[f"{c.key}.status"] = c.status
    return out


def run_case(inputs: dict) -> dict:
    g = compile_local()
    name = inputs["name"]
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}, "tags": [f"dataset:{name}", "intake"], "metadata": {"dataset": name}}
    payload, values = _drain(g, {"dataset": name, "csv": str(ROOT / inputs["csv"]), "docs": {"context": inputs["context"]}, "question": None}, cfg)
    replies, fallback, uncovered, ready_at = [], [], [], None
    v = g.get_state(cfg).values
    refuted0 = [c.key for c in v["claims"].claims.values() if c.status == "refuted"]
    turn = 0
    turns = list(inputs["turns"])
    while payload is not None:
        replies.append(payload["text"])
        v = g.get_state(cfg).values
        if any("fell back" in e for e in v.get("respond_errors") or []):
            fallback.append(turn)
        asked = {k for q in (v["reply"].questions if v.get("reply") else []) for k in q.keys}
        uncovered += [f"turn {turn}: {k}" for k in (v["status"].open if v.get("status") else []) if k not in asked]
        if payload["ready"] and ready_at is None:
            ready_at = turn
        if payload["ready"] or not turns:
            break
        payload, values = _drain(g, Command(resume=turns.pop(0)), cfg)
        turn += 1
    v = g.get_state(cfg).values
    st = v.get("status")
    return {
        "ready": bool(st and st.ready),
        "ready_at_turn": ready_at,
        "turns_used": turn,
        "claims": _flat(v["claims"]),
        "surviving": list(st.surviving) if st else [],
        "struck": list(st.struck) if st else [],
        "open": list(st.open) if st else [],
        "replies": replies,
        "fallback_turns": fallback,
        "uncovered_open": uncovered,
        "refuted_at_turn0": refuted0,
        "thoughts": [{"node": t.node, "text": t.text} for t in v.get("debug") or [] if t.text],
    }


def main() -> None:
    results = evaluate(run_case, data=DATASET, evaluators=ALL, experiment_prefix="intake", max_concurrency=1)
    print(results)


if __name__ == "__main__":
    main()
