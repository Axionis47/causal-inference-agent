"""The decision model against the labelled turns, before it is trusted with any:

    uv run python -m causal_agent.evals.turns                 # the backend DECIDER names, the thresholds swept
    uv run python -m causal_agent.evals.turns --threshold 0.7 # one threshold, the confusion by kind

Each labelled turn is a message with what the desk had asked and where the conversation stood, and the kind it is. The
model answers every turn once; the thresholds are applied to those answers, so a sweep costs one pass. Reported per
threshold: the share decided (not escalated to the Reader), and the accuracy among the decided; then the confusion by kind
at the threshold decider.yaml declares, or the one given.

`turns.jsonl` is the held-out set and the one scored by default; `turns_train.jsonl` is the set `evals/decider.py` fine-tunes
on, and a sentence in the held-out set is never in it. Both are in the desk's own row shape; `to_laya` renders a row as the
decision model is trained on it, from the same `state_of` and the same questions the desk sends."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from causal_agent.common import llm, trace
from causal_agent.desk import turns as TD

SET = Path(__file__).resolve().parents[1] / "desk" / "evals" / "turns.jsonl"
TRAIN_SET = SET.with_name("turns_train.jsonl")
SWEEP = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def rows_of(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def to_laya(row: dict) -> dict:
    """One labelled turn in the shape the decision model trains on: the state the desk would send, the questions
    decider.yaml spells, and the expected answer to each (the kind; about the data exactly when the kind is update)."""
    state = TD.state_of(row["message"], row.get("asked"), bool(row.get("at_ready")), row.get("last_reply") or "")
    return {"state": state, "questions": TD.spec()["questions"], "expected": {"kind": row["kind"], "about_data": row["kind"] == "update"}}


def answers(rows: list[dict]) -> list[tuple[str | None, float]]:
    """Every turn decided once at threshold zero: the kind the model picked and its confidence, by the same rule the desk uses."""
    out: list[tuple[str | None, float]] = []
    for r in rows:
        kind, conf = TD.classify(r["message"], asked=r.get("asked"), at_ready=bool(r.get("at_ready")), last_reply=r.get("last_reply") or "", threshold=0.0)
        out.append((kind, conf))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--set", default=str(SET))
    args = ap.parse_args(argv)
    rows = rows_of(Path(args.set))
    with trace.span("turns:eval", inputs={"turns": len(rows), "model": llm.current_decider().name}):
        got = answers(rows)
    print(f"{len(rows)} turns, decided by {llm.current_decider().name}")
    print("threshold  decided  accuracy among decided")
    for t in SWEEP:
        decided = [(r["kind"], k) for r, (k, c) in zip(rows, got) if c >= t and k is not None]
        acc = sum(1 for want, k in decided if want == k) / len(decided) if decided else 0.0
        print(f"{t:9.1f}  {len(decided) / len(rows):7.0%}  {acc:7.0%}")
    t = args.threshold if args.threshold is not None else float(TD.spec().get("threshold", 0.6))
    print(f"\nat threshold {t:g}: want -> got (confidence)")
    confusion: Counter[tuple[str, str]] = Counter()
    for r, (k, c) in zip(rows, got):
        chosen = k if (k is not None and c >= t) else "escalated"
        confusion[(r["kind"], chosen)] += 1
        if chosen != r["kind"]:
            print(f"  {r['kind']:8} -> {chosen:9} ({c:.2f})  {r['message'][:90]!r}")
    for want in TD.KINDS:
        line = ", ".join(f"{got_k}: {n}" for (w, got_k), n in sorted(confusion.items()) if w == want)
        print(f"{want:8} {line}")


if __name__ == "__main__":
    main()
