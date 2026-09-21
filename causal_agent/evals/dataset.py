"""Upload a family's cases.yaml to its LangSmith dataset. Idempotent on case ids; expected outputs kept in step.

uv run python -m causal_agent.evals.dataset <family>
"""

from __future__ import annotations

import json
import sys

import yaml
from langsmith import Client

from causal_agent.evals.families import spec
from causal_agent.evals.spec import EvalSpec


def load_cases(s: EvalSpec) -> list[dict]:
    cases = yaml.safe_load((s.cases_dir / "cases.yaml").read_text())
    for c in cases:
        if c.get("handoff"):
            c["handoff_json"] = json.loads((s.cases_dir / c["handoff"]).read_text())
    return cases


def upload(s: EvalSpec) -> None:
    cases = load_cases(s)
    client = Client()
    if client.has_dataset(dataset_name=s.dataset):
        ds = client.read_dataset(dataset_name=s.dataset)
    else:
        ds = client.create_dataset(dataset_name=s.dataset, description=s.description)
    by_id = {e.metadata.get("case_id"): e for e in client.list_examples(dataset_id=ds.id) if e.metadata}
    changed = 0
    for c in cases:  # keep expected outputs in step with cases.yaml
        e = by_id.get(c["id"])
        if e is not None and (e.outputs or {}) != c["expected"]:
            client.update_example(example_id=e.id, outputs=c["expected"])
            changed += 1
    new = [c for c in cases if c["id"] not in by_id]
    if new:
        client.create_examples(
            dataset_id=ds.id,
            inputs=[{"dataset": c["dataset"], "question": c["question"], "handoff": c.get("handoff_json")} for c in new],
            outputs=[c["expected"] for c in new],
            metadata=[{"case_id": c["id"]} for c in new],
        )
    print(f"dataset {s.dataset}: {len(by_id)} existing ({changed} updated), {len(new)} added")


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        raise SystemExit("usage: python -m causal_agent.evals.dataset <family>")
    upload(spec(args[0]))


if __name__ == "__main__":
    main()
