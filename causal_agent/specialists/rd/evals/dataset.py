"""Upload evals/cases.yaml to a LangSmith dataset. Idempotent on case ids; expected outputs kept in step."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

DATASET = "causal-rd-v0"
HERE = Path(__file__).parent


def load_cases() -> list[dict]:
    cases = yaml.safe_load((HERE / "cases.yaml").read_text())
    for c in cases:
        if c.get("handoff"):
            c["handoff_json"] = json.loads((HERE / c["handoff"]).read_text())
    return cases


def main() -> None:
    cases = load_cases()
    client = Client()
    if client.has_dataset(dataset_name=DATASET):
        ds = client.read_dataset(dataset_name=DATASET)
    else:
        ds = client.create_dataset(
            dataset_name=DATASET,
            description="Discontinuity lane on rdrobust and rddensity: five real datasets through the router, Card and Krueger and students forced",
        )
    by_id = {e.metadata.get("case_id"): e for e in client.list_examples(dataset_id=ds.id) if e.metadata}
    changed = 0
    for c in cases:
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
    print(f"dataset {DATASET}: {len(by_id)} existing ({changed} updated), {len(new)} added")


if __name__ == "__main__":
    main()
