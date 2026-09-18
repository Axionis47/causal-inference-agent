"""Upload evals/cases.yaml to a LangSmith dataset. Idempotent on example ids."""

from __future__ import annotations

from pathlib import Path

import yaml
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

DATASET = "causal-router-v0"


def main() -> None:
    cases = yaml.safe_load((Path(__file__).parent / "cases.yaml").read_text())
    client = Client()
    if client.has_dataset(dataset_name=DATASET):
        ds = client.read_dataset(dataset_name=DATASET)
    else:
        ds = client.create_dataset(dataset_name=DATASET, description="Router eval: one question per lane on real datasets")
    by_id = {e.metadata.get("case_id"): e for e in client.list_examples(dataset_id=ds.id) if e.metadata}
    existing = set(by_id)
    changed = 0
    for c in cases:  # keep expected outputs in step with cases.yaml
        e = by_id.get(c["id"])
        if e is not None and (e.outputs or {}) != c["expected"]:
            client.update_example(example_id=e.id, outputs=c["expected"])
            changed += 1
    new = [c for c in cases if c["id"] not in existing]
    if new:
        client.create_examples(
            dataset_id=ds.id,
            inputs=[{"dataset": c["dataset"], "question": c["question"]} for c in new],
            outputs=[c["expected"] for c in new],
            metadata=[{"case_id": c["id"]} for c in new],
        )
    print(f"dataset {DATASET}: {len(existing)} existing ({changed} updated), {len(new)} added")


if __name__ == "__main__":
    main()
