"""Upload evals/cases.yaml to a LangSmith dataset. Idempotent on case ids."""

from __future__ import annotations

from pathlib import Path

import yaml
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

DATASET = "causal-chat-v0"
HERE = Path(__file__).parent


def load_cases() -> list[dict]:
    cases = yaml.safe_load((HERE / "cases.yaml").read_text())
    for c in cases:
        c["context_text"] = (HERE / c["context"]).read_text()
    return cases


def main() -> None:
    cases = load_cases()
    client = Client()
    if client.has_dataset(dataset_name=DATASET):
        ds = client.read_dataset(dataset_name=DATASET)
    else:
        ds = client.create_dataset(dataset_name=DATASET, description="The desk: scripted conversations before, through, and after the analysis, with revisions and new questions")
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
            inputs=[{"csv": c["csv"], "context": c["context_text"], "question": c["question"], "turns": c["turns"], "name": c["id"]} for c in new],
            outputs=[c["expected"] for c in new],
            metadata=[{"case_id": c["id"]} for c in new],
        )
    print(f"dataset {DATASET}: {len(by_id)} existing ({changed} updated), {len(new)} added")


if __name__ == "__main__":
    main()
