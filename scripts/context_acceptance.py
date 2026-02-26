"""Opt-in real model journey against a running API/worker. Creates review datasets."""

import argparse
import json
import time
from pathlib import Path
from uuid import uuid4

import httpx
from dotenv import dotenv_values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=Path("."))
    parser.add_argument(
        "--cases", type=Path, help="JSON list of title, csv and document paths relative to reference."
    )
    parser.add_argument("--url", default="http://127.0.0.1:8008")
    parser.add_argument("--label", default="Context review")
    parser.add_argument("--output", type=Path, default=Path(".runtime/context-acceptance.json"))
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse the report's datasets, starting fresh preparations only for failed operations.",
    )
    args = parser.parse_args()
    env = dotenv_values(".env")
    cases = [
        (
            "Students Performance",
            "data/raw/students-performance-in-exams/StudentsPerformance.csv",
            "data/context/students.md",
        ),
        (
            "Government Transfers",
            "data/raw/gov-transfers-uruguay/gov_transfers.csv",
            "data/context/gov_transfers.md",
        ),
    ]
    if args.cases:
        cases = [(c["title"], c["csv"], c["document"]) for c in json.loads(args.cases.read_text())]
    records = []
    existing = {r["title"]: r for r in json.loads(args.output.read_text())} if args.reuse else {}
    with httpx.Client(base_url=args.url, headers={"X-Causal-Client": "web"}, timeout=30) as client:
        client.post(
            "/api/auth/login", json={"email": env["BOOTSTRAP_EMAIL"], "password": env["BOOTSTRAP_PASSWORD"]}
        ).raise_for_status()
        for title, csv, document in cases:
            if args.reuse:
                previous = existing[title]
                dataset, source = previous["dataset"], previous["source"]
                base = f"/api/versions/{dataset['version']['id']}/context"
                status = client.get(base + "/preparations/" + previous["operation"]["id"])
                status.raise_for_status()
                if status.json()["status"] != "failed":
                    records.append(previous)
                    continue
                workspace = client.get(base)
                workspace.raise_for_status()
                assert workspace.json()["current_revision_id"] is None, "Reuse requires an unprepared pack."
                response = client.post(
                    base + "/preparations",
                    json={
                        "submission_id": str(uuid4()),
                        "expected_revision_id": None,
                        "source_ids": [source["id"]],
                    },
                )
                response.raise_for_status()
                records.append(
                    {"title": title, "dataset": dataset, "source": source, "operation": response.json()}
                )
                continue
            path = args.reference / csv
            uploaded = client.post(
                "/api/datasets",
                data={"title": f"{args.label} / {title}", "submission_id": str(uuid4())},
                files={"file": (path.name, path.read_bytes(), "text/csv")},
            )
            uploaded.raise_for_status()
            dataset = uploaded.json()
            base = f"/api/versions/{dataset['version']['id']}/context"
            path = args.reference / document
            uploaded = client.post(
                base + "/sources",
                data={"title": title + " study account", "kind": "file", "submission_id": str(uuid4())},
                files={"file": (path.name, path.read_bytes(), "text/markdown")},
            )
            uploaded.raise_for_status()
            source = uploaded.json()
            request = {
                "submission_id": str(uuid4()),
                "expected_revision_id": None,
                "source_ids": [source["id"]],
            }
            response = client.post(base + "/preparations", json=request)
            response.raise_for_status()
            operation = response.json()
            records.append({"title": title, "dataset": dataset, "source": source, "operation": operation})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2) + "\n")
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            active = False
            for record in records:
                base = f"/api/versions/{record['dataset']['version']['id']}/context"
                result = client.get(base + "/preparations/" + record["operation"]["id"])
                result.raise_for_status()
                operation = result.json()
                record["operation"] = operation
                if operation["status"] in ["queued", "running"]:
                    active = True
                elif operation["status"] == "succeeded" and "revision" not in record:
                    revision = client.get(base + "/revisions/" + operation["result_revision_id"])
                    revision.raise_for_status()
                    record["revision"] = revision.json()
                    original = client.get(base + "/sources/" + record["source"]["id"] + "/file")
                    original.raise_for_status()
                    import hashlib

                    assert hashlib.sha256(original.content).hexdigest() == record["source"]["sha256"]
                    print(
                        record["title"],
                        operation["status"],
                        "calls",
                        operation["calls"],
                        "revision",
                        operation["result_revision_id"],
                        flush=True,
                    )
                elif operation["status"] == "failed" and not record.get("reported"):
                    print(record["title"], "failed:", operation["error"], flush=True)
                    record["reported"] = True
            args.output.write_text(json.dumps(records, indent=2) + "\n")
            if not active:
                break
            time.sleep(3)
        assert all(r["operation"]["status"] == "succeeded" for r in records), (
            "Acceptance preparations did not all succeed; inspect the saved report."
        )
        students = next((r for r in records if r["title"] == "Students Performance"), None)
        if students:
            columns = {c["identity"]["name"]: c for c in students["revision"]["content"]["columns"].values()}
            # The supplied account describes these three columns in one paragraph.
            # Preserving their identities while dropping two meanings is not coverage.
            for name in ["math score", "reading score", "writing score"]:
                claims = [
                    s["value"].lower()
                    for field, statements in columns[name].items()
                    if field != "identity"
                    for s in statements.values()
                ]
                assert any("exam" in claim or "mark" in claim for claim in claims), (
                    f"Missing shared exam meaning for {name}"
                )
                assert any("100" in claim for claim in claims), f"Missing shared range for {name}"
            print(
                "Shared score-column meanings and ranges are present under all three original columns.",
                flush=True,
            )
        print(
            "Real-model preparation journeys completed. Inspect the reports and traces for semantic acceptance.",
            flush=True,
        )


if __name__ == "__main__":
    main()
