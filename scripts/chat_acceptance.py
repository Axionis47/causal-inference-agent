"""Opt-in real-model, isolated-runner and saved multi-turn acceptance."""

import argparse
import json
import time
from pathlib import Path
from uuid import uuid4

import httpx
from dotenv import dotenv_values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8088")
    parser.add_argument("--output", type=Path, default=Path(".runtime/chat-acceptance.json"))
    parser.add_argument("--context-report", type=Path, default=Path(".runtime/context-acceptance.json"))
    args = parser.parse_args()
    env = dotenv_values(".env")
    reference = json.loads(args.context_report.read_text())[0]
    dataset = reference["dataset"]
    with httpx.Client(base_url=args.url, headers={"X-Causal-Client": "web"}, timeout=30) as client:
        client.post(
            "/api/auth/login", json={"email": env["BOOTSTRAP_EMAIL"], "password": env["BOOTSTRAP_PASSWORD"]}
        ).raise_for_status()
        workspace = client.get(f"/api/versions/{dataset['version']['id']}/context").json()
        response = client.post(
            f"/api/versions/{dataset['version']['id']}/conversations",
            json={
                "submission_id": str(uuid4()),
                "title": "Reading scores: exploration and follow-up",
                "context_revision_id": workspace["current_revision_id"],
            },
        )
        response.raise_for_status()
        chat = response.json()
        base = f"/api/conversations/{chat['id']}"
        report = {"dataset": dataset, "conversation": chat, "turns": [], "artifacts": []}
        args.output.write_text(json.dumps(report, indent=2))
        messages = [
            "Plot mean reading score by lunch category and include the group sample sizes. This is a descriptive comparison.",
            "Keep that same comparison, but make the bars horizontal and label each mean and group count.",
            "Does that establish that lunch causes higher reading scores? Explain using the supplied account and what we just computed. Do not run another computation.",
        ]
        for number, message in enumerate(messages):
            selected = [report["artifacts"][-1]["id"]] if number == 1 and report["artifacts"] else []
            response = client.post(
                base + "/turns",
                json={
                    "submission_id": str(uuid4()),
                    "expected_turn": number,
                    "text": message,
                    "artifact_ids": selected,
                },
            )
            response.raise_for_status()
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                saved = client.get(base)
                saved.raise_for_status()
                data = saved.json()
                last = data["turns"][-1]
                if last["status"] in {"succeeded", "failed"}:
                    break
                time.sleep(2)
            else:
                raise RuntimeError("Chat turn exceeded its acceptance wait.")
            report.update({"turns": data["turns"], "artifacts": data["artifacts"]})
            args.output.write_text(json.dumps(report, indent=2))
            print(
                json.dumps(
                    {
                        "turn": number + 1,
                        "status": last["status"],
                        "calls": last["calls"],
                        "error": last["error"],
                        "trace_id": last["trace_id"],
                    }
                ),
                flush=True,
            )
            assert last["status"] == "succeeded", last
        assert len(report["artifacts"]) == 2, "Only the first two messages request new computations."
        print(
            f"Review {args.url}/datasets/{dataset['id']}/chat/{dataset['version']['id']}?conversation={chat['id']}"
        )


if __name__ == "__main__":
    main()
