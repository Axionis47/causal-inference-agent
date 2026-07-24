"""Opt-in real-model acceptance on a new, explicitly synthetic randomized study."""

import argparse
import csv
import io
import json
import random
import time
from pathlib import Path
from uuid import uuid4

import httpx
from dotenv import dotenv_values

OUT = Path(".runtime/checkpoint-acceptance.json")
report = {}


def save():
    OUT.write_text(json.dumps(report, indent=2))


def wait(client, url, terminal, field=None):
    previous = None
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        try:
            response = client.get(url)
        except httpx.TransportError:
            time.sleep(2)
            continue
        response.raise_for_status()
        body = response.json()
        value = body[field][-1] if field else body
        state = (value["status"], value.get("stage"), value.get("calls"))
        if state != previous:
            print(json.dumps({"url": url, "state": state}), flush=True)
            previous = state
        if value["status"] in terminal:
            return body
        time.sleep(2)
    raise RuntimeError("Acceptance wait expired")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opaque-columns", action="store_true")
    parser.add_argument("--reuse-context", action="store_true")
    parser.add_argument("--manual-approval", action="store_true")
    parser.add_argument("--resume-analysis", action="store_true")
    args = parser.parse_args()
    global OUT
    if args.opaque_columns:
        OUT = Path(".runtime/checkpoint-acceptance-opaque.json")
    previous_report = json.loads(OUT.read_text()) if args.reuse_context or args.resume_analysis else None
    if args.resume_analysis:
        report.update(previous_report)
    env = dotenv_values(".env")
    rng = random.Random(1729)
    rows = []
    for index in range(120):
        treatment = rng.choice([0, 1])
        score = 50 + 4 * treatment + rng.gauss(0, 6)
        rows.append((index + 1, treatment, round(score, 6)))
    stream = io.StringIO()
    writer = csv.writer(stream)
    names = ["participant_id", "assigned_program", "followup_score"]
    mapping = (
        dict(zip(names, ["r9", "z17", "p4"], strict=True))
        if args.opaque_columns
        else {name: name for name in names}
    )
    writer.writerow(
        [mapping["followup_score"], mapping["participant_id"], mapping["assigned_program"]]
        if args.opaque_columns
        else names
    )
    writer.writerows([(r[2], r[0], r[1]) for r in rows] if args.opaque_columns else rows)
    means = {t: sum(r[2] for r in rows if r[1] == t) / sum(r[1] == t for r in rows) for t in [0, 1]}
    report["independent_mean_difference"] = means[1] - means[0]
    note = """This is a synthetic randomized trial for application verification, not real clinical or educational evidence.
Each row is one distinct participant. participant_id is an opaque identifier, not a cause or adjustment variable.
The target population is the enrolled study sample. No subgroup restriction or weighting is intended.
assigned_program records random assignment at baseline: 1 means offered the program, 0 means no program offer.
Assignment was a simulated independent fair random draw per participant. The intervention changes only the program offer.
The target contrast is assignment to the program offer versus no offer, an intention-to-treat average effect in the study sample.
followup_score measures participant performance on a simulated evaluation task. Higher values mean better performance. It is a continuous outcome measured in score points on a common instrument one month after assignment.
All observations have the same measurement schedule. No missing-value codes or empty cells are present. Scores are not bounded by design.
For this synthetic exercise assume consistency, no interference and no differential follow-up measurement. These are working premises, not empirically demonstrated facts.
No baseline covariates were measured. Use the unadjusted difference in mean followup_score (assigned_program=1 minus 0), a two-sided 95% Welch t confidence interval and all rows. Do not search over models or fit alternatives.
"""
    for original, alias in mapping.items():
        note = note.replace(original, alias)
    question = "Using the supplied synthetic randomized study account, estimate the unadjusted intention-to-treat mean difference in followup_score for assigned_program=1 versus 0 in all enrolled participants, with a two-sided 95% Welch t confidence interval. Preserve the study assumptions and use no adjustment variables."
    for original, alias in mapping.items():
        question = question.replace(original, alias)
    report["column_mapping"] = mapping
    with httpx.Client(
        base_url="http://127.0.0.1:8088", headers={"X-Causal-Client": "web"}, timeout=30
    ) as client:
        for _ in range(30):
            if client.get("/api/health").status_code == 200:
                break
            time.sleep(1)
        client.post(
            "/api/auth/login", json={"email": env["BOOTSTRAP_EMAIL"], "password": env["BOOTSTRAP_PASSWORD"]}
        ).raise_for_status()
        if previous_report:
            dataset = previous_report["dataset"]
            version = dataset["version"]["id"]
            revision = previous_report["context_revision_id"]
            chat = previous_report["conversation"]
            report["dataset"] = dataset
            report["context_revision_id"] = revision
        else:
            result = client.post(
                "/api/datasets",
                data={
                    "title": "Synthetic opaque columns"
                    if args.opaque_columns
                    else "Synthetic randomized program: one-shot acceptance",
                    "submission_id": str(uuid4()),
                },
                files={"file": ("synthetic_trial.csv", stream.getvalue().encode(), "text/csv")},
            )
            result.raise_for_status()
            dataset = result.json()
            report["dataset"] = dataset
            save()
            version = dataset["version"]["id"]
            base = f"/api/versions/{version}/context"
            source = client.post(
                base + "/sources",
                data={"title": "Synthetic study account", "kind": "note", "submission_id": str(uuid4())},
                files={"file": ("study.txt", note.encode(), "text/plain")},
            )
            source.raise_for_status()
            prepared = client.post(
                base + "/preparations",
                json={
                    "submission_id": str(uuid4()),
                    "expected_revision_id": None,
                    "source_ids": [source.json()["id"]],
                },
            )
            prepared.raise_for_status()
            # Context workspace orders preparations newest first.
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                workspace = client.get(base).json()
                prep = workspace["preparations"][0]
                if prep["status"] in ["succeeded", "failed"]:
                    break
                time.sleep(2)
            assert prep["status"] == "succeeded", prep
            revision = workspace["current_revision_id"]
            report["context_revision_id"] = revision
            save()
            response = client.post(
                f"/api/versions/{version}/conversations",
                json={
                    "submission_id": str(uuid4()),
                    "title": "Synthetic causal analysis and context discussion",
                    "context_revision_id": revision,
                },
            )
            response.raise_for_status()
            chat = response.json()
        report["conversation"] = chat
        save()
        chat_url = f"/api/conversations/{chat['id']}"
        if args.resume_analysis:
            identity = previous_report["analysis_id"]
            state = client.get(f"/api/analyses/{identity}").json()
            if state["status"] == "failed" and state["retryable"]:
                client.post(f"/api/analyses/{identity}/retry").raise_for_status()
        else:
            response = client.post(
                chat_url + "/analyses",
                json={
                    "submission_id": str(uuid4()),
                    "question": question,
                    "expected_context_revision_id": revision,
                },
            )
            response.raise_for_status()
            identity = response.json()["id"]
        report["analysis_id"] = identity
        save()
        run_url = f"/api/analyses/{identity}"
        saved = wait(client, run_url, ["awaiting_approval", "blocked", "failed", "succeeded"])
        report["prepared"] = saved
        save()
        assert saved["status"] in ["awaiting_approval", "succeeded"], (
            f"{saved['status']} at {saved['stage']}; inspect {OUT}"
        )
        if not args.resume_analysis:
            assert not any(e["stage"] == "execute" for e in saved["events"])
        # This approval is solely for the synthetic acceptance fixture. Real user plans still pause.
        if saved["status"] == "succeeded":
            pass
        elif args.manual_approval:
            print(f"Awaiting browser approval for synthetic fixture {identity}", flush=True)
            wait(client, run_url, ["queued", "running", "succeeded", "failed", "blocked"])
        else:
            response = client.post(
                run_url + "/approve",
                json={"snapshot_hash": saved["snapshot_hash"], "program_hash": saved["programs"][-1]["hash"]},
            )
            response.raise_for_status()
        completed = wait(client, run_url, ["succeeded", "failed", "awaiting_approval", "blocked"])
        while completed["status"] == "awaiting_approval" and completed["programs"][-1]["revision"] <= 2:
            if args.manual_approval:
                print(f"Awaiting browser repair approval for synthetic fixture {identity}", flush=True)
                wait(client, run_url, ["queued", "running", "succeeded", "failed", "blocked"])
            else:
                client.post(
                    run_url + "/approve",
                    json={
                        "snapshot_hash": completed["snapshot_hash"],
                        "program_hash": completed["programs"][-1]["hash"],
                    },
                ).raise_for_status()
            completed = wait(client, run_url, ["succeeded", "failed", "awaiting_approval", "blocked"])
        report["completed"] = completed
        save()
        assert completed["status"] == "succeeded", (
            f"{completed['status']} at {completed['stage']}; inspect {OUT}"
        )
        effect_id = next(
            effect["id"]
            for effect in completed["snapshot"]["frame"]["treatments"]["effects"]
            if effect["priority"] == "primary"
        )
        assert (
            abs(completed["result"]["facts"][effect_id + "_estimate"] - report["independent_mean_difference"])
            < 1e-6
        )
        assert all(cp["status"] == "accepted" for cp in completed["checkpoints"])
        first_turn = len(client.get(chat_url).json()["turns"])
        response = client.post(
            chat_url + "/turns",
            json={
                "submission_id": str(uuid4()),
                "expected_turn": first_turn,
                "text": "Explain the saved analysis estimate and its assumptions. Do not compute again.",
            },
        )
        response.raise_for_status()
        data = wait(client, chat_url, ["succeeded", "failed"], "turns")
        report["discussion"] = data["turns"][-1]
        save()
        assert data["turns"][-1]["status"] == "succeeded", data["turns"][-1]
        for number, message in [
            (
                1,
                "Could interference between participants undermine the interpretation? Let us debate it without changing context.",
            ),
            (
                2,
                "Please add this qualification to the dataset context: The synthetic population represents only enrolled participants, with no claim of transportability to another population.",
            ),
        ]:
            response = client.post(
                chat_url + "/turns",
                json={
                    "submission_id": str(uuid4()),
                    "expected_turn": first_turn + number,
                    "text": message,
                    "mode": "context",
                },
            )
            response.raise_for_status()
            data = wait(client, chat_url, ["succeeded", "failed"], "turns")
            report.setdefault("context_turns", []).append(data["turns"][-1])
            save()
            assert data["turns"][-1]["status"] == "succeeded", data["turns"][-1]
            if number == 1:
                assert not data["turns"][-1]["reply"]["context_edits"]
        assert data["turns"][-1]["reply"]["context_edits"]
        # Leave the visible proposal unapplied for the browser acceptance step.
        print(
            json.dumps(
                {
                    "review": f"http://127.0.0.1:8088/datasets/{dataset['id']}/chat/{version}?conversation={chat['id']}&analysis={identity}",
                    "difference": report["independent_mean_difference"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
