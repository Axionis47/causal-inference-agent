"""Opt-in local acceptance. Real models receive the account, never the numerical oracle.

Run one explicit step at a time: start, inspect, discuss, apply, launch, audit. Approval is
deliberately a separate browser action after review of the frozen program.
"""

import argparse
import csv
import io
import json
import random
import re
import statistics
from pathlib import Path
from uuid import uuid4

import httpx
from dotenv import dotenv_values

OUT = Path(".runtime/frozen-context-v19")
QUESTION = "Estimate the unadjusted mean difference in visit for history=1 minus history=0 across all enrolled synthetic participants, with a two-sided 95% Welch t interval. Use no adjustment, interaction or subgroup analysis."
ACCOUNT = """This is a private synthetic fixture, not a public marketing dataset. Familiar column labels carry only the meanings given here.
Each row is one distinct enrolled synthetic participant. The target population is all enrolled synthetic participants.
history records randomized assignment at baseline, not prior purchases: 1 means offer of intervention A, 0 means no offer. Assignment is an independent fair coin for each participant. Assignment and received offer coincide in this fixture.
visit records a continuous simulated performance score in points at a common one-month follow-up after assignment. It is not a binary visit indicator.
mens records a baseline inventory category, recorded before assignment. Its name does not denote sex or purchasing history. No effect-modification claim is supplied for this column.
The requested comparison is history=1 minus history=0, an intention-to-treat mean contrast in all enrolled participants. Use the unadjusted difference in means with a two-sided 95% Welch t interval. No adjustment variables, interactions, subgroup analyses, weighting or searches are requested.
No missing values are present in this generated fixture. Scores are not bounded by design. There is one observation per participant.
Consistency and no interference are working causal assumptions for this synthetic exercise, not empirically established facts. Preserve them as conditional premises. No significance-driven change is permitted.
"""


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["start", "inspect", "discuss", "apply", "launch", "audit"])
    args = parser.parse_args()
    env = dotenv_values(".env")
    with httpx.Client(base_url="http://127.0.0.1:8088", headers={"X-Causal-Client": "web"}, timeout=60) as client:
        def request(method, path, **kwargs):
            result = client.request(method, path, **kwargs)
            result.raise_for_status()
            return result.json()

        request("POST", "/api/auth/login", json={"email": env["BOOTSTRAP_EMAIL"], "password": env["BOOTSTRAP_PASSWORD"]})
        if args.step == "start":
            if (OUT / "state.json").exists():
                raise SystemExit("This trajectory already exists. Inspect it; do not silently reroll failed work.")
            rng = random.Random(901731)
            rows = [(rng.randrange(2), rng.randrange(3), rng.gauss(0, 1)) for _ in range(160)]
            values = [(t, category, round(17.5 + 2.25 * t + 3 * noise, 8)) for t, category, noise in rows]
            stream = io.StringIO()
            writer = csv.writer(stream)
            writer.writerow(["history", "mens", "visit"])
            writer.writerows(values)
            groups = {t: [y for treatment, _, y in values if treatment == t] for t in (0, 1)}
            components = [statistics.variance(groups[t]) / len(groups[t]) for t in (0, 1)]
            save("hidden-oracle.json", {"estimate": statistics.mean(groups[1]) - statistics.mean(groups[0]),
                                       "standard_error": sum(components) ** .5,
                                       "welch_df": sum(components) ** 2 / sum(components[t] ** 2 / (len(groups[t]) - 1) for t in (0, 1)),
                                       "n": len(values), "exposure": "Never sent to context or analysis models."})
            data = request("POST", "/api/datasets", data={"title": "Private frozen-context mechanism test", "submission_id": str(uuid4())},
                           files={"file": ("private_mechanism.csv", stream.getvalue().encode(), "text/csv")})
            base = f"/api/versions/{data['version']['id']}/context"
            source = request("POST", base + "/sources", data={"title": "Explicit synthetic mechanism", "kind": "note", "submission_id": str(uuid4())},
                             files={"file": ("mechanism.txt", ACCOUNT.encode(), "text/plain")})
            prepared = request("POST", base + "/preparations", json={"submission_id": str(uuid4()), "expected_revision_id": None, "source_ids": [source["id"]]})
            state = {"dataset": data, "source_id": source["id"], "preparation_id": prepared["id"]}
            save("state.json", state)
            print(json.dumps({"stage": "context_preparation", "preparation_id": prepared["id"]}))
            return
        state = json.loads((OUT / "state.json").read_text())
        base = f"/api/versions/{state['dataset']['version']['id']}/context"
        workspace = request("GET", base)
        save("context-workspace.json", workspace)
        if workspace["current_revision_id"]:
            save("context-revision.json", request("GET", base + "/revisions/" + workspace["current_revision_id"]))
        if args.step == "discuss":
            if state.get("conversation"):
                raise SystemExit("A discussion already exists; inspect its saved attempt first.")
            revision = workspace["current_revision_id"]
            if not revision:
                raise SystemExit("Context preparation has not produced a revision.")
            chat = request("POST", f"/api/versions/{state['dataset']['version']['id']}/conversations",
                           json={"title": "Freeze private study context", "submission_id": str(uuid4()), "context_revision_id": revision})
            text = ("Finalize this analysis brief through a proposal I can Apply. Review or add the essential source statements for target, intervention, change, comparison, population, unit and timing using my exact account below. "
                    "Use existing compatible context entries where possible. Optional unknown roles need not be settled. Propose the exact analysis question below, and select exact context keys for the brief. Do not launch analysis.\n\n"
                    + ACCOUNT + "\nFinal question: " + QUESTION)
            turn = request("POST", f"/api/conversations/{chat['id']}/turns", json={"submission_id": str(uuid4()), "expected_turn": 0, "mode": "context", "text": text})
            state.update(conversation=chat, turn_id=turn["id"])
        if state.get("conversation"):
            path = f"/api/conversations/{state['conversation']['id']}"
            if args.step == "apply":
                request("POST", path + f"/turns/{state['turn_id']}/apply-context")
            ready = request("GET", path + "/analysis-readiness")
            save("readiness.json", ready)
            if args.step == "launch":
                if state.get("analysis_id"):
                    raise SystemExit("An analysis already exists; inspect its saved attempt first.")
                if ready["status"] == "needs_context":
                    raise SystemExit("Applied context is not ready. Inspect the readiness checks.")
                run = request("POST", path + "/analyses", json={"submission_id": str(uuid4()), "expected_context_revision_id": ready["context_revision_id"],
                                                                "expected_readiness_fingerprint": ready["fingerprint"]})
                state["analysis_id"] = run["id"]
            save("conversation.json", request("GET", path))
        if state.get("analysis_id"):
            run = request("GET", f"/api/analyses/{state['analysis_id']}")
            save("analysis.json", run)
            print(json.dumps({"stage": run["stage"], "status": run["status"], "calls": run["calls"], "error": run["error"]}))
            if args.step == "audit":
                if run["status"] not in {"failed", "blocked", "awaiting_approval", "succeeded"}:
                    raise SystemExit("Audit only a settled trajectory.")
                root = f"/api/analyses/{run['id']}"
                frozen = request("GET", root + "/files/frozen-context.json")
                assert frozen == run["frozen_context"]
                events = run["events"]
                sequence = [event["sequence"] for event in events]
                assert sequence == list(range(1, len(sequence) + 1))
                cursor = len(sequence) // 2
                response = client.get(root + "/events", headers={"Last-Event-ID": str(cursor)})
                response.raise_for_status()
                replay = [int(value) for value in re.findall(r"^id: (\d+)$", response.text, re.M)]
                assert replay == sequence[cursor:]
                for attempt in run["shared_attempts"]:
                    save("provider-" + attempt["id"] + ".json", request("GET", root + "/provider-attempts/" + attempt["id"]))
                audit = {"analysis_id": run["id"], "context_hash": frozen["context_hash"],
                         "readiness_fingerprint": frozen["readiness_fingerprint"], "calls": run["calls"],
                         "event_count": len(sequence), "replayed_sequences": replay,
                         "exact_frozen_artifact": True, "snapshot_hash": run["snapshot_hash"],
                         "program_count": len(run["programs"]), "status": run["status"]}
                previous = OUT / "audit.json"
                if previous.exists():
                    assert json.loads(previous.read_text()) == audit, "Settled records changed during reopening/restart."
                save("audit.json", audit)
                print(json.dumps(audit))
        else:
            print(json.dumps({"preparations": [(p["status"], p.get("stage")) for p in workspace["preparations"]], "chat": state.get("conversation", {}).get("id")}))
        save("state.json", state)


if __name__ == "__main__":
    main()
