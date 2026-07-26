#!/usr/bin/env python
"""Drive the API the way the browser will: submit, watch, choose, read.

    uvicorn causal.api:app --port 8200      # in one terminal
    python verify_api.py                    # in another

Proves the whole loop: a job parks at the design gate, the live tape reports it,
a choice resumes the run, and the result carries the benchmark number.
"""
from __future__ import annotations

import json
import sys
import time

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8200"

CASES = [
    dict(dataset="heart_failure",
         question="Does high blood pressure affect how long patients survive?",
         context="DEATH_EVENT marks death during follow-up; time is days observed.",
         lane="survival",
         kwargs=dict(treatment="high_blood_pressure", duration="time", event="DEATH_EVENT"),
         expect=1.546),
    dict(dataset="card_krueger",
         question="Did New Jersey's minimum wage rise change fast-food employment?",
         context="Stores in NJ and PA surveyed before and after. period 0 is before, 1 is after.",
         lane="did",
         kwargs=dict(outcome="fte", group="state", period="period",
                     treated_group="NJ", unit="store_id"),
         expect=2.754),
    dict(dataset="lalonde",
         question="Did the job training program raise earnings in 1978?",
         context="NSW training trial treated units, combined with a PSID comparison group.",
         lane="matching",
         kwargs=dict(outcome="re78", treatment="treat",
                     covariates=["age","educ","black","hispan","married","nodegree","re74","re75"]),
         expect=1110.12),
    dict(dataset="ihdp",
         question="What is the effect of the intervention on the child's test score?",
         context="Infant Health and Development Program. y_factual is the observed score.",
         lane="observational",
         kwargs=dict(outcome="y_factual", treatment="treatment",
                     covariates=[f"x{i}" for i in range(1, 26)]),
         expect=3.92867),
    dict(dataset="card",
         question="What is the effect of years of schooling on log hourly wages?",
         context="US men in 1976. nearc4 marks growing up near a four-year college. Use lwage.",
         lane="iv",
         kwargs=dict(outcome="lwage", treatment="educ", instrument="nearc4",
                     covariates=["exper","expersq","black","south","smsa"]),
         expect=0.132289),
    dict(dataset="bank",
         question="Does the higher recovery strategy increase the amount recovered?",
         context="Customers above an expected recovery of 1000 get a more intensive strategy.",
         lane="rdd",
         kwargs=dict(outcome="actual_recovery_amount",
                     running="expected_recovery_amount", cutoff=1000.0),
         expect=263.8),
    dict(dataset="student",
         question="Does study time affect final grades through past failures?",
         context="Portuguese secondary school records. G3 is the final grade.",
         lane="mediation",
         kwargs=dict(outcome="G3", treatment="studytime", mediator="failures"),
         expect=0.2093),
    dict(dataset="visitors",
         question="Did anything change site traffic at the start of 2018?",
         context="Daily website statistics. Unique.Visits is the visitor count.",
         lane="time_series",
         kwargs=dict(outcome="Unique.Visits", time="Date", intervention="2018-01-01"),
         expect=389.4),
]


def wait_for(client: httpx.Client, job: str, status: str, timeout: float = 120) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"{BASE}/jobs/{job}").json()
        if body["status"] == status:
            return body
        if body["status"] == "failed":
            return body
        time.sleep(0.5)
    raise TimeoutError(f"job {job} never reached {status}")


def main() -> int:
    ok = True
    with httpx.Client(timeout=30) as client:
        print("GET /health  ", client.get(f"{BASE}/health").json())
        sets = client.get(f"{BASE}/datasets").json()
        print("GET /datasets", f"{len(sets)} available:", [s["name"] for s in sets])
        print()

        for case in CASES:
            print("=" * 74)
            print(case["question"])

            created = client.post(f"{BASE}/jobs", json={
                "dataset": case["dataset"],
                "question": case["question"],
                "context": case["context"],
            })
            job = created.json()["id"]
            print(f"  POST /jobs            -> {created.status_code} id={job}")

            # the live tape, read until the run parks
            seen = []
            with client.stream("GET", f"{BASE}/jobs/{job}/stream", timeout=90) as tape:
                for line in tape.iter_lines():
                    if line.startswith("event:"):
                        seen.append(line.split(":", 1)[1].strip())
                    if seen and seen[-1] in ("waiting_for_you", "completed", "failed"):
                        break
            print(f"  GET  .../stream       -> {seen}")

            parked = wait_for(client, job, "waiting_for_you")
            if parked["status"] != "waiting_for_you":
                print(f"  FAILED at intake: {parked.get('error')}")
                ok = False
                continue
            offered = [o["lane"] for o in parked["menu"] if o["available"]]
            print(f"  GET  /jobs/{{id}}       -> parked; available={offered}")
            print(f"                           exposure={parked['intake']['exposure']}"
                  f" -> {parked['intake']['outcome']}")

            chose = client.post(f"{BASE}/jobs/{job}/design",
                                json={"lane": case["lane"], "kwargs": case["kwargs"]})
            print(f"  POST .../design       -> {chose.status_code} lane={case['lane']}")

            done = wait_for(client, job, "completed", timeout=180)
            res = client.get(f"{BASE}/jobs/{job}/result").json()
            if res["status"] != "completed":
                print(f"  FAILED: {res.get('error')}")
                ok = False
                continue

            value = res["estimate"]["value"]
            close = abs(value - case["expect"]) / abs(case["expect"]) < 0.02
            ok = ok and close
            print(f"  GET  .../result       -> {res['estimate']['estimand']}="
                  f"{value:.4g} (expected ~{case['expect']}) "
                  f"{'MATCH' if close else 'MISMATCH'}")
            print(f"                           strength={res['strength']}")
            print(f"  readout: {res['narrative'][:150].strip()}...")
            print()

        # the gate refuses a second answer
        print("=" * 74)
        body = client.post(f"{BASE}/jobs/{job}/design", json={"lane": "survival"})
        print(f"  choosing twice        -> {body.status_code} "
              f"{'PASS (rejected)' if body.status_code == 409 else 'FAIL (accepted)'}")
        ok = ok and body.status_code == 409

        missing = client.get(f"{BASE}/jobs/doesnotexist")
        print(f"  unknown job           -> {missing.status_code} "
              f"{'PASS' if missing.status_code == 404 else 'FAIL'}")
        ok = ok and missing.status_code == 404

    print()
    print("API GREEN" if ok else "PROBLEMS ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
