#!/usr/bin/env python
"""Run the whole pipeline on all eight datasets: question in, readout out.

    python verify_pipeline.py            # no narration, one LLM call per dataset
    python verify_pipeline.py --narrate  # adds the written readout, two calls

Each case is a real question in English against a real dataset. The intake step
must find the right columns; the design menu must offer the right options; the
chosen lane must produce the benchmark number from phase 1.

The lane is named here rather than inferred, because that is the gate: code
says what is possible, a person says what to do.
"""
from __future__ import annotations

import sys
import warnings

import pandas as pd

from causal.run import estimate, plan

warnings.filterwarnings("ignore")

CASES = [
    dict(
        data="heart_failure", lane="survival",
        question="Does high blood pressure affect how long patients survive?",
        context="Clinical records. DEATH_EVENT marks death during follow-up; time is days observed.",
        expect_treatment="high_blood_pressure",
        kwargs=dict(treatment="high_blood_pressure", duration="time", event="DEATH_EVENT"),
    ),
    dict(
        data="lalonde", lane="matching",
        question="Did the job training program raise earnings in 1978?",
        context="NSW training trial treated units, combined with a PSID comparison group.",
        expect_treatment="treat",
        kwargs=dict(outcome="re78", treatment="treat",
                    covariates=("age", "educ", "black", "hispan", "married",
                                "nodegree", "re74", "re75")),
    ),
    dict(
        data="ihdp", lane="observational",
        question="What is the effect of the intervention on the child's test score?",
        context="Infant Health and Development Program. y_factual is the observed score.",
        expect_treatment="treatment",
        kwargs=dict(outcome="y_factual", treatment="treatment",
                    covariates=tuple(f"x{i}" for i in range(1, 26))),
    ),
    dict(
        data="card", lane="iv",
        question="What is the effect of years of schooling on log hourly wages?",
        context="US men in 1976. nearc4 marks growing up near a four-year college. Use lwage.",
        expect_treatment="educ",
        kwargs=dict(outcome="lwage", treatment="educ", instrument="nearc4",
                    covariates=("exper", "expersq", "black", "south", "smsa")),
    ),
    dict(
        data="card_krueger", lane="did",
        question="Did New Jersey's minimum wage rise change fast-food employment?",
        context="Stores in NJ and PA surveyed before and after. period 0 is before, 1 is after.",
        expect_treatment=None,
        kwargs=dict(outcome="fte", group="state", period="period",
                    treated_group="NJ", unit="store_id"),
    ),
    dict(
        data="bank", lane="rdd",
        question="Does the higher recovery strategy increase the amount recovered?",
        context="Customers above an expected recovery of 1000 get a more intensive strategy.",
        expect_treatment=None,
        kwargs=dict(outcome="actual_recovery_amount",
                    running="expected_recovery_amount", cutoff=1000.0),
    ),
    dict(
        data="student", lane="mediation",
        question="Does study time affect final grades through past failures?",
        context="Portuguese secondary school records. G3 is the final grade.",
        expect_treatment="studytime",
        kwargs=dict(outcome="G3", treatment="studytime", mediator="failures"),
    ),
    dict(
        data="visitors", lane="time_series",
        question="Did anything change site traffic at the start of 2018?",
        context="Daily website statistics. Unique.Visits is the visitor count.",
        expect_treatment=None,
        kwargs=dict(outcome="Unique.Visits", time="Date", intervention="2018-01-01"),
    ),
]


def main() -> int:
    do_narrate = "--narrate" in sys.argv
    ok = True
    for case in CASES:
        df = pd.read_csv(f"data/{case['data']}.csv")
        print("=" * 78)
        print(f"{case['data']}  --  {case['question']}")
        a = plan(df, case["question"], case["context"])

        if a.stopped:
            print(f"  STOPPED at intake: {a.stopped}")
            ok = False
            continue

        read_ok = (case["expect_treatment"] is None
                   or a.intake.treatment == case["expect_treatment"])
        print(f"  intake   {a.intake.exposure} -> {a.intake.outcome} "
              f"({a.intake.question_family}, {a.intake.confidence} confidence)"
              f"{'' if read_ok else '   [expected ' + case['expect_treatment'] + ']'}")
        print(f"  menu     available: {[o.lane for o in a.menu if o.available]}")
        print(f"  chosen   {case['lane']}")

        a = estimate(df, a, case["lane"], narrate_result=do_narrate, **case["kwargs"])
        if a.estimate is None:
            print(f"  FAILED: {a.stopped}")
            ok = False
            continue
        print(f"  result   {a.estimate}")
        print(f"  strength {a.strength}")
        if do_narrate:
            print()
            for line in a.narrative.splitlines():
                print("    " + line)
        print()

    print("=" * 78)
    print("PIPELINE GREEN" if ok else "PROBLEMS ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
