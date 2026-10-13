"""Known-protocol datasets for real Gemini trajectory validation, never scripted judgements.

These are simulated studies with explicitly supplied generative assumptions. They
exercise method trajectories; they are not evidence that assumptions hold in real studies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from causal_agent.common import trace
from causal_agent.common.contracts import Belief, Candidate, Cited, ColumnBrief, ColumnFacts, Fact, Handoff, Provenance, Said, Scope
from causal_agent.families.adjustment.design import AdjustmentDesign
from causal_agent.families.adjustment.lane.knowledge import load_endpoint_catalogue
from causal_agent.lane.endpoints import admit

HERE = Path(__file__).parent


def scenario(name: str, csv: str) -> tuple[pd.DataFrame, Handoff]:
    rng = np.random.default_rng(7)
    n = 1200
    c = rng.normal(size=n)
    u = rng.normal(size=n)
    binary = rng.binomial(1, 0.5, n)
    scope = "average"
    instrument = None
    instruments = []
    controls = []
    mediator = None
    mods = []
    hidden = False
    assumptions = {}
    true_edges = []
    if name in ("backdoor", "randomized", "att", "atc"):
        scope = "on_treated" if name == "att" else "on_untreated" if name == "atc" else "average"
        t = binary if name == "randomized" else (c + rng.normal(size=n) > 0).astype(int)
        x = rng.normal(size=n)
        y = 5 * t + 2 * c + x + rng.normal(size=n)
        table = pd.DataFrame({"t": t, "c": c, "x": x, "y": y})
        true_edges = ([] if name == "randomized" else [("c", "t")]) + [("c", "y"), ("x", "y"), ("t", "y")]
        story = "Rows are independently sampled people. Baseline c causes treatment uptake and outcome; baseline x causes outcome only. Treatment may affect outcome additively; its magnitude is unknown. No other common causes exist. c and x are measured before treatment and cannot be changed by it. Outcome is measured afterwards. No interference or selection on outcomes."
        if name == "randomized":
            story = story.replace(
                "Baseline c causes treatment uptake and outcome;",
                "Treatment is randomly assigned, independently of all baseline variables. Baseline c causes outcome only;",
            )
    elif name == "wald":
        z = binary
        complier = rng.binomial(1, 0.7, n)
        t = z * complier
        y = 4 * t + u + rng.normal(size=n)
        table = pd.DataFrame({"t": t, "z": z, "y": y})
        true_edges = [("z", "t"), ("t", "y")]
        scope = "local"
        instrument = "z"
        story = "Rows are independent people. A randomized binary offer z pushes treatment t. The offer is fixed before uptake and has no direct or indirect outcome route except through t. No one takes treatment without an offer. Some people comply, some never take treatment; there are no defiers. The treatment effect on compliers is unknown. Outcome is recorded after uptake; no spillovers or missingness."
        assumptions = {"claim:exclusion.exogenous": True, "claim:monotonicity.believed": True}
    elif name in ("iv", "iv_conditional", "iv_single", "weak_iv"):
        z1 = c + rng.normal(size=n) if name == "iv_conditional" else rng.normal(size=n)
        z2 = rng.normal(size=n)
        t = (0.9 * z1 + 0.7 * z2 + 0.7 * u + (0.5 * c if name == "iv_conditional" else 0) + rng.normal(size=n) > 0).astype(int)
        y = 11 + 4 * t + u + (2 * c if name == "iv_conditional" else 0) + rng.normal(size=n)
        table = pd.DataFrame({"t": t, "z1": z1, "z2": z2, "c": c, "y": y})
        instrument = "z1"
        instruments = ["z1", "z2"]
        controls = ["c"] if name == "iv_conditional" else []
        hidden = True
        true_edges = [("z1", "t"), ("z2", "t"), ("t", "y")] + ([("c", "z1"), ("c", "t"), ("c", "y")] if controls else [])
        story = (
            "Independently sampled people receive binary treatment t. Hidden u drives uptake and outcome. Instruments z1 and z2 affect treatment only, with no outcome route except treatment. The outcome follows an additive linear model with an unknown intercept and unknown constant treatment effect. No treatment-effect heterogeneity, spillovers or missingness. z1,z2 and c are measured before treatment. "
            + (
                "Baseline c causes z1, uptake and outcome, so z1 is exogenous only conditional on c; c is independent of z2 and hidden u."
                if controls
                else "Both instruments are randomized independently of u and outcome disturbances; c is an unrelated baseline measurement."
            )
        )
        assumptions = {
            "claim:exclusion.exogenous": True,
            "claim:exclusion.homogeneous": True,
            "claim:exclusion.columns": instruments,
            "claim:exclusion.controls": controls,
        }
        if name in ("iv_single", "weak_iv"):
            t = ((0.01 if name == "weak_iv" else 0.9) * z1 + 0.7 * u + rng.normal(size=n) > 0).astype(int)
            y = 11 + 4 * t + u + rng.normal(size=n)
            table = pd.DataFrame({"t": t, "z1": z1, "c": c, "y": y})
            instruments = ["z1"]
            true_edges = [("z1", "t"), ("t", "y")]
            story = "Independent people receive binary treatment t. Hidden u drives treatment and outcome. Randomized numeric instrument z1 causes treatment only, and has no other route to outcome. z1 is independent of hidden u and all outcome disturbances. c is an unrelated baseline measurement. The outcome follows an additive linear model with an unknown intercept and unknown constant treatment effect. z1 and c precede treatment; outcome is recorded afterwards. No spillovers or missingness."
            assumptions["claim:exclusion.columns"] = instruments
    elif name in ("nie", "nde", "frontdoor"):
        mediator = "m"
        t = (u + rng.normal(size=n) > 0).astype(int) if name == "frontdoor" else binary
        m = 1.5 * t + rng.normal(size=n)
        y = 2 * m + (u if name == "frontdoor" else 2 * t) + rng.normal(size=n)
        table = pd.DataFrame({"t": t, "m": m, "y": y})
        true_edges = [("t", "m"), ("m", "y")] + ([] if name == "frontdoor" else [("t", "y")])
        hidden = name == "frontdoor"
        scope = "natural_indirect" if name == "nie" else "natural_direct" if name == "nde" else "average"
        story = (
            "Rows are independently sampled people. Treatment t precedes numeric mediator m, which precedes continuous outcome y. All stages are additive and linear, with no treatment-mediator interaction. t may change m, and m may change y; the effect magnitudes are unknown. No hidden factors confound treatment-mediator or mediator-outcome after conditioning on treatment, and no treatment-induced mediator-outcome confounder exists. "
            + (
                "Hidden u drives treatment and outcome; every directed treatment-outcome path passes through m, with no direct bypass."
                if hidden
                else "Treatment is randomized. Treatment may also directly change outcome. No hidden common causes exist. The natural direct and indirect effect magnitudes are unknown."
            )
            + " No spillovers or missingness."
        )
        assumptions = {"claim:mediator.linear_additive": True, "claim:mediator.sequential_ignorability": True, "claim:mediator.no_treatment_confounder": True}
    elif name == "conditional":
        x = c
        t = binary
        y = (2 + 0.5 * x) * t + x + rng.normal(size=n)
        table = pd.DataFrame({"t": t, "x": x, "y": y})
        true_edges = [("x", "y"), ("t", "y")]
        scope = "conditional"
        mods = ["x"]
        story = "Independent people receive randomized binary treatment t. Baseline numeric x is measured before assignment and is not affected by treatment. The outcome model is linear with an unknown treatment effect that changes linearly with x and an unknown baseline main effect of x. No other common causes, spillovers or missingness. Outcome is measured after treatment. The requested target is the conditional treatment effect across x."
    elif name == "unidentified":
        t = (u + rng.normal(size=n) > 0).astype(int)
        y = 4 * t + u + rng.normal(size=n)
        table = pd.DataFrame({"t": t, "y": y})
        hidden = True
        true_edges = [("t", "y")]
        story = "Independent people choose treatment t. An unobserved common cause drives both uptake and outcome. Treatment may affect outcome; its magnitude is unknown. No measured common cause, instrument or mediator is available. Outcome follows treatment. No spillovers or missingness."
    elif name == "zero":
        t = binary
        y = 2 * c + rng.normal(size=n)
        table = pd.DataFrame({"t": t, "c": c, "y": y})
        true_edges = [("c", "y")]
        story = "Independent people receive randomized binary treatment t. Baseline c causes outcome y. Treatment has no effect on any measured or hidden variable causing outcome. There is no directed treatment-outcome path, direct or mediated. Outcome is measured after assignment. No spillovers, missingness or selection on outcome."
    else:
        raise ValueError(name)
    facts = {}

    def fact(address, value):
        facts[address] = Fact(value=value, status="confirmed", source="user:turn:1", said=story, reason="explicit supplied simulated-study protocol")

    for address, value in {
        "claim:grain.row_is": "person",
        "claim:assignment.kind": "lottery" if name in ("randomized", "wald", "nie", "nde", "conditional", "zero") else "own_choice",
        "claim:assignment.rule": story,
        "claim:assignment.depends_on": [a for a, b in true_edges if b == "t"],
        "claim:sampling.how": "all",
        "claim:missing.why": "none",
        "claim:unobserved.exists": hidden,
        "claim:spillover.possible": False,
        "claim:story.text": story,
    }.items():
        fact(address, value)
    if instrument:
        fact("claim:exclusion.exists", True)
        fact("claim:exclusion.column", instrument)
        fact("claim:exclusion.why", story)
    if mediator:
        fact("claim:mediator.exists", True)
        fact("claim:mediator.column", mediator)
        fact("claim:mediator.why", story)
    if name == "zero":
        fact("claim:change.no_causal_path", True)
    for a, v in assumptions.items():
        fact(a, v)
    briefs = []
    for col in table:
        when = "at" if col == "t" else "after" if col in ("m", "y") else "before"
        fact(f"col:{col}.when", when)
        fact(f"col:{col}.meaning", f"{col}: " + story)
        if col not in ("t", "y"):
            for field, pair in [("feeds_treatment", (col, "t")), ("moves_outcome", (col, "y")), ("moved_by_change", ("t", col))]:
                fact(f"col:{col}.{field}", pair in true_edges)
            fact(f"col:{col}.measures_outcome", False)
        fact(f"col:{col}.may_modify", col in mods)
        briefs.append(
            ColumnBrief(
                name=col,
                key=col,
                role="treatment"
                if col == "t"
                else "outcome"
                if col == "y"
                else "instrument"
                if col in instruments or col == instrument
                else "mediator"
                if col == mediator
                else "candidate",
                meaning=f"{col}: " + story,
                when=when,
                may_modify=col in mods,
                provenance={"when": Provenance(status="confirmed", source="user:turn:1", said=story)},
                facts=ColumnFacts(kind="numeric", distinct=int(table[col].nunique()), nulls=0),
            )
        )
    beliefs = {"unobserved": Belief(kind="unobserved", value=hidden, status="confirmed", source="user:turn:1", what="u" if hidden else "", why=story)}
    if mediator:
        beliefs["mediator"] = Belief(kind="mediator", value=True, column="m", status="confirmed", source="user:turn:1", why=story)
    if instrument:
        beliefs["exclusion"] = Belief(kind="exclusion", value=True, column=instrument, status="confirmed", source="user:turn:1", why=story)
    h = Handoff(
        family="adjustment",
        specialist="dowhy",
        supported_now=True,
        outcome="y",
        treatment="t",
        scope=Scope(target=scope, modifiers=mods),
        pack_name="protocol_" + name,
        relevant_columns=[Candidate(column=c, reason="known protocol variable", cites=["claim:story.text"]) for c in table],
        chosen_assumption=story,
        reasons=[Cited(reason=story, cites=["claim:story.text"])],
        question={
            "nie": "What is the natural indirect effect of t through m on y?",
            "nde": "What is the natural direct effect of t on y?",
            "conditional": "How does the treatment effect on y differ by x?",
            "wald": "What is the effect of t on y for people whose uptake changes with the offer?",
        }.get(name, "What is the effect of t on y?"),
        story=story,
        csv=csv,
        columns=briefs,
        facts=facts,
        beliefs=beliefs,
        said=[Said(turn=1, about="study protocol", text=story)],
        grain={"row_is": "person"},
        sampling={"how": "all"},
        missing={"why": "none"},
        assignment={"kind": facts["claim:assignment.kind"].value, "depends_on": facts["claim:assignment.depends_on"].value, "rule": story},
        treated_level="1",
        control_level="0",
        design=AdjustmentDesign(
            instrument=instrument, instruments=instruments, iv_controls=controls, mediator=mediator, modifiers=mods, unobserved_confounding=hidden
        ),
    )
    return table, admit(h, load_endpoint_catalogue())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "name",
        choices=[
            "randomized",
            "backdoor",
            "att",
            "atc",
            "wald",
            "iv",
            "iv_single",
            "weak_iv",
            "iv_conditional",
            "nie",
            "nde",
            "frontdoor",
            "conditional",
            "zero",
            "unidentified",
        ],
    )
    args = ap.parse_args()
    csv = HERE / "data" / f"{args.name}.csv"
    handoff = HERE / "handoffs" / f"{args.name}_protocol.json"
    csv.parent.mkdir(exist_ok=True)
    table, h = scenario(args.name, str(csv.resolve()))
    trace.write(csv, table.to_csv(index=False))
    h = admit(h, load_endpoint_catalogue())
    trace.write(handoff, h.model_dump_json(indent=2))
    print(json.dumps({"handoff": str(handoff), "ready": h.admission.ready, "missing": h.admission.missing}))


if __name__ == "__main__":
    main()
