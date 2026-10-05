"""The context pack builder. One function projects the Handoff a lane receives from a memory: the question frame, the
decision, the records, the beliefs, the probes, the person's words. The routing calls it at run time; the CLI below writes
the forced hand-offs the specialist evals and tests use. No second copy anywhere.

    uv run python -m causal_agent.desk.handoff students --family adjustment --outcome "math score" \\
        --treatment "test preparation course" --columns "lunch,parental level of education" --question "..." -o handoff.json

A memory that has never been mined or interviewed carries only the file's facts; the briefs then say "(not described)" and
the lane's own judgements fill what they can. Nothing here reads a note.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from causal_agent.common.addresses import key
from causal_agent.common.contracts import (
    Candidate,
    DesignBrief,
    Fact,
    FamilyDecision,
    Handoff,
    Probe,
    QuestionFrame,
    Scope,
)
from causal_agent.families import registry as R
from causal_agent.families.registry import BlockInputs, FamilyDef
from causal_agent.memory import facts as FX
from causal_agent.memory import ops, store
from causal_agent.memory.claims import ProbeResult
from causal_agent.memory.records import Memory
from causal_agent.memory.views import BELIEF_KINDS, belief_of, brief_of, fields_of, in_play, index_records, said_of
from causal_agent.profile import datasets as DS

# ------------------------------------------------------------------ the builder


def build(
    *,
    question: str,
    frame: QuestionFrame,
    decision: FamilyDecision,
    family: FamilyDef,
    memory: Memory,
    probes: list[ProbeResult] | list[Probe] = (),
    design_id: int = 0,
    brief: DesignBrief | None = None,
    df: pd.DataFrame | None = None,
) -> Handoff:
    """The one hand-off, projected from the memory. The memory is not changed. The brief, when there is one, is what the design
    bets on; without one the decision's assumption stands. Every column of the file but ids and constants is carried, with every
    word about it and its status (ADR 0011); the data facts are computed for all of them here, in one pass over the file."""
    m = memory
    entry = DS.dataset_entries().get(m.name) or {}
    a, ch, g, samp, miss = (fields_of(m, k) for k in ("assignment", "change", "grain", "sampling", "missing"))
    outcome = frame.outcome or ""
    treatment = a.get("treatment_column") or frame.cause
    if a.get("kind") == "cutoff_rule" and not a.get("treatment_column") and treatment and a.get("score_column") and key(treatment) == key(a["score_column"]):
        treatment = None  # the change is the rule itself; the score is not the treatment
    role = ops.roles(m, outcome=outcome, treatment=treatment, time=entry.get("time"))
    for n in entry.get("entity") or []:  # the index entry's flags stand in for what the memory does not carry yet
        if n and m.column(n) is not None:
            role.setdefault(m.column(n).key, "unit")

    # every column is carried (ADR 0011): the pair first, then the columns the question or the rule named, then the rest of the
    # file but ids and constants, in the file's order
    pair = [m.column(n).name for n in (outcome, treatment) if n and m.column(n) is not None]
    named = [n for n in in_play(m, frame, entry) if n not in pair]
    rest = [c.name for c in index_records(m) if c.name not in pair and c.name not in named]
    names = pair + named + rest
    briefs = [brief_of(m, m.column(n), role.get(key(n))) for n in names]
    carried = {b.key for b in briefs}
    facts = {
        a: Fact(value=f.value, status=f.status, source=f.source, said=f.said, reason=f.reason)
        for a, f in m.fields.items()
        if (f.value is not None or f.status != "empty") and (not a.startswith("col:") or Memory.parse(a)[1] in carried)
    }

    beliefs = {k: b for k in BELIEF_KINDS if (b := belief_of(m, k)) is not None}
    probe_list = [
        p if isinstance(p, Probe) else Probe(family=p.family, name=p.name, value=p.value, passed=p.passed, detail=p.detail)
        for p in probes
        if p.family != "data"  # the caller's data facts covered the columns in play; the pack's cover every carried column
    ] + data_facts(m, outcome, treatment, names, df)
    tb = next((b for b in briefs if b.role == "treatment"), None)
    treated_level = str(a["treated_level"]) if a.get("treated_level") is not None else None
    control_level = None
    if tb and treated_level is not None:
        others = [v for v in tb.facts.levels() if v != treated_level]
        control_level = others[0] if len(tb.facts.levels()) == 2 and others else None

    inputs = BlockInputs(
        briefs=briefs,
        claims={"assignment": a, "change": ch, "grain": g, "sampling": samp, "missing": miss},
        beliefs=beliefs,
        probes=probe_list,
        entry=entry,
        scope=frame.scope,
        treatment=treatment,
    )
    design = family.design_block(inputs) if family.design_block else None
    return Handoff(
        family=family.name,
        specialist=family.specialist,
        supported_now=family.built,
        outcome=outcome,
        treatment=treatment,
        scope=frame.scope,
        pack_name=m.name,
        relevant_columns=list(frame.relevant_columns),
        chosen_assumption=brief.bets_on if brief is not None else decision.chosen_assumption,
        reasons=[Candidate(column=c.column, reason=c.reason, cites=c.cites) for c in frame.outcome_candidates[:1] + frame.cause_candidates[:1]],
        question=question,
        story=m.value("claim:story.text"),
        intent=frame.intent,
        design_id=design_id,
        memory_version=m.version,
        why=decision.why_over_alternatives,
        over={r.family: r.reason for r in decision.rejected},
        csv=m.csv or entry.get("csv"),
        dataset_facts=m.facts,
        grain=g,
        sampling=samp,
        missing=miss,
        treated_level=treated_level,
        control_level=control_level,
        columns=briefs,
        change=ch,
        assignment=a,
        facts=facts,
        beliefs=beliefs,
        said=said_of(m),
        probes=probe_list,
        design=design,
        brief=brief,
    )


def data_facts(m: Memory, outcome: str, treatment: str | None, names: list[str], df: pd.DataFrame | None = None) -> list[Probe]:
    """The `probe:data.*` facts for every carried column, from the file: one read, unless the caller has the table already."""
    if df is None:
        path = DS.csv_path(m.name, m.csv)
        if path is None or not path.exists():
            return []
        df = pd.read_csv(path)
    out = FX.facts(df, m, m.to_claims(), outcome=outcome, treatment=treatment, columns=names)
    return [Probe(family=p.family, name=p.name, value=p.value, passed=p.passed, detail=p.detail) for p in out]


def forced(
    pack_name: str,
    question: str,
    family: str,
    outcome: str,
    treatment: str | None,
    columns: list[str],
    *,
    scope: Scope | None = None,
    assumption: str = "forced hand-off",
    cite: str | None = None,
    memory: Memory | None = None,
) -> Handoff:
    """A hand-off without a frame or a decision: the family, the outcome, the treatment, and the columns are given.
    For tests and evals. The memory on disk, when the dataset has one, fills the briefs and the family block."""
    fam = R.family(family)
    m = memory or store.memory_for(pack_name)
    cite_of = lambda c: [cite] if cite else [f"col:{key(c)}.note"]  # noqa: E731
    cands = [Candidate(column=c, reason="named in the forced hand-off", cites=cite_of(c)) for c in columns]
    frame = QuestionFrame(
        intent="effect_of_change",
        decision_served="a forced run",
        outcome_candidates=[Candidate(column=outcome, reason="the outcome the question names", cites=cite_of(outcome))],
        cause_candidates=[Candidate(column=treatment, reason="the change asked about", cites=cite_of(treatment))] if treatment else [],
        scope=scope or Scope(),
        relevant_columns=cands,
        reasons=[],
    )
    decision = FamilyDecision(admissible=[family], chosen=family, chosen_assumption=assumption, why_over_alternatives="forced", rejected=[])
    path = DS.csv_path(pack_name, m.csv)
    df = pd.read_csv(path) if path is not None and path.exists() else None
    probes: list = ops.probe(m, df, R.REGISTRY.values()) if df is not None else []
    return build(question=question, frame=frame, decision=decision, family=fam, memory=m, probes=probes, df=df)


# ------------------------------------------------------------------ CLI


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="write a forced hand-off for a dataset")
    ap.add_argument("dataset")
    ap.add_argument("--family", required=True)
    ap.add_argument("--outcome", required=True)
    ap.add_argument("--treatment", default=None)
    ap.add_argument("--columns", default="", help="comma-separated relevant columns")
    ap.add_argument("--question", default="")
    ap.add_argument("--target", default="average")
    ap.add_argument("--contrast", default="switch")
    ap.add_argument("--window", default=None)
    ap.add_argument("--assumption", default="forced hand-off")
    ap.add_argument(
        "--mine", action="store_true", help="read the attached note into drafts first, when the memory holds only the file's facts (one model call)"
    )
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args(argv)
    if args.mine:
        from causal_agent.desk.nodes.frame import mine

        mine({"dataset": args.dataset, "question": args.question})
    cols = [c.strip() for c in args.columns.split(",") if c.strip()]
    h = forced(
        args.dataset,
        args.question,
        args.family,
        args.outcome,
        args.treatment,
        cols,
        scope=Scope(target=args.target, window=args.window, contrast=args.contrast),
        assumption=args.assumption,
    )
    text = json.dumps(h.model_dump(), indent=2, default=str)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
