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
from typing import Any

from causal_agent.common.addresses import key
from causal_agent.common.contracts import (
    AdjustmentDesign,
    Belief,
    Candidate,
    ColumnBrief,
    DidDesign,
    FamilyDecision,
    Handoff,
    Probe,
    QuestionFrame,
    RdDesign,
    Said,
    Scope,
)
from causal_agent.knowledge import Family, load_registry
from causal_agent.memory import ops, store
from causal_agent.memory.claims import ProbeResult
from causal_agent.memory.records import Column, Memory
from causal_agent.profile import datasets as DS

BELIEF_KINDS = ("unobserved", "exclusion", "spillover", "trend_continues", "cutoff_only", "mediator")
_VALUE_FIELD = {"unobserved": "exists", "exclusion": "exists", "spillover": "possible", "trend_continues": "believed", "cutoff_only": "believed", "mediator": "exists"}


# ------------------------------------------------------------------ pieces


def fields_of(memory: Memory, kind: str) -> dict[str, Any]:
    return memory.values_of(f"claim:{kind}")


def brief_of(memory: Memory, col: Column, role: str | None = None) -> ColumnBrief:
    """A column as the lane reads it: the file's facts beside what is known about it. The role is the caller's view."""
    fs = memory.fields_of(col.address)
    v = {n: f.value for n, f in fs.items() if f.value is not None}
    src = next((fs[n].source for n in ("meaning", "when") if n in v and fs[n].source), None)
    return ColumnBrief(name=col.name, key=col.key, role=role or "candidate", meaning=v.get("meaning"), when=v.get("when") or "unknown", set_by=v.get("set_by"),
                       moved_by_change=v.get("moved_by_change"), source=src, facts=col.facts)


def belief_of(memory: Memory, kind: str) -> Belief | None:
    fs = {n: f for n, f in memory.fields_of(f"claim:{kind}").items() if f.value is not None or f.status != "empty"}
    if not fs:
        return None
    vf = fs.get(_VALUE_FIELD[kind]) or next(iter(fs.values()))
    known = {n: f.value for n, f in fs.items() if f.value is not None}
    return Belief(kind=kind, value=known.get(_VALUE_FIELD[kind]), what=known.get("what"), why=known.get("why") or known.get("why_believed"),
                  column=known.get("column"), status=vf.status, source=vf.source, said=vf.said)


def said_of(memory: Memory) -> list[Said]:
    """The person's words, plus the sentence each known field rests on when the transcript does not carry it."""
    out = list(memory.said)
    seen = {(s.turn, s.text) for s in out}
    for address, f in memory.fields.items():
        if f.said and f.source and f.source.startswith("user:turn:"):
            turn = int(f.source.rsplit(":", 1)[1])
            if (turn, f.said) not in seen:
                out.append(Said(turn=turn, about=address, text=f.said))
                seen.add((turn, f.said))
    return sorted(out, key=lambda s: s.turn)


# ------------------------------------------------------------------ the family blocks (derived by code; never a constraint)


def _adjustment(briefs: list[ColumnBrief], a: dict, beliefs: dict[str, Belief], scope: Scope) -> AdjustmentDesign:
    dep = [key(c) for c in a.get("depends_on") or []]
    before = [b.key for b in briefs if b.role not in ("outcome", "treatment") and b.when == "before" and b.moved_by_change is not True]
    forbidden = [b.key for b in briefs if b.role not in ("outcome", "treatment", "depends_on") and (b.when in ("after", "at") or b.moved_by_change is True)]
    excl, med, unob = beliefs.get("exclusion"), beliefs.get("mediator"), beliefs.get("unobserved")
    instrument = key(excl.column) if excl and excl.known() and excl.value and excl.column else None
    mediator = key(med.column) if med and med.known() and med.value and med.column else None
    kind = a.get("kind")
    voluntary = True if kind == "own_choice" else False if kind in ("cutoff_rule", "date_by_others", "lottery") else (True if a.get("movable") else None)
    return AdjustmentDesign(adjustment_candidates=list(dict.fromkeys(dep + before)), forbidden=list(dict.fromkeys(forbidden)),
                            identification_allowed=["backdoor"] + (["instrument"] if instrument else []) + (["frontdoor"] if mediator else []),
                            instrument=instrument, mediator=mediator, unobserved_confounding=unob.value if unob and unob.known() else None,
                            voluntary_uptake=voluntary, target_units=scope.target, contrast=scope.contrast)


def _did(briefs: list[ColumnBrief], a: dict, ch: dict, g: dict, beliefs: dict[str, Belief], probes: list[Probe], entry: dict, treatment: str | None) -> DidDesign:
    time = ch.get("date_column") or entry.get("time")
    tkey = key(time) if time else None
    units = [c for c in (g.get("key_columns") or []) if key(c) != tkey] if g.get("panel") is True else []
    unit = units[0] if units else (entry.get("entity") or [None])[0]
    tb = next((b for b in briefs if tkey and b.key == tkey), None)
    period_kind = ("date" if tb.facts.kind == "datetime" else "integer") if tb else None
    treated_group = {"column": a["treatment_column"], "level": a["treated_level"]} if a.get("treatment_column") and a.get("treated_level") is not None else \
        ({"column": treatment, "level": a["treated_level"]} if treatment and a.get("treated_level") is not None else {})
    pre = next((p for p in probes if p.family == "diff_in_diff" and p.name == "pre_periods"), None)
    controls = [b.key for b in briefs if b.role == "candidate" and b.moved_by_change is not True and (b.when == "before" or b.facts.varies_over in ("entity", "time", "both"))]
    return DidDesign(unit=key(unit) if unit else None, time=tkey, period_kind=period_kind, change_period=str(ch["period_value"]) if ch.get("period_value") is not None else None,
                     treated_group=treated_group, pre_periods=int(pre.value) if pre and pre.value is not None else None, controls_allowed=controls,
                     cluster_level=key(a["level_column"]) if a.get("level_column") else (key(unit) if unit else None),
                     trend_belief=beliefs.get("trend_continues"), spillover=beliefs.get("spillover"))


def _rd(briefs: list[ColumnBrief], a: dict, samp: dict, beliefs: dict[str, Belief], entry: dict, treatment: str | None) -> RdDesign:
    score = key(a["score_column"]) if a.get("score_column") else None
    sb = next((b for b in briefs if score and b.key == score), None)
    takeup = {"column": treatment, "level": a.get("treated_level")} if treatment and (not score or key(treatment) != score) and a.get("treated_level") is not None else None
    covs = [b.key for b in briefs if b.role == "candidate" and b.when == "before" and b.moved_by_change is not True]
    cluster = a.get("level_column") or (entry.get("entity") or [None])[0]
    return RdDesign(score=score, cutoff=float(a["cutoff"]) if a.get("cutoff") is not None else None, treated_side=a.get("treated_side"), cutoff_value_treated=a.get("cutoff_value_treated"),
                    score_fixed_before=(sb.when == "before") if sb and sb.when != "unknown" else None, movable=a.get("movable"), takeup=takeup, covariates_allowed=covs,
                    cluster=key(cluster) if cluster else None, sampled_by_side=samp.get("how") == "by_side" or bool(entry.get("sampled_by_side", False)),
                    cutoff_only=beliefs.get("cutoff_only"))


# ------------------------------------------------------------------ the builder


def build(*, question: str, frame: QuestionFrame, decision: FamilyDecision, family: Family, memory: Memory,
          probes: list[ProbeResult] | list[Probe] = ()) -> Handoff:
    """The one hand-off, projected from the memory. The memory is not changed."""
    m = memory
    entry = DS.dataset_entries().get(m.name) or {}
    a, ch, g, samp, miss = (fields_of(m, k) for k in ("assignment", "change", "grain", "sampling", "missing"))
    outcome = frame.outcome or ""
    treatment = a.get("treatment_column") or frame.cause
    if a.get("kind") == "cutoff_rule" and not a.get("treatment_column") and treatment and a.get("score_column") and key(treatment) == key(a["score_column"]):
        treatment = None  # the change is the rule itself; the score is not the treatment
    role = ops.roles(m, outcome=outcome, treatment=treatment)
    for n in [entry.get("time")] + list(entry.get("entity") or []):  # the index entry's flags stand in for what the memory does not carry yet
        if n and m.column(n) is not None:
            role.setdefault(m.column(n).key, "time" if n == entry.get("time") else "unit")

    names: list[str] = []
    for n in [outcome, treatment] + [c.column for c in frame.relevant_columns] + list(a.get("depends_on") or []) + \
            [a.get("score_column"), ch.get("date_column"), a.get("level_column"), fields_of(m, "exclusion").get("column"), fields_of(m, "mediator").get("column")] + \
            list(g.get("key_columns") or []) + [entry.get("time")] + list(entry.get("entity") or []):
        if n and key(n) not in {key(x) for x in names} and m.column(n) is not None:
            names.append(n)
    briefs = [brief_of(m, m.column(n), role.get(key(n))) for n in names]

    beliefs = {k: b for k in BELIEF_KINDS if (b := belief_of(m, k)) is not None}
    probe_list = [p if isinstance(p, Probe) else Probe(family=p.family, name=p.name, value=p.value, passed=p.passed, detail=p.detail) for p in probes]
    tb = next((b for b in briefs if b.role == "treatment"), None)
    treated_level = str(a["treated_level"]) if a.get("treated_level") is not None else None
    control_level = None
    if tb and treated_level is not None:
        others = [v for v in tb.facts.levels() if v != treated_level]
        control_level = others[0] if len(tb.facts.levels()) == 2 and others else None

    if family.name == "adjustment":
        design = _adjustment(briefs, a, beliefs, frame.scope)
    elif family.name == "diff_in_diff":
        design = _did(briefs, a, ch, g, beliefs, probe_list, entry, treatment)
    elif family.name == "discontinuity":
        design = _rd(briefs, a, samp, beliefs, entry, treatment)
    else:
        design = None

    table = m.to_claims()
    unknowns = [address for address, f in m.fields.items() if f.status == "unknown"]
    return Handoff(
        family=family.name, specialist=family.specialist, supported_now=family.status == "built",
        outcome=outcome, treatment=treatment, scope=frame.scope, pack_name=m.name, relevant_columns=list(frame.relevant_columns),
        chosen_assumption=decision.chosen_assumption,
        reasons=[Candidate(column=c.column, reason=c.reason, cites=c.cites) for c in frame.outcome_candidates[:1] + frame.cause_candidates[:1]],
        question=question, intent=frame.intent, why=decision.why_over_alternatives, over={r.family: r.reason for r in decision.rejected},
        csv=m.csv or entry.get("csv"), docs={}, dataset_facts=m.facts, grain=g, sampling=samp, missing=miss,
        treated_level=treated_level, control_level=control_level, columns=briefs,
        change=ch, assignment=a, beliefs=beliefs, unknowns=unknowns, said=said_of(m), probes=probe_list,
        claims={c.key: {"kind": c.kind, "fields": {k: v for k, v in c.fields.items() if v is not None}, "status": c.status, "source": c.source}
                for c in table.claims.values() if c.status != "empty"},
        design=design,
    )


def forced(pack_name: str, question: str, family: str, outcome: str, treatment: str | None, columns: list[str], *,
           scope: Scope | None = None, assumption: str = "forced hand-off", cite: str | None = None, memory: Memory | None = None) -> Handoff:
    """A hand-off without a frame or a decision: the family, the outcome, the treatment, and the columns are given.
    For tests and evals. The memory on disk, when the dataset has one, fills the briefs and the family block."""
    import pandas as pd

    fam = next(f for f in load_registry() if f.name == family)
    m = memory or store.memory_for(pack_name)
    cite_of = lambda c: [cite] if cite else [f"col:{key(c)}.note"]  # noqa: E731
    cands = [Candidate(column=c, reason="named in the forced hand-off", cites=cite_of(c)) for c in columns]
    frame = QuestionFrame(intent="effect_of_change", decision_served="a forced run", outcome_candidates=[Candidate(column=outcome, reason="the outcome the question names", cites=cite_of(outcome))],
                          cause_candidates=[Candidate(column=treatment, reason="the change asked about", cites=cite_of(treatment))] if treatment else [],
                          scope=scope or Scope(), relevant_columns=cands, reasons=[])
    decision = FamilyDecision(admissible=[family], chosen=family, chosen_assumption=assumption, why_over_alternatives="forced", rejected=[])
    csv = m.csv or (DS.dataset_entries().get(pack_name) or {}).get("csv")
    path = Path(csv) if csv and Path(csv).is_absolute() else (Path(DS.ROOT) / csv if csv else None)
    probes = ops.probe(m, pd.read_csv(path)) if path is not None and path.exists() else []
    return build(question=question, frame=frame, decision=decision, family=fam, memory=m, probes=probes)


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
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args(argv)
    cols = [c.strip() for c in args.columns.split(",") if c.strip()]
    h = forced(args.dataset, args.question, args.family, args.outcome, args.treatment, cols, scope=Scope(target=args.target, window=args.window, contrast=args.contrast), assumption=args.assumption)
    text = json.dumps(h.model_dump(), indent=2, default=str)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
