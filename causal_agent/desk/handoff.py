"""The context pack builder. One function makes the Handoff a lane receives: from the question frame, the decision, the
claim table, the probes, and the profile. The router calls it at run time; the CLI below writes the forced hand-offs the
specialist evals and tests use. No second copy anywhere.

    uv run python -m causal_agent.desk.handoff students --family adjustment --outcome "math score" \\
        --treatment "test preparation course" --columns "lunch,parental level of education" --question "..." -o handoff.json

Transitional rule, until every dataset has claims: a column with no measured claim takes its meaning from the note card,
marked source doc:<name>. The lanes never read a note; only this builder does, once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from causal_agent.common.addresses import key
from causal_agent.common.contracts import (
    AdjustmentDesign,
    Belief,
    Candidate,
    ColumnBrief,
    ColumnFacts,
    DidDesign,
    FamilyDecision,
    Handoff,
    Probe,
    QuestionFrame,
    RdDesign,
    Said,
    Scope,
)
from causal_agent.profile.datasets import ROOT, dataset_entries, load_dataset_pack
from causal_agent.memory.claims import Claim, ClaimTable, ProbeResult
from causal_agent.profile.pack import Pack
from causal_agent.knowledge import Family, load_registry

BELIEF_KINDS = ("unobserved", "exclusion", "spillover", "trend_continues", "cutoff_only")
_VALUE_FIELD = {"unobserved": "exists", "exclusion": "exists", "spillover": "possible", "trend_continues": "believed", "cutoff_only": "believed"}


# ------------------------------------------------------------------ claims on disk


def load_claims(path: str | Path) -> tuple[ClaimTable, list[ProbeResult]]:
    """The claims document the interview writes: {claims: [...], probes: [...]}."""
    doc = yaml.safe_load(Path(path).read_text()) or {}
    table = ClaimTable(claims={c["key"]: Claim.model_validate(c) for c in doc.get("claims") or []})
    probes = [ProbeResult.model_validate(p) for p in doc.get("probes") or []]
    return table, probes


def claims_for(pack_name: str) -> tuple[ClaimTable | None, list[ProbeResult]]:
    e = dataset_entries().get(pack_name) or {}
    if e.get("claims") and (ROOT / e["claims"]).exists():
        return load_claims(ROOT / e["claims"])
    return None, []


# ------------------------------------------------------------------ the builder


def _fields(table: ClaimTable, k: str) -> dict[str, Any]:
    c = table.get(k)
    return {n: v for n, v in (c.fields if c else {}).items() if v is not None}


def _brief(pack: Pack, name: str, role: str, table: ClaimTable) -> ColumnBrief | None:
    card = pack.column(name)
    if card is None:
        return None
    facts = ColumnFacts.from_profile(card.profile)
    claim = table.get(f"col:{card.key}")
    if claim is not None and claim.status != "empty" and claim.fields.get("meaning"):
        f = claim.fields
        return ColumnBrief(name=card.name, key=card.key, role=role, meaning=f.get("meaning"), when=f.get("when") or "unknown", set_by=f.get("set_by"),
                           affected_by_treatment=f.get("affected_by_treatment"), source=claim.source, facts=facts)
    when = (claim.fields.get("when") if claim else None) or "unknown"
    return ColumnBrief(name=card.name, key=card.key, role=role, meaning=card.note or None, when=when, source=(f"doc:{card.source}" if card.source else "doc:note") if card.note else None, facts=facts)


def _belief(table: ClaimTable, kind: str) -> Belief | None:
    c = table.get(kind)
    if c is None or c.status == "empty":
        return None
    f = c.fields
    return Belief(kind=kind, value=f.get(_VALUE_FIELD[kind]), what=f.get("what"), why=f.get("why") or f.get("why_believed"), column=f.get("column"), status=c.status, source=c.source)


def _adjustment(briefs: list[ColumnBrief], a: dict, beliefs: dict[str, Belief], scope: Scope) -> AdjustmentDesign:
    dep = [key(c) for c in a.get("depends_on") or []]
    before = [b.key for b in briefs if b.role not in ("outcome", "treatment") and b.when == "before" and b.affected_by_treatment is not True]
    forbidden = [b.key for b in briefs if b.role not in ("outcome", "treatment", "depends_on") and (b.when in ("after", "at") or b.affected_by_treatment is True)]
    allowed: list = ["backdoor"]
    excl = beliefs.get("exclusion")
    instrument = key(excl.column) if excl and excl.known() and excl.value and excl.column else None
    if instrument:
        allowed.append("instrument")
    unob = beliefs.get("unobserved")
    kind = a.get("kind")
    voluntary = True if kind == "own_choice" else False if kind in ("cutoff_rule", "date_by_others", "lottery") else (True if a.get("movable") else None)
    return AdjustmentDesign(adjustment_candidates=list(dict.fromkeys(dep + before)), forbidden=list(dict.fromkeys(forbidden)), identification_allowed=allowed,
                            instrument=instrument, unobserved_confounding=unob.value if unob and unob.known() else None, voluntary_uptake=voluntary,
                            target_units=scope.target, contrast=scope.contrast)


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
    controls = [b.key for b in briefs if b.role == "candidate" and b.affected_by_treatment is not True and (b.when == "before" or b.facts.varies_over in ("entity", "time", "both"))]
    return DidDesign(unit=key(unit) if unit else None, time=tkey, period_kind=period_kind, change_period=str(ch["period_value"]) if ch.get("period_value") is not None else None,
                     treated_group=treated_group, pre_periods=int(pre.value) if pre and pre.value is not None else None, controls_allowed=controls,
                     cluster_level=key(a["level_column"]) if a.get("level_column") else (key(unit) if unit else None),
                     trend_belief=beliefs.get("trend_continues"), spillover=beliefs.get("spillover"))


def _rd(briefs: list[ColumnBrief], a: dict, samp: dict, beliefs: dict[str, Belief], entry: dict, treatment: str | None) -> RdDesign:
    score = key(a["score_column"]) if a.get("score_column") else None
    sb = next((b for b in briefs if score and b.key == score), None)
    takeup = {"column": treatment, "level": a.get("treated_level")} if treatment and (not score or key(treatment) != score) and a.get("treated_level") is not None else None
    covs = [b.key for b in briefs if b.role == "candidate" and b.when == "before" and b.affected_by_treatment is not True]
    cluster = a.get("level_column") or (entry.get("entity") or [None])[0]
    return RdDesign(score=score, cutoff=float(a["cutoff"]) if a.get("cutoff") is not None else None, treated_side=a.get("treated_side"), cutoff_value_treated=a.get("cutoff_value_treated"),
                    score_fixed_before=(sb.when == "before") if sb and sb.when != "unknown" else None, movable=a.get("movable"), takeup=takeup, covariates_allowed=covs,
                    cluster=key(cluster) if cluster else None, sampled_by_side=samp.get("how") == "by_side" or bool(entry.get("sampled_by_side", False)),
                    cutoff_only=beliefs.get("cutoff_only"))


def build(*, question: str, frame: QuestionFrame, decision: FamilyDecision, family: Family, pack: Pack, claims: ClaimTable | None = None,
          probes: list[ProbeResult] | list[Probe] = (), said: list[Said] = ()) -> Handoff:
    """The one hand-off. The person's claims win over the frame's reading wherever both speak."""
    table = claims or ClaimTable()
    entry = dataset_entries().get(pack.name) or {}
    a, ch, g, samp, miss = (_fields(table, k) for k in ("assignment", "change", "grain", "sampling", "missing"))
    if not ch and pack.changes:  # transitional: a shipped note with no change claim
        c0 = pack.changes[0]
        ch = {"what": f"{c0.title}. {c0.note}".strip(". ") + "."}
    outcome = frame.outcome or ""
    treatment = a.get("treatment_column") or frame.cause
    if a.get("kind") == "cutoff_rule" and not a.get("treatment_column") and treatment and a.get("score_column") and key(treatment) == key(a["score_column"]):
        treatment = None  # the change is the rule itself; the score is not the treatment

    roles: dict[str, str] = {}

    def role(name: str | None, r: str) -> None:
        if name:
            roles.setdefault(key(name), r)

    role(outcome, "outcome")
    role(treatment, "treatment")
    for c in a.get("depends_on") or []:
        role(c, "depends_on")
    role(a.get("score_column"), "score")
    role(ch.get("date_column"), "time")
    if g.get("panel") is True:
        for c in g.get("key_columns") or []:
            if key(c) != key(ch.get("date_column") or ""):
                role(c, "unit")
    role(a.get("level_column"), "group")
    role(entry.get("time"), "time")  # the index entry's flags, when the claims do not name them
    for c in entry.get("entity") or []:
        role(c, "unit")
    excl = _fields(table, "exclusion")
    if excl.get("exists") and excl.get("column"):
        role(excl["column"], "instrument")

    names: list[str] = []
    for n in [outcome, treatment] + [c.column for c in frame.relevant_columns] + [c for c in (a.get("depends_on") or [])] + \
            [a.get("score_column"), ch.get("date_column"), a.get("level_column"), excl.get("column")] + list(g.get("key_columns") or []):
        if n and key(n) not in {key(x) for x in names} and pack.column(n) is not None:
            names.append(n)
    briefs = [b for n in names if (b := _brief(pack, n, roles.get(key(n), "candidate"), table)) is not None]

    beliefs = {k: b for k in BELIEF_KINDS if (b := _belief(table, k)) is not None}
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

    dp = pack.dataset.profile
    facts = {"rows": dp.rows, "columns": dp.columns, "duplicate_rows": dp.duplicate_rows, "grain": list(dp.grain or []),
             "time_coverage": dp.time_coverage.model_dump() if dp.time_coverage else None,
             "entity_summary": dp.entity_summary.model_dump() if dp.entity_summary else None, "format_issues": list(dp.format_issues or [])}
    docs = {"about": pack.dataset.note} if pack.dataset.note and not g.get("row_is") else {}
    return Handoff(
        family=family.name, specialist=family.specialist, supported_now=family.status == "built",
        outcome=outcome, treatment=treatment, scope=frame.scope, pack_name=pack.name, relevant_columns=list(frame.relevant_columns),
        chosen_assumption=decision.chosen_assumption,
        reasons=[Candidate(column=c.column, reason=c.reason, cites=c.cites) for c in frame.outcome_candidates[:1] + frame.cause_candidates[:1]],
        question=question, intent=frame.intent, why=decision.why_over_alternatives, over={r.family: r.reason for r in decision.rejected},
        csv=entry.get("csv"), docs=docs, dataset_facts=facts, grain=g, sampling=samp, missing=miss,
        treated_level=treated_level, control_level=control_level, columns=briefs,
        change=ch, assignment=a, beliefs=beliefs, unknowns=[c.key for c in table.claims.values() if c.status == "unknown"], said=list(said),
        probes=probe_list, claims={c.key: {"kind": c.kind, "fields": {k: v for k, v in c.fields.items() if v is not None}, "status": c.status, "source": c.source}
                                   for c in table.claims.values() if c.status != "empty"},
        design=design,
    )


def forced(pack_name: str, question: str, family: str, outcome: str, treatment: str | None, columns: list[str], *,
           scope: Scope | None = None, assumption: str = "forced hand-off", cite: str | None = None) -> Handoff:
    """A hand-off without a frame or a decision: the family, the outcome, the treatment, and the columns are given.
    For tests and evals. Claims on disk, when the dataset has them, still fill the briefs and the family block."""
    fam = next(f for f in load_registry() if f.name == family)
    pack = load_dataset_pack(pack_name)
    cite_of = lambda c: [cite] if cite else [f"col:{key(c)}.note"]  # noqa: E731
    cands = [Candidate(column=c, reason="named in the forced hand-off", cites=cite_of(c)) for c in columns]
    frame = QuestionFrame(intent="effect_of_change", decision_served="a forced run", outcome_candidates=[Candidate(column=outcome, reason="the outcome the question names", cites=cite_of(outcome))],
                          cause_candidates=[Candidate(column=treatment, reason="the change asked about", cites=cite_of(treatment))] if treatment else [],
                          scope=scope or Scope(), relevant_columns=cands, reasons=[])
    decision = FamilyDecision(admissible=[family], chosen=family, chosen_assumption=assumption, why_over_alternatives="forced", rejected=[])
    claims, probes = claims_for(pack_name)
    return build(question=question, frame=frame, decision=decision, family=fam, pack=pack, claims=claims, probes=probes)


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
