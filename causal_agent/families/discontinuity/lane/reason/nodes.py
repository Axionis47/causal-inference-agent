"""Admitted context to cited RD assumptions and frozen rival designs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from langgraph.types import Command, Send

from causal_agent.common import trace
from causal_agent.common.addresses import key, norm_address
from causal_agent.common.contracts import CheckResult, Checks, Cited, ContextDeficiency, Contrast, Feasibility
from causal_agent.families.discontinuity.design import RdDesign
from causal_agent.families.discontinuity.lane.contracts import (
    AssumptionDecisions,
    Bandwidths,
    Candidate,
    CovariateRelation,
    CovariateRoles,
    Covariates,
    Design,
    Excluded,
    Ladder,
    Line,
    Score,
)
from causal_agent.families.discontinuity.lane.engine import diagnostics as D
from causal_agent.families.discontinuity.lane.engine import shape as S
from causal_agent.families.discontinuity.lane.knowledge import (
    CATALOGUE,
    SPEC,
    estimator,
    load_checks,
    load_falsifications,
    load_sensitivities,
    pick_inference,
)
from causal_agent.families.discontinuity.lane.reason import prompts as P
from causal_agent.families.discontinuity.lane.reason.provenance import continuity_entry
from causal_agent.families.discontinuity.lane.reason.selection import select_lead
from causal_agent.families.discontinuity.lane.reason.views import build
from causal_agent.families.discontinuity.lane.state import SpecialistState
from causal_agent.lane import intake
from causal_agent.lane import nodes as L
from causal_agent.lane.endpoints import MAX_DESIGNS, design_identity, validate_context
from causal_agent.lane.episode import EpisodeLog, run_episode
from causal_agent.lane.ledger import Entry, Evidence, apply
from causal_agent.lane.ledger import seed as seed_context


def _stop(stage, reason, facts=(), extra=None, *, terminal_kind):
    return Command(
        goto="feasibility", update={"feasibility": Feasibility(stage=stage, reason=reason, facts=list(facts), terminal_kind=terminal_kind), **(extra or {})}
    )


def _transition(state, stage, candidates):
    return {
        "stage": stage,
        "ledger_revision": state["ledger"].revision,
        "context_id": state["handoff"].context_id,
        "candidates": [c.model_dump() for c in candidates],
        "endpoint_obligations": {
            c.endpoint_id: {
                "requirements": [r.model_dump() for r in CATALOGUE.endpoint(c.endpoint_id).requirements],
                "context_admission": next(e.model_dump() for e in state["handoff"].admission.endpoints if e.endpoint_id == c.endpoint_id),
                "assumptions": CATALOGUE.endpoint(c.endpoint_id).assumptions,
                "checks": CATALOGUE.endpoint(c.endpoint_id).checks,
                "trajectory": CATALOGUE.endpoint(c.endpoint_id).trajectory,
            }
            for c in candidates
        },
    }


def _canon(state):
    return pd.read_csv(state["canon_path"])


def _episode(state, name, schema, system, view, gate):
    def checked(rec, log):
        def canonical(address):
            normal = norm_address(address)
            if view.resolve(normal) or log.resolve(normal):
                return normal
            qualified = f"claim:{normal}"
            return qualified if view.resolve(qualified) else normal

        if hasattr(rec, "cites"):
            rec.cites = [canonical(a) for a in rec.cites]
        for item in [*getattr(rec, "items", []), *getattr(rec, "risks", [])]:
            if hasattr(item, "cites"):
                item.cites = [canonical(a) for a in item.cites]
            for reason in getattr(item, "reasons", []):
                reason.cites = [canonical(a) for a in reason.cites]
        errors = gate(rec, log)
        cites = []
        if hasattr(rec, "cites"):
            cites += rec.cites
        for item in getattr(rec, "items", []):
            cites += getattr(item, "cites", [])
            for reason in getattr(item, "reasons", []):
                cites += reason.cites
        errors += [f"{a}: not in this decision's view" for a in cites if not (view.resolve(a) or log.resolve(a))]
        return errors

    table = _canon(state)
    selected = [k for k in state["column_candidates"] if f"col:{k}" in view.addresses and S.covcol(k) in table]
    aliases = {key(state["handoff"].outcome): "y", state["score"].column: "x", **{k: S.covcol(k) for k in selected}}
    if state["score"].takeup_column:
        aliases[state["score"].takeup_column] = "t"
    columns = [c for c in ("x", "t", *[S.covcol(k) for k in selected]) if c in table]
    tools = L.data_tools(state, table["x"] >= 0, table_=table[columns], aliases=aliases) if SPEC.budget(name) else None
    record, log, thoughts, errors = run_episode(schema, system, view.text, tools=tools, budget=SPEC.budget(name), gate=checked, node=name, notes=True)
    return (
        record,
        {
            "debug": thoughts,
            "episodes": {name: log},
            "reasoning_history": [
                {
                    "node": name,
                    "context_id": state["handoff"].context_id,
                    "ledger_revision": state["ledger"].revision,
                    "record": record.model_dump() if record else None,
                    "errors": errors,
                    "episode_visit": log.visit,
                }
            ],
            "episode_visits": [{"node": name, "view": view.text, "record": record.model_dump() if record else None, "log": log.model_dump(), "errors": errors}],
        },
        errors,
    )


def load(state: SpecialistState) -> Command:
    h = state["handoff"]
    deficiency = validate_context(h, CATALOGUE)
    if deficiency:
        return _stop("load", deficiency.reason, deficiency.addresses, {"context_deficiency": deficiency}, terminal_kind="context_deficiency")
    b = h.design
    if not isinstance(b, RdDesign):
        deficiency = ContextDeficiency(stage="load", addresses=["design.kind"], reason="the admitted context has no RD block")
        return _stop("load", deficiency.reason, deficiency.addresses, {"context_deficiency": deficiency}, terminal_kind="context_deficiency")
    try:
        it = intake.load(h, "rd", extra=b.columns(), dropna=False)
    except intake.IntakeStop as ex:
        return Command(goto="feasibility", update={"feasibility": ex.feasibility})
    if not pd.api.types.is_numeric_dtype(it.table[key(h.outcome)]):
        return _stop("load", "RD requires a numeric outcome", terminal_kind="unsupported_method")
    return Command(
        goto="validate_context",
        update={
            "run_dir": str(it.run_dir),
            "table_path": str(it.table_path),
            "columns": it.columns,
            "cluster_column": key(b.cluster) if b.cluster else None,
            "sampled_by_side": b.sampled_by_side,
            "declines": it.declines,
            "ladder": Ladder(),
        },
    )


def validate_admitted_context(state):
    b = state["handoff"].design
    required = {
        "claim:assignment.score_column": b.score,
        "claim:assignment.cutoff": b.cutoff,
        "claim:assignment.treated_side": b.treated_side,
        "claim:assignment.cutoff_value_treated": b.cutoff_value_treated,
    }
    missing = [a for a, v in required.items() if v is None]
    if b.score not in state["columns"]:
        missing.append("claim:assignment.score_column")
    if missing:
        d = ContextDeficiency(stage="validate_context", addresses=missing, reason="the admitted RD rule is incomplete")
        return _stop("validate_context", d.reason, missing, {"context_deficiency": d}, terminal_kind="context_deficiency")
    return Command(goto="seed")


def seed(state):
    return {"ledger": seed_context(state["handoff"])}


def score(state):
    b = state["handoff"].design
    tk = b.takeup or {}
    rec = Score(
        column=key(b.score),
        cutoff=b.cutoff,
        treated_side=b.treated_side,
        cutoff_value_treated=b.cutoff_value_treated,
        takeup_column=key(tk["column"]) if tk.get("column") else None,
        takeup_level=str(tk["level"]) if tk.get("level") is not None else None,
        reason="the fixed, admitted assignment rule",
        cites=["claim:assignment.score_column", "claim:assignment.cutoff", "claim:assignment.treated_side"],
        by="pack",
    )
    raw = L.table(state)
    x = pd.to_numeric(raw[rec.column], errors="coerce")
    side = (
        (x >= rec.cutoff)
        if rec.treated_side == "above" and rec.cutoff_value_treated
        else (x > rec.cutoff)
        if rec.treated_side == "above"
        else (x <= rec.cutoff)
        if rec.cutoff_value_treated
        else (x < rec.cutoff)
    )
    if not np.isfinite(x).any() or not (side[np.isfinite(x)].any() and (~side[np.isfinite(x)]).any()):
        return _stop("score", "the cutoff has no observed score support on both sides", terminal_kind="inadequate_support")
    if rec.takeup_column:
        if rec.takeup_column not in raw or rec.takeup_level is None:
            return _stop("score", "the admitted receipt column or positive level is not observed", terminal_kind="context_deficiency")
        got = S.receipt_indicator(raw[rec.takeup_column], rec.takeup_level)
        if got[side].mean() < got[~side].mean():
            return _stop("score", "receipt is lower on the admitted treated side; resolve the contradictory rule", terminal_kind="context_deficiency")
    return Command(goto="shape_table", update={"score": rec, "ladder": state["ladder"].model_copy(update={"score": rec})})


def shape_table(state):
    h, sc = state["handoff"], state["score"]
    candidates = [key(c.column) for c in h.relevant_columns if key(c.column) not in {key(h.outcome), sc.column, sc.takeup_column, state.get("cluster_column")}]
    try:
        canon, xall, facts = S.canonical(L.table(state), sc, key(h.outcome), candidates, state.get("cluster_column"), load_checks())
    except S.ShapeError as ex:
        return _stop("shape_table", ex.reason, ex.facts, terminal_kind="inadequate_support")
    folder = Path(state["run_dir"])
    trace.write(folder / "canon.csv", canon.to_csv(index=False))
    trace.write(folder / "scores.csv", xall.to_frame("x").to_csv(index=False))
    contrast = Contrast(control="control_side", treated="treated_side", reason=sc.reason, cites=sc.cites)
    return Command(
        goto="covariates",
        update={
            "canon_path": str(folder / "canon.csv"),
            "xall_path": str(folder / "scores.csv"),
            "shape": facts,
            "column_candidates": candidates,
            "contrast": contrast,
            "ladder": state["ladder"].model_copy(update={"shape": facts}),
        },
    )


def covariates(state):
    h = state["handoff"]
    fixed, asked = [], []
    for k in state["column_candidates"]:
        c = h.column(k)
        if c and c.when in {"before", "at", "after"} and h.status(f"col:{k}.when") == "confirmed":
            before = c.when == "before" and not c.moved_by_change and not c.measures_outcome and c.facts.kind != "id"
            fixed.append(
                CovariateRelation(
                    column=k,
                    predetermined=before,
                    affected_by_treatment=c.moved_by_change is True,
                    is_outcome_measure=c.measures_outcome is True,
                    modifier_candidate=before and c.may_modify is True,
                    reasons=[Cited(reason="the admitted column measurement account", cites=[f"col:{k}.when"])],
                )
            )
        else:
            asked.append(k)
    extra = {}
    if asked:
        view = build(h, columns=asked, claims=("change",))

        def gate(rec, log):
            errs = []
            if sorted(x.column for x in rec.items) != sorted(asked):
                errs.append("return every offered column exactly once")
            for x in rec.items:
                if x.predetermined and (x.affected_by_treatment or x.is_outcome_measure):
                    errs.append(f"{x.column}: a covariate cannot be post-treatment or an outcome")
                if (x.predetermined or x.affected_by_treatment or x.is_outcome_measure) and not any(r.cites for r in x.reasons):
                    errs.append(f"{x.column}: positive claims require evidence")
            return errs

        rec, extra, errors = _episode(state, "covariates", CovariateRoles, P.COVARIATES, view, gate)
        if rec is None:
            return _stop("covariates", "column roles could not be recorded", errors, extra, terminal_kind="exhausted_reasoning")
        fixed += rec.items
    roles = CovariateRoles(items=fixed)
    canon = _canon(state)
    good = [r.column for r in fixed if r.predetermined and not r.is_outcome_measure and not r.affected_by_treatment and S.covcol(r.column) in canon]
    covs = Covariates(balance_tested=good, adjusted=good, excluded=[Excluded(column=r.column, why=r.word()) for r in fixed if r.column not in good])
    entries = [
        Entry(
            id=f"rel:covariate.{r.column}",
            kind="covariate_role",
            value=r.model_dump(),
            status="supported" if r.reasons else "proposed",
            reason=r.word(),
            evidence=[
                Evidence(source="pack" if h.resolve(a) else "data", address=a, category="measurement_timing") for reason in r.reasons for a in reason.cites
            ],
        )
        for r in fixed
    ]
    episode_facts = {fact.address for visit in extra.get("episode_visits", []) for fact in EpisodeLog.model_validate(visit["log"]).facts}
    ledger, errors = apply(state["ledger"], entries, by="covariates", resolves=lambda a: h.resolve(a) or a in episode_facts)
    if errors:
        return _stop("covariates", "ledger rejected column roles", errors, extra, terminal_kind="exhausted_reasoning")
    return Command(goto="density", update={"covariates": covs, "ledger": ledger, "ladder": state["ladder"].model_copy(update={"covariates": roles}), **extra})


def density(state):
    facts, raw = D.density_evidence(pd.read_csv(state["xall_path"])["x"], state["shape"], load_checks(), sampled_by_side=state["sampled_by_side"])
    return {"density_facts": raw, "ladder": state["ladder"].model_copy(update={"density": facts})}


def balance(state):
    canon, shape = _canon(state), state["shape"]
    inf = pick_inference(cluster_column=bool(shape.cluster_column), clusters=shape.clusters)
    facts = D.balance_evidence(
        canon, pd.DataFrame(), state["covariates"].balance_tested, shape, load_checks(), cluster=bool(shape.cluster_column), vce=inf.vce, window=None
    )
    raw = dict(state["density_facts"])
    raw["windows"] = [w.model_dump() for w in state["ladder"].density.windows]
    checks, extra = D.run_checks(
        canon,
        pd.read_csv(state["xall_path"])["x"],
        shape,
        state["covariates"],
        state["contrast"].key,
        load_checks(),
        cluster=bool(shape.cluster_column),
        vce=inf.vce,
        sampled_by_side=state["sampled_by_side"],
        density=raw,
        balance=facts,
    )
    return {"checks": checks, "check_facts": extra, "ladder": state["ladder"].model_copy(update={"balance": facts})}


def line(state):
    h, lad = state["handoff"], state["ladder"]
    lines = (
        lad.score.lines()
        + lad.shape.lines()
        + lad.density.lines()
        + lad.balance.lines()
        + [(c.address, f"{c.level}: {c.detail}") for c in state["checks"] if c.name not in {"first_stage", "first_stage_weak", "no_first_stage"}]
    )
    view = build(h, claims=("assignment", "cutoff_only", "sampling"), columns=[state["score"].column], lines=lines)

    def gate(rec, log):
        errors = []
        if not rec.why.strip() or not rec.cites:
            errors.append("state a cited reason for the decision")
        if not rec.clean and not rec.risks:
            errors.append("an invalid comparison needs an explicit cited risk")
        for r in rec.risks:
            if not r.cites:
                errors.append(f"{r.name}: missing citation")
            errors += [f"{a}: outside the line view" for a in r.cites if not (view.resolve(a) or log.resolve(a))]
        flagged = [c.address for c in state["checks"] if c.level != "pass" and c.name in {"density", "covariate_continuity"}]
        if rec.clean:
            allc = (
                set(rec.cites)
                | {a for r in rec.risks for a in r.cites}
                | {norm_address(u.about) for u in rec.unsure if u.reason.strip() and (view.resolve(u.about) or log.resolve(u.about))}
            )
            for a in flagged:
                covered = a in allc
                if a.endswith(".density"):
                    covered = covered or "ladder:density.test" in allc
                elif a.endswith(".covariate_continuity"):
                    sources = [f"ladder:balance.{item.column}" for item in lad.balance.items if item.flagged(lad.balance.threshold)]
                    covered = covered or bool(sources) and all(source in allc for source in sources)
                if not covered:
                    errors.append(f"answer the diagnostic {a} or its shown underlying evidence")
        return errors

    rec, extra, errors = _episode(state, "line", Line, P.LINE, view, gate)
    if rec is None:
        return _stop("line", "continuity reasoning could not be recorded", errors, extra, terminal_kind="exhausted_reasoning")
    episode_facts = {fact.address for visit in extra["episode_visits"] for fact in EpisodeLog.model_validate(visit["log"]).facts}
    ledger, errs = apply(
        state["ledger"],
        [continuity_entry(h, rec)],
        by="line",
        resolves=lambda a: view.resolve(a) or a in episode_facts,
    )
    if errs:
        return _stop("line", "continuity ledger rejected the record", errs, extra, terminal_kind="exhausted_reasoning")
    return Command(goto="assumptions", update={"ledger": ledger, "ladder": lad.model_copy(update={"line": rec}), **extra})


def assumptions(state):
    h = state["handoff"]
    offered = [
        CATALOGUE.endpoint(a.endpoint_id)
        for a in h.admission.endpoints
        if a.admitted and CATALOGUE.endpoint(a.endpoint_id).configuration["kind"] == state["shape"].kind
    ]
    if not offered:
        return _stop("assumptions", "no admitted endpoint matches observed compliance", terminal_kind="context_deficiency")
    lines = [
        (
            f"endpoint:{e.id}",
            json.dumps({"title": e.title, "configuration": e.configuration, "targets": e.targets, "assumptions": e.assumptions, "context_admitted": True}),
        )
        for e in offered
    ]
    lines += state["ladder"].line.lines() + [(c.address, f"{c.level}: {c.detail}") for c in state["checks"]]
    view = build(
        h,
        claims=("assignment", "cutoff_only", "exclusion", "monotonicity", "local_randomisation", "sampling", "grain", "target"),
        columns=[state["score"].column],
        lines=lines,
    )

    def gate(rec, log):
        errors = []
        if sorted(x.endpoint_id for x in rec.items) != sorted(e.id for e in offered):
            errors.append("evaluate every offered endpoint exactly once")
        for x in rec.items:
            if not x.cites or not x.reason.strip():
                errors.append(f"{x.endpoint_id}: a cited reason is required")
            e = next((e for e in offered if e.id == x.endpoint_id), None)
            if e and x.admissible and not x.assumptions:
                errors.append(f"{x.endpoint_id}: record identifying assumptions")
            if e and not x.admissible:
                contradictions = {chk.address for chk in state["checks"] if chk.level == "hard"}
                if not state["ladder"].line.clean:
                    contradictions |= {"ladder:line.clean", "ladder:line.why"}
                if state.get("cluster_column") and e.configuration["framework"] == "local_randomisation":
                    contradictions |= {"ladder:shape.clusters"}
                if not set(x.cites) & contradictions:
                    errors.append(
                        f"{x.endpoint_id}: rejecting an admitted conditional endpoint needs shown contradictory or hard support evidence; missing optional balance/density proof does not refute its premises"
                    )
        return errors

    rec, extra, errors = _episode(state, "assumptions", AssumptionDecisions, P.ASSUMPTIONS, view, gate)
    if rec is None:
        return _stop("assumptions", "endpoint assumptions could not be recorded", errors, extra, terminal_kind="exhausted_reasoning")
    entries = [
        Entry(
            id=f"rel:endpoint.{x.endpoint_id}",
            kind="assumption",
            value=x.admissible,
            status="proposed" if x.admissible else "rejected",
            reason=x.reason,
            evidence=[
                Evidence(source="pack" if h.resolve(a) or any(f.source == a for f in h.facts.values()) else "judgement", address=a, category="assumption")
                for a in x.cites
            ],
        )
        for x in rec.items
    ]
    ledger, errors = apply(state["ledger"], entries, by="assumptions", resolves=view.resolve)
    if errors:
        return _stop("assumptions", "ledger rejected endpoint reasoning", errors, extra, terminal_kind="exhausted_reasoning")
    return Command(goto="rivals", update={"assumption_decisions": rec, "ledger": ledger, **extra})


def rivals(state):
    candidates = []
    for decision in state["assumption_decisions"].items:
        endpoint = CATALOGUE.endpoint(decision.endpoint_id)
        for adjusted in [False, True] if state["covariates"].adjusted and endpoint.configuration["framework"] == "continuity" else [False]:
            config = {**endpoint.configuration, "adjusted": adjusted, "estimator": endpoint.estimators[0]}
            cid = design_identity(endpoint.id, config, state["handoff"].context_id)
            candidates.append(
                Candidate(
                    id=cid,
                    endpoint_id=endpoint.id,
                    status="surviving" if decision.admissible else "blocked",
                    reason=decision.reason,
                    terminal_kind=None if decision.admissible else "unidentified_effect",
                    cites=decision.cites,
                    assumptions=list(dict.fromkeys([*endpoint.assumptions, *decision.assumptions])),
                    adjusted=adjusted,
                    estimator=endpoint.estimators[0],
                )
            )
    # Deterministic preference: requested receipt/assignment, polynomial, unadjusted, then window and eligibility companions.
    candidates.sort(key=lambda c: (c.endpoint_id.startswith("eligibility"), "randomisation" in c.endpoint_id, c.adjusted, c.id))
    for c in candidates[MAX_DESIGNS:]:
        c.status = "blocked"
        c.reason = "declared maximum of eight designs"
        c.terminal_kind = "exhausted_reasoning"
    return {"candidates": candidates, "candidate_history": [_transition(state, "rivals", candidates)]}


def check_candidates(state):
    records = []
    for candidate in state["candidates"]:
        c = candidate.model_copy(deep=True)
        if c.status == "surviving":
            reasons = []
            if "randomisation" in c.endpoint_id and state.get("cluster_column"):
                reasons.append("row permutations do not implement clustered assignment")
            if "polynomial" in c.endpoint_id and not state["ladder"].line.clean:
                reasons.append("continuity account was rejected")
            for chk in state["checks"]:
                if chk.level == "hard" and chk.name not in {"no_first_stage", "first_stage_weak"}:
                    reasons.append(chk.detail)
            if c.endpoint_id == "fuzzy_local_polynomial" and state.get("check_facts", {}).get("first_stage_status") != "strong":
                reasons.append("the polynomial ratio requires the declared strong first stage")
            if c.adjusted:
                cols = [S.covcol(k) for k in state["covariates"].adjusted]
                complete = _canon(state).dropna(subset=cols)
                if not len(complete) or np.linalg.matrix_rank(np.column_stack([np.ones(len(complete)), complete[cols].to_numpy()])) != len(cols) + 1:
                    reasons.append("adjustment covariates are rank deficient; automatic dropping is disabled")
            if reasons:
                c.terminal_kind = (
                    "unsupported_method"
                    if any("clustered" in r or "rank deficient" in r for r in reasons)
                    else "inadequate_support"
                    if any(chk.level == "hard" and chk.name not in {"no_first_stage", "first_stage_weak"} for chk in state["checks"])
                    else "unidentified_effect"
                )
                c.status = "blocked"
                c.reason = "; ".join(reasons)
        records.append(c)
    return {"candidates": records, "candidate_history": [_transition(state, "check_candidates", records)]}


def freeze_designs(state):
    canon = _canon(state)
    designs = {}
    candidates = []
    cfg = load_checks()
    fingerprint = hashlib.sha256(Path(state["canon_path"]).read_bytes()).hexdigest()
    for raw in state["candidates"]:
        c = raw.model_copy(deep=True)
        if c.status != "surviving":
            candidates.append(c)
            continue
        endpoint = CATALOGUE.endpoint(c.endpoint_id)
        entry = estimator(c.estimator)
        adjusted = state["covariates"].adjusted if c.adjusted else []
        fuzzy = c.endpoint_id.startswith("fuzzy")
        sample_columns = [S.covcol(k) for k in (state["covariates"].balance_tested if entry.engine == "local_randomisation" else adjusted)] + (
            ["t"] if fuzzy else []
        )
        table = canon.dropna(subset=sample_columns).copy()
        inf = pick_inference(cluster_column=bool(state.get("cluster_column")), clusters=state["shape"].clusters)
        if entry.engine == "local_randomisation":
            window = D.randomisation_window(table, state["covariates"].balance_tested, cfg, entry.params)
        else:
            window = D.primary_window(
                entry.params,
                table,
                state["shape"],
                cfg,
                fuzzy=fuzzy,
                covs=[S.covcol(k) for k in adjusted],
                cluster=bool(state.get("cluster_column")),
                vce=inf.vce,
            )
        if window.get("error"):
            c.status = "blocked"
            c.reason = window["error"]
            c.terminal_kind = "inadequate_support"
            candidates.append(c)
            continue
        bw = Bandwidths(**window["bandwidths"])
        if min(bw.n_h_left, bw.n_h_right) < cfg["window"]["min_rows_each_side"]:
            c.status = "blocked"
            c.reason = "the selected window has insufficient effective rows"
            c.terminal_kind = "inadequate_support"
            candidates.append(c)
            continue
        target = {
            "estimand": endpoint.targets[0],
            "population": "compliers inside the window"
            if c.endpoint_id == "fuzzy_local_randomisation"
            else "units inside the window"
            if entry.engine == "local_randomisation"
            else "compliers at the cutoff"
            if fuzzy
            else "units at the cutoff",
            "outcome": state["handoff"].outcome,
            "contrast": "eligibility" if c.endpoint_id.startswith("eligibility") else "receipt" if fuzzy else "assignment",
            "locality": "window" if entry.engine == "local_randomisation" else "cutoff",
            "cutoff": state["score"].cutoff,
            "window": bw.h if entry.engine == "local_randomisation" else None,
            "assumptions": c.assumptions,
        }
        covs = state["covariates"].model_copy(update={"adjusted": adjusted})
        checks = [
            chk.model_copy(update={"design_id": c.id})
            for chk in state["checks"]
            if chk.name not in {"first_stage", "first_stage_weak", "no_first_stage"}
            or (entry.engine != "local_randomisation" and c.endpoint_id.startswith("fuzzy"))
        ]
        if entry.engine == "local_randomisation" and fuzzy:
            inside = table[(table.x >= -bw.h_left) & (table.x <= bw.h_right)]
            left_receipt = float(inside.loc[inside.x < 0, "t"].mean())
            right_receipt = float(inside.loc[inside.x >= 0, "t"].mean())
            difference = right_receipt - left_receipt
            checks.append(
                CheckResult(
                    design_id=c.id,
                    contrast=state["contrast"].key,
                    name="first_stage_window",
                    level="soft" if abs(difference) <= 1e-9 else "pass",
                    value=difference,
                    detail=f"Frozen-window receipt rates: control {left_receipt:.6g}, treated {right_receipt:.6g}; difference {difference:.6g}. Anderson-Rubin inversion does not require the polynomial strong-first-stage threshold.",
                )
            )
        d = Design(
            id=c.id,
            endpoint_id=c.endpoint_id,
            context_id=state["handoff"].context_id,
            ledger_revision=state["ledger"].revision,
            target=target,
            contrast=state["contrast"],
            score=state["score"],
            shape=state["shape"],
            covariates=covs,
            checks=Checks(results=checks),
            estimator=c.estimator,
            estimand=endpoint.targets[0],
            spec=dict(entry.params),
            inference="randomisation" if entry.engine == "local_randomisation" else inf.name,
            vce="randomisation" if entry.engine == "local_randomisation" else inf.vce,
            cluster=state.get("cluster_column") if entry.engine != "local_randomisation" else None,
            bandwidths=bw,
            fuzzy=fuzzy,
            assumptions=c.assumptions,
            falsifications=[
                e.name
                for e in load_falsifications()
                if e.applies(engine=entry.engine, kind=state["shape"].kind, rule=bw.rule, distinct_scores=state["shape"].distinct_scores)
            ],
            sensitivities=[
                e.name
                for e in load_sensitivities()
                if e.applies(engine=entry.engine, kind=state["shape"].kind, rule=bw.rule, distinct_scores=state["shape"].distinct_scores)
            ],
            modifiers=[r.column for r in state["ladder"].covariates.items if r.modifier_candidate],
            rows=[int(r) for r in table["row"]],
            table_fingerprint=fingerprint,
            identification_certificate={
                "endpoint_id": endpoint.id,
                "context_id": state["handoff"].context_id,
                "admission": next(a.model_dump() for a in state["handoff"].admission.endpoints if a.endpoint_id == endpoint.id),
                "assumption_entry": f"rel:endpoint.{endpoint.id}",
                "conditional_on": c.assumptions,
            },
            support={
                "analysis_rows": len(table),
                "effective_rows_control": bw.n_h_left,
                "effective_rows_treated": bw.n_h_right,
                "score_regime": "coarse" if state["shape"].distinct_scores < cfg["support"]["distinct_min"]["soft"] else "dense",
            },
            weights_policy={"kernel": entry.params.get("kernel", "uniform" if entry.engine == "local_randomisation" else "tri"), "external_weights": None},
            selector_policy={"selector": bw.selector, "outcome_dependent": entry.engine == "local_polynomial", "selected_after_specification": True},
        )
        designs[d.id] = d
        trace.write(Path(state["run_dir"]) / d.id / "design.json", d.model_dump_json(indent=2))
        c.check_results = window.get("diagnostics", [])
        candidates.append(c)
    if not designs:
        return _stop(
            "freeze_designs",
            "no candidate can be frozen",
            [f"{c.endpoint_id}: {c.reason}" for c in candidates],
            {"candidates": candidates, "candidate_history": [_transition(state, "freeze_designs", candidates)]},
            terminal_kind=next(iter({c.terminal_kind for c in candidates}))
            if len({c.terminal_kind for c in candidates}) == 1 and candidates[0].terminal_kind
            else "unidentified_effect",
        )
    lead_selection = select_lead(state["handoff"], designs, state["ledger"])
    trace.write(Path(state["run_dir"]) / "lead_selection.json", json.dumps(lead_selection, indent=2))
    limitation = (
        {
            "feasibility": Feasibility(
                stage="freeze_designs", reason=lead_selection["reason"], facts=lead_selection["cites"], terminal_kind="unidentified_effect"
            )
        }
        if lead_selection["chosen_id"] is None
        else {}
    )
    return Command(
        goto=[Send("execute", {"design": d.model_dump(), "canon_path": state["canon_path"]}) for d in designs.values()],
        update={
            "candidate_history": [_transition(state, "freeze_designs", candidates)],
            "designs": designs,
            "lead_design_id": lead_selection["chosen_id"],
            "lead_selection": lead_selection,
            "reasoning_history": [{"node": "freeze_designs", "context_id": state["handoff"].context_id, "lead_selection": lead_selection}],
            "candidates": candidates,
            "checks": [chk for d in designs.values() for chk in d.checks.results],
            **limitation,
        },
    )
