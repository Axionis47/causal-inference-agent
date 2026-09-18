"""Diff-in-diff lane nodes. Facts compute; judgements call the model once and are gated.

Stops are typed: a node that cannot go on routes to "feasibility" with a Feasibility record.
Nothing here names a column, a method, or a dataset.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from langgraph.config import get_stream_writer
from langgraph.types import Command, Send

from causal_agent.common.addresses import key as _key
from causal_agent.common.contracts import CheckResult, Checks, Contrast, DidDesign, Estimate, Feasibility, Handoff, Interpretation, Refutation
from causal_agent.common.llm import structured
from causal_agent.profile.datasets import ROOT, dataset_entries
from causal_agent.specialists.did import adapter, checks as CK, prompts as P, shape as SH
from causal_agent.specialists.did.contracts import (
    ControlRelation,
    Controls,
    Design,
    DesignAssessment,
    EstimatorPick,
    Excluded,
    Groups,
    Periods,
    Revision,
)
from causal_agent.specialists.did.knowledge import (
    estimator as estimator_entry,
    load_checks,
    load_estimators,
    load_placebos,
    pick_inference,
    placebo as placebo_entry,
    render_preferences,
)
from causal_agent.specialists.did.state import InterpretTask, PlaceboTask, RelateTask, SpecialistState

MAX_RELATE_ATTEMPTS = 3
MAX_REVISIONS = 3
MAX_MODEL_RETRIES = 3
MAX_PICK_ATTEMPTS = 2
MAX_LEVELS = 100  # levels shown to the model; a level list cut short once hid California (state 5) behind 30 string-sorted numbers

# ------------------------------------------------------------------ helpers

def _writer():
    try:
        return get_stream_writer()
    except Exception:
        return lambda _x: None


def _question(state: SpecialistState) -> str:
    return state.get("question") or ""


def _stop(stage: str, reason: str, facts: list[str], fix: str, extra: dict | None = None) -> Command:
    f = Feasibility(stage=stage, reason=reason, facts=facts, what_would_fix=fix)
    return Command(goto="feasibility", update={"feasibility": f, **(extra or {})})


def _feas(stage: str, reason: str, facts: list[str], fix: str) -> Feasibility:
    return Feasibility(stage=stage, reason=reason, facts=facts, what_would_fix=fix)


def _card(h: Handoff, key: str) -> str:
    return h.brief_text(key)


def _block(h: Handoff) -> DidDesign | None:
    return h.design if isinstance(h.design, DidDesign) else None


def _cites(h: Handoff, *addresses: str) -> list[str]:
    """The claim addresses that resolve in the pack, else the change card."""
    ok = [a for a in addresses if h.resolve(a)]
    return ok or ["change:1.note"]


def _frame_text(state: SpecialistState) -> str:
    h = state["handoff"]
    s = h.scope
    g, p = state.get("groups"), state.get("periods")
    lines = [f"family: {h.family}; outcome: {h.outcome}; treatment: {h.treatment}; filter={s.population_filter or 'none'}; window={s.window or 'none'}; target={s.target}",
             f"assumption the router bet on: {h.chosen_assumption}"]
    if _block(h):
        lines.append("what the pack settled:\n" + h.design.render())
    if g:
        lines.append(f"groups: {g.column} = {g.treated_level!r} treated, other levels control")
    if p:
        lines.append(f"periods: {p.kind}; " + (f"time {p.time_column}, first post {p.first_post}" if p.kind == "long" else f"before {p.before_column}, after {p.after_column}"))
    return "\n".join(lines)


def _rejected(errors: list[str]) -> str:
    return ("\nPREVIOUS ANSWER WAS REJECTED\n" + "\n".join(f"- {e}" for e in errors) + "\n") if errors else ""


def _keys(state: SpecialistState) -> tuple[str, str, list[str]]:
    h = state["handoff"]
    t, y = _key(h.treatment or ""), _key(h.outcome)
    rel = [_key(c.column) for c in h.relevant_columns]
    return t, y, [k for k in dict.fromkeys(rel) if k in state.get("columns", {})]


def _sorted_levels(col: pd.Series) -> list:
    vals = col.dropna().unique()
    try:
        return sorted(vals, key=lambda v: float(v))
    except (TypeError, ValueError):
        return sorted(vals, key=lambda v: str(v))


# ------------------------------------------------------------------ load (fact)


def load(state: SpecialistState) -> Command:
    h = state["handoff"]
    entry = dataset_entries()[h.pack_name]
    raw = pd.read_csv(ROOT / (h.csv or entry["csv"]))
    columns = {_key(c): c for c in raw.columns}
    raw.columns = [_key(c) for c in raw.columns]
    t, y = _key(h.treatment or ""), _key(h.outcome)
    wanted = [k for k in dict.fromkeys([t, y] + [_key(c.column) for c in h.relevant_columns]) if k]
    b = _block(h)
    unit_col = _key(b.unit) if b and b.unit else (_key(entry["entity"][0]) if entry.get("entity") else None)
    if unit_col and unit_col in raw.columns and unit_col not in wanted:
        wanted.append(unit_col)
    missing = [k for k in wanted if k not in raw.columns]
    if not h.treatment or missing:
        return _stop("load", "a column the hand-off names is not in the file" if missing else "the hand-off names no treatment",
                     [f"missing: {missing}"] if missing else [], "a hand-off whose columns exist in the file")
    table = raw[wanted].dropna()
    if not pd.api.types.is_numeric_dtype(table[y]):
        return _stop("load", "the outcome is not numeric", [f"outcome {y} is {table[y].dtype}"], "a numeric outcome")
    target = load_checks()["target_units"].get(h.scope.target)
    if target is None:
        return _stop("load", f"target '{h.scope.target}' is not supported by this lane", [], "a question asking for the average effect, or the effect on the treated")
    levels = [str(v) for v in _sorted_levels(table[t])][:MAX_LEVELS]
    run_dir = Path(os.getenv("RUN_DIR", ".artifacts/runs")) / f"{h.pack_name}-did-{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    table_path = run_dir / "table.csv"
    table.to_csv(table_path, index=False)
    _writer()({"load": {"rows": len(table), "columns": wanted, "target_units": target, "unit_column": unit_col, "run_dir": str(run_dir)}})
    return Command(goto="groups", update={
        "run_dir": str(run_dir), "table_path": str(table_path), "columns": {k: columns[k] for k in wanted}, "unit_column": unit_col,
        "target_units": target, "group_levels": levels, "relate_attempts": 0, "revisions": 0, "pick_attempts": 0,
        "relate_errors": {}, "excluded_estimators": [], "applied_revisions": [],
    })


# ------------------------------------------------------------------ groups (judgement)


def _block_groups(h: Handoff) -> Groups | None:
    b = _block(h)
    tg = b.treated_group if b else {}
    if not (tg.get("column") and tg.get("level") is not None):
        return None
    return Groups(column=_key(tg["column"]), treated_level=str(tg["level"]), reason="the pack names the column and the level that mean the unit got the change",
                  cites=_cites(h, "claim:assignment.treatment_column", "claim:assignment.treated_level"))


def groups(state: SpecialistState) -> dict:
    h = state["handoff"]
    t, _, _ = _keys(state)
    table_cols = set(pd.read_csv(state["table_path"], nrows=0).columns)
    errors: list[str] = []
    debug = []
    block = _block_groups(h)
    for attempt in range(MAX_MODEL_RETRIES + (1 if block else 0)):
        if block is not None and attempt == 0:  # a fact from the pack; the same checks apply, and the model is asked only if they fail
            parsed = block
        else:
            user = P.GROUPS_USER.format(question=_question(state), frame=_frame_text(state), dataset_card=h.render_dataset(), changes=h.render_change(),
                                        treatment_card=_card(h, t), levels=", ".join(repr(v) for v in state["group_levels"])) + _rejected(errors)
            parsed, th = structured(Groups, P.GROUPS_SYSTEM, user, node="groups")
            debug.append(th)
        parsed.column = _key(parsed.column)
        errors = []
        if parsed.column not in table_cols:
            errors.append(f"{parsed.column!r} is not a column in the table")
        else:
            col = pd.read_csv(state["table_path"], usecols=[parsed.column])[parsed.column]
            observed = set(col.astype(str))
            if str(parsed.treated_level) not in observed:
                errors.append(f"level {parsed.treated_level!r} is not observed in {parsed.column!r}; observed: {[str(v) for v in _sorted_levels(col)][:MAX_LEVELS]}")
            elif len(observed) < 2:
                errors.append(f"{parsed.column!r} has a single level; nothing to compare")
        for c in parsed.cites:
            if not h.resolve(c):
                errors.append(f"{c} is not a pack address")
        if not errors:
            _writer()({"groups": parsed.model_dump()})
            return {"groups": parsed, "debug": debug}
        if parsed is block:
            _writer()({"groups": {"pack_block_rejected": errors}})
            errors = []
    return {"feasibility": _feas("groups", "could not name who got the change", errors, "a column and level the notes tie to the change"), "debug": debug}


def after_groups(state: SpecialistState) -> str:
    return "feasibility" if state.get("feasibility") else "periods"


# ------------------------------------------------------------------ periods (judgement)


def _block_periods(h: Handoff) -> Periods | None:
    b = _block(h)
    if not (b and b.time and b.change_period):
        return None
    return Periods(kind="long", time_column=_key(b.time), first_post=str(b.change_period), window_start=None, window_end=None,
                   reason="the pack names the period column and the first period at or after the change",
                   cites=_cites(h, "claim:change.date_column", "claim:change.period_value"))


def periods(state: SpecialistState) -> dict:
    h = state["handoff"]
    g: Groups = state["groups"]
    _, y, rel = _keys(state)
    time_cards = "\n\n".join(_card(h, k) for k in rel if k not in (g.column,))
    errors: list[str] = []
    debug = []
    table_cols = set(pd.read_csv(state["table_path"], nrows=0).columns)
    block = _block_periods(h)
    for attempt in range(MAX_MODEL_RETRIES + (1 if block else 0)):
        if block is not None and attempt == 0:
            parsed = block
        else:
            user = P.PERIODS_USER.format(question=_question(state), changes=h.render_change(), dataset_card=h.render_dataset(),
                                         outcome_card=_card(h, y), time_cards=time_cards or "(none besides the outcome)") + _rejected(errors)
            parsed, th = structured(Periods, P.PERIODS_SYSTEM, user, node="periods")
            debug.append(th)
        errors = []
        if parsed.kind == "long":
            parsed.time_column = _key(parsed.time_column or "")
            if parsed.time_column not in table_cols:
                errors.append(f"time column {parsed.time_column!r} is not in the table")
            if not parsed.first_post:
                errors.append("long shape needs first_post")
        else:
            parsed.before_column, parsed.after_column = _key(parsed.before_column or ""), _key(parsed.after_column or "")
            for c in (parsed.before_column, parsed.after_column):
                if c not in table_cols:
                    errors.append(f"{c!r} is not in the table")
        for c in parsed.cites:
            if not h.resolve(c):
                errors.append(f"{c} is not a pack address")
        if not errors:
            _writer()({"periods": parsed.model_dump(exclude_none=True)})
            return {"periods": parsed, "debug": debug}
        if parsed is block:
            _writer()({"periods": {"pack_block_rejected": errors}})
            errors = []
    return {"feasibility": _feas("periods", "could not locate before and after", errors, "a time column or a before and an after measure the notes describe"), "debug": debug}


def after_periods(state: SpecialistState) -> str:
    return "feasibility" if state.get("feasibility") else "shape_table"


# ------------------------------------------------------------------ shape (fact)


def shape_table(state: SpecialistState) -> Command:
    g: Groups = state["groups"]
    p: Periods = state["periods"]
    _, y, rel = _keys(state)
    table = pd.read_csv(state["table_path"])
    candidates = [k for k in rel if k not in (g.column, y, p.time_column, p.before_column, p.after_column, state.get("unit_column"))]
    try:
        panel, facts = SH.canonical(table, g, p, y, candidates, unit_column=state.get("unit_column"))
    except SH.ShapeError as ex:
        return _stop("shape_table", ex.reason, ex.facts, ex.fix)
    panel_path = Path(state["run_dir"]) / "panel.csv"
    panel.to_csv(panel_path, index=False)
    others = sorted(set(state["group_levels"]) - {str(g.treated_level)})
    control = others[0] if len(others) == 1 else "other"
    contrast = Contrast(control=control, treated=str(g.treated_level), reason=g.reason, cites=g.cites)
    _writer()({"shape": facts.model_dump(exclude={"time_values"})})
    update = {"panel_path": str(panel_path), "shape": facts, "contrast": contrast}
    sends = _relate_sends(state, candidates)
    return Command(goto=sends or "merge_controls", update=update)


def _relate_sends(state: SpecialistState, candidates: list[str], errors: dict[str, list[str]] | None = None) -> list[Send]:
    h = state["handoff"]
    errs = errors or {}
    targets = [k for k in candidates if k in errs] if errs else candidates
    return [Send("relate", RelateTask(question=_question(state), frame=_frame_text(state), column=k, card=_card(h, k), errors=_rejected(errs.get(k, []))))
            for k in targets]


def _candidates(state: SpecialistState) -> list[str]:
    g: Groups = state["groups"]
    p: Periods = state["periods"]
    _, y, rel = _keys(state)
    return [k for k in rel if k not in (g.column, y, p.time_column, p.before_column, p.after_column, state.get("unit_column"))]


# ------------------------------------------------------------------ relate (judgement, fan-out)


def relate(task: RelateTask) -> dict:
    user = P.RELATE_USER.format(**task)
    parsed, th = structured(ControlRelation, P.RELATE_SYSTEM, user, node=f"relate:{task['column']}")
    parsed.column = task["column"]
    return {"relations": [parsed], "debug": [th]}


# ------------------------------------------------------------------ merge + verify (facts)


def merge_controls(state: SpecialistState) -> dict:
    h = state["handoff"]
    shape = state["shape"]
    latest: dict[str, ControlRelation] = {r.column: r for r in state.get("relations") or []}
    included: list[str] = []
    dropped: list[Excluded] = []
    excluded: list[Excluded] = []
    for k in _candidates(state):
        r = latest.get(k)
        if r is None:
            continue
        card = h.brief(k)
        varies = card.facts.varies_over if card else None
        # facts first: a column the fixed effects absorb is never a control, whatever the model said
        if shape.kind == "wide":
            dropped.append(Excluded(column=k, why="one row per unit in a two-period table; absorbed by the unit effects"))
            continue
        if varies in ("entity", "neither"):
            dropped.append(Excluded(column=k, why=f"fixed within a unit (varies over {varies}); absorbed by the unit effects"))
            continue
        if varies == "time":
            dropped.append(Excluded(column=k, why="the same for every unit in a period; absorbed by the period effects"))
            continue
        # then the judgements
        if r.affected_by_treatment:
            excluded.append(Excluded(column=k, why="could be changed by the treatment; adjusting for it would remove part of the effect"))
        elif r.usable_as_control:
            included.append(k)
        else:
            excluded.append(Excluded(column=k, why="judged not a driver of the outcome that differs between the groups over time"))
    for rv in state.get("applied_revisions") or []:
        if rv.change == "remove_control":
            if rv.column in included:
                included.remove(rv.column)
                excluded.append(Excluded(column=rv.column, why=f"revised out: {rv.reason}"))
        elif rv.column not in included and rv.column in _candidates(state):
            included.append(rv.column)
            excluded = [x for x in excluded if x.column != rv.column]
            dropped = [x for x in dropped if x.column != rv.column]
    c = Controls(included=included, dropped_fixed=dropped, excluded=excluded)
    _writer()({"controls": c.render()})
    return {"controls": c}


def verify(state: SpecialistState) -> Command:
    h = state["handoff"]
    latest: dict[str, ControlRelation] = {r.column: r for r in state.get("relations") or []}
    errs: dict[str, list[str]] = {}
    for k in _candidates(state):
        r = latest.get(k)
        if r is None:
            errs.setdefault(k, []).append("no relation returned")
            continue
        if (r.affected_by_treatment or r.usable_as_control) and not r.reasons:
            errs.setdefault(k, []).append("claims marked true but no reasons given")
        if r.affected_by_treatment and r.usable_as_control:
            errs.setdefault(k, []).append("cannot be both changed by the treatment and a control")
        for reason in r.reasons:
            if not reason.cites:
                errs.setdefault(k, []).append("a reason has no citation")
            for c in reason.cites:
                if not h.resolve(c):
                    errs.setdefault(k, []).append(f"{c} is not a pack address")
    attempts = state.get("relate_attempts", 0) + 1
    if errs:
        if attempts >= MAX_RELATE_ATTEMPTS:
            return _stop("verify", "the control relations could not be made to pass verification", [f"{k}: {'; '.join(v)}" for k, v in errs.items()],
                         "clearer column notes about what moves over time and what the treatment touches", {"relate_attempts": attempts})
        _writer()({"verify": {"attempt": attempts, "errors": errs}})
        return Command(goto=_relate_sends(state, _candidates(state), errs), update={"relate_errors": errs, "relate_attempts": attempts})
    _writer()({"verify": "ok"})
    return Command(goto="check_design", update={"relate_errors": {}, "relate_attempts": attempts})


# ------------------------------------------------------------------ checks (fact)


def check_design(state: SpecialistState) -> dict:
    panel = pd.read_csv(state["panel_path"])
    results = CK.run_checks(panel, state["shape"], state["controls"].included, state["contrast"].key, load_checks())
    _writer()({"checks": [f"{r.level} {r.address} {r.detail}" for r in results]})
    return {"checks": results}


def after_checks(state: SpecialistState) -> str:
    return "assess" if any(r.level != "pass" for r in state["checks"]) else "pick_estimator"


# ------------------------------------------------------------------ assess (judgement, repair loop)


def _design_text(state: SpecialistState) -> str:
    return _frame_text(state) + "\n" + state["controls"].render() + "\nshape: " + json.dumps(state["shape"].model_dump(exclude={"time_values"}))


def assess(state: SpecialistState) -> Command:
    checks = Checks(results=state["checks"])
    flags, hard = checks.flags, checks.hard
    candidates = set(_candidates(state))
    flag_text = "\n".join(f"[{r.address}] {r.level.upper()}: {r.detail}" for r in flags)
    errors: list[str] = []
    debug = []
    for _ in range(MAX_MODEL_RETRIES):
        user = P.ASSESS_USER.format(question=_question(state), design=_design_text(state), flags=flag_text, errors=_rejected(errors))
        parsed, th = structured(DesignAssessment, P.ASSESS_SYSTEM, user, node="assess")
        debug.append(th)
        errors = []
        if parsed.action == "proceed" and hard:
            errors.append("proceed is not allowed while a hard flag stands: " + ", ".join(r.address for r in hard))
        if parsed.action == "revise":
            if not parsed.revisions:
                errors.append("revise needs at least one revision")
            for rv in parsed.revisions:
                rv.column = _key(rv.column)
                if rv.column not in candidates:
                    errors.append(f"revision names {rv.column!r}, which is not a candidate control; candidates: {sorted(candidates)}")
        for c in parsed.cites:
            if not (any(c == r.address for r in state["checks"]) or state["handoff"].resolve(c)):
                errors.append(f"{c} is not a check or pack address")
        if errors:
            continue
        _writer()({"assess": parsed.model_dump()})
        if parsed.action == "proceed":
            return Command(goto="pick_estimator", update={"assessment": parsed, "debug": debug})
        if parsed.action == "stop":
            return _stop("assess", parsed.reason, [f"{r.address}: {r.detail}" for r in flags], "a comparison group that moved like the treated group before the change, or a design that does not need one",
                         {"assessment": parsed, "debug": debug})
        n = state.get("revisions", 0) + 1
        if n > MAX_REVISIONS:
            return _stop("assess", "three revisions did not clear the flags", [f"{r.address}: {r.detail}" for r in flags], "a different design", {"assessment": parsed, "debug": debug, "revisions": n})
        return Command(goto="merge_controls", update={"assessment": parsed, "debug": debug, "revisions": n,
                                                      "applied_revisions": (state.get("applied_revisions") or []) + parsed.revisions})
    return _stop("assess", "the design assessment could not be validated", errors, "see the gate errors", {"debug": debug})


# ------------------------------------------------------------------ pick estimator (judgement)


def _facts(state: SpecialistState) -> dict[str, Any]:
    s = state["shape"]
    return {"kind": s.kind, "units_treated": s.units_treated, "units_control": s.units_control, "periods_pre": s.periods_pre,
            "periods_post": s.periods_post, "cohorts": s.cohorts, "controls": state["controls"].included}


def pick_estimator(state: SpecialistState) -> Command:
    facts = _facts(state)
    s = state["shape"]
    excluded = set(state.get("excluded_estimators") or [])
    allowed = [e for e in load_estimators() if e.applies(cohorts=s.cohorts, periods_pre=s.periods_pre, periods_post=s.periods_post) and e.name not in excluded]
    if not allowed:
        return _stop("pick_estimator", "no estimator in the catalogue applies to this design", [f"facts: {facts}", f"excluded after failures: {sorted(excluded)}"],
                     "an estimator entry for this adoption pattern and period count")
    names = [e.name for e in allowed]
    check_text = "\n".join(f"[{r.address}] {r.level}: {r.detail}" for r in state["checks"])
    errors: list[str] = []
    debug = []
    for _ in range(MAX_MODEL_RETRIES):
        user = P.PICK_USER.format(facts=json.dumps(facts), checks=check_text, estimators="\n\n".join(e.render() for e in allowed),
                                  preferences=render_preferences(allowed), names=", ".join(names), errors=_rejected(errors))
        parsed, th = structured(EstimatorPick, P.PICK_SYSTEM, user, node="pick_estimator")
        debug.append(th)
        errors = []
        if parsed.name not in names:
            errors.append(f"{parsed.name!r} is not one of {names}")
        for c in parsed.cites:
            if not (any(c == r.address for r in state["checks"]) or state["handoff"].resolve(c)):
                errors.append(f"{c} is not a check or pack address")
        if not errors:
            _writer()({"estimator": parsed.model_dump()})
            return Command(goto="freeze_design", update={"estimator": parsed.name, "estimator_pick": parsed, "debug": debug,
                                                         "pick_attempts": state.get("pick_attempts", 0) + 1})
    return _stop("pick_estimator", "the estimator pick could not be validated", errors, "see the gate errors", {"debug": debug})


# ------------------------------------------------------------------ freeze (fact)


def freeze_design(state: SpecialistState) -> dict:
    s = state["shape"]
    entry = estimator_entry(state["estimator"])
    controls = state["controls"].included
    also = None
    if entry.also_run:
        alt = estimator_entry(entry.also_run)
        if alt.applies(cohorts=s.cohorts, periods_pre=s.periods_pre, periods_post=s.periods_post):
            also = alt
    inf = pick_inference(units_treated=s.units_treated, kind=s.kind)
    units = s.units_treated + s.units_control
    placebos = [p.name for p in load_placebos() if p.applies(units=units, periods_pre=s.periods_pre)]
    d = Design(contrast=state["contrast"], groups=state["groups"], periods=state["periods"], shape=s, controls=state["controls"],
               checks=Checks(results=state["checks"]), estimator=entry.name, formula=adapter.formula_for(entry, controls),
               also_run=also.name if also else None, also_formula=adapter.formula_for(also, controls) if also else None,
               inference=inf.name, vcov=inf.vcov, placebos=placebos, target_units=state["target_units"])
    run_dir = Path(state["run_dir"])
    (run_dir / "design.json").write_text(d.model_dump_json(indent=2))
    (run_dir / "design.md").write_text(d.render())
    _writer()({"design": d.render()})
    return {"design": d}


# ------------------------------------------------------------------ estimate (fact) + placebos (fact, fan-out)


def estimate(state: SpecialistState) -> Command:
    d: Design = state["design"]
    panel = pd.read_csv(state["panel_path"])
    ests, model = adapter.estimate(estimator_entry(d.estimator), d.formula, panel, d.vcov, d.contrast.key, d.target_units)
    refs: list[Refutation] = []
    dynamic: dict = {}
    primary = ests[0]
    if primary.error is None and d.also_run:
        more, alt_model = adapter.estimate(estimator_entry(d.also_run), d.also_formula, panel, d.vcov, d.contrast.key, d.target_units, secondary=True)
        ests += more
        if alt_model is not None and d.also_run == "twfe_dynamic":
            dynamic = {str(k): v for k, v in adapter.dynamic_coefficients(alt_model).items()}
    inf = pick_inference(units_treated=d.shape.units_treated, kind=d.shape.kind)
    if primary.error is None and inf.resample == "wild_bootstrap" and model is not None:
        p = adapter.wild_bootstrap(model, int(inf.params.get("reps", 999)), int(inf.params.get("seed", 7)))
        refs.append(Refutation(contrast=d.contrast.key, refuter="wild_bootstrap", kind="sensitivity", p_value=p,
                               detail=f"wild cluster bootstrap p-value for the effect: {p}" if p is not None else "wild bootstrap failed"))
    _writer()({"estimate": [e.model_dump(exclude_none=True) for e in ests]})
    update: dict[str, Any] = {"estimates": ests, "refutations": refs, "dynamic": dynamic}
    if primary.error:
        if state.get("pick_attempts", 0) < MAX_PICK_ATTEMPTS:
            update["excluded_estimators"] = (state.get("excluded_estimators") or []) + [d.estimator]
            return Command(goto="pick_estimator", update=update)
        return _stop("estimate", "the estimator failed to fit and the re-pick failed too", [primary.error], "a different estimator entry", update)
    sends = [Send("placebo", PlaceboTask(name=n, design=d.model_dump(), panel_path=state["panel_path"], observed=primary.value)) for n in d.placebos]
    return Command(goto=sends or "interpret", update=update)


def placebo(task: PlaceboTask) -> dict:
    d = Design.model_validate(task["design"])
    panel = pd.read_csv(task["panel_path"])
    entry = placebo_entry(task["name"])
    base = adapter.formula_for(estimator_entry(d.estimator), [])  # placebos refit the bare shape; controls do not change what a placebo tests
    if task["name"] == "placebo_group":
        r = adapter.placebo_group(base, panel, task["observed"], entry, d.contrast.key)
    else:
        r = adapter.placebo_timing(base, panel, entry, d.contrast.key)
    _writer()({"placebo": {r.refuter: r.detail}})
    return {"refutations": [r]}


# ------------------------------------------------------------------ interpret (judgement)


def _addresses(state: SpecialistState) -> list[str]:
    d: Design = state["design"]
    c = d.contrast.key
    out = ["design.assumption", "design.periods", "design.controls"] + [b.address for b in state["handoff"].beliefs.values() if b.known() or b.status == "unknown"] + [r.address for r in d.checks.results]
    for e in state.get("estimates") or []:
        if e.error is None:
            tag = f"estimate:{c}" if e.method == d.estimator else f"estimate:{c}.{e.method}"
            out += [f"{tag}.value", f"{tag}.ci", f"{tag}.n"]
    for r in state.get("refutations") or []:
        out += [f"placebo:{c}.{r.refuter}.p_value", f"placebo:{c}.{r.refuter}.new_effect", f"placebo:{c}.{r.refuter}.passed"]
    return list(dict.fromkeys(out))


def _material(state: SpecialistState) -> str:
    d: Design = state["design"]
    names = state.get("columns") or {}
    c = d.contrast.key
    p = d.periods
    lines = [f"[design.assumption] the design bets on: without the change, the treated units would have moved like the control units; router's reading: {state['handoff'].chosen_assumption}"]
    lines += [b.render() for b in state["handoff"].beliefs.values() if b.known() or b.status == "unknown"]
    lines += [f"[design.periods] " + (f"before {names.get(p.before_column, p.before_column)}, after {names.get(p.after_column, p.after_column)}" if p.kind == "wide"
                                     else f"time {names.get(p.time_column, p.time_column)}, first post {p.first_post}; {d.shape.periods_pre} pre periods, {d.shape.periods_post} post"),
             f"[design.controls] {', '.join(names.get(k, k) for k in d.controls.included) or 'none'}",
             f"comparison: {names.get(d.groups.column, d.groups.column)} = {d.contrast.treated!r} versus {d.contrast.control!r}; outcome: {state['handoff'].outcome}; "
             f"{d.shape.units_treated} treated units, {d.shape.units_control} control"]
    for r in d.checks.results:
        lines.append(f"[{r.address}] {r.level}: {r.detail}")
    for e in state.get("estimates") or []:
        if e.error is None:
            tag = f"estimate:{c}" if e.method == d.estimator else f"estimate:{c}.{e.method}"
            what = "primary" if e.method == d.estimator else ("with controls added cumulatively" if e.method.startswith(d.estimator + "+") else "secondary, mean of post-period coefficients")
            lines.append(f"[{tag}.value] {e.value:.4g} ({what}: {e.method}, target {e.target_units})")
            lines.append(f"[{tag}.ci] 95% interval {e.ci_low:.4g} to {e.ci_high:.4g}" if e.ci_low is not None else f"[{tag}.ci] no interval")
            lines.append(f"[{tag}.n] {e.n_treated} treated units, {e.n_control} control")
    for r in state.get("refutations") or []:
        lines.append(f"[placebo:{c}.{r.refuter}.passed] {r.passed}  [placebo:{c}.{r.refuter}.new_effect] {r.new_effect}  [placebo:{c}.{r.refuter}.p_value] {r.p_value}  ({r.detail})")
    return "\n".join(lines)


def interpret(state: SpecialistState) -> dict:
    d: Design = state["design"]
    primary = next((e for e in state.get("estimates") or [] if e.method == d.estimator and e.error is None), None)
    allowed = set(_addresses(state))
    tol = float(load_checks().get("effect_tolerance", 0.01))
    errors: list[str] = []
    debug = []
    parsed = None
    for _ in range(MAX_MODEL_RETRIES):
        user = P.INTERPRET_USER.format(question=_question(state), contrast=d.contrast.key, material=_material(state), addresses="\n".join(sorted(allowed)), errors=_rejected(errors))
        parsed, th = structured(Interpretation, P.INTERPRET_SYSTEM, user, node="interpret")
        debug.append(th)
        parsed.contrast = d.contrast.key
        errors = [f"{c} is not an address you may cite" for c in parsed.cites if c not in allowed]
        if not parsed.cites:
            errors.append("no citations given")
        if primary is not None and abs(parsed.effect_stated - primary.value) > max(abs(primary.value), 1e-9) * tol:
            errors.append(f"effect_stated {parsed.effect_stated} does not match the primary estimate {primary.value:.4g}")
        if not errors:
            break
    out: dict[str, Any] = {"interpretations": [parsed], "debug": debug}
    if errors:
        out["interpret_errors"] = {d.contrast.key: errors}
    return out


# ------------------------------------------------------------------ feasibility + assemble (facts)


def feasibility(state: SpecialistState) -> dict:
    _writer()({"feasibility": state["feasibility"].model_dump()})
    return {}


def assemble(state: SpecialistState) -> dict:
    h = state["handoff"]
    names = state.get("columns") or {}
    d: Design | None = state.get("design")
    f: Feasibility | None = state.get("feasibility")
    lines = [f"QUESTION     {_question(state)}", f"LANE         {h.family} → {h.specialist}", f"OUTCOME      {h.outcome}    TREATMENT   {h.treatment}", ""]
    if d:
        lines += d.render().splitlines() + ["", "RESULTS"]
        for e in state.get("estimates") or []:
            if e.error:
                lines.append(f"    {e.method:34} FAILED: {e.error}")
            else:
                ci = f" [{e.ci_low:.3g}, {e.ci_high:.3g}]" if e.ci_low is not None else ""
                lines.append(f"    {e.method:34} {e.value:+.4g}{ci}  units={e.n_treated}/{e.n_control}  {'primary' if e.method == d.estimator else 'secondary'}")
        dyn = state.get("dynamic") or {}
        if dyn:
            lines.append("    dynamic effects by period relative to the change:")
            for k in sorted(dyn, key=lambda x: int(x)):
                v, lo, hi = dyn[k]
                lines.append(f"      {int(k):+d}: {v:+.3g} [{lo:.3g}, {hi:.3g}]")
        for r in state.get("refutations") or []:
            lines.append(f"    placebo {r.refuter:26} {r.detail}")
        for i in state.get("interpretations") or []:
            lines.append(f"    ANSWER  {i.answer}")
            lines += [f"    CAVEAT  {cv}" for cv in i.caveats]
            lines.append(f"    CITES   {', '.join(i.cites)}")
        errs = (state.get("interpret_errors") or {}).get(d.contrast.key)
        if errs:
            lines.append(f"    INTERPRETATION GATE FAILED: {'; '.join(errs)}")
    else:
        g, p, s = state.get("groups"), state.get("periods"), state.get("shape")
        if g:
            lines.append(f"GROUPS       {names.get(g.column, g.column)} = {g.treated_level!r} treated")
        if p:
            lines.append(f"PERIODS      {p.kind}: " + (f"before {p.before_column}, after {p.after_column}" if p.kind == "wide" else f"time {p.time_column}, first post {p.first_post}"))
        if s:
            lines.append(f"SHAPE        {s.units_treated} treated units, {s.units_control} control; {s.periods_pre} pre, {s.periods_post} post")
        for r in state.get("checks") or []:
            lines.append(f"CHECK        {r.level:4} {r.address}  {r.detail}")
    if f:
        lines += ["", f"STOPPED AT   {f.stage}", f"REASON       {f.reason}"] + [f"FACT         {x}" for x in f.facts] + [f"WOULD FIX    {f.what_would_fix}"]
    debug = state.get("debug") or []
    if any(t.text for t in debug):
        lines += ["", "MODEL THOUGHTS (debug only)"] + [f"  [{t.node}] {t.text.strip()[:2000]}" for t in debug if t.text]
    report = "\n".join(lines)
    run_dir = state.get("run_dir")
    if run_dir:
        Path(run_dir, "report.md").write_text(report)
        Path(run_dir, "artifacts.json").write_text(json.dumps({
            "estimates": [e.model_dump() for e in state.get("estimates") or []],
            "refutations": [r.model_dump() for r in state.get("refutations") or []],
            "dynamic": state.get("dynamic") or {},
            "interpretations": [i.model_dump() for i in state.get("interpretations") or []],
            "feasibility": f.model_dump() if f else None,
        }, indent=2, default=str))
    result = {
        "status": "infeasible" if f else "done", "family": h.family, "specialist": h.specialist, "run_dir": run_dir, "report": report,
        "design": d.model_dump() if d else None, "shape": state["shape"].model_dump() if state.get("shape") else None,
        "controls": state["controls"].model_dump() if state.get("controls") else None,
        "checks": [c.model_dump() for c in state.get("checks") or []],
        "estimates": [e.model_dump() for e in state.get("estimates") or []],
        "refutations": [r.model_dump() for r in state.get("refutations") or []],
        "interpretations": [i.model_dump() for i in state.get("interpretations") or []],
        "feasibility": f.model_dump() if f else None,
    }
    _writer()({"report": report})
    return {"report": report, "specialist_result": result}
