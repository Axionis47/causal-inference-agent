"""Discontinuity lane nodes. Facts compute; judgements call the model once and are gated.

Stops are typed: a node that cannot go on routes to "feasibility" with a Feasibility record.
Nothing here names a column, a method, a cutoff, or a dataset.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from langgraph.config import get_stream_writer
from langgraph.types import Command, Send

from causal_agent.common.contracts import CheckResult, Checks, Contrast, Estimate, Feasibility, Refutation
from causal_agent.common.llm import structured
from causal_agent.intake.datasets import ROOT, dataset_entries, load_dataset_pack
from causal_agent.intake.pack import Pack, _key
from causal_agent.specialists.rd import adapter, checks as CK, prompts as P, shape as SH
from causal_agent.specialists.rd.contracts import (
    Bandwidths,
    CovariateRelation,
    Covariates,
    Design,
    DesignAssessment,
    EstimatorPick,
    Excluded,
    RDInterpretation,
    Score,
)
from causal_agent.specialists.rd.knowledge import (
    EstimatorEntry,
    estimator as estimator_entry,
    load_checks,
    load_estimators,
    load_placebos,
    pick_inference,
    placebo as placebo_entry,
    render_preferences,
)
from causal_agent.specialists.rd.state import PlaceboTask, RelateTask, SpecialistState

MAX_RELATE_ATTEMPTS = 3
MAX_MODEL_RETRIES = 3
MAX_PICK_ATTEMPTS = 2

# ------------------------------------------------------------------ helpers

_pack_cache: dict[str, Pack] = {}


def _pack(state: SpecialistState) -> Pack:
    name = state["handoff"].pack_name
    if name not in _pack_cache:
        _pack_cache[name] = load_dataset_pack(name)
    return _pack_cache[name]


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


def _card(pack: Pack, key: str) -> str:
    c = pack.column(key)
    return c.render() if c else f"[col:{key}] (no card)"


def _frame_text(state: SpecialistState) -> str:
    h = state["handoff"]
    s = h.scope
    lines = [f"family: {h.family}; outcome: {h.outcome}; treatment: {h.treatment or 'none named (the change may be the cutoff rule itself)'}; "
             f"filter={s.population_filter or 'none'}; window={s.window or 'none'}; target={s.target}",
             f"assumption the router bet on: {h.chosen_assumption}"]
    sc = state.get("score")
    if sc and sc.column:
        rule = f"{'at or ' if sc.cutoff_value_treated else ''}{sc.treated_side} {sc.cutoff:g}"
        lines.append(f"score: {sc.column}, treated when {rule}" + (f"; take-up {sc.takeup_column} = {sc.takeup_level!r}" if sc.takeup_column else "; treatment is the cutoff rule itself"))
    sh = state.get("shape")
    if sh:
        lines.append(f"shape: {sh.kind}; {sh.n_left} rows on the control side, {sh.n_right} on the treated side")
    return "\n".join(lines)


def _rejected(errors: list[str]) -> str:
    return ("\nPREVIOUS ANSWER WAS REJECTED\n" + "\n".join(f"- {e}" for e in errors) + "\n") if errors else ""


def _keys(state: SpecialistState) -> tuple[str | None, str, list[str]]:
    h = state["handoff"]
    t = _key(h.treatment) if h.treatment else None
    y = _key(h.outcome)
    rel = [_key(c.column) for c in h.relevant_columns]
    return t, y, [k for k in dict.fromkeys(rel) if k in state.get("columns", {})]


def _table(state: SpecialistState) -> pd.DataFrame:
    return pd.read_csv(state["table_path"])


def _canon(state_or_path) -> pd.DataFrame:
    path = state_or_path if isinstance(state_or_path, str) else state_or_path["canon_path"]
    return pd.read_csv(path)


def _cfg() -> dict:
    return load_checks()


def _is_fuzzy(entry: EstimatorEntry, primary_fuzzy: bool) -> bool:
    return primary_fuzzy if entry.fuzzy == "inherit" else bool(entry.fuzzy)


def _side_rule_identical(col: pd.Series, x_raw: pd.Series, cutoff: float, above: bool, inclusive: bool) -> bool:
    """True when a column is a deterministic function of the side of the cutoff (it is the assignment, not take-up)."""
    if col.nunique(dropna=True) != 2:
        return False
    side = (x_raw >= cutoff) if (above and inclusive) else (x_raw > cutoff) if above else (x_raw <= cutoff) if inclusive else (x_raw < cutoff)
    ok = np.isfinite(x_raw) & col.notna()
    a, b = col[ok].astype(str), side[ok]
    levels = sorted(a.unique())
    for lvl in levels:
        if ((a == lvl) == b).all():
            return True
    return False


# ------------------------------------------------------------------ load (fact)


def load(state: SpecialistState) -> Command:
    h = state["handoff"]
    entry = dataset_entries()[h.pack_name]
    raw = pd.read_csv(ROOT / entry["csv"])
    columns = {_key(c): c for c in raw.columns}
    raw.columns = [_key(c) for c in raw.columns]
    t, y = (_key(h.treatment) if h.treatment else None), _key(h.outcome)
    named = [k for k in dict.fromkeys([y] + ([t] if t else []) + [_key(c.column) for c in h.relevant_columns]) if k]
    missing = [k for k in named if k not in raw.columns]
    if missing:
        return _stop("load", "a column the hand-off names is not in the file", [f"missing: {missing}"], "a hand-off whose columns exist in the file")
    if not pd.api.types.is_numeric_dtype(raw[y]):
        return _stop("load", "the outcome is not numeric", [f"outcome {y} is {raw[y].dtype}"], "a numeric outcome")
    target = _cfg()["target_units"].get(h.scope.target)
    if target is None:
        return _stop("load", f"target '{h.scope.target}' is not supported by this lane", [], "a question asking for the average effect, or the effect on the treated")
    cluster = _key(entry["entity"][0]) if entry.get("entity") else None
    if cluster and cluster not in raw.columns:
        cluster = None
    run_dir = Path(os.getenv("RUN_DIR", ".artifacts/runs")) / f"{h.pack_name}-rd-{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    table_path = run_dir / "table.csv"
    raw.to_csv(table_path, index=False)
    _writer()({"load": {"rows": len(raw), "columns": list(raw.columns), "target_units": target, "cluster_column": cluster, "run_dir": str(run_dir)}})
    return Command(goto="score", update={
        "run_dir": str(run_dir), "table_path": str(table_path), "columns": columns, "cluster_column": cluster,
        "sampled_by_side": bool(entry.get("sampled_by_side", False)), "target_units": target,
        "relate_attempts": 0, "pick_attempts": 0, "relate_errors": {}, "excluded_estimators": [], "check_facts": {},
    })


# ------------------------------------------------------------------ score (judgement)


def score(state: SpecialistState) -> Command:
    pack = _pack(state)
    t, y, _ = _keys(state)
    table = _table(state)
    cards = "\n\n".join(c.render() for c in pack.columns)
    treatment_card = _card(pack, t) if t else "(the hand-off names no treatment column: the change may be the cutoff rule itself)"
    errors: list[str] = []
    debug = []
    for _ in range(MAX_MODEL_RETRIES):
        user = P.SCORE_USER.format(question=_question(state), frame=_frame_text(state), dataset_card=pack.dataset.render(), changes=pack.render_changes(),
                                   treatment_card=treatment_card, cards=cards, errors=_rejected(errors))
        parsed, th = structured(Score, P.SCORE_SYSTEM, user, node="score")
        debug.append(th)
        errors = []
        if parsed.column is None or str(parsed.column).strip().lower() in ("", "null", "none"):
            _writer()({"score": {"column": None, "reason": parsed.reason}})
            return _stop("score", "no cutoff rule on a numeric score is stated", [f"the model's reading: {parsed.reason}"],
                         "a note naming the score column and the cutoff value that decided who got the change", {"score": parsed, "debug": debug})
        parsed.column = _key(parsed.column)
        # the model sometimes writes the word null instead of a null; that is a null
        if str(parsed.takeup_column).strip().lower() in ("", "null", "none"):
            parsed.takeup_column, parsed.takeup_level = None, None
        parsed.takeup_column = _key(parsed.takeup_column) if parsed.takeup_column else None
        if parsed.takeup_column and parsed.column and parsed.takeup_column == _key(parsed.column):
            parsed.takeup_column, parsed.takeup_level = None, None  # the score cannot record its own take-up: the rule is the change
        for c in parsed.cites:
            if not pack.resolve(c):
                errors.append(f"{c} is not a pack address")
        if parsed.column not in table.columns:
            errors.append(f"{parsed.column!r} is not a column in the file")
        elif not pd.api.types.is_numeric_dtype(table[parsed.column]):
            errors.append(f"{parsed.column!r} is not numeric ({table[parsed.column].dtype}); a score must be a number")
        else:
            x = pd.to_numeric(table[parsed.column], errors="coerce")
            fin = x[np.isfinite(x)]
            if fin.nunique() <= 2:
                errors.append(f"{parsed.column!r} takes only {fin.nunique()} distinct values; a score must vary")
            if parsed.cutoff is None:
                errors.append("cutoff is missing")
            elif not (fin.min() < parsed.cutoff < fin.max()):
                errors.append(f"cutoff {parsed.cutoff!r} is not strictly inside the score's range {fin.min():g} to {fin.max():g}")
            else:
                above, below = int((fin > parsed.cutoff).sum()), int((fin < parsed.cutoff).sum())
                if above == 0 or below == 0:
                    errors.append(f"no rows on one side of the cutoff: {below} below, {above} above")
        if parsed.takeup_column:
            if parsed.takeup_column not in table.columns:
                errors.append(f"take-up column {parsed.takeup_column!r} is not in the file")
            elif parsed.takeup_level is None:
                errors.append("takeup_level is missing: give the exact value that means the unit received the change")
            else:
                observed = set(table[parsed.takeup_column].dropna().astype(str))
                if str(parsed.takeup_level) not in observed:
                    errors.append(f"level {parsed.takeup_level!r} is not observed in {parsed.takeup_column!r}; observed: {sorted(observed)[:20]}")
        if t and not errors and t != parsed.column:
            x = pd.to_numeric(table[parsed.column], errors="coerce")
            identical = _side_rule_identical(table[t], x, float(parsed.cutoff), parsed.treated_side == "above", parsed.cutoff_value_treated)
            if not identical and parsed.takeup_column != t:
                errors.append(f"the hand-off names {t!r} as the treatment and it is not a function of the cutoff; name it as the take-up column with its treated level, or name the score differently")
        if errors:
            continue
        # facts that end the judgement rather than retry it
        if parsed.takeup_column:
            x = pd.to_numeric(table[parsed.column], errors="coerce")
            tu = (table[parsed.takeup_column].astype(str) == str(parsed.takeup_level)).astype(float)
            tu[table[parsed.takeup_column].isna()] = np.nan
            treated_mask = (x > parsed.cutoff) if parsed.treated_side == "above" else (x < parsed.cutoff)
            other_mask = (x < parsed.cutoff) if parsed.treated_side == "above" else (x > parsed.cutoff)
            s_t, s_o = float(tu[treated_mask].mean()), float(tu[other_mask].mean())
            if not (s_t > s_o):
                return _stop("score", "the notes and the data disagree about which side got the change",
                             [f"take-up {s_t:.2f} on the side the notes call treated ({parsed.treated_side} {parsed.cutoff:g}), {s_o:.2f} on the other side"],
                             "a note whose account of the rule matches the take-up recorded in the data", {"score": parsed, "debug": debug})
            at = tu[x == parsed.cutoff]
            if len(at):
                share_at = float(at.mean())
                if parsed.cutoff_value_treated and share_at < 0.5:
                    errors.append(f"cutoff_value_treated is true but only {share_at:.2f} of the {len(at)} units exactly at the cutoff received the change")
                if not parsed.cutoff_value_treated and share_at > 0.5:
                    errors.append(f"cutoff_value_treated is false but {share_at:.2f} of the {len(at)} units exactly at the cutoff received the change")
                if errors:
                    continue
        _writer()({"score": parsed.model_dump()})
        return Command(goto="shape_table", update={"score": parsed, "debug": debug})
    return _stop("score", "could not name the score and cutoff", errors, "a note that names the score column, the cutoff value, and which side got the change", {"debug": debug})


# ------------------------------------------------------------------ shape (fact)


def _candidates(state: SpecialistState) -> list[str]:
    sc: Score = state["score"]
    _, y, rel = _keys(state)
    return [k for k in rel if k not in (sc.column, y, sc.takeup_column, state.get("cluster_column"))]


def shape_table(state: SpecialistState) -> Command:
    sc: Score = state["score"]
    _, y, _ = _keys(state)
    table = _table(state)
    candidates = _candidates(state)
    try:
        canon, x_all, facts = SH.canonical(table, sc, y, candidates, state.get("cluster_column"), _cfg())
    except SH.ShapeError as ex:
        return _stop("shape_table", ex.reason, ex.facts, ex.fix)
    run_dir = Path(state["run_dir"])
    canon_path, xall_path = run_dir / "canon.csv", run_dir / "scores.csv"
    canon.to_csv(canon_path, index=False)
    x_all.to_frame("x").to_csv(xall_path, index=False)
    treated_label = "above_cutoff" if sc.treated_side == "above" else "below_cutoff"
    control_label = "below_cutoff" if sc.treated_side == "above" else "above_cutoff"
    contrast = Contrast(control=control_label, treated=treated_label, reason=sc.reason, cites=sc.cites)
    _writer()({"shape": facts.model_dump()})
    update = {"canon_path": str(canon_path), "xall_path": str(xall_path), "shape": facts, "contrast": contrast, "candidates": candidates}
    sends = _relate_sends(state, candidates)
    return Command(goto=sends or "merge_covariates", update=update)


def _relate_sends(state: SpecialistState, candidates: list[str], errors: dict[str, list[str]] | None = None) -> list[Send]:
    pack = _pack(state)
    errs = errors or {}
    targets = [k for k in candidates if k in errs] if errs else candidates
    return [Send("relate", RelateTask(question=_question(state), frame=_frame_text(state), column=k, card=_card(pack, k), errors=_rejected(errs.get(k, []))))
            for k in targets]


# ------------------------------------------------------------------ relate (judgement, fan-out)


def relate(task: RelateTask) -> dict:
    user = P.RELATE_USER.format(**task)
    parsed, th = structured(CovariateRelation, P.RELATE_SYSTEM, user, node=f"relate:{task['column']}")
    parsed.column = task["column"]
    return {"relations": [parsed], "debug": [th]}


# ------------------------------------------------------------------ merge + verify (facts)


def merge_covariates(state: SpecialistState) -> dict:
    canon = _canon(state)
    numeric = {c for c in canon.columns if pd.api.types.is_numeric_dtype(canon[c])}
    latest: dict[str, CovariateRelation] = {r.column: r for r in state.get("relations") or []}
    tested: list[str] = []
    excluded: list[Excluded] = []
    pack = _pack(state)
    for k in state.get("candidates") or []:
        r = latest.get(k)
        if r is None:
            continue
        card = pack.column(k)
        if card is not None and card.profile.kind == "id":
            excluded.append(Excluded(column=k, why="an identifier names a unit; it is not a characteristic that could be continuous or jump at the cutoff"))
            continue
        if r.is_outcome_measure:
            excluded.append(Excluded(column=k, why="another measure of the outcome, or a later one; not a covariate"))
        elif r.affected_by_treatment:
            excluded.append(Excluded(column=k, why="could be changed by the treatment; adjusting for it would remove part of the effect"))
        elif not r.predetermined:
            excluded.append(Excluded(column=k, why="not judged fixed before the score was set; nothing says it should be continuous at the cutoff"))
        elif SH.covcol(k) not in numeric:
            excluded.append(Excluded(column=k, why="not numeric; needs encoding before it can enter a local fit"))
        else:
            tested.append(k)
    c = Covariates(balance_tested=tested, adjusted=list(tested), excluded=excluded)
    shape = state["shape"].model_copy(update={"rows_covariates": SH.rows_complete_on(canon, tested)})
    _writer()({"covariates": c.render()})
    return {"covariates": c, "shape": shape}


def verify(state: SpecialistState) -> Command:
    pack = _pack(state)
    latest: dict[str, CovariateRelation] = {r.column: r for r in state.get("relations") or []}
    errs: dict[str, list[str]] = {}
    for k in state.get("candidates") or []:
        r = latest.get(k)
        if r is None:
            errs.setdefault(k, []).append("no relation returned")
            continue
        if (r.predetermined or r.affected_by_treatment or r.is_outcome_measure) and not r.reasons:
            errs.setdefault(k, []).append("claims marked true but no reasons given")
        if r.predetermined and r.affected_by_treatment:
            errs.setdefault(k, []).append("cannot be both fixed before the change and changed by it")
        for reason in r.reasons:
            if not reason.cites:
                errs.setdefault(k, []).append("a reason has no citation")
            for c in reason.cites:
                if not pack.resolve(c):
                    errs.setdefault(k, []).append(f"{c} is not a pack address")
    attempts = state.get("relate_attempts", 0) + 1
    if errs:
        if attempts >= MAX_RELATE_ATTEMPTS:
            return _stop("verify", "the covariate relations could not be made to pass verification", [f"{k}: {'; '.join(v)}" for k, v in errs.items()],
                         "clearer column notes about when each column was fixed and what the treatment touches", {"relate_attempts": attempts})
        _writer()({"verify": {"attempt": attempts, "errors": errs}})
        return Command(goto=_relate_sends(state, state.get("candidates") or [], errs), update={"relate_errors": errs, "relate_attempts": attempts})
    _writer()({"verify": "ok"})
    return Command(goto="check_design", update={"relate_errors": {}, "relate_attempts": attempts})


# ------------------------------------------------------------------ checks (fact)


def check_design(state: SpecialistState) -> dict:
    canon = _canon(state)
    x_all = pd.read_csv(state["xall_path"])["x"]
    shape = state["shape"]
    inf = pick_inference(cluster_column=bool(shape.cluster_column))
    results, extra = CK.run_checks(canon, x_all, shape, state["covariates"], state["contrast"].key, _cfg(),
                                   cluster=bool(shape.cluster_column), vce=inf.vce, sampled_by_side=bool(state.get("sampled_by_side")))
    _writer()({"checks": [f"{r.level} {r.address} {r.detail}" for r in results]})
    facts = {k: v for k, v in extra.items() if k != "first_stage"}
    fs = extra.get("first_stage")
    if fs is not None:
        facts["first_stage"] = None if fs.error else dict(value=fs.value, ci_low=fs.ci_low, ci_high=fs.ci_high, se=fs.se, n_h_left=fs.n_h_left, n_h_right=fs.n_h_right, h=fs.h)
    return {"checks": results, "check_facts": facts}


def after_checks(state: SpecialistState) -> str:
    return "assess" if any(r.level != "pass" for r in state["checks"]) else "pick_estimator"


# ------------------------------------------------------------------ assess (judgement)


def _design_text(state: SpecialistState) -> str:
    return _frame_text(state) + "\n" + state["covariates"].render() + "\nshape: " + json.dumps(state["shape"].model_dump())


def assess(state: SpecialistState) -> Command:
    pack = _pack(state)
    sc: Score = state["score"]
    checks = Checks(results=state["checks"])
    flags, hard = checks.flags, checks.hard
    argue = set(_cfg().get("argue_from_notes") or [])
    flag_text = "\n".join(f"[{r.address}] {r.level.upper()}: {r.detail}" for r in flags)
    covs: Covariates = state["covariates"]
    cards = "\n\n".join([_card(pack, sc.column), pack.dataset.render()] + [_card(pack, k) for k in covs.balance_tested])
    check_addresses = {r.address for r in state["checks"]}
    errors: list[str] = []
    debug = []
    for _ in range(MAX_MODEL_RETRIES):
        user = P.ASSESS_USER.format(question=_question(state), design=_design_text(state), flags=flag_text, cards=cards, errors=_rejected(errors))
        parsed, th = structured(DesignAssessment, P.ASSESS_SYSTEM, user, node="assess")
        debug.append(th)
        errors = [f"{c} is not a check or pack address" for c in parsed.cites if not (c in check_addresses or pack.resolve(c))]
        if parsed.action == "proceed":
            if hard:
                errors.append("proceed is not allowed while a hard flag stands: " + ", ".join(r.address for r in hard))
            missing = [r.address for r in flags if r.address not in parsed.cites]
            if missing:
                errors.append("proceed does not address: " + ", ".join(missing))
            if any(r.name in argue for r in flags) and not any(pack.resolve(c) for c in parsed.cites):
                errors.append("no note cited for the argument about " + ", ".join(sorted({r.name for r in flags if r.name in argue})) + "; cite the card that says how the score was set or that the covariate was fixed before the change")
        if errors:
            continue
        _writer()({"assess": parsed.model_dump()})
        if parsed.action == "proceed":
            return Command(goto="pick_estimator", update={"assessment": parsed, "debug": debug})
        return _stop("assess", parsed.reason, [f"{r.address}: {r.detail}" for r in flags],
                     "units on both sides of the cutoff that are alike in everything the notes call fixed, and a score no unit could move", {"assessment": parsed, "debug": debug})
    return _stop("assess", "the design assessment could not be validated", errors, "see the gate errors", {"debug": debug})


# ------------------------------------------------------------------ pick estimator (judgement)


def _facts(state: SpecialistState) -> dict[str, Any]:
    s = state["shape"]
    cf = state.get("check_facts") or {}
    return {"kind": s.kind, "n_left": s.n_left, "n_right": s.n_right, "distinct_scores": s.distinct_scores, "first_stage": cf.get("first_stage_status"),
            "first_stage_F": cf.get("first_stage_F"), "adjusted_covariates": state["covariates"].adjusted, "cluster": s.cluster_column}


def _allowed(state: SpecialistState) -> list[EstimatorEntry]:
    s = state["shape"]
    excluded = set(state.get("excluded_estimators") or [])
    status = (state.get("check_facts") or {}).get("first_stage_status")
    return [e for e in load_estimators() if e.pickable and e.applies(kind=s.kind, first_stage=status, distinct_scores=s.distinct_scores) and e.name not in excluded]


def pick_estimator(state: SpecialistState) -> Command:
    facts = _facts(state)
    allowed = _allowed(state)
    if not allowed:
        return _stop("pick_estimator", "no estimator in the catalogue applies to this design", [f"facts: {json.dumps(facts, default=str)}", f"excluded after failures: {sorted(state.get('excluded_estimators') or [])}"],
                     "an estimator entry for this kind of design")
    names = [e.name for e in allowed]
    check_text = "\n".join(f"[{r.address}] {r.level}: {r.detail}" for r in state["checks"])
    check_addresses = {r.address for r in state["checks"]}
    errors: list[str] = []
    debug = []
    for _ in range(MAX_MODEL_RETRIES):
        user = P.PICK_USER.format(facts=json.dumps(facts, default=str), checks=check_text, estimators="\n\n".join(e.render() for e in allowed),
                                  preferences=render_preferences(allowed), names=", ".join(names), errors=_rejected(errors))
        parsed, th = structured(EstimatorPick, P.PICK_SYSTEM, user, node="pick_estimator")
        debug.append(th)
        errors = []
        if parsed.name not in names:
            errors.append(f"{parsed.name!r} is not one of {names}")
        for c in parsed.cites:
            if not (c in check_addresses or _pack(state).resolve(c)):
                errors.append(f"{c} is not a check or pack address")
        if not errors:
            _writer()({"estimator": parsed.model_dump()})
            return Command(goto="freeze_design", update={"estimator": parsed.name, "estimator_pick": parsed, "debug": debug,
                                                         "pick_attempts": state.get("pick_attempts", 0) + 1})
    return _stop("pick_estimator", "the estimator pick could not be validated", errors, "see the gate errors", {"debug": debug})


# ------------------------------------------------------------------ freeze (fact)


def freeze_design(state: SpecialistState) -> Command:
    s = state["shape"]
    entry = estimator_entry(state["estimator"])
    covs: Covariates = state["covariates"]
    inf = pick_inference(cluster_column=bool(s.cluster_column))
    fuzzy = _is_fuzzy(entry, False)
    canon = _canon(state)
    plan = CK.bandwidth_plan(entry.params, canon, s, _cfg(), fuzzy=fuzzy, cluster=bool(s.cluster_column), vce=inf.vce)
    if "error" in plan:
        return _stop("freeze_design", "bandwidth selection failed for the picked estimator", [plan["error"]], "an estimator whose bandwidth can be selected on this data")
    status = (state.get("check_facts") or {}).get("first_stage_status")
    also = [n for n in entry.also_run if estimator_entry(n).applies(kind=s.kind, first_stage=status, distinct_scores=s.distinct_scores)]
    for e in load_estimators():
        if not e.pickable and e.runs_when.get("adjusted_covariates") and covs.adjusted and e.name not in also:
            also.append(e.name)
    d = Design(contrast=state["contrast"], score=state["score"], shape=s, covariates=covs, checks=Checks(results=state["checks"]),
               estimator=entry.name, estimand=entry.estimand, spec=dict(entry.params), also_run=also, inference=inf.name, vce=inf.vce,
               cluster=s.cluster_column if inf.cluster else None,
               bandwidths=Bandwidths(h=plan["h"], b=plan["b"], h_cer=plan["h_cer"], rule=plan["rule"], n_h_left=plan["n_h_left"], n_h_right=plan["n_h_right"]),
               sharp_bandwidth_used=bool(fuzzy and (s.takeup_left == 0.0 or s.takeup_right == 1.0)),
               placebos=[p.name for p in load_placebos() if p.applies()], target_units=state["target_units"])
    run_dir = Path(state["run_dir"])
    (run_dir / "design.json").write_text(d.model_dump_json(indent=2))
    (run_dir / "design.md").write_text(d.render())
    b = adapter.bins(canon["y"], canon["x"])
    if b is not None:
        b.to_csv(run_dir / "bins.csv", index=False)
    _writer()({"design": d.render()})
    return Command(goto="estimate", update={"design": d})


# ------------------------------------------------------------------ estimate (fact) + placebos (fact, fan-out)


def _pinned(d: Design) -> dict:
    return {"h": d.bandwidths.h, "b": d.bandwidths.b} if d.bandwidths.rule == "support_points" else {}


def _fit_entry(d: Design, entry: EstimatorEntry, canon: pd.DataFrame, primary_fuzzy: bool) -> adapter.Fit:
    return adapter.fit(entry.params, canon, fuzzy=_is_fuzzy(entry, primary_fuzzy), covs=[SH.covcol(k) for k in d.covariates.adjusted] if entry.covs else None,
                       cluster=bool(d.cluster), vce=d.vce, **_pinned(d))


def estimate(state: SpecialistState) -> Command:
    d: Design = state["design"]
    canon = _canon(state)
    entry = estimator_entry(d.estimator)
    primary_fuzzy = _is_fuzzy(entry, False)
    key = d.contrast.key
    f = _fit_entry(d, entry, canon, primary_fuzzy)
    ests: list[Estimate] = [adapter.to_estimate(f, key, d.estimator, d.target_units)]
    update: dict[str, Any] = {"estimates": ests}
    if f.error:
        _writer()({"estimate": {"error": f.error}})
        if state.get("pick_attempts", 0) < MAX_PICK_ATTEMPTS:
            update["excluded_estimators"] = (state.get("excluded_estimators") or []) + [d.estimator]
            return Command(goto="pick_estimator", update=update)
        return _stop("estimate", "the estimator failed to fit and the re-pick failed too", [f.error], "a different estimator entry", update)
    for name in d.also_run:
        e2 = estimator_entry(name)
        ests.append(adapter.to_estimate(_fit_entry(d, e2, canon, primary_fuzzy), key, name, d.target_units, secondary=True))
    if f.first_stage:
        fs = f.first_stage
        ests.append(Estimate(contrast=key, method="first_stage", value=fs["value"], ci_low=fs["ci_low"], ci_high=fs["ci_high"],
                             n_treated=f.n_h_right, n_control=f.n_h_left, target_units="take_up_jump", secondary=True))
    primary = dict(value=f.value, ci_low=f.ci_low, ci_high=f.ci_high, p=f.p, h=f.h, b=f.b, n_h_left=f.n_h_left, n_h_right=f.n_h_right,
                   n_left=f.n_left, n_right=f.n_right, vce=f.vce, model=f.model, notes=f.notes)
    update["primary"] = primary
    _writer()({"estimate": [e.model_dump(exclude_none=True) for e in ests]})
    sends = [Send("placebo", PlaceboTask(name=n, design=d.model_dump(), canon_path=state["canon_path"], primary=primary)) for n in d.placebos]
    return Command(goto=sends or "interpret", update=update)


def _overlaps(a: adapter.Fit, prim: dict) -> bool:
    return a.ci_low <= prim["ci_high"] and prim["ci_low"] <= a.ci_high


def _same_sign(a: adapter.Fit, prim: dict) -> bool:
    return (a.value >= 0) == (prim["value"] >= 0)


def _refit_rule(points: list[tuple[str, adapter.Fit | None, bool, str]], prim: dict, entry) -> tuple[bool | None, str]:
    """points: (label, fit, informative, note). Returns (passed, detail) under the entry's pass rule."""
    primary_excludes_zero = not (prim["ci_low"] <= 0 <= prim["ci_high"])
    verdicts: list[bool] = []
    parts: list[str] = []
    for label, f, informative, note in points:
        if f is None or f.error:
            parts.append(f"{label}: could not run ({note or (f.error if f else '')})")
            continue
        if not informative:
            parts.append(f"{label}: {f.value:+.3g} [{f.ci_low:.3g}, {f.ci_high:.3g}], {f.n_h_left}/{f.n_h_right} rows; uninformative ({note})")
            continue
        ok = _overlaps(f, prim) if entry.pass_when.get("interval_overlaps_primary") else True
        if entry.pass_when.get("sign_stable_when_primary_excludes_zero") and primary_excludes_zero:
            ok = ok and _same_sign(f, prim)
        if entry.pass_when.get("interval_covers_zero"):
            ok = bool(f.covers_zero())
        verdicts.append(ok)
        flip = "" if _same_sign(f, prim) else "; sign differs from the primary"
        parts.append(f"{label}: {f.value:+.3g} [{f.ci_low:.3g}, {f.ci_high:.3g}], {f.n_h_left}/{f.n_h_right} rows{flip}" + ("" if ok else " (FAIL)"))
    passed = all(verdicts) if verdicts else None
    return passed, "; ".join(parts)


def placebo(task: PlaceboTask) -> dict:
    d = Design.model_validate(task["design"])
    canon = _canon(task["canon_path"])
    prim = task["primary"]
    entry = placebo_entry(task["name"])
    pentry = estimator_entry(d.estimator)
    primary_fuzzy = _is_fuzzy(pentry, False)
    floor = int(_cfg()["effective_rows"]["min"]["soft"])
    key = d.contrast.key
    points: list[tuple[str, adapter.Fit | None, bool, str]] = []

    def informative(f: adapter.Fit) -> tuple[bool, str]:
        if f.error:
            return False, f.error
        if min(f.n_h_left, f.n_h_right) < floor:
            return False, f"fewer than {floor} effective rows on a side"
        return True, ""

    if entry.name == "placebo_cutoffs":
        for label, mask in (("control side", canon["x"] < 0), ("treated side", canon["x"] >= 0)):
            sub = canon[mask]
            c_med = float(sub["x"].median())
            if not (sub["x"].min() < c_med < sub["x"].max()):
                points.append((f"{label} at {c_med:.4g}", None, False, "the placebo cutoff is not strictly inside that side's scores"))
                continue
            f = adapter.fit(CK.SHARP, sub, cluster=bool(d.cluster), vce=d.vce, c=c_med, **_pinned(d))
            ok, note = informative(f)
            points.append((f"{label} at {c_med:.4g}", f, ok, note))
    elif entry.name == "bandwidth_grid":
        bws = d.bandwidths
        grid = {"h_mse": bws.h, "2h_mse": 2 * bws.h}
        if bws.h_cer is not None:
            grid.update({"h_cer": bws.h_cer, "2h_cer": 2 * bws.h_cer})
        for name in entry.grid:
            if name not in grid:
                points.append((name, None, False, "no coverage-error bandwidth under the support-points rule"))
                continue
            h = grid[name]
            f = adapter.fit(d.spec, canon, fuzzy=primary_fuzzy, cluster=bool(d.cluster), vce=d.vce, h=h, b=bws.b if entry.hold_b else None)
            ok, note = informative(f)
            points.append((f"{name} = {h:.4g}", f, ok, note))
    elif entry.name == "donut":
        for share in entry.radii_share_of_h:
            r = share * d.bandwidths.h
            mask = canon["x"].abs() >= r
            dropped = int((~mask).sum())
            if dropped == 0:
                points.append((f"radius {share:.0%} of h", None, False, "no rows lie within the radius"))
                continue
            f = adapter.fit(d.spec, canon, fuzzy=primary_fuzzy, cluster=bool(d.cluster), vce=d.vce, mask=mask, **_pinned(d))
            ok, note = informative(f)
            points.append((f"radius {share:.0%} of h ({dropped} rows dropped)", f, ok, note))
    passed, detail = _refit_rule(points, prim, entry)
    values = [f.value for _, f, ok, _ in points if f is not None and not f.error and ok]
    r = Refutation(contrast=key, refuter=entry.name, kind="falsification", new_effect=float(np.mean(values)) if values else None, passed=passed,
                   detail=detail + ("" if passed is None else " (pass)" if passed else " (FAIL)"))
    _writer()({"placebo": {r.refuter: r.detail}})
    return {"refutations": [r]}


# ------------------------------------------------------------------ interpret (judgement)


def _addresses(state: SpecialistState) -> list[str]:
    d: Design = state["design"]
    c = d.contrast.key
    out = ["design.assumption", "design.score", "design.bandwidth"] + [r.address for r in d.checks.results]
    for e in state.get("estimates") or []:
        if e.error is None:
            tag = f"estimate:{c}" if e.method == d.estimator else f"estimate:{c}.{e.method}"
            out += [f"{tag}.value", f"{tag}.ci", f"{tag}.n"] + ([f"{tag}.p", f"{tag}.bandwidth"] if e.method == d.estimator else [])
    for r in state.get("refutations") or []:
        out += [f"placebo:{c}.{r.refuter}.passed", f"placebo:{c}.{r.refuter}.detail"]
    return list(dict.fromkeys(out))


def _required(state: SpecialistState) -> list[str]:
    d: Design = state["design"]
    c = d.contrast.key
    req = ["design.bandwidth", f"estimate:{c}.ci", f"estimate:{c}.n"]
    req += [r.address for r in d.checks.results if r.level != "pass"]
    req += [f"placebo:{c}.{r.refuter}.passed" for r in state.get("refutations") or [] if r.passed is False]
    return list(dict.fromkeys(req))


def _material(state: SpecialistState) -> str:
    d: Design = state["design"]
    names = state.get("columns") or {}
    sc = d.score
    c = d.contrast.key
    prim = state.get("primary") or {}
    rule = f"{'at or ' if sc.cutoff_value_treated else ''}{sc.treated_side} {sc.cutoff:g}"
    lines = [
        f"[design.assumption] the design bets on: units just either side of the cutoff are alike in everything except the change; the score's density and every predetermined characteristic are continuous at the cutoff; router's reading: {state['handoff'].chosen_assumption}",
        f"[design.score] score {names.get(sc.column, sc.column)}, treated when {rule}" + (f"; take-up recorded in {names.get(sc.takeup_column, sc.takeup_column)}" if sc.takeup_column else "; treatment is the cutoff rule itself")
        + f"; {d.shape.kind} design; scores run from {d.shape.score_min:.4g} to {d.shape.score_max:.4g}",
        f"[design.bandwidth] estimation bandwidth h = {d.bandwidths.h:.4g} in the score's units (bias bandwidth b = {d.bandwidths.b:.4g}); the effect is estimated from rows within h of the cutoff",
        f"comparison: {d.contrast.treated} versus {d.contrast.control}; outcome: {state['handoff'].outcome}",
    ]
    for r in d.checks.results:
        lines.append(f"[{r.address}] {r.level}: {r.detail}")
    for e in state.get("estimates") or []:
        if e.error is not None:
            continue
        if e.method == d.estimator:
            tag = f"estimate:{c}"
            lines.append(f"[{tag}.value] {e.value:.4g} (primary: {e.method}, {d.estimand})")
            lines.append(f"[{tag}.ci] 95% robust interval {e.ci_low:.4g} to {e.ci_high:.4g}")
            lines.append(f"[{tag}.p] robust p = {prim.get('p', float('nan')):.3g}")
            lines.append(f"[{tag}.n] {e.n_control} control-side and {e.n_treated} treated-side rows inside the bandwidth")
            lines.append(f"[{tag}.bandwidth] h = {prim.get('h', d.bandwidths.h):.4g}")
        else:
            tag = f"estimate:{c}.{e.method}"
            what = {"first_stage": "the jump in take-up at the cutoff", "local_linear_adjusted": "with the adjusted covariates partialled out",
                    "local_quadratic": "quadratic on each side", "local_linear_itt": "effect of crossing the cutoff, whatever was taken up"}.get(e.method, "secondary")
            lines.append(f"[{tag}.value] {e.value:.4g} ({what}: {e.method})")
            lines.append(f"[{tag}.ci] 95% robust interval {e.ci_low:.4g} to {e.ci_high:.4g}" if e.ci_low is not None else f"[{tag}.ci] no interval")
            lines.append(f"[{tag}.n] {e.n_control} control-side and {e.n_treated} treated-side rows")
    for r in state.get("refutations") or []:
        lines.append(f"[placebo:{c}.{r.refuter}.passed] {r.passed}  [placebo:{c}.{r.refuter}.detail] {r.detail}")
    return "\n".join(lines)


def interpret(state: SpecialistState) -> dict:
    d: Design = state["design"]
    prim = state.get("primary") or {}
    allowed = set(_addresses(state))
    required = _required(state)
    tol = float(_cfg().get("effect_tolerance", 0.01))
    scale = max(abs(prim.get("value", 0.0)), abs(prim.get("ci_low", 0.0)), abs(prim.get("ci_high", 0.0)), 1e-9)
    errors: list[str] = []
    debug = []
    parsed = None
    for _ in range(MAX_MODEL_RETRIES):
        user = P.INTERPRET_USER.format(question=_question(state), contrast=d.contrast.key, material=_material(state), addresses="\n".join(sorted(allowed)),
                                       required="\n".join(required), errors=_rejected(errors))
        parsed, th = structured(RDInterpretation, P.INTERPRET_SYSTEM, user, node="interpret")
        debug.append(th)
        parsed.contrast = d.contrast.key
        errors = [f"{c} is not an address you may cite" for c in parsed.cites if c not in allowed]
        missing = [c for c in required if c not in parsed.cites]
        if missing:
            errors.append("required addresses not cited: " + ", ".join(missing))
        if parsed.estimand != d.estimand:
            errors.append(f"estimand {parsed.estimand!r} does not match the design's {d.estimand!r}")
        if prim:
            if abs(parsed.effect_stated - prim["value"]) > scale * tol:
                errors.append(f"effect_stated {parsed.effect_stated} does not match the primary estimate {prim['value']:.4g}")
            if abs(parsed.ci_low_stated - prim["ci_low"]) > scale * tol or abs(parsed.ci_high_stated - prim["ci_high"]) > scale * tol:
                errors.append(f"the stated interval does not match {prim['ci_low']:.4g} to {prim['ci_high']:.4g}")
            if abs(parsed.bandwidth_stated - prim["h"]) > max(abs(prim["h"]), 1e-9) * tol:
                errors.append(f"bandwidth_stated {parsed.bandwidth_stated} does not match h = {prim['h']:.4g}")
            if parsed.n_left_stated != prim["n_h_left"] or parsed.n_right_stated != prim["n_h_right"]:
                errors.append(f"effective rows must be {prim['n_h_left']} control-side and {prim['n_h_right']} treated-side")
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
    lines = [f"QUESTION     {_question(state)}", f"LANE         {h.family} → {h.specialist}", f"OUTCOME      {h.outcome}    TREATMENT   {h.treatment or '(the cutoff rule)'}", ""]
    if d:
        lines += d.render().splitlines() + ["", "RESULTS"]
        for e in state.get("estimates") or []:
            if e.error:
                lines.append(f"    {e.method:28} FAILED: {e.error}")
            else:
                ci = f" [{e.ci_low:.3g}, {e.ci_high:.3g}]" if e.ci_low is not None else ""
                lines.append(f"    {e.method:28} {e.value:+.4g}{ci}  rows={e.n_control}/{e.n_treated}  {'primary' if e.method == d.estimator else 'secondary'}")
        for r in state.get("refutations") or []:
            lines.append(f"    placebo {r.refuter:18} {r.detail}")
        for i in state.get("interpretations") or []:
            lines.append(f"    ANSWER  {i.answer}")
            lines += [f"    CAVEAT  {cv}" for cv in i.caveats]
            lines.append(f"    CITES   {', '.join(i.cites)}")
        errs = (state.get("interpret_errors") or {}).get(d.contrast.key)
        if errs:
            lines.append(f"    INTERPRETATION GATE FAILED: {'; '.join(errs)}")
    else:
        sc, s = state.get("score"), state.get("shape")
        if sc and sc.column:
            rule = f"{'at or ' if sc.cutoff_value_treated else ''}{sc.treated_side} {sc.cutoff:g}"
            lines.append(f"SCORE        {names.get(sc.column, sc.column)} treated when {rule}" + (f"; take-up {names.get(sc.takeup_column, sc.takeup_column)} = {sc.takeup_level!r}" if sc.takeup_column else ""))
        if s:
            lines.append(f"SHAPE        {s.kind}; {s.n_left} rows on the control side, {s.n_right} on the treated side; {s.distinct_scores} distinct scores")
        for r in state.get("checks") or []:
            lines.append(f"CHECK        {r.level:4} {r.address}  {r.detail}")
        a = state.get("assessment")
        if a:
            lines.append(f"ASSESS       {a.action}: {a.reason}  cites {', '.join(a.cites)}")
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
            "primary": state.get("primary"),
            "check_facts": state.get("check_facts"),
            "interpretations": [i.model_dump() for i in state.get("interpretations") or []],
            "feasibility": f.model_dump() if f else None,
        }, indent=2, default=str))
    a = state.get("assessment")
    result = {
        "status": "infeasible" if f else "done", "family": h.family, "specialist": h.specialist, "run_dir": run_dir, "report": report,
        "design": d.model_dump() if d else None, "score": state["score"].model_dump() if state.get("score") else None,
        "shape": state["shape"].model_dump() if state.get("shape") else None,
        "covariates": state["covariates"].model_dump() if state.get("covariates") else None,
        "checks": [c.model_dump() for c in state.get("checks") or []],
        "assessment": a.model_dump() if a else None,
        "estimates": [e.model_dump() for e in state.get("estimates") or []],
        "refutations": [r.model_dump() for r in state.get("refutations") or []],
        "interpretations": [i.model_dump() for i in state.get("interpretations") or []],
        "feasibility": f.model_dump() if f else None,
    }
    _writer()({"report": report})
    return {"report": report, "specialist_result": result}
