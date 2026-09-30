"""Adjustment-lane nodes. Facts compute; judgements are bounded episodes, gated.

The pack is weighed by code first (the harness's `case`): a settled column field is a fact the graph takes, an open one
is placed by a rung of the ladder, a belief is a flag the assessment must answer. The ladder climbs in the order an
analyst reads the problem: the pair, the mechanism, time, the pre-treatment roles all together, the post-treatment roles
all together, then the graph from them. Each rung reads the rungs below it and may look at the data through the read-only
tools; every claim cites the pack, a fact it asked for, or a rung below. Stops are typed: a node that cannot go on returns
Command(goto="feasibility") with a Feasibility record; a question for the person is the same with a LaneAsk.
Nothing here names a column, a method, or a dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from langgraph.types import Command, Send

from causal_agent.common.addresses import key as _key
from causal_agent.common.contracts import (
    CheckResult,
    Checks,
    Cited,
    Contrast,
    Decline,
    Estimate,
    Feasibility,
    Handoff,
    Interpretation,
    LaneAsk,
    Refutation,
)
from causal_agent.common.llm import structured
from causal_agent.families.adjustment.design import AdjustmentDesign
from causal_agent.families.adjustment.lane import adapter
from causal_agent.families.adjustment.lane import checks as CK
from causal_agent.families.adjustment.lane import prompts as P
from causal_agent.families.adjustment.lane.contracts import (
    Contrasts,
    Design,
    DesignAssessment,
    Edge,
    Estimand,
    EstimatorPick,
    Excluded,
    Graph,
    Ladder,
    Mechanism,
    Pair,
    PostRoles,
    Relation,
    Revision,
    Role,
    Roles,
    Timing,
)
from causal_agent.families.adjustment.lane.knowledge import (
    estimator as estimator_entry,
)
from causal_agent.families.adjustment.lane.knowledge import (
    load_beliefs,
    load_checks,
    load_estimators,
    load_refuters,
    render_preferences,
)
from causal_agent.families.adjustment.lane.state import ContrastTask, InterpretTask, SpecialistState
from causal_agent.lane import asks, intake, records
from causal_agent.lane import case as C
from causal_agent.lane import figures as LF
from causal_agent.lane import nodes as L
from causal_agent.lane import verify as V
from causal_agent.lane import words as W
from causal_agent.lane.episode import EpisodeLog, run_episode
from causal_agent.lane.nodes import MAX_MODEL_RETRIES, MAX_PICK_ATTEMPTS, MAX_REVISIONS
from causal_agent.viz.postviz import common as PV

MAX_DOSE_LEVELS = 12
CLAIMS = ("affects_treatment", "affects_outcome", "affected_by_treatment", "is_outcome_measure")
# a model claim that contradicts a pack fact on a claim the pack did not itself settle
CONTRADICTION_RULES: list[V.Rule] = [("affects_treatment", True, "when", ("after",))]


_writer, _question, _stop, _card, _case, _cites, _rejected, _keys, _table = (
    L.writer,
    L.question,
    L.stop,
    L.card,
    L.case_of,
    L.cites,
    L.rejected,
    L.keys,
    L.table,
)
after_checks, feasibility = L.after_checks, L.feasibility
case = L.make_case(load_beliefs)


# ------------------------------------------------------------------ helpers


def _block(h: Handoff) -> AdjustmentDesign | None:
    return h.design if isinstance(h.design, AdjustmentDesign) else None


def _thought(t, node: str):
    t.node = node
    return t


# ------------------------------------------------------------------ load (fact) and the case (fact)


def load(state: SpecialistState) -> Command:
    h = state["handoff"]
    if h.intent != "effect_of_change":
        return _stop(
            "load",
            "this lane answers the effect of a change on an outcome; the question was read differently",
            [f"intent: {h.intent}"],
            "a question about the effect of a change on an outcome",
        )
    if not h.treatment:
        return _stop("load", "the hand-off names no treatment", [], "a hand-off whose columns exist in the file")
    try:
        it = intake.load(h, "dowhy")
    except intake.IntakeStop as e:
        return Command(goto="feasibility", update={"feasibility": e.feasibility})
    t, y = _key(h.treatment), _key(h.outcome)
    table = it.table
    ok = CK.outcome_kind(table[y])
    if ok is None:
        return _stop(
            "load",
            "the outcome is neither numeric nor two-valued",
            [f"outcome {y} has {table[y].nunique()} distinct non-numeric values"],
            "an outcome measured as a number or a yes/no",
            {"declines": it.declines},
        )
    cfg = load_checks()
    target = cfg["target_units"].get(h.scope.target)
    if target is None:
        return _stop(
            "load",
            f"target '{h.scope.target}' is not supported by this lane yet",
            [],
            "a question asking for the average effect, or the effect on the treated",
            {"declines": it.declines},
        )
    tcol = table[t]
    numeric_dose = pd.api.types.is_numeric_dtype(tcol) and tcol.nunique() > MAX_DOSE_LEVELS
    if numeric_dose:
        return _stop(
            "load",
            "the treatment is a dose with many values; this lane compares levels",
            [f"{t} has {tcol.nunique()} distinct values"],
            "a two-level or few-level treatment, or a dose-response lane",
            {"declines": it.declines},
        )
    levels = [str(v) for v in sorted(tcol.unique(), key=lambda v: str(v))]
    _writer()(
        {
            "load": {
                "rows": len(table),
                "columns": list(it.columns),
                "outcome_kind": ok,
                "target_units": target,
                "run_dir": str(it.run_dir),
                "declines": [d.render() for d in it.declines],
                **it.facts,
            }
        }
    )
    return Command(
        goto="case",
        update={
            "run_dir": str(it.run_dir),
            "table_path": str(it.table_path),
            "columns": it.columns,
            "declines": it.declines,
            "check_facts": {"intake": it.facts} if it.facts else {},
            "outcome_kind": ok,
            "target_units": target,
            "treatment_levels": levels,
            "ladder": Ladder(),
            "revisions": 0,
            "pick_attempts": 0,
            "interpret_attempts": 0,
            "excluded_estimators": [],
            "applied_revisions": [],
        },
    )


# ------------------------------------------------------------------ the ladder: rungs 0 to 4, then the graph from them


def _ladder(state: SpecialistState) -> Ladder:
    lad = state.get("ladder")
    return lad if isinstance(lad, Ladder) else Ladder()


def _budget(node: str) -> int:
    return int((load_checks().get("episode_budget") or {}).get(node, 4))


def _treated_mask(state: SpecialistState) -> pd.Series | None:
    """The treated rows once the pair is set: the treatment at the first contrast's treated level."""
    t, _, _ = _keys(state)
    cs = state.get("contrasts") or []
    if not t or not cs:
        return None
    return _table(state)[t].astype(str) == str(cs[0].treated)


def _resolver(h: Handoff, log: EpisodeLog, ladder: Ladder):
    """What a rung may cite: the pack, the facts this episode asked for, and the rungs below."""

    def ok(address: str) -> bool:
        return h.resolve(address) or log.resolve(address) or ladder.resolve(address)

    return ok


def _column_key(state: SpecialistState, name: Any) -> str | None:
    """The key of a column in play, from its key or its raw name; None when it is not in play or the word is none."""
    if not name or str(name).strip().lower() == "none":
        return None
    cols = state.get("columns") or {}
    k = _key(str(name))
    if k in cols:
        return k
    return next((kk for kk, raw in cols.items() if raw == name), None)


def _as_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [x.strip() for x in v.split(",") if x.strip()]
    return [str(x) for x in v]


# rung 0: the pair (code from the pack; a judgement only when the pack does not name the treated level)


def pair(state: SpecialistState) -> Command:
    h = state["handoff"]
    t, y, _ = _keys(state)
    assert t is not None
    levels = state["treatment_levels"]
    lad = _ladder(state)

    def done(contrasts: list[Contrast], by: Any, debug: list, log: EpisodeLog | None) -> Command:
        p = Pair(outcome=y, treatment=t, contrasts=contrasts, target_asked=h.scope.target, by=by)
        _writer()({"contrasts": [c.model_dump() for c in contrasts]})
        update: dict[str, Any] = {"contrasts": contrasts, "ladder": lad.model_copy(update={"pair": p}), "debug": debug}
        if log is not None:
            update["episodes"] = {"pair": log}
        return Command(goto="mechanism", update=update)

    if h.treated_level is not None and str(h.treated_level) in levels:  # the pack says which level means treated: a fact
        treated = str(h.treated_level)
        control = str(h.control_level) if h.control_level is not None and str(h.control_level) in levels and str(h.control_level) != treated else None
        if control is None and len(levels) == 2:
            control = next(v for v in levels if v != treated)
        if control is not None:
            c = Contrast(
                control=control,
                treated=treated,
                reason="the pack names the level that means the unit got the change",
                cites=_cites(h, "claim:assignment.treated_level", "claim:assignment.treatment_column"),
            )
            return done([c], "pack", [], None)
    s = h.scope
    scope = f"filter={s.population_filter or 'none'}; window={s.window or 'none'}; contrast={s.contrast}; target={s.target}"
    user = P.CONTRAST_USER.format(question=_question(state), scope=scope, treatment_card=_card(h, t), levels=", ".join(repr(v) for v in levels))
    rec, log, thoughts, errors = run_episode(
        Contrasts,
        P.CONTRAST_SYSTEM,
        user,
        tools=L.data_tools(state, None),
        budget=_budget("pair"),
        gate=lambda r, _log: _validate_contrasts(r.items, levels),
        node="pair",
    )
    if rec is None:
        return _stop(
            "pair",
            "could not define a comparison from the treatment's levels",
            errors,
            "a treatment whose levels the note explains",
            {"debug": thoughts, "episodes": {"pair": log}},
        )
    return done(rec.items, "judgement", thoughts, log)


def _validate_contrasts(items: list[Contrast], levels: list[str]) -> list[str]:
    errs = []
    if not items:
        errs.append("no contrast given")
    seen = set()
    for c in items:
        for v in (c.control, c.treated):
            if str(v) not in levels:
                errs.append(f"level {v!r} is not an observed level; observed: {levels}")
        if c.control == c.treated:
            errs.append(f"control and treated are the same level {c.control!r}")
        if c.key in seen:
            errs.append(f"duplicate contrast {c.key}")
        seen.add(c.key)
    return errs


# rung 1: the mechanism (code from the pack; a judgement fills what the pack leaves open)


def mechanism(state: SpecialistState) -> Command:
    h = state["handoff"]
    lad = _ladder(state)
    blk = _block(h)
    a = h.assignment or {}
    kind = a.get("kind")
    drivers = [k for d in _as_list(a.get("depends_on")) if (k := _column_key(state, d))]
    offer, uptake = _column_key(state, a.get("offer_column")), _column_key(state, a.get("uptake_column"))
    self_sel = blk.voluntary_uptake if blk is not None else None
    cites = _cites(h, "claim:assignment.kind", "claim:assignment.rule", "claim:assignment.depends_on")
    if kind == "lottery" or drivers:  # the pack states what the decision looked at, or that it looked at nothing
        m = Mechanism(
            kind=kind,
            drivers=drivers,
            offer_column=offer,
            uptake_column=uptake,
            self_selection=self_sel if self_sel is not None else kind == "own_choice",
            reason="as the pack states it",
            cites=cites,
            by="pack",
        )
        _writer()({"mechanism": m.model_dump()})
        return Command(goto="time", update={"ladder": lad.model_copy(update={"mechanism": m})})
    t, _, others = _keys(state)

    def gate(r: Mechanism, log: EpisodeLog) -> list[str]:
        ok = _resolver(h, log, lad)
        errs = [f"driver {d!r} is not a column in play" for d in r.drivers if _column_key(state, d) is None]
        for f in ("offer_column", "uptake_column"):
            v = getattr(r, f)
            if v is not None and _column_key(state, v) is None:
                errs.append(f"{f} names {v!r}, which is not a column in play; use null when there is no such column")
        if r.offer_column and r.uptake_column and _column_key(state, r.offer_column) == _column_key(state, r.uptake_column):
            errs.append("the offer and the uptake are the same column; name two or leave both null")
        if r.self_selection and kind == "lottery":
            errs.append("a lottery leaves no room for units to move their own assignment")
        if not r.cites:
            errs.append("no citations given")
        return errs + V.cites_resolve(r.cites, h, ok)

    user = P.MECHANISM_USER.format(
        question=_question(state),
        frame=L.frame_text(state),
        treatment_card=_card(h, t) if t else "(none)",
        columns="\n".join(b.line() for k in others if (b := h.column(k)) is not None) or "(none)",
        kind=kind or "not said",
        errors="",
    )
    rec, log, thoughts, errors = run_episode(
        Mechanism, P.MECHANISM_SYSTEM, user, tools=L.data_tools(state, _treated_mask(state)), budget=_budget("mechanism"), gate=gate, node="mechanism"
    )
    if rec is None:
        return _stop(
            "mechanism",
            "how the change reached the units could not be read from the pack",
            errors,
            "a clearer account of what the decision or the offer looked at",
            {"debug": thoughts, "episodes": {"mechanism": log}},
        )
    m = rec.model_copy(
        update={
            "kind": kind,
            "drivers": [k for d in rec.drivers if (k := _column_key(state, d))],
            "offer_column": _column_key(state, rec.offer_column),
            "uptake_column": _column_key(state, rec.uptake_column),
            "by": "judgement",
        }
    )
    _writer()({"mechanism": m.model_dump()})
    return Command(goto="time", update={"ladder": lad.model_copy(update={"mechanism": m}), "debug": thoughts, "episodes": {"mechanism": log}})


# rung 2: time (code from the pack)


def timing(state: SpecialistState) -> dict:
    h = state["handoff"]
    case = _case(state)
    _, _, others = _keys(state)
    tm = Timing()
    for k in others:
        b = h.column(k)
        w = case.fact(f"col:{k}.when") or case.drafts.get(f"col:{k}.when") or (b.when if b is not None else "unknown")
        getattr(tm, w if w in ("before", "at", "after") else "unknown").append(k)
    _writer()({"timing": tm.model_dump()})
    return {"ladder": _ladder(state).model_copy(update={"timing": tm})}


# rungs 3 and 4: the roles, every column of a rung together (judgement only for what the pack leaves open)


def settled_claims(h: Handoff, k: str, case: C.Case) -> tuple[dict[str, bool], dict[str, str]]:
    """The claims about a column the pack settles, each with the address it rests on. The instrument and the mediator the
    person named settle all four; a column the offer depended on fed the treatment; a column fixed before the change was not
    moved by it; a column the person called a measure of the outcome is one; the person's word on `moved_by_change` stands."""
    b = h.column(k)
    if b is None:
        return {}, {}
    a = b.address
    blk = _block(h)
    if blk is not None and blk.instrument == k:
        claims = dict(affects_treatment=True, affects_outcome=False, affected_by_treatment=False, is_outcome_measure=False)
        return claims, {c: "claim:exclusion" for c in claims}
    if blk is not None and blk.mediator == k:
        claims = dict(affects_treatment=False, affects_outcome=True, affected_by_treatment=True, is_outcome_measure=False)
        return claims, {c: "claim:mediator" for c in claims}
    claims: dict[str, bool] = {}
    cites: dict[str, str] = {}
    if b.role == "depends_on":
        claims["affects_treatment"], cites["affects_treatment"] = True, "claim:assignment.depends_on"
    elif case.is_fact(f"{a}.feeds_treatment"):
        claims["affects_treatment"], cites["affects_treatment"] = bool(case.fact(f"{a}.feeds_treatment")), f"{a}.feeds_treatment"
    if case.is_fact(f"{a}.moves_outcome"):
        claims["affects_outcome"], cites["affects_outcome"] = bool(case.fact(f"{a}.moves_outcome")), f"{a}.moves_outcome"
    if case.is_fact(f"{a}.measures_outcome"):
        claims["is_outcome_measure"], cites["is_outcome_measure"] = bool(case.fact(f"{a}.measures_outcome")), f"{a}.measures_outcome"
    if case.is_fact(f"{a}.moved_by_change"):
        claims["affected_by_treatment"], cites["affected_by_treatment"] = bool(case.fact(f"{a}.moved_by_change")), f"{a}.moved"
    elif case.fact(f"{a}.when") == "before":
        claims["affected_by_treatment"], cites["affected_by_treatment"] = False, f"{a}.when"
    return claims, cites


def fact_relation(h: Handoff, k: str, case: C.Case | None = None) -> Relation | None:
    """A column's relation when the pack settles every claim, so no judgement is made: the instrument, the mediator, a measure
    of the outcome, and a column the rule or the offer looked at that was fixed before the change (a parent of both)."""
    case = case or C.Case()
    b = h.column(k)
    if b is None:
        return None
    claims, cites = settled_claims(h, k, case)
    if claims.get("is_outcome_measure") is True:
        return Relation(
            column=k,
            affects_treatment=False,
            affects_outcome=False,
            affected_by_treatment=False,
            is_outcome_measure=True,
            reasons=[Cited(reason=f"{b.name}: the person said it measures the outcome", cites=[cites["is_outcome_measure"]])],
        )
    if all(c in claims for c in CLAIMS):
        return Relation(column=k, reasons=[Cited(reason=f"{b.name}: {c} settled by the pack", cites=[cites[c]]) for c in CLAIMS if claims[c]], **claims)
    if b.role == "depends_on" and case.fact(f"{b.address}.when") == "before":
        return Relation(
            column=k,
            affects_treatment=True,
            affects_outcome=True,
            affected_by_treatment=False,
            is_outcome_measure=False,
            reasons=[
                Cited(reason=f"{b.name}: the rule or the offer looked at it", cites=["claim:assignment.depends_on"]),
                Cited(reason=f"{b.name}: fixed before the change, a background attribute", cites=[f"{b.address}.when"]),
            ],
        )
    return None


_DRAFT_FIELDS = {
    "feeds_treatment": "affects_treatment",
    "moves_outcome": "affects_outcome",
    "moved_by_change": "affected_by_treatment",
    "measures_outcome": "is_outcome_measure",
}


def drafted_claims(h: Handoff, k: str, case: C.Case) -> tuple[dict[str, bool], dict[str, str]]:
    """The claims about a column an earlier run read and nobody has confirmed: the last reading, each with its address."""
    b = h.column(k)
    if b is None:
        return {}, {}
    settled, _ = settled_claims(h, k, case)
    claims: dict[str, bool] = {}
    cites: dict[str, str] = {}
    for field, claim in _DRAFT_FIELDS.items():
        addr = f"{b.address}.{field}"
        if claim not in settled and addr in case.drafts and case.drafts[addr] is not None:
            claims[claim], cites[claim] = bool(case.drafts[addr]), addr if field != "moved_by_change" else f"{b.address}.moved"
    return claims, cites


_settled_text, apply_settled = L.make_settled(settled_claims)
_drafted_text = L.make_drafted(drafted_claims)
LINK_CITES = ("probe:data.redundancy.", ".same_as", ".nested_in")  # what may back a redundancy or a nesting, beside a redundancy tool result


def _column_block(h: Handoff, k: str, case: C.Case) -> str:
    return f"COLUMN {k!r}\n{_card(h, k)}\n{_settled_text(h, k, case)}{_drafted_text(h, k, case)}"


def _asked_columns(state: SpecialistState, whens: tuple[str, ...]) -> list[str]:
    """The columns of these timings the pack does not settle, in the order the timing rung lists them."""
    h = state["handoff"]
    case = _case(state)
    tm = _ladder(state).timing or Timing()
    return [k for w in whens for k in getattr(tm, w) if fact_relation(h, k, case) is None]


def _relation_errors(r: Relation, k: str, h: Handoff, case: C.Case, ok: Any) -> list[str]:
    """The checks every relation must pass: a reason with a cite per claim marked true, every cite resolving, not both feeding
    the treatment and changed by it, no contradiction with a pack fact, no unexplained departure from the last reading."""
    errs: list[str] = []
    if sum([r.affects_treatment, r.affects_outcome, r.affected_by_treatment, r.is_outcome_measure]) and not r.reasons:
        errs.append("claims marked true but no reasons given")
    for reason in r.reasons:
        if not reason.cites:
            errs.append("a reason has no citation")
        errs += [f"{c} is not an address you may cite" for c in reason.cites if not ok(c)]
    if r.affects_treatment and r.affected_by_treatment:
        errs.append("cannot both feed the treatment and be changed by it")
    settled, _ = settled_claims(h, k, case)
    rules = [rule for rule in CONTRADICTION_RULES if rule[0] not in settled]
    cited = [c for reason in r.reasons for c in reason.cites]
    errs += V.contradictions({c: getattr(r, c) for c in CLAIMS}, k, case, rules, cited, h)
    drafted, _ = drafted_claims(h, k, case)
    errs += V.departures({c: getattr(r, c) for c in CLAIMS}, drafted, r.departures, h, ok)
    return errs


def _presence_errors(asked: list[str], got: list[str]) -> list[str]:
    errs = []
    for k in asked:
        n = got.count(k)
        if n == 0:
            errs.append(f"{k}: no role returned")
        elif n > 1:
            errs.append(f"{k}: returned {n} times; once")
    errs += [f"{k!r} was not asked for" for k in dict.fromkeys(got) if k not in asked]
    return errs


def _with_pack_links(state: SpecialistState, x: Role, h: Handoff) -> Role:
    """The pack's own word on redundancy and nesting wins over the model's, and every link names a key."""
    b = h.column(x.column)
    same = _column_key(state, b.same_as) if b is not None else None
    nested = _column_key(state, b.nested_in) if b is not None else None
    return x.model_copy(update={"redundant_with": same or _column_key(state, x.redundant_with), "nested_in": nested or _column_key(state, x.nested_in)})


def roles(state: SpecialistState) -> Command:
    """Rung 3: every pre-treatment column (and every column whose timing is unknown) that the pack leaves open, placed together."""
    h = state["handoff"]
    case = _case(state)
    lad = _ladder(state)
    asked = _asked_columns(state, ("before", "unknown"))
    if not asked:
        return Command(goto="post_roles", update={"ladder": lad.model_copy(update={"roles": Roles(items=[])})})
    t, y, _ = _keys(state)

    def gate(r: Roles, log: EpisodeLog) -> list[str]:
        ok = _resolver(h, log, lad)
        errs = _presence_errors(asked, [x.column for x in r.items])
        for x in r.items:
            if x.column not in asked:
                continue
            errs += [f"{x.column}: {e}" for e in _relation_errors(apply_settled(x, h, case), x.column, h, case, ok)]
            link_cites = [c for link in x.links for c in link.cites]
            backed = any(any(pat in c for pat in LINK_CITES) or ((f := log.find(c)) is not None and f.tool == "redundancy") for c in link_cites)
            for field in ("redundant_with", "nested_in"):
                v = getattr(x, field)
                if v is None:
                    continue
                if _column_key(state, v) is None or _column_key(state, v) == x.column:
                    errs.append(f"{x.column}: {field} names {v!r}, which is not another column in play")
                if not backed:
                    errs.append(
                        f"{x.column}: {field} = {v!r} needs a fact under links: a redundancy tool result, [probe:data.redundancy.*], "
                        "or the column's same_as or nested_in line"
                    )
            for link in x.links:
                errs += [f"{x.column}: {e}" for e in V.cites_resolve(link.cites, h, ok)]
        return errs

    user = P.ROLES_USER.format(
        question=_question(state),
        frame=L.frame_text(state),
        treatment_card=_card(h, t) if t else "(none)",
        outcome_card=_card(h, y),
        count=len(asked),
        columns="\n\n".join(_column_block(h, k, case) for k in asked),
        errors="",
    )
    rec, log, thoughts, errors = run_episode(
        Roles, P.ROLES_SYSTEM, user, tools=L.data_tools(state, _treated_mask(state)), budget=_budget("roles"), gate=gate, node="roles"
    )
    if rec is None:
        return _stop(
            "roles",
            "the columns fixed before the change could not be placed",
            errors,
            "clearer column notes about what fed the decision",
            {"debug": thoughts, "episodes": {"roles": log}},
        )
    out = Roles(items=[_with_pack_links(state, apply_settled(x, h, case), h) for x in rec.items])
    _writer()({"roles": [f"[{a}] {text}" for a, text in out.lines()]})
    return Command(goto="post_roles", update={"ladder": lad.model_copy(update={"roles": out}), "debug": thoughts, "episodes": {"roles": log}})


def post_roles(state: SpecialistState) -> Command:
    """Rung 4: every column set at or after the treatment that the pack leaves open, placed together."""
    h = state["handoff"]
    case = _case(state)
    lad = _ladder(state)
    asked = _asked_columns(state, ("at", "after"))
    if not asked:
        return Command(goto="merge_graph", update={"ladder": lad.model_copy(update={"post_roles": PostRoles(items=[])})})
    t, y, _ = _keys(state)

    def gate(r: PostRoles, log: EpisodeLog) -> list[str]:
        ok = _resolver(h, log, lad)
        errs = _presence_errors(asked, [x.column for x in r.items])
        for x in r.items:
            if x.column not in asked:
                continue
            if not x.cites:
                errs.append(f"{x.column}: no citation")
            errs += [f"{x.column}: {e}" for e in _relation_errors(apply_settled(x.relation(), h, case), x.column, h, case, ok)]
        return errs

    user = P.POST_ROLES_USER.format(
        question=_question(state),
        frame=L.frame_text(state),
        treatment_card=_card(h, t) if t else "(none)",
        outcome_card=_card(h, y),
        count=len(asked),
        columns="\n\n".join(_column_block(h, k, case) for k in asked),
        errors="",
    )
    rec, log, thoughts, errors = run_episode(
        PostRoles, P.POST_ROLES_SYSTEM, user, tools=L.data_tools(state, _treated_mask(state)), budget=_budget("post_roles"), gate=gate, node="post_roles"
    )
    if rec is None:
        return _stop(
            "post_roles",
            "the columns set at or after the change could not be placed",
            errors,
            "clearer column notes about what the change moved",
            {"debug": thoughts, "episodes": {"post_roles": log}},
        )
    _writer()({"post_roles": [f"[{a}] {text}" for a, text in rec.lines()]})
    return Command(goto="merge_graph", update={"ladder": lad.model_copy(update={"post_roles": rec}), "debug": thoughts, "episodes": {"post_roles": log}})


def _latest(state: SpecialistState) -> dict[str, Relation]:
    """Every column's relation as it stands: the pack's facts, then the rungs' answers with the settled claims overwritten."""
    h = state["handoff"]
    case = _case(state)
    _, _, others = _keys(state)
    latest: dict[str, Relation] = {k: r for k in others if (r := fact_relation(h, k, case)) is not None}
    lad = _ladder(state)
    for x in lad.roles.items if lad.roles is not None else []:
        latest[x.column] = apply_settled(x, h, case)
    for pr in lad.post_roles.items if lad.post_roles is not None else []:
        latest[pr.column] = apply_settled(pr.relation(), h, case)
    return latest


# ------------------------------------------------------------------ merge + verify (facts)


def merge_graph(state: SpecialistState) -> dict:
    h = state["handoff"]
    t, y, others = _keys(state)
    latest = _latest(state)
    edges: list[Edge] = [Edge(src=t, dst=y)]
    excluded: list[Excluded] = []
    nodes = [t, y]
    revs: list[Revision] = state.get("applied_revisions") or []
    blk = _block(h)
    mediator = blk.mediator if blk else None
    forbidden = set(blk.forbidden) if blk else set()
    if blk is not None and blk.unobserved_confounding is True and not state.get("hidden_dropped"):
        nodes.append(adapter.HIDDEN)  # the person says something outside the file drove both; the graph says so, and only a road that avoids it identifies
        edges += [Edge(src=adapter.HIDDEN, dst=t, cites=["claim:unobserved"]), Edge(src=adapter.HIDDEN, dst=y, cites=["claim:unobserved"])]
    for k in others:
        r = latest.get(k)
        if r is None:
            continue
        cites = sorted({c for reason in r.reasons for c in reason.cites})
        if r.is_outcome_measure:
            excluded.append(Excluded(column=k, why="another measurement of the outcome"))
            continue
        if r.affected_by_treatment and r.affects_outcome and k == mediator:
            nodes.append(k)
            edges = [e for e in edges if not (e.src == t and e.dst == y)]  # the person says the whole effect runs through it: no direct road
            edges += [Edge(src=t, dst=k, cites=cites), Edge(src=k, dst=y, cites=cites)]
            continue
        if k in forbidden:
            excluded.append(Excluded(column=k, why="set at or after the change; the pack forbids adjusting for it [design.forbidden]"))
            continue
        if r.affected_by_treatment and not r.affects_treatment:
            excluded.append(Excluded(column=k, why="comes after the treatment; adjusting for it would remove part of the effect"))
            continue
        nodes.append(k)
        if r.affects_treatment:
            edges.append(Edge(src=k, dst=t, cites=cites))
        if r.affects_outcome:
            edges.append(Edge(src=k, dst=y, cites=cites))

    def drop(col: str, why: str) -> None:
        nonlocal edges
        nodes.remove(col)
        edges = [e for e in edges if col not in (e.src, e.dst)]
        excluded.append(Excluded(column=col, why=why))

    lad = _ladder(state)
    for x in lad.roles.items if lad.roles is not None else []:  # two columns that carry one thing: one is kept
        k = x.column
        if x.redundant_with and k in nodes and x.redundant_with in nodes and x.redundant_with != mediator:
            drop(k, f"carries the same information as {x.redundant_with}, which is kept [ladder:roles.{k}]")
        if x.nested_in and k in nodes and x.nested_in in nodes and x.nested_in != mediator:
            drop(x.nested_in, f"{k} sits inside it and is kept; the coarser column adds nothing [ladder:roles.{k}]")
    for rv in revs:
        k = rv.column
        if rv.change == "exclude":
            edges = [e for e in edges if k not in (e.src, e.dst)]
            nodes = [n for n in nodes if n != k]
            excluded.append(Excluded(column=k, why=f"revised out: {rv.reason}"))
            continue
        if k not in nodes:
            nodes.append(k)
        dst = t if rv.change.endswith("treatment") else y
        edges = [e for e in edges if not (e.src == k and e.dst == dst)]
        if rv.change.startswith("add"):
            edges.append(Edge(src=k, dst=dst, cites=rv.cites))
    g = Graph(treatment=t, outcome=y, nodes=nodes, edges=edges, excluded=excluded)
    _writer()({"graph": g.render()})
    return {"graph": g}


def verify_graph(state: SpecialistState) -> Command:
    """The graph as a whole: acyclic, every node a table column, a role for every column. The per-column checks were made in the
    rungs' gates; a graph that fails here is an honest stop."""
    g: Graph = state["graph"]
    _, _, others = _keys(state)
    table_cols = set(pd.read_csv(state["table_path"], nrows=0).columns)
    import networkx as nx

    general: list[str] = []
    if not nx.is_directed_acyclic_graph(g.to_networkx()):
        general.append("graph has a cycle")
    general += [f"node {n!r} is not a table column" for n in g.nodes if n not in table_cols and n != adapter.HIDDEN]
    latest = _latest(state)
    missing = [k for k in others if k not in latest]
    if missing:
        general.append("no role for " + ", ".join(missing))
    if general:
        return _stop("verify_graph", "the graph could not be made to pass verification", general, "clearer column notes about what fed the decision")
    _writer()({"verify_graph": "ok"})
    return Command(goto="identify")


# ------------------------------------------------------------------ identify + checks (facts)

_ROAD_QUESTIONS = {
    "mediator": (
        "claim:mediator.exists",
        "You said something outside the file drove both who got the change and the outcome. Is there a column the change altered, "
        "through which its whole effect on the outcome runs? If so, which, and why?",
        "a column the change altered, through which its whole effect on the outcome runs",
    ),
    "exclusion": (
        "claim:exclusion.exists",
        "Is there a column that pushed units toward the change but could not have affected the outcome any other way? If so, which, and why?",
        "a column that pushed units toward the change and could not have affected the outcome any other way",
    ),
}


def identify(state: SpecialistState) -> Command[Literal["check_design", "feasibility"]]:
    """Every road DoWhy finds. When the person says a hidden factor exists and no road avoids it, the lane asks about the one
    belief that could open a road and is not settled: whether it exists, or which column it is. Once the person has said
    there is neither, the design takes the back door with the hidden factor left in as a sensitivity range and a caveat."""
    g: Graph = state["graph"]
    h = state["handoff"]
    table = _table(state)
    sub = adapter.contrast_table(table, g.treatment, state["contrasts"][0])
    est = adapter.identify(adapter.build_model(sub, g, g.outcome))
    hidden = adapter.HIDDEN in g.nodes
    declines: list[Decline] = []
    if est.kind == "none" and hidden:
        because = "nothing identifies the effect while a hidden factor stands; one question could open a road"
        facts = [f"roads found: none; graph: {g.render().splitlines()[0]}"]
        for kind, (address, question, what) in _ROAD_QUESTIONS.items():
            b = h.beliefs.get(kind)
            st = C.belief_status(b)
            if st in ("empty", "drafted"):
                return asks.ask_back(
                    "identify",
                    LaneAsk(address=address, question=question, options=["yes", "no"], because=because),
                    reason=f"nothing identifies the effect while a hidden factor stands; one question could open a road: {kind}",
                    facts=facts,
                    extra={"estimand": est},
                )
            if st == "confirmed_true":
                col = _key(b.column) if b and b.column else None
                if not col:
                    return asks.ask_back(
                        "identify",
                        LaneAsk(address=f"claim:{kind}.column", question=f"You said there is {what}. Which column is it?", because=because),
                        reason=f"the person says there is {what}, but not which column",
                        facts=facts,
                        extra={"estimand": est},
                    )
                if col not in state.get("columns", {}):
                    declines.append(
                        Decline(
                            stage="identify",
                            kind="declined",
                            about=f"claim:{kind}.column",
                            pack_value=b.column,
                            check="intake.column_missing",
                            reason="the pack names it and the file has no such column; the road it would open is closed",
                        )
                    )
        # the person has said there is no instrument and no mediator, or named one the file cannot carry: the back door it is, with the hidden factor as a range
        g2 = Graph(
            treatment=g.treatment,
            outcome=g.outcome,
            nodes=[n for n in g.nodes if n != adapter.HIDDEN],
            edges=[e for e in g.edges if adapter.HIDDEN not in (e.src, e.dst)],
            excluded=g.excluded,
        )
        est = adapter.identify(adapter.build_model(sub, g2, g2.outcome))
        est.sensitivity_required = True
        _writer()({"estimand": est.model_dump(exclude={"dowhy_text"}), "hidden_dropped": True})
        return Command(goto="check_design", update={"estimand": est, "graph": g2, "hidden_dropped": True, "declines": declines})
    if state.get("hidden_dropped"):
        est.sensitivity_required = True  # a revise loop re-identifies without the hidden node; the range and the caveat stay
    road = h.brief.road if h.brief is not None else None
    if road is not None:  # the design brief names the road: the graph must have it, and the design takes it
        if road not in est.roads:
            return _stop(
                "identify",
                f"the design brief names the {road} road and the graph has no such road",
                [f"roads found: {', '.join(est.roads) or 'none'}", f"graph: {g.render().splitlines()[0]}"],
                "a brief whose road the graph opens, or a graph with that road",
                {"estimand": est},
            )
        est = est.model_copy(update={"kind": road})
    _writer()({"estimand": est.model_dump(exclude={"dowhy_text"})})
    return Command(goto="check_design", update={"estimand": est})


def check_design(state: SpecialistState) -> dict:
    g: Graph = state["graph"]
    est: Estimand = state["estimand"]
    h = state["handoff"]
    cfg = load_checks()
    table = _table(state)
    results: list[CheckResult] = []
    facts: dict[str, Any] = {"balance": {}, "propensity": {}}
    if est.kind == "none":
        results.append(
            CheckResult(contrast="all", name="identification", level="hard", detail="no adjustment set makes the comparison identifiable with this graph")
        )
    else:
        for c in state["contrasts"]:
            sub = adapter.contrast_table(table, g.treatment, c)
            res, f = CK.run_checks(sub, est.adjustment_set, c.key, cfg)
            results += res
            facts["balance"][c.key], facts["propensity"][c.key] = f.get("balance", {}), f.get("propensity", {})
    blk = _block(h)
    if blk is not None and blk.adjustment_candidates and est.adjustment_set:
        outside = [k for k in est.adjustment_set if k not in blk.adjustment_candidates]
        if outside:
            results.append(
                CheckResult(
                    contrast="all",
                    name="adjusts_outside_candidates",
                    level="soft",
                    detail=f"the adjustment set reaches beyond the columns the pack named as candidates: {', '.join(outside)} [design.adjustment_candidates]",
                )
            )
    W.say(results, cfg, state.get("columns") or {})  # the sentence before the number, for the reader
    results += C.as_checks(_case(state))
    _writer()({"checks": [f"{r.level} {r.address} {r.detail}" for r in results]})
    return {"checks": results, "check_facts": facts}


# ------------------------------------------------------------------ assess (the yaml first, then a judgement, repair loop)


def assess(state: SpecialistState) -> Command:
    g: Graph = state["graph"]
    est: Estimand = state["estimand"]
    h = state["handoff"]
    action, payload, results = C.decide_by_code(_case(state), state["checks"], load_beliefs(), h)
    if action == "stop":
        return _stop("assess", payload["reason"], payload["facts"], payload["what_would_fix"], {"checks": results})
    if action == "ask":
        return asks.ask_back(
            "assess",
            payload,
            reason=payload.because or "the design needs one more thing from the person",
            facts=[f"[{a}]" for a in payload.evidence],
            extra={"checks": results},
        )
    checks = Checks(results=results)
    flags = checks.flags
    if not flags:
        return Command(goto="pick_estimator", update={"checks": results})
    hard = checks.hard
    flagged_cols = {r.name.split(".", 1)[1] for r in flags if r.name.startswith("balance.")}
    if any(r.name in ("overlap", "separation", "identification") for r in flags):
        flagged_cols |= set(est.adjustment_set) | {n for n in g.nodes if n not in (g.treatment, g.outcome)}
    flag_text = "\n".join(f"[{r.address}] {r.level.upper()}: {r.detail}" for r in flags)
    errors: list[str] = []
    debug = []
    for _ in range(MAX_MODEL_RETRIES):
        user = P.ASSESS_USER.format(
            frame=L.frame_text(state),
            question=_question(state),
            graph=g.render(),
            estimand=(", ".join(est.adjustment_set) or "nothing") if est.kind == "backdoor" else "none found",
            flags=flag_text,
            errors=_rejected(errors),
        )
        parsed, th = structured(DesignAssessment, P.ASSESS_SYSTEM, user, node="assess")
        debug.append(th)
        errors = []
        if parsed.action == "proceed" and hard:
            errors.append("proceed is not allowed while a hard flag stands: " + ", ".join(r.address for r in hard))
        if parsed.action == "revise":
            if not parsed.revisions:
                errors.append("revise needs at least one revision")
            for rv in parsed.revisions:
                if rv.column not in flagged_cols:
                    errors.append(f"revision names {rv.column!r}, which no flag mentions; flagged: {sorted(flagged_cols)}")
                for c in rv.cites:
                    if not h.resolve(c):
                        errors.append(f"{c} is not a pack address")
        for c in parsed.cites:
            if not (c.startswith("check:") and any(c == r.address for r in results)) and not h.resolve(c):
                errors.append(f"{c} is not a check or pack address")
        if errors:
            continue
        _writer()({"assess": parsed.model_dump()})
        if parsed.action == "proceed":
            return Command(goto="pick_estimator", update={"assessment": parsed, "debug": debug, "checks": results})
        if parsed.action == "stop":
            return _stop(
                "assess",
                parsed.reason,
                [f"{r.address}: {r.detail}" for r in flags],
                "a design whose comparison the data can support; see the flags",
                {"assessment": parsed, "debug": debug, "checks": results},
            )
        n = state.get("revisions", 0) + 1
        if n > MAX_REVISIONS:
            return _stop(
                "assess",
                "three revisions did not clear the flags",
                [f"{r.address}: {r.detail}" for r in flags],
                "a different design or better column notes",
                {"assessment": parsed, "debug": debug, "revisions": n, "checks": results},
            )
        return Command(
            goto="merge_graph",
            update={
                "assessment": parsed,
                "debug": debug,
                "revisions": n,
                "checks": results,
                "applied_revisions": (state.get("applied_revisions") or []) + parsed.revisions,
            },
        )
    return _stop("assess", "the design assessment could not be validated", errors, "see the gate errors", {"debug": debug, "checks": results})


# ------------------------------------------------------------------ pick estimator (judgement)


def _design_facts(state: SpecialistState) -> dict[str, Any]:
    est: Estimand = state["estimand"]
    checks: list[CheckResult] = state["checks"]
    arms = [r for r in checks if r.name == "arms"]
    h = state["handoff"]
    b = _block(h)
    return {
        "estimand": est.kind,
        "roads": est.roads,
        "road_from_brief": h.brief.road if h.brief is not None else None,
        "instruments": est.instruments,
        "frontdoor_set": est.frontdoor_set,
        "sensitivity_required": est.sensitivity_required,
        "treatment": "binary",
        "outcome": state["outcome_kind"],
        "adjustment_set": "nonempty" if est.adjustment_set else "empty",
        "adjustment_columns": est.adjustment_set,
        "contrasts": len(state["contrasts"]),
        "smallest_arm": min((int(r.value) for r in arms if r.value is not None), default=0),
        # from the pack: what the person said, so the pick is not made by habit
        "instrument_named": b.instrument if b else None,
        "mediator_named": b.mediator if b else None,
        "hidden_confounding_per_person": b.unobserved_confounding if b else None,
        "voluntary_uptake": b.voluntary_uptake if b else None,
    }


def pick_estimator(state: SpecialistState) -> Command:
    """The catalogue filtered by the facts; when the brief names the road, only that road's estimators are offered."""
    facts = _design_facts(state)
    excluded = set(state.get("excluded_estimators") or [])
    roads = [facts["road_from_brief"]] if facts["road_from_brief"] else facts["roads"]
    allowed = [
        e
        for e in load_estimators()
        if e.applies(estimand=facts["estimand"], treatment=facts["treatment"], outcome=facts["outcome"], adjustment_set=facts["adjustment_set"], roads=roads)
        and e.name not in excluded
    ]
    if not allowed:
        return _stop(
            "pick_estimator",
            "no estimator in the catalogue applies to this design",
            [f"facts: {facts}", f"excluded after failures: {sorted(excluded)}"],
            "an estimator entry for this estimand, treatment, and outcome kind",
        )
    names = [e.name for e in allowed]
    check_text = "\n".join(f"[{r.address}] {r.level}: {r.detail}" for r in state["checks"])
    errors: list[str] = []
    debug = []
    for _ in range(MAX_MODEL_RETRIES):
        user = P.PICK_USER.format(
            frame=L.frame_text(state),
            facts=json.dumps(facts),
            checks=check_text,
            estimators="\n\n".join(e.render() for e in allowed),
            preferences=render_preferences(allowed),
            names=", ".join(names),
            errors=_rejected(errors),
        )
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
            return Command(
                goto="freeze_design",
                update={"estimator": parsed.name, "estimator_pick": parsed, "debug": debug, "pick_attempts": state.get("pick_attempts", 0) + 1},
            )
    return _stop("pick_estimator", "the estimator pick could not be validated", errors, "see the gate errors", {"debug": debug})


# ------------------------------------------------------------------ freeze (fact)


def freeze_design(state: SpecialistState) -> dict:
    est: Estimand = state["estimand"]
    entry = estimator_entry(state["estimator"])
    if entry.estimand in est.roads and entry.estimand != est.kind:  # the pick took another open road: the design records it
        est = est.model_copy(update={"kind": entry.estimand, "adjustment_set": est.adjustment_set if entry.estimand == "backdoor" else []})
    adj = "nonempty" if est.adjustment_set else "empty"
    refuters = [r.name for r in load_refuters() if r.applies(estimand=est.kind, adjustment_set=adj, hidden=est.sensitivity_required)]
    also = (
        entry.also_run
        if entry.also_run and estimator_entry(entry.also_run).applies(estimand=est.kind, treatment="binary", outcome=state["outcome_kind"], adjustment_set=adj)
        else None
    )
    d = Design(
        contrasts=state["contrasts"],
        graph=state["graph"],
        estimand=est,
        checks=Checks(results=state["checks"]),
        estimator=entry.name,
        params=entry.params,
        also_run=also,
        refuters=refuters,
        target_units=state["target_units"],
    )
    run_dir = Path(state["run_dir"])
    (run_dir / "design.json").write_text(d.model_dump_json(indent=2))
    (run_dir / "design.md").write_text(d.render())
    _writer()({"design": d.render()})
    return {"design": d}


# ------------------------------------------------------------------ analyse (fact, fan-out per contrast)


def fan_out_analyse(state: SpecialistState):
    return [Send("analyse", ContrastTask(contrast=c.key, design=state["design"].model_dump(), table_path=state["table_path"])) for c in state["contrasts"]]


def analyse(task: ContrastTask) -> dict:
    d = Design.model_validate(task["design"])
    c = next(x for x in d.contrasts if x.key == task["contrast"])
    table = pd.read_csv(task["table_path"])
    sub = adapter.contrast_table(table, d.graph.treatment, c)
    model = adapter.build_model(sub, d.graph, d.graph.outcome)
    estimates: list[Estimate] = []
    refutations: list[Refutation] = []
    primary, bundle = adapter.estimate(model, estimator_entry(d.estimator), c.key, d.target_units)
    estimates.append(primary)
    if primary.error is None:
        for name in d.refuters:
            rf = next(r for r in load_refuters() if r.name == name)
            refutations.append(adapter.refute(model, bundle, primary, rf, c.key))
    if d.also_run:
        sec, _ = adapter.estimate(model, estimator_entry(d.also_run), c.key, d.target_units, secondary=True)
        estimates.append(sec)
    _writer()({"analyse": {c.key: {"estimate": primary.model_dump(exclude_none=True), "refutations": [r.detail for r in refutations]}}})
    return {"estimates": estimates, "refutations": refutations}


def _latest_primary(state: SpecialistState) -> dict[str, Estimate]:
    out: dict[str, Estimate] = {}
    for e in state.get("estimates") or []:
        if not e.secondary and e.method == state["estimator"]:
            out[e.contrast] = e
    return out


def after_analyse(state: SpecialistState) -> Command:
    failed = [e for e in _latest_primary(state).values() if e.error]
    if not failed:
        sends = fan_out_interpret(state)
        return Command(goto=sends or "figures")
    if state.get("pick_attempts", 0) < MAX_PICK_ATTEMPTS:
        _writer()({"fit_failure": [e.error for e in failed]})
        return Command(goto="pick_estimator", update={"excluded_estimators": (state.get("excluded_estimators") or []) + [state["estimator"]]})
    return _stop(
        "estimate",
        "the estimator failed to fit and the re-pick failed too",
        [f"{e.contrast}: {e.error}" for e in failed],
        "a different estimator entry or a smaller adjustment set",
    )


# ------------------------------------------------------------------ interpret (judgement, fan-out per contrast)


def _episodes(state: SpecialistState) -> dict[str, EpisodeLog]:
    return {k: v for k, v in (state.get("episodes") or {}).items() if isinstance(v, EpisodeLog)}


def _addresses(state: SpecialistState, contrast_key: str) -> list[str]:
    d: Design = state["design"]
    out = ["design.estimand.adjustment_set", "design.assumption"] + [b.address for b in state["handoff"].beliefs.values() if b.known() or b.status == "unknown"]
    if state["handoff"].brief is not None:
        out += sorted(state["handoff"].brief.addresses())
    out += [r.address for r in d.checks.results if r.contrast in (contrast_key, "all")]
    out += [x.address for x in state.get("declines") or []]
    out += sorted(_ladder(state).addresses())
    out += [f.address for log in _episodes(state).values() for f in log.facts]
    for e in state.get("estimates") or []:
        if e.contrast == contrast_key and e.error is None:
            tag = f"estimate:{contrast_key}" + (f".{e.method}" if e.secondary else "")
            out += [f"{tag}.value", f"{tag}.ci", f"{tag}.n"]
    for r in state.get("refutations") or []:
        if r.contrast == contrast_key:
            out += [f"refute:{contrast_key}.{r.refuter}.p_value", f"refute:{contrast_key}.{r.refuter}.new_effect", f"refute:{contrast_key}.{r.refuter}.passed"]
    return out


def _required(state: SpecialistState, contrast_key: str) -> list[str]:
    """What the interpretation must cite: every flagged check (the person's flags among them) and the estimate's interval."""
    d: Design = state["design"]
    out = [r.address for r in d.checks.results if r.contrast in (contrast_key, "all") and r.level != "pass"]
    if any(e.contrast == contrast_key and e.error is None and not e.secondary for e in state.get("estimates") or []):
        out.append(f"estimate:{contrast_key}.ci")
    return out


def _material(state: SpecialistState, contrast_key: str) -> str:
    d: Design = state["design"]
    c = next(x for x in d.contrasts if x.key == contrast_key)
    names = state.get("columns") or {}
    lines = [f"[design.assumption] the design bets on: {state['handoff'].chosen_assumption}"]
    if state["handoff"].brief is not None:
        lines.append(state["handoff"].brief.render())
    lines += [b.render() for b in state["handoff"].beliefs.values() if b.known() or b.status == "unknown"]
    lines += [
        f"[design.estimand.adjustment_set] adjusted for: {', '.join(names.get(k, k) for k in d.estimand.adjustment_set) or 'nothing (no confounders in the graph)'}",
        f"comparison: {names.get(d.graph.treatment, d.graph.treatment)} = {c.treated!r} versus {c.control!r}; outcome: {names.get(d.graph.outcome, d.graph.outcome)}",
    ]
    for r in d.checks.results:
        if r.contrast in (contrast_key, "all"):
            lines.append(f"[{r.address}] {r.level}: {r.detail}")
    lines += [x.render() for x in state.get("declines") or []]
    lines += [f"[{a}] {text}" for a, text in _ladder(state).lines()]
    lines += [f.render() for log in _episodes(state).values() for f in log.facts]
    for e in state.get("estimates") or []:
        if e.contrast == contrast_key and e.error is None:
            tag = f"estimate:{contrast_key}" + (f".{e.method}" if e.secondary else "")
            lines.append(f"[{tag}.value] {e.value:.4g} ({'secondary' if e.secondary else 'primary'} estimator {e.method}, target {e.target_units})")
            lines.append(f"[{tag}.ci] 95% interval {e.ci_low:.4g} to {e.ci_high:.4g}" if e.ci_low is not None else f"[{tag}.ci] no interval")
            lines.append(f"[{tag}.n] {e.n_treated} treated, {e.n_control} control")
    for r in state.get("refutations") or []:
        if r.contrast == contrast_key:
            lines.append(
                f"[refute:{contrast_key}.{r.refuter}.passed] {r.passed}  [refute:{contrast_key}.{r.refuter}.new_effect] {r.new_effect}  "
                f"[refute:{contrast_key}.{r.refuter}.p_value] {r.p_value}  ({r.detail})"
            )
    return "\n".join(lines)


def fan_out_interpret(state: SpecialistState):
    return [
        Send(
            "interpret",
            InterpretTask(
                question=_question(state),
                frame=L.frame_text(state),
                contrast=c.key,
                material=_material(state, c.key),
                addresses="\n".join(_addresses(state, c.key)),
                required="\n".join(_required(state, c.key)),
                errors="",
                primary_value=_latest_primary(state)[c.key].value,
                tolerance=load_checks().get("effect_tolerance", 0.01),
            ),
        )
        for c in state["contrasts"]
        if c.key in _latest_primary(state)
    ]


def interpret(task: InterpretTask) -> dict:
    allowed = set(task["addresses"].splitlines())
    required = [a for a in task["required"].splitlines() if a]
    errors: list[str] = []
    debug = []
    parsed = None
    for _ in range(MAX_MODEL_RETRIES):
        user = P.INTERPRET_USER.format(
            frame=task["frame"],
            question=task["question"],
            contrast=task["contrast"],
            material=task["material"],
            addresses=task["addresses"],
            required="\n".join(required) or "(none)",
            errors=_rejected(errors),
        )
        parsed, th = structured(Interpretation, P.INTERPRET_SYSTEM, user, node=f"interpret:{task['contrast']}")
        debug.append(th)
        parsed.contrast = task["contrast"]
        errors = [f"{c} is not an address you may cite" for c in parsed.cites if c not in allowed]
        if not parsed.cites:
            errors.append("no citations given")
        missing = [a for a in required if a not in parsed.cites]
        if missing:
            errors.append("these must be cited: " + ", ".join(missing))
        v = task["primary_value"]
        if v is not None and abs(parsed.effect_stated - v) > max(abs(v), 1e-9) * task["tolerance"]:
            errors.append(f"effect_stated {parsed.effect_stated} does not match the primary estimate {v:.4g}")
        if not errors:
            break
    out: dict[str, Any] = {"interpretations": [parsed], "debug": debug}
    if errors:
        out["interpret_errors"] = {task["contrast"]: errors}
    return out


# ------------------------------------------------------------------ feasibility, figures, assemble (facts)


def figures(state: SpecialistState) -> dict:
    """What this run drew, checked against the addresses it produced: the graph it built, the balance of every adjustment
    column before and after weighting on the score, and the estimate against its falsifications."""
    from causal_agent.families.adjustment import postviz as PA

    h = state["handoff"]
    names = state.get("columns") or {}
    g = state.get("graph")
    specs = [PA.causal_graph(g.model_dump() if g is not None else {}, names)]
    thr = float(load_checks()["balance"]["smd"]["soft"])
    for c, facts in ((state.get("check_facts") or {}).get("balance") or {}).items():
        specs.append(PA.balance(facts, c, thr, names))
    specs.append(
        PV.effect_and_refutations([e.model_dump() for e in state.get("estimates") or []], [r.model_dump() for r in state.get("refutations") or []], "refute")
    )
    kept, declines = LF.write(state.get("run_dir"), specs, LF.ok_addresses(h, state, "refute"))
    _writer()({"figures": [s.id for s in kept]})
    return {"figures": [s.model_dump() for s in kept], "declines": declines}


def assemble(state: SpecialistState) -> dict:
    h = state["handoff"]
    names = state.get("columns") or {}
    d: Design | None = state.get("design")
    f: Feasibility | None = state.get("feasibility")
    lines = [f"QUESTION     {_question(state)}", f"LANE         {h.family} → {h.specialist}", f"OUTCOME      {h.outcome}    TREATMENT   {h.treatment}", ""]
    lad = _ladder(state)
    if lad.lines():
        lines += ["THE LADDER"] + [f"  [{a}] {text}" for a, text in lad.lines()] + [""]
    episodes = _episodes(state)
    if any(log.facts or log.refusals for log in episodes.values()):
        lines.append("WHAT THE EPISODES LOOKED AT")
        for name, log in episodes.items():
            lines.append(f"  {name}: {log.calls} call{'s' if log.calls != 1 else ''}, {log.tries} tr{'ies' if log.tries != 1 else 'y'}")
            lines += [f"    {f.render()}" for f in log.facts]
            lines += [f"    refused {r.tool}({', '.join(f'{k}={v!r}' for k, v in r.args.items())}): {r.reason}" for r in log.refusals]
        lines.append("")
    if d:
        lines += d.render().splitlines()
        lines.append("")
        lines.append("RESULTS")
        for c in d.contrasts:
            lines.append(f"  {names.get(d.graph.treatment, d.graph.treatment)} = {c.treated!r} vs {c.control!r}")
            for e in state.get("estimates") or []:
                if e.contrast == c.key:
                    if e.error:
                        lines.append(f"    {e.method:34} FAILED: {e.error}")
                    else:
                        ci = f" [{e.ci_low:.3g}, {e.ci_high:.3g}]" if e.ci_low is not None else ""
                        lines.append(f"    {e.method:34} {e.value:+.4g}{ci}  n={e.n_treated}/{e.n_control}  {'secondary' if e.secondary else 'primary'}")
            for r in state.get("refutations") or []:
                if r.contrast == c.key:
                    lines.append(f"    refute {r.refuter:27} {r.detail}")
            for i in state.get("interpretations") or []:
                if i.contrast == c.key:
                    lines.append(f"    ANSWER  {i.answer}")
                    for cv in i.caveats:
                        lines.append(f"    CAVEAT  {cv}")
                    lines.append(f"    CITES   {', '.join(i.cites)}")
            errs = (state.get("interpret_errors") or {}).get(c.key)
            if errs:
                lines.append(f"    INTERPRETATION GATE FAILED: {'; '.join(errs)}")
    if f:
        lines += ["", f"STOPPED AT   {f.stage}", f"REASON       {f.reason}"]
        lines += [f"FACT         {x}" for x in f.facts]
        lines.append(f"WOULD FIX    {f.what_would_fix}")
    lines += records.report_tail(state)
    debug = state.get("debug") or []
    if any(t.text for t in debug):
        lines += ["", "MODEL THOUGHTS (debug only)"]
        for t in debug:
            if t.text:
                lines.append(f"  [{t.node}] {t.text.strip()[:2000]}")
    report = "\n".join(lines)
    g = state.get("graph")
    records.write(state.get("run_dir"), records.artifacts(state, {"graph": g.model_dump() if g else None, "estimand": state.get("estimand")}), report)
    result = records.result(
        state,
        report,
        {
            "ladder": [[a, text] for a, text in lad.lines()],
            "facts": [{"address": f.address, "text": f.render().split("] ", 1)[1], "value": f.value} for log in episodes.values() for f in log.facts],
        },
    )
    _writer()({"report": report})
    return {"report": report, "specialist_result": result}
