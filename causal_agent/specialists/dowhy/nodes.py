"""Adjustment-lane nodes. Facts compute; judgements call the model once and are gated.

The pack is weighed by code first (the harness's `case`): a settled column field is a fact the graph takes, an open one
is asked of the model, a belief is a flag the assessment must answer. Stops are typed: a node that cannot go on returns
Command(goto="feasibility") with a Feasibility record; a question for the person is the same with a LaneAsk.
Nothing here names a column, a method, or a dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from langgraph.config import get_stream_writer
from langgraph.types import Command, Send

from causal_agent.common.addresses import key as _key
from causal_agent.common.contracts import AdjustmentDesign, CheckResult, Checks, Cited, Contrast, Decline, Estimate, Feasibility, Handoff, Interpretation, LaneAsk, Refutation
from causal_agent.common.llm import structured
from causal_agent.lane import asks, case as C, figures as LF, intake, records, verify as V
from causal_agent.specialists.dowhy import adapter, checks as CK
from causal_agent.specialists.dowhy import prompts as P
from causal_agent.specialists.dowhy.contracts import (
    Contrasts,
    Design,
    DesignAssessment,
    Edge,
    Estimand,
    EstimatorPick,
    Excluded,
    Graph,
    Relation,
    Revision,
)
from causal_agent.specialists.dowhy.knowledge import (
    estimator as estimator_entry,
    load_beliefs,
    load_checks,
    load_estimators,
    load_refuters,
    render_preferences,
)
from causal_agent.specialists.dowhy.state import ContrastTask, InterpretTask, RelateTask, SpecialistState
from causal_agent.viz.postviz import common as PV

MAX_RELATE_ATTEMPTS = 3
MAX_REVISIONS = 3
MAX_MODEL_RETRIES = 3
MAX_PICK_ATTEMPTS = 2
MAX_DOSE_LEVELS = 12
CLAIMS = ("affects_treatment", "affects_outcome", "affected_by_treatment", "is_outcome_measure")
# a model claim that contradicts a pack fact on a claim the pack did not itself settle
CONTRADICTION_RULES: list[V.Rule] = [("affects_treatment", True, "when", ("after",))]


# ------------------------------------------------------------------ helpers

def _writer():
    try:
        return get_stream_writer()
    except Exception:  # outside a graph run
        return lambda _x: None


def _table(state: SpecialistState) -> pd.DataFrame:
    return pd.read_csv(state["table_path"])


def _keys(state: SpecialistState) -> tuple[str, str, list[str]]:
    """The treatment, the outcome, and every other loaded column: the frame's and the ones the family block names."""
    h = state["handoff"]
    t, y = _key(h.treatment or ""), _key(h.outcome)
    return t, y, [k for k in state.get("columns", {}) if k not in (t, y)]


def _stop(stage: str, reason: str, facts: list[str], fix: str, extra: dict | None = None) -> Command:
    f = Feasibility(stage=stage, reason=reason, facts=facts, what_would_fix=fix)
    return Command(goto="feasibility", update={"feasibility": f, **(extra or {})})


def _card(h: Handoff, key: str) -> str:
    return h.brief_text(key)


def _block(h: Handoff) -> AdjustmentDesign | None:
    return h.design if isinstance(h.design, AdjustmentDesign) else None


def _case(state: SpecialistState) -> C.Case:
    c = state.get("case")
    return c if isinstance(c, C.Case) else C.Case()


def _cites(h: Handoff, *addresses: str) -> list[str]:
    """The claim addresses that resolve in the pack, else the change card."""
    ok = [a for a in addresses if h.resolve(a)]
    return ok or ["change:1.note"]


def _frame_text(state: SpecialistState) -> str:
    """What every judgement of this lane reads beside its own cards: the question as read, the dataset and the change,
    the family block, the probes, the case the code made of the pack, and the person's words."""
    h = state["handoff"]
    s = h.scope
    parts = [
        f"family: {h.family}; outcome: {h.outcome}; treatment: {h.treatment}; "
        f"filter={s.population_filter or 'none'}; window={s.window or 'none'}; contrast={s.contrast}; target={s.target}\n"
        f"assumption the router bet on: {h.chosen_assumption}",
        h.render_dataset(),
        h.render_change(),
        (f"what the pack settled:\n{h.design.render()}" if _block(h) else ""),
        ("PROBES\n" + "\n".join(p.render() for p in h.probes)) if h.probes else "",
        _case(state).render() if state.get("case") else "",
        h.render_words(),
    ]
    return "\n".join(x for x in parts if x)


def _thought(t, node: str):
    t.node = node
    return t


def _question(state: SpecialistState) -> str:
    return state.get("question") or ""


# ------------------------------------------------------------------ load (fact) and the case (fact)


def load(state: SpecialistState) -> Command:
    h = state["handoff"]
    if h.intent != "effect_of_change":
        return _stop("load", "this lane answers the effect of a change on an outcome; the question was read differently", [f"intent: {h.intent}"],
                     "a question about the effect of a change on an outcome")
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
        return _stop("load", "the outcome is neither numeric nor two-valued", [f"outcome {y} has {table[y].nunique()} distinct non-numeric values"],
                     "an outcome measured as a number or a yes/no", {"declines": it.declines})
    cfg = load_checks()
    target = cfg["target_units"].get(h.scope.target)
    if target is None:
        return _stop("load", f"target '{h.scope.target}' is not supported by this lane yet", [], "a question asking for the average effect, or the effect on the treated",
                     {"declines": it.declines})
    tcol = table[t]
    numeric_dose = pd.api.types.is_numeric_dtype(tcol) and tcol.nunique() > MAX_DOSE_LEVELS
    if numeric_dose:
        return _stop("load", "the treatment is a dose with many values; this lane compares levels", [f"{t} has {tcol.nunique()} distinct values"],
                     "a two-level or few-level treatment, or a dose-response lane", {"declines": it.declines})
    levels = [str(v) for v in sorted(tcol.unique(), key=lambda v: str(v))]
    _writer()({"load": {"rows": len(table), "columns": list(it.columns), "outcome_kind": ok, "target_units": target, "run_dir": str(it.run_dir),
                        "declines": [d.render() for d in it.declines], **it.facts}})
    return Command(
        goto="case",
        update={
            "run_dir": str(it.run_dir), "table_path": str(it.table_path), "columns": it.columns, "declines": it.declines,
            "check_facts": {"intake": it.facts} if it.facts else {},
            "outcome_kind": ok, "target_units": target, "treatment_levels": levels,
            "relate_attempts": 0, "revisions": 0, "pick_attempts": 0, "interpret_attempts": 0,
            "relate_errors": {}, "excluded_estimators": [], "applied_revisions": [],
        },
    )


def case(state: SpecialistState) -> dict:
    """The pack weighed by code: facts, open fields, flags from what the person said."""
    c = C.weigh(state["handoff"], load_beliefs())
    _writer()({"case": c.render()})
    return {"case": c}


# ------------------------------------------------------------------ contrast (judgement, unless the pack settles it)


def contrast(state: SpecialistState) -> dict:
    h = state["handoff"]
    t, _, _ = _keys(state)
    levels = state["treatment_levels"]
    if h.treated_level is not None and str(h.treated_level) in levels:  # the pack says which level means treated: a fact
        treated = str(h.treated_level)
        control = str(h.control_level) if h.control_level is not None and str(h.control_level) in levels and str(h.control_level) != treated else None
        if control is None and len(levels) == 2:
            control = next(v for v in levels if v != treated)
        if control is not None:
            c = Contrast(control=control, treated=treated, reason="the pack names the level that means the unit got the change",
                         cites=_cites(h, "claim:assignment.treated_level", "claim:assignment.treatment_column"))
            _writer()({"contrasts": [c.model_dump()]})
            return {"contrasts": [c], "debug": []}
    s = h.scope
    scope = f"filter={s.population_filter or 'none'}; window={s.window or 'none'}; contrast={s.contrast}; target={s.target}"
    errors: list[str] = []
    debug = []
    for _ in range(MAX_MODEL_RETRIES):
        user = P.CONTRAST_USER.format(question=_question(state), scope=scope,
                                      treatment_card=_card(h, t), levels=", ".join(repr(v) for v in levels))
        if errors:
            user += "\nPREVIOUS ANSWER WAS REJECTED\n" + "\n".join(f"- {e}" for e in errors) + "\n"
        parsed, th = structured(Contrasts, P.CONTRAST_SYSTEM, user, node="contrast")
        debug.append(th)
        errors = _validate_contrasts(parsed.items, levels)
        if not errors:
            _writer()({"contrasts": [c.model_dump() for c in parsed.items]})
            return {"contrasts": parsed.items, "debug": debug}
    return {"feasibility": Feasibility(stage="contrast", reason="could not define a comparison from the treatment's levels", facts=errors,
                                       what_would_fix="a treatment whose levels the note explains"), "debug": debug}


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


# ------------------------------------------------------------------ relate (judgement only for what the pack leaves open)


def settled_claims(h: Handoff, k: str, case: C.Case) -> tuple[dict[str, bool], dict[str, str]]:
    """The claims about a column the pack settles, each with the address it rests on. The instrument and the mediator the
    person named settle all four; a column the offer depended on fed the treatment; a column fixed before the change was not
    moved by it; a column the person called a measure of the outcome is one; the person's word on `moved_by_change` stands."""
    b = h.brief(k)
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
    b = h.brief(k)
    if b is None:
        return None
    claims, cites = settled_claims(h, k, case)
    if claims.get("is_outcome_measure") is True:
        return Relation(column=k, affects_treatment=False, affects_outcome=False, affected_by_treatment=False, is_outcome_measure=True,
                        reasons=[Cited(reason=f"{b.name}: the person said it measures the outcome", cites=[cites["is_outcome_measure"]])])
    if all(c in claims for c in CLAIMS):
        return Relation(column=k, reasons=[Cited(reason=f"{b.name}: {c} settled by the pack", cites=[cites[c]]) for c in CLAIMS if claims[c]], **claims)
    if b.role == "depends_on" and case.fact(f"{b.address}.when") == "before":
        return Relation(column=k, affects_treatment=True, affects_outcome=True, affected_by_treatment=False, is_outcome_measure=False,
                        reasons=[Cited(reason=f"{b.name}: the rule or the offer looked at it", cites=["claim:assignment.depends_on"]),
                                 Cited(reason=f"{b.name}: fixed before the change, a background attribute", cites=[f"{b.address}.when"])])
    return None


def _settled_text(h: Handoff, k: str, case: C.Case) -> str:
    claims, cites = settled_claims(h, k, case)
    if not claims:
        return ""
    return "\nSETTLED BY THE PACK (copy these answers; cite the address)\n" + "\n".join(f"  {c} = {str(v).lower()} [{cites[c]}]" for c, v in claims.items()) + "\n"


def apply_settled(r: Relation, h: Handoff, case: C.Case) -> Relation:
    """The model's relation with every claim the pack settles overwritten, and a cited reason for each settled claim."""
    claims, cites = settled_claims(h, r.column, case)
    if not claims:
        return r
    out = r.model_copy(deep=True)
    for c, v in claims.items():
        setattr(out, c, v)
        if v and not any(cites[c] in reason.cites for reason in out.reasons):
            out.reasons.append(Cited(reason=f"{r.column}: {c} settled by the pack", cites=[cites[c]]))
    return out


def _latest(state: SpecialistState) -> dict[str, Relation]:
    """Every column's relation as it stands: the pack's facts, then the model's answers with the settled claims overwritten."""
    h = state["handoff"]
    case = _case(state)
    _, _, others = _keys(state)
    latest: dict[str, Relation] = {k: r for k in others if (r := fact_relation(h, k, case)) is not None}
    for r in state.get("relations") or []:
        latest[r.column] = apply_settled(r, h, case)  # later answers replace earlier ones
    return latest


def _relate_send(state: SpecialistState, k: str, errors: list[str] | None = None) -> Send:
    h = state["handoff"]
    t, y, _ = _keys(state)
    return Send("relate", RelateTask(question=_question(state), frame=_frame_text(state), treatment_card=_card(h, t), outcome_card=_card(h, y),
                                     column=k, card=_card(h, k), settled=_settled_text(h, k, _case(state)),
                                     errors=("\nPREVIOUS ANSWER WAS REJECTED\n" + "\n".join(f"- {e}" for e in errors) + "\n") if errors else ""))


def fan_out_relate(state: SpecialistState):
    if state.get("feasibility"):
        return "feasibility"
    h = state["handoff"]
    _, _, others = _keys(state)
    errs = state.get("relate_errors") or {}
    case = _case(state)
    targets = [k for k in others if k in errs] if errs else [k for k in others if fact_relation(h, k, case) is None]
    if not targets:
        return "merge_graph"
    return [_relate_send(state, k, errs.get(k)) for k in targets]


def relate(task: RelateTask) -> dict:
    user = P.RELATE_USER.format(**task)
    parsed, th = structured(Relation, P.RELATE_SYSTEM, user, node=f"relate:{task['column']}")
    parsed.column = task["column"]
    return {"relations": [parsed], "debug": [th]}


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
    h = state["handoff"]
    g: Graph = state["graph"]
    t, y, others = _keys(state)
    case = _case(state)
    table_cols = set(pd.read_csv(state["table_path"], nrows=0).columns)
    errs: dict[str, list[str]] = {}
    general: list[str] = []
    nx_g = g.to_networkx()
    import networkx as nx

    if not nx.is_directed_acyclic_graph(nx_g):
        general.append("graph has a cycle")
    for n in g.nodes:
        if n not in table_cols and n != adapter.HIDDEN:
            general.append(f"node {n!r} is not a table column")
    latest: dict[str, Relation] = {k: r for k in others if (r := fact_relation(h, k, case)) is not None}
    latest.update({r.column: r for r in state.get("relations") or []})
    for k in others:
        r = latest.get(k)
        if r is None:
            errs.setdefault(k, []).append("no relation returned")
            continue
        claims = sum([r.affects_treatment, r.affects_outcome, r.affected_by_treatment, r.is_outcome_measure])
        if claims and not r.reasons:
            errs.setdefault(k, []).append("claims marked true but no reasons given")
        for reason in r.reasons:
            if not reason.cites:
                errs.setdefault(k, []).append("a reason has no citation")
            for c in reason.cites:
                if not h.resolve(c):
                    errs.setdefault(k, []).append(f"{c} is not a pack address")
        if r.affects_treatment and r.affected_by_treatment:
            errs.setdefault(k, []).append("cannot both feed the treatment and be changed by it")
        settled, _ = settled_claims(h, k, case)
        rules = [rule for rule in CONTRADICTION_RULES if rule[0] not in settled]
        for e in V.contradictions({c: getattr(r, c) for c in CLAIMS}, k, case, rules, [c for reason in r.reasons for c in reason.cites], h):
            errs.setdefault(k, []).append(e)
    attempts = state.get("relate_attempts", 0) + 1
    if errs or general:
        if attempts >= MAX_RELATE_ATTEMPTS or (general and not errs):
            return _stop("verify_graph", "the graph could not be made to pass verification",
                         general + [f"{k}: {'; '.join(v)}" for k, v in errs.items()], "clearer column notes about what fed the decision",
                         {"relate_attempts": attempts})
        _writer()({"verify_graph": {"attempt": attempts, "errors": errs}})
        return Command(goto=[_relate_send(state, k, v) for k, v in errs.items()], update={"relate_errors": errs, "relate_attempts": attempts})
    _writer()({"verify_graph": "ok"})
    return Command(goto="identify", update={"relate_errors": {}, "relate_attempts": attempts})


# ------------------------------------------------------------------ identify + checks (facts)

_ROAD_QUESTIONS = {
    "mediator": ("claim:mediator.exists", "You said something outside the file drove both who got the change and the outcome. Is there a column the change altered, "
                                          "through which its whole effect on the outcome runs? If so, which, and why?",
                 "a column the change altered, through which its whole effect on the outcome runs"),
    "exclusion": ("claim:exclusion.exists", "Is there a column that pushed units toward the change but could not have affected the outcome any other way? If so, which, and why?",
                  "a column that pushed units toward the change and could not have affected the outcome any other way"),
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
                return asks.ask_back("identify", LaneAsk(address=address, question=question, options=["yes", "no"], because=because),
                                     reason=f"nothing identifies the effect while a hidden factor stands; one question could open a road: {kind}", facts=facts, extra={"estimand": est})
            if st == "confirmed_true":
                col = _key(b.column) if b and b.column else None
                if not col:
                    return asks.ask_back("identify", LaneAsk(address=f"claim:{kind}.column", question=f"You said there is {what}. Which column is it?", because=because),
                                         reason=f"the person says there is {what}, but not which column", facts=facts, extra={"estimand": est})
                if col not in state.get("columns", {}):
                    declines.append(Decline(stage="identify", kind="declined", about=f"claim:{kind}.column", pack_value=b.column, check="intake.column_missing",
                                            reason="the pack names it and the file has no such column; the road it would open is closed"))
        # the person has said there is no instrument and no mediator, or named one the file cannot carry: the back door it is, with the hidden factor as a range
        g2 = Graph(treatment=g.treatment, outcome=g.outcome, nodes=[n for n in g.nodes if n != adapter.HIDDEN], edges=[e for e in g.edges if adapter.HIDDEN not in (e.src, e.dst)], excluded=g.excluded)
        est = adapter.identify(adapter.build_model(sub, g2, g2.outcome))
        est.sensitivity_required = True
        _writer()({"estimand": est.model_dump(exclude={"dowhy_text"}), "hidden_dropped": True})
        return Command(goto="check_design", update={"estimand": est, "graph": g2, "hidden_dropped": True, "declines": declines})
    if state.get("hidden_dropped"):
        est.sensitivity_required = True  # a revise loop re-identifies without the hidden node; the range and the caveat stay
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
        results.append(CheckResult(contrast="all", name="identification", level="hard",
                                   detail="no adjustment set makes the comparison identifiable with this graph"))
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
            results.append(CheckResult(contrast="all", name="adjusts_outside_candidates", level="soft",
                                       detail=f"the adjustment set reaches beyond the columns the pack named as candidates: {', '.join(outside)} [design.adjustment_candidates]"))
    results += C.as_checks(_case(state))
    _writer()({"checks": [f"{r.level} {r.address} {r.detail}" for r in results]})
    return {"checks": results, "check_facts": facts}


def after_checks(state: SpecialistState) -> str:
    return "assess" if any(r.level != "pass" for r in state["checks"]) else "pick_estimator"


# ------------------------------------------------------------------ assess (the yaml first, then a judgement, repair loop)


def assess(state: SpecialistState) -> Command:
    g: Graph = state["graph"]
    est: Estimand = state["estimand"]
    h = state["handoff"]
    action, payload, results = C.decide_by_code(_case(state), state["checks"], load_beliefs(), h)
    if action == "stop":
        return _stop("assess", payload["reason"], payload["facts"], payload["what_would_fix"], {"checks": results})
    if action == "ask":
        return asks.ask_back("assess", payload, reason=payload.because or "the design needs one more thing from the person", facts=[f"[{a}]" for a in payload.evidence], extra={"checks": results})
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
        user = P.ASSESS_USER.format(question=_question(state), graph=g.render(),
                                    estimand=(", ".join(est.adjustment_set) or "nothing") if est.kind == "backdoor" else "none found",
                                    flags=flag_text, errors=("\nPREVIOUS ANSWER WAS REJECTED\n" + "\n".join(f"- {e}" for e in errors) + "\n") if errors else "")
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
            return _stop("assess", parsed.reason, [f"{r.address}: {r.detail}" for r in flags],
                         "a design whose comparison the data can support; see the flags", {"assessment": parsed, "debug": debug, "checks": results})
        n = state.get("revisions", 0) + 1
        if n > MAX_REVISIONS:
            return _stop("assess", "three revisions did not clear the flags", [f"{r.address}: {r.detail}" for r in flags],
                         "a different design or better column notes", {"assessment": parsed, "debug": debug, "revisions": n, "checks": results})
        return Command(goto="merge_graph", update={"assessment": parsed, "debug": debug, "revisions": n, "checks": results,
                                                   "applied_revisions": (state.get("applied_revisions") or []) + parsed.revisions})
    return _stop("assess", "the design assessment could not be validated", errors, "see the gate errors", {"debug": debug, "checks": results})


# ------------------------------------------------------------------ pick estimator (judgement)


def _design_facts(state: SpecialistState) -> dict[str, Any]:
    est: Estimand = state["estimand"]
    checks: list[CheckResult] = state["checks"]
    arms = [r for r in checks if r.name == "arms"]
    b = _block(state["handoff"])
    return {
        "estimand": est.kind,
        "roads": est.roads,
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
    facts = _design_facts(state)
    excluded = set(state.get("excluded_estimators") or [])
    allowed = [e for e in load_estimators()
               if e.applies(estimand=facts["estimand"], treatment=facts["treatment"], outcome=facts["outcome"], adjustment_set=facts["adjustment_set"], roads=facts["roads"])
               and e.name not in excluded]
    if not allowed:
        return _stop("pick_estimator", "no estimator in the catalogue applies to this design", [f"facts: {facts}", f"excluded after failures: {sorted(excluded)}"],
                     "an estimator entry for this estimand, treatment, and outcome kind")
    names = [e.name for e in allowed]
    check_text = "\n".join(f"[{r.address}] {r.level}: {r.detail}" for r in state["checks"])
    errors: list[str] = []
    debug = []
    for _ in range(MAX_MODEL_RETRIES):
        user = P.PICK_USER.format(facts=json.dumps(facts), checks=check_text, estimators="\n\n".join(e.render() for e in allowed),
                                  preferences=render_preferences(allowed), names=", ".join(names),
                                  errors=("\nPREVIOUS ANSWER WAS REJECTED\n" + "\n".join(f"- {e}" for e in errors) + "\n") if errors else "")
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
    est: Estimand = state["estimand"]
    entry = estimator_entry(state["estimator"])
    if entry.estimand in est.roads and entry.estimand != est.kind:  # the pick took another open road: the design records it
        est = est.model_copy(update={"kind": entry.estimand, "adjustment_set": est.adjustment_set if entry.estimand == "backdoor" else []})
    adj = "nonempty" if est.adjustment_set else "empty"
    refuters = [r.name for r in load_refuters() if r.applies(estimand=est.kind, adjustment_set=adj, hidden=est.sensitivity_required)]
    also = entry.also_run if entry.also_run and estimator_entry(entry.also_run).applies(
        estimand=est.kind, treatment="binary", outcome=state["outcome_kind"], adjustment_set=adj) else None
    d = Design(contrasts=state["contrasts"], graph=state["graph"], estimand=est, checks=Checks(results=state["checks"]),
               estimator=entry.name, params=entry.params, also_run=also, refuters=refuters, target_units=state["target_units"])
    run_dir = Path(state["run_dir"])
    (run_dir / "design.json").write_text(d.model_dump_json(indent=2))
    (run_dir / "design.md").write_text(d.render())
    _writer()({"design": d.render()})
    return {"design": d}


# ------------------------------------------------------------------ analyse (fact, fan-out per contrast)


def fan_out_analyse(state: SpecialistState):
    return [Send("analyse", ContrastTask(contrast=c.key, design=state["design"].model_dump(), table_path=state["table_path"]))
            for c in state["contrasts"]]


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
    return _stop("estimate", "the estimator failed to fit and the re-pick failed too", [f"{e.contrast}: {e.error}" for e in failed],
                 "a different estimator entry or a smaller adjustment set")


# ------------------------------------------------------------------ interpret (judgement, fan-out per contrast)


def _addresses(state: SpecialistState, contrast_key: str) -> list[str]:
    d: Design = state["design"]
    out = ["design.estimand.adjustment_set", "design.assumption"] + [b.address for b in state["handoff"].beliefs.values() if b.known() or b.status == "unknown"]
    out += [r.address for r in d.checks.results if r.contrast in (contrast_key, "all")]
    out += [x.address for x in state.get("declines") or []]
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
    lines += [b.render() for b in state["handoff"].beliefs.values() if b.known() or b.status == "unknown"]
    lines += [f"[design.estimand.adjustment_set] adjusted for: {', '.join(names.get(k, k) for k in d.estimand.adjustment_set) or 'nothing (no confounders in the graph)'}",
              f"comparison: {names.get(d.graph.treatment, d.graph.treatment)} = {c.treated!r} versus {c.control!r}; outcome: {names.get(d.graph.outcome, d.graph.outcome)}"]
    for r in d.checks.results:
        if r.contrast in (contrast_key, "all"):
            lines.append(f"[{r.address}] {r.level}: {r.detail}")
    lines += [x.render() for x in state.get("declines") or []]
    for e in state.get("estimates") or []:
        if e.contrast == contrast_key and e.error is None:
            tag = f"estimate:{contrast_key}" + (f".{e.method}" if e.secondary else "")
            lines.append(f"[{tag}.value] {e.value:.4g} ({'secondary' if e.secondary else 'primary'} estimator {e.method}, target {e.target_units})")
            lines.append(f"[{tag}.ci] 95% interval {e.ci_low:.4g} to {e.ci_high:.4g}" if e.ci_low is not None else f"[{tag}.ci] no interval")
            lines.append(f"[{tag}.n] {e.n_treated} treated, {e.n_control} control")
    for r in state.get("refutations") or []:
        if r.contrast == contrast_key:
            lines.append(f"[refute:{contrast_key}.{r.refuter}.passed] {r.passed}  [refute:{contrast_key}.{r.refuter}.new_effect] {r.new_effect}  "
                         f"[refute:{contrast_key}.{r.refuter}.p_value] {r.p_value}  ({r.detail})")
    return "\n".join(lines)


def fan_out_interpret(state: SpecialistState):
    return [Send("interpret", InterpretTask(question=_question(state), contrast=c.key, material=_material(state, c.key),
                                            addresses="\n".join(_addresses(state, c.key)), required="\n".join(_required(state, c.key)), errors="",
                                            primary_value=_latest_primary(state)[c.key].value, tolerance=load_checks().get("effect_tolerance", 0.01)))
            for c in state["contrasts"] if c.key in _latest_primary(state)]


def interpret(task: InterpretTask) -> dict:
    allowed = set(task["addresses"].splitlines())
    required = [a for a in task["required"].splitlines() if a]
    errors: list[str] = []
    debug = []
    parsed = None
    for _ in range(MAX_MODEL_RETRIES):
        user = P.INTERPRET_USER.format(question=task["question"], contrast=task["contrast"], material=task["material"], addresses=task["addresses"],
                                       required="\n".join(required) or "(none)",
                                       errors=("\nPREVIOUS ANSWER WAS REJECTED\n" + "\n".join(f"- {e}" for e in errors) + "\n") if errors else "")
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


def feasibility(state: SpecialistState) -> dict:
    f: Feasibility = state["feasibility"]
    _writer()({"feasibility": f.model_dump()})
    return {}


def figures(state: SpecialistState) -> dict:
    """What this run drew, checked against the addresses it produced: the estimate against its falsifications."""
    h = state["handoff"]
    specs = [PV.effect_and_refutations([e.model_dump() for e in state.get("estimates") or []], [r.model_dump() for r in state.get("refutations") or []], "refute")]
    kept, declines = LF.write(state.get("run_dir"), specs, LF.ok_addresses(h, state, "refute"))
    _writer()({"figures": [s.id for s in kept]})
    return {"figures": [s.model_dump() for s in kept], "declines": declines}


def assemble(state: SpecialistState) -> dict:
    h = state["handoff"]
    names = state.get("columns") or {}
    d: Design | None = state.get("design")
    f: Feasibility | None = state.get("feasibility")
    lines = [f"QUESTION     {_question(state)}", f"LANE         {h.family} → {h.specialist}", f"OUTCOME      {h.outcome}    TREATMENT   {h.treatment}", ""]
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
    result = records.result(state, report)
    _writer()({"report": report})
    return {"report": report, "specialist_result": result}
