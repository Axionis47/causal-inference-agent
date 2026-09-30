"""Diff-in-diff lane nodes. Facts compute; judgements are bounded episodes, gated.

The pack is weighed by code first (the harness's `case`): the person's belief that the paths would have stayed together
meets the pre-trends test by code; a column the person says the change moved is never a control; the controls the pack
allows are honoured; and every place the lane does not take the pack as given is a Decline. The ladder climbs in the order
an analyst reads a before-and-after comparison: who got the change, the clock, the shape of the panel, whether the
comparison group is a fair stand-in and what could break it, the candidate controls all together, where the effect could
differ, the threats, the clustering. Each rung reads the rungs below it and may look at the data through the read-only
tools; every claim cites the pack, a fact it asked for, or a rung below. Stops are typed: a node that cannot go on routes to
"feasibility" with a Feasibility record; a question for the person is the same with a LaneAsk.
Nothing here names a column, a method, or a dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from langgraph.types import Command, Send

from causal_agent.common.addresses import key as _key
from causal_agent.common.contracts import (
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
from causal_agent.families.diff_in_diff.design import DidDesign
from causal_agent.families.diff_in_diff.lane import adapter
from causal_agent.families.diff_in_diff.lane import checks as CK
from causal_agent.families.diff_in_diff.lane import prompts as P
from causal_agent.families.diff_in_diff.lane import shape as SH
from causal_agent.families.diff_in_diff.lane.contracts import (
    Cluster,
    Comparison,
    ControlRelation,
    ControlRoles,
    Controls,
    Design,
    DesignAssessment,
    EstimatorPick,
    Excluded,
    Groups,
    Ladder,
    Periods,
    ShapeFacts,
)
from causal_agent.families.diff_in_diff.lane.knowledge import (
    EstimatorEntry,
    load_beliefs,
    load_checks,
    load_estimators,
    load_placebos,
    pick_inference,
    render_preferences,
)
from causal_agent.families.diff_in_diff.lane.knowledge import (
    estimator as estimator_entry,
)
from causal_agent.families.diff_in_diff.lane.knowledge import (
    placebo as placebo_entry,
)
from causal_agent.families.diff_in_diff.lane.state import PlaceboTask, SpecialistState
from causal_agent.lane import asks, intake, records
from causal_agent.lane import case as C
from causal_agent.lane import figures as LF
from causal_agent.lane import ladder as LAD
from causal_agent.lane import nodes as L
from causal_agent.lane import verify as V
from causal_agent.lane import words as W
from causal_agent.lane.episode import EpisodeLog, run_episode
from causal_agent.lane.ladder import Heterogeneity, Threat, Threats
from causal_agent.lane.nodes import MAX_MODEL_RETRIES, MAX_PICK_ATTEMPTS, MAX_REVISIONS
from causal_agent.profile.datasets import dataset_entries
from causal_agent.viz.postviz import common as PV

MAX_LEVELS = 100  # levels shown to the model; a level list cut short once hid California (state 5) behind 30 string-sorted numbers

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


def _feas(stage: str, reason: str, facts: list[str], fix: str) -> Feasibility:
    return Feasibility(stage=stage, reason=reason, facts=facts, what_would_fix=fix)


def _block(h: Handoff) -> DidDesign | None:
    return h.design if isinstance(h.design, DidDesign) else None


def _sorted_levels(col: pd.Series) -> list:
    vals = col.dropna().unique()
    try:
        return sorted(vals, key=lambda v: float(v))
    except (TypeError, ValueError):
        return sorted(vals, key=lambda v: str(v))


# ------------------------------------------------------------------ load (fact) and the case (fact)


def load(state: SpecialistState) -> Command:
    h = state["handoff"]
    if not h.treatment:
        return _stop("load", "the hand-off names no treatment", [], "a hand-off whose columns exist in the file")
    entry = dataset_entries().get(h.pack_name) or {}
    b = _block(h)
    unit_col = _key(b.unit) if b and b.unit else (_key(entry["entity"][0]) if entry.get("entity") else None)
    cluster_col = _key(b.cluster_level) if b and b.cluster_level else None
    try:
        it = intake.load(h, "did", extra=[c for c in (unit_col, cluster_col) if c])
    except intake.IntakeStop as e:
        return Command(goto="feasibility", update={"feasibility": e.feasibility})
    if unit_col and unit_col not in it.columns:
        unit_col = None
    if cluster_col and cluster_col not in it.columns:
        cluster_col = None
    t, y = _key(h.treatment), _key(h.outcome)
    table = it.table
    if not pd.api.types.is_numeric_dtype(table[y]):
        return _stop("load", "the outcome is not numeric", [f"outcome {y} is {table[y].dtype}"], "a numeric outcome", {"declines": it.declines})
    target = load_checks()["target_units"].get(h.scope.target)
    if target is None:
        return _stop(
            "load",
            f"target '{h.scope.target}' is not supported by this lane",
            [],
            "a question asking for the average effect, or the effect on the treated",
            {"declines": it.declines},
        )
    levels = [str(v) for v in _sorted_levels(table[t])][:MAX_LEVELS]
    _writer()(
        {
            "load": {
                "rows": len(table),
                "columns": list(it.columns),
                "target_units": target,
                "unit_column": unit_col,
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
            "unit_column": unit_col,
            "cluster_column": cluster_col,
            "declines": it.declines,
            "check_facts": {"intake": it.facts} if it.facts else {},
            "target_units": target,
            "group_levels": levels,
            "ladder": Ladder(),
            "revisions": 0,
            "pick_attempts": 0,
            "excluded_estimators": [],
            "applied_revisions": [],
        },
    )


# ------------------------------------------------------------------ the ladder


def _ladder(state: SpecialistState) -> Ladder:
    lad = state.get("ladder")
    return lad if isinstance(lad, Ladder) else Ladder()


def _budget(node: str) -> int:
    return int((load_checks().get("episode_budget") or {}).get(node, 4))


def _treated_mask(state: SpecialistState) -> pd.Series | None:
    """The treated rows once the groups are set: the group column at the treated level."""
    g = state.get("groups")
    if g is None:
        return None
    return _table(state)[g.column].astype(str) == str(g.treated_level)


def _resolver(h: Handoff, log: EpisodeLog, ladder: Ladder):
    """What a rung may cite: the pack, the facts this episode asked for, and the rungs below."""

    def ok(address: str) -> bool:
        return h.resolve(address) or log.resolve(address) or ladder.resolve(address)

    return ok


# rung 0: who got the change (a fact from the pack, else a judgement)


def _block_groups(h: Handoff) -> Groups | None:
    b = _block(h)
    tg = b.treated_group if b else {}
    if not (tg.get("column") and tg.get("level") is not None):
        return None
    return Groups(
        column=_key(tg["column"]),
        treated_level=str(tg["level"]),
        reason="the pack names the column and the level that mean the unit got the change",
        cites=_cites(h, "claim:assignment.treatment_column", "claim:assignment.treated_level"),
        by="pack",
    )


def _groups_errors(state: SpecialistState, parsed: Groups, h: Handoff, ok: Any) -> list[str]:
    parsed.column = _key(parsed.column)
    table_cols = set(pd.read_csv(state["table_path"], nrows=0).columns)
    errors: list[str] = []
    if parsed.column not in table_cols:
        errors.append(f"{parsed.column!r} is not a column in the table")
    else:
        col = pd.read_csv(state["table_path"], usecols=[parsed.column])[parsed.column]
        observed = set(col.astype(str))
        if str(parsed.treated_level) not in observed:
            errors.append(
                f"level {parsed.treated_level!r} is not observed in {parsed.column!r}; observed: {[str(v) for v in _sorted_levels(col)][:MAX_LEVELS]}"
            )
        elif len(observed) < 2:
            errors.append(f"{parsed.column!r} has a single level; nothing to compare")
    return errors + [f"{c} is not an address you may cite" for c in parsed.cites if not ok(c)]


def groups(state: SpecialistState) -> dict:
    h = state["handoff"]
    t, _, _ = _keys(state)
    lad = _ladder(state)
    declines: list[Decline] = []
    block = _block_groups(h)
    if block is not None:  # a fact from the pack; the same checks apply, and the model is asked only if they fail
        errors = _groups_errors(state, block, h, h.resolve)
        if not errors:
            _writer()({"groups": block.model_dump()})
            return {"groups": block, "ladder": lad.model_copy(update={"groups": block}), "declines": declines}
        declines.append(
            Decline(
                stage="groups",
                kind="replaced",
                about="claim:assignment.treated_level",
                pack_value=f"{block.column} = {block.treated_level!r}",
                check="groups.level_observed",
                reason="; ".join(errors),
            )
        )
        _writer()({"groups": {"pack_block_rejected": errors}})
    user = P.GROUPS_USER.format(
        question=_question(state),
        frame=L.frame_text(state),
        dataset_card=h.render_dataset(),
        changes=h.render_change(),
        treatment_card=_card(h, t),
        levels=", ".join(repr(v) for v in state["group_levels"]),
    )
    first = [f"PREVIOUS ANSWER WAS REJECTED\n- {d.reason}" for d in declines]  # the pack's failure is what the model is told first
    rec, log, thoughts, errors = run_episode(
        Groups,
        P.GROUPS_SYSTEM,
        user + ("\n" + "\n".join(first) if first else ""),
        tools=L.data_tools(state, None),
        budget=_budget("groups"),
        gate=lambda r, lg: _groups_errors(state, r, h, _resolver(h, lg, lad)),
        node="groups",
    )
    if rec is None:
        return {
            "feasibility": _feas("groups", "could not name who got the change", errors, "a column and level the notes tie to the change"),
            "debug": thoughts,
            "declines": declines,
            "episodes": {"groups": log},
        }
    rec = rec.model_copy(update={"by": "judgement"})
    _writer()({"groups": rec.model_dump()})
    return {"groups": rec, "ladder": lad.model_copy(update={"groups": rec}), "debug": thoughts, "declines": declines, "episodes": {"groups": log}}


def after_groups(state: SpecialistState) -> str:
    return "feasibility" if state.get("feasibility") else "periods"


# rung 1: the clock (a fact from the pack, else a judgement, else a question back)


def _block_periods(h: Handoff) -> Periods | None:
    b = _block(h)
    if not (b and b.time and b.change_period):
        return None
    lo = hi = None
    if h.scope.window and (w := intake.parse_window(h.scope.window)):
        lo, _, hi, _ = w
    return Periods(
        kind="long",
        time_column=_key(b.time),
        first_post=str(b.change_period),
        window_start=lo,
        window_end=hi,
        reason="the pack names the period column and the first period at or after the change",
        cites=_cites(h, "claim:change.date_column", "claim:change.period_value"),
        by="pack",
    )


def _periods_errors(state: SpecialistState, parsed: Periods, ok: Any) -> list[str]:
    table_cols = set(pd.read_csv(state["table_path"], nrows=0).columns)
    errors: list[str] = []
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
    return errors + [f"{c} is not an address you may cite" for c in parsed.cites if not ok(c)]


def periods(state: SpecialistState) -> dict:
    h = state["handoff"]
    g: Groups = state["groups"]
    lad = _ladder(state)
    _, y, rel = _keys(state)
    table_cols = set(pd.read_csv(state["table_path"], nrows=0).columns)
    declines: list[Decline] = []

    def done(p: Periods, extra: dict[str, Any]) -> dict:
        if p.kind == "long" and h.scope.window and not (p.window_start or p.window_end) and (w := intake.parse_window(h.scope.window)):
            p.window_start, _, p.window_end, _ = w  # the table is already windowed by intake; the design says so
        _writer()({"periods": p.model_dump(exclude_none=True)})
        return {"periods": p, "ladder": lad.model_copy(update={"periods": p}), "declines": declines, **extra}

    block = _block_periods(h)
    if block is not None:
        errors = _periods_errors(state, block, h.resolve)
        if not errors:
            return done(block, {})
        declines.append(
            Decline(
                stage="periods",
                kind="replaced",
                about="claim:change.period_value",
                pack_value=f"{block.time_column} from {block.first_post!r}",
                check="periods.first_post_in_table",
                reason="; ".join(errors),
            )
        )
        _writer()({"periods": {"pack_block_rejected": errors}})
    time_cards = "\n\n".join(_card(h, k) for k in rel if k not in (g.column,))
    user = P.PERIODS_USER.format(
        question=_question(state),
        changes=h.render_change(),
        dataset_card=h.render_dataset(),
        outcome_card=_card(h, y),
        time_cards=time_cards or "(none besides the outcome)",
    )
    last: dict[str, Periods | None] = {"parsed": None}

    def gate(r: Periods, lg: EpisodeLog) -> list[str]:
        last["parsed"] = r
        return _periods_errors(state, r, _resolver(h, lg, lad))

    rec, log, thoughts, errors = run_episode(
        Periods, P.PERIODS_SYSTEM, user, tools=L.data_tools(state, _treated_mask(state)), budget=_budget("periods"), gate=gate, node="periods"
    )
    extra: dict[str, Any] = {"debug": thoughts, "episodes": {"periods": log}}
    if rec is not None:
        return done(rec.model_copy(update={"by": "judgement"}), extra)
    parsed = last["parsed"]
    if (
        parsed is not None and parsed.kind == "long" and parsed.time_column in table_cols
    ):  # a time column the file has but a first period nobody could settle: one question, not a stop
        time_name = (state.get("columns") or {}).get(parsed.time_column, parsed.time_column)
        ask = LaneAsk(
            address="claim:change.period_value",
            question=f"Which value of {time_name!r} is the first period at or after the change?",
            because="the period column is known but not the first period after the change",
        )
        f = Feasibility(
            stage="ask",
            reason="the first period at or after the change could not be settled from the notes",
            facts=errors,
            what_would_fix=f"an answer to [{ask.address}]",
        )
        return {"feasibility": f, "ask": ask.model_copy(update={"stage": "periods"}), "declines": declines, **extra}
    return {
        "feasibility": _feas("periods", "could not locate before and after", errors, "a time column or a before and an after measure the notes describe"),
        "declines": declines,
        **extra,
    }


def after_periods(state: SpecialistState) -> str:
    return "feasibility" if state.get("feasibility") else "shape_table"


# rung 2: the shape (fact)


def _candidates(state: SpecialistState) -> list[str]:
    g: Groups = state["groups"]
    p: Periods = state["periods"]
    _, y, rel = _keys(state)
    return [k for k in rel if k not in (g.column, y, p.time_column, p.before_column, p.after_column, state.get("unit_column"), state.get("cluster_column"))]


def shape_table(state: SpecialistState) -> Command:
    h = state["handoff"]
    g: Groups = state["groups"]
    p: Periods = state["periods"]
    t, y, _ = _keys(state)
    table = pd.read_csv(state["table_path"])
    candidates = _candidates(state)
    tg = (_block(h).treated_group if _block(h) else {}) or {}
    cohort_col = _key(tg["cohort_column"]) if tg.get("cohort_column") else None
    try:
        panel, facts = SH.canonical(
            table,
            g,
            p,
            y,
            candidates,
            unit_column=state.get("unit_column"),
            cluster_column=state.get("cluster_column"),
            cohort_column=cohort_col if cohort_col and cohort_col in table.columns else None,
        )
    except SH.ShapeError as ex:
        return _stop("shape_table", ex.reason, ex.facts, ex.fix)
    panel_path = Path(state["run_dir"]) / "panel.csv"
    panel.to_csv(panel_path, index=False)
    # the control level from the settled group column, and the pack's word on it when the group column is the treatment column
    levels = [str(v) for v in _sorted_levels(table[g.column])]
    others = [v for v in levels if v != str(g.treated_level)]
    control = (
        str(h.control_level)
        if h.control_level is not None and g.column == t and str(h.control_level) in others
        else (others[0] if len(others) == 1 else "other")
    )
    contrast = Contrast(control=control, treated=str(g.treated_level), reason=g.reason, cites=g.cites)
    # what the pack said about the panel against what the table shows: a difference is recorded, never hidden
    declines: list[Decline] = []
    b = _block(h)
    if b is not None:
        if b.pre_periods is not None and b.pre_periods != facts.periods_pre:
            declines.append(
                Decline(
                    stage="shape_table",
                    kind="replaced",
                    about="design.pre_periods",
                    pack_value=str(b.pre_periods),
                    took=str(facts.periods_pre),
                    check="shape.pre_periods",
                    reason="counted on the table after the filter and the window",
                )
            )
        if b.post_periods is not None and b.post_periods != facts.periods_post:
            declines.append(
                Decline(
                    stage="shape_table",
                    kind="replaced",
                    about="design.post_periods",
                    pack_value=str(b.post_periods),
                    took=str(facts.periods_post),
                    check="shape.post_periods",
                    reason="counted on the table after the filter and the window",
                )
            )
        if b.staggered is False and facts.cohorts > 1:
            declines.append(
                Decline(
                    stage="shape_table",
                    kind="replaced",
                    about="design.staggered",
                    pack_value="false",
                    took=f"{facts.cohorts} first-treated periods",
                    check="shape.cohorts",
                    reason="the units did not all get the change in the same period",
                )
            )
    for pr in h.probes:
        if pr.family == "diff_in_diff" and pr.name == "pre_periods" and pr.value is not None and int(pr.value) != facts.periods_pre:
            declines.append(
                Decline(
                    stage="shape_table",
                    kind="replaced",
                    about=pr.address,
                    pack_value=str(int(pr.value)),
                    took=str(facts.periods_pre),
                    check="shape.pre_periods",
                    reason="the desk counted on the whole file; the lane counts on the table after the filter and the window",
                )
            )
    _writer()({"shape": facts.model_dump(exclude={"time_values"}), "declines": [d.render() for d in declines]})
    return Command(
        goto="comparison",
        update={
            "panel_path": str(panel_path),
            "shape": facts,
            "contrast": contrast,
            "declines": declines,
            "ladder": _ladder(state).model_copy(update={"shape": facts}),
        },
    )


# rung 3: the comparison (a judgement from the story and the facts)


def comparison(state: SpecialistState) -> Command:
    h = state["handoff"]
    lad = _ladder(state)
    g: Groups = state["groups"]
    s: ShapeFacts = state["shape"]

    def gate(r: Comparison, log: EpisodeLog) -> list[str]:
        ok = _resolver(h, log, lad)
        errs: list[str] = []
        if not r.why.strip():
            errs.append("say why, from the story and the facts")
        if not r.fair and not r.risks:
            errs.append("a comparison judged unfair names at least one risk")
        for risk in r.risks:
            if not risk.cites:
                errs.append(f"risk {risk.name}: no citation")
            errs += [f"risk {risk.name}: {e}" for e in V.cites_resolve(risk.cites, h, ok)]
        return errs + V.cites_resolve(r.cites, h, ok)

    groups_text = f"{g.column} = {g.treated_level!r} treated, every other level control; {s.units_treated} treated units, {s.units_control} control; {s.periods_pre} periods before the change, {s.periods_post} after"
    user = P.COMPARISON_USER.format(question=_question(state), frame=L.frame_text(state), groups=groups_text, errors="")
    rec, log, thoughts, errors = run_episode(
        Comparison, P.COMPARISON_SYSTEM, user, tools=L.data_tools(state, _treated_mask(state)), budget=_budget("comparison"), gate=gate, node="comparison"
    )
    if rec is None:
        return _stop(
            "comparison",
            "the comparison could not be judged",
            errors,
            "a story that says how the two groups came to differ",
            {"debug": thoughts, "episodes": {"comparison": log}},
        )
    _writer()({"comparison": [f"[{a}] {text}" for a, text in rec.lines()]})
    return Command(goto="controls", update={"ladder": lad.model_copy(update={"comparison": rec}), "debug": thoughts, "episodes": {"comparison": log}})


# rung 4: the controls, placed together (a judgement only for what the pack leaves open)


def settled_claims(h: Handoff, k: str, case: C.Case) -> tuple[dict[str, bool], dict[str, str]]:
    """What the pack settles about a candidate control: the person's word on whether the change could have moved it, or
    that it was fixed before the change. A column the change could have moved is never a control, so nothing is asked."""
    a = f"col:{k}"
    if case.is_fact(f"{a}.moved_by_change"):
        moved = bool(case.fact(f"{a}.moved_by_change"))
        claims = {"affected_by_treatment": moved} | ({"usable_as_control": False} if moved else {})
        return claims, {c: f"{a}.moved" for c in claims}
    if case.fact(f"{a}.when") == "before":
        return {"affected_by_treatment": False}, {"affected_by_treatment": f"{a}.when"}
    return {}, {}


def fact_relation(h: Handoff, k: str, case: C.Case) -> ControlRelation | None:
    claims, cites = settled_claims(h, k, case)
    if "affected_by_treatment" in claims and "usable_as_control" in claims:
        name = h.column(k).name if h.column(k) else k
        return ControlRelation(
            column=k, reasons=[Cited(reason=f"{name}: the person says the change could have moved it", cites=[cites["affected_by_treatment"]])], **claims
        )
    return None


_settled_text, apply_settled = L.make_settled(settled_claims)


def _column_block(h: Handoff, k: str, case: C.Case) -> str:
    return f"COLUMN {k!r}\n{_card(h, k)}\n{_settled_text(h, k, case)}"


def _relation_errors(r: ControlRelation, ok: Any) -> list[str]:
    errs: list[str] = []
    if (r.affected_by_treatment or r.usable_as_control) and not r.reasons:
        errs.append("claims marked true but no reasons given")
    if r.affected_by_treatment and r.usable_as_control:
        errs.append("cannot be both changed by the treatment and a control")
    for reason in r.reasons:
        if not reason.cites:
            errs.append("a reason has no citation")
        errs += [f"{c} is not an address you may cite" for c in reason.cites if not ok(c)]
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


def controls(state: SpecialistState) -> Command:
    h = state["handoff"]
    case = _case(state)
    lad = _ladder(state)
    asked = [k for k in _candidates(state) if fact_relation(h, k, case) is None]
    if not asked:
        return Command(goto="merge_controls", update={"ladder": lad.model_copy(update={"controls": ControlRoles(items=[])})})

    def gate(r: ControlRoles, log: EpisodeLog) -> list[str]:
        ok = _resolver(h, log, lad)
        errs = _presence_errors(asked, [x.column for x in r.items])
        for x in r.items:
            if x.column in asked:
                errs += [f"{x.column}: {e}" for e in _relation_errors(apply_settled(x, h, case), ok)]
        return errs

    user = P.CONTROLS_USER.format(
        question=_question(state), frame=L.frame_text(state), count=len(asked), columns="\n\n".join(_column_block(h, k, case) for k in asked), errors=""
    )
    rec, log, thoughts, errors = run_episode(
        ControlRoles, P.CONTROLS_SYSTEM, user, tools=L.data_tools(state, _treated_mask(state)), budget=_budget("controls"), gate=gate, node="controls"
    )
    if rec is None:
        return _stop(
            "controls",
            "the candidate controls could not be placed",
            errors,
            "clearer column notes about what moves over time and what the treatment touches",
            {"debug": thoughts, "episodes": {"controls": log}},
        )
    out = ControlRoles(items=[apply_settled(x, h, case) for x in rec.items], unsure=rec.unsure)
    _writer()({"controls_rung": [f"[{a}] {text}" for a, text in out.lines()]})
    return Command(goto="merge_controls", update={"ladder": lad.model_copy(update={"controls": out}), "debug": thoughts, "episodes": {"controls": log}})


def _latest(state: SpecialistState) -> dict[str, ControlRelation]:
    h = state["handoff"]
    case = _case(state)
    latest: dict[str, ControlRelation] = {k: r for k in _candidates(state) if (r := fact_relation(h, k, case)) is not None}
    lad = _ladder(state)
    for r in lad.controls.items if lad.controls is not None else []:
        latest[r.column] = apply_settled(r, h, case)
    return latest


# ------------------------------------------------------------------ merge + verify (facts)


def merge_controls(state: SpecialistState) -> dict:
    h = state["handoff"]
    shape = state["shape"]
    latest = _latest(state)
    b = _block(h)
    allowed = {_key(c) for c in b.controls_allowed} if b else set()
    included: list[str] = []
    dropped: list[Excluded] = []
    excluded: list[Excluded] = []
    for k in _candidates(state):
        r = latest.get(k)
        if r is None:
            continue
        card = h.column(k)
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
        # then the person's word, then the judgements
        if r.affected_by_treatment:
            why = (
                "the person says the change could have moved it [col:%s.moved]" % k
                if _case(state).fact(f"col:{k}.moved_by_change") is True
                else "could be changed by the treatment; adjusting for it would remove part of the effect"
            )
            excluded.append(Excluded(column=k, why=why))
        elif r.usable_as_control:
            if allowed and k not in allowed:
                excluded.append(Excluded(column=k, why="the pack does not allow it as a control [design.controls_allowed]"))
            else:
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
    """A relation for every candidate; the per-column checks were made in the controls rung's gate."""
    latest = _latest(state)
    missing = [k for k in _candidates(state) if k not in latest]
    if missing:
        return _stop(
            "verify", "the control relations could not be made to pass verification", ["no relation for " + ", ".join(missing)], "clearer column notes"
        )
    _writer()({"verify": "ok"})
    return Command(goto="heterogeneity")


# rung 5: heterogeneity (a judgement over the candidates; the target by code)


def _modifier_candidates(state: SpecialistState) -> dict[str, list[str]]:
    """Every unit trait the controls rung, or the person, marked as one the effect could differ by, with what marked it. A trait is
    fixed within a unit, read off the panel itself."""
    case = _case(state)
    lad = _ladder(state)
    panel = pd.read_csv(state["panel_path"])
    out: dict[str, list[str]] = {}

    def is_trait(k: str) -> bool:
        return k in panel.columns and bool((panel.groupby("unit")[k].nunique(dropna=False) <= 1).all())

    for x in lad.controls.items if lad.controls is not None else []:
        if x.modifier_candidate and is_trait(x.column):
            out.setdefault(x.column, []).append(f"ladder:controls.{x.column}")
    for k in _candidates(state):
        if case.fact(f"col:{k}.may_modify") is True and is_trait(k):
            out.setdefault(k, []).append(f"col:{k}.may_modify")
    return out


def heterogeneity(state: SpecialistState) -> Command:
    h = state["handoff"]
    lad = _ladder(state)
    cands = _modifier_candidates(state)
    target = state["target_units"]
    cfg = load_checks().get("modifiers") or {}
    max_m = int(cfg.get("max", 3))
    if not cands:
        het = Heterogeneity(modifiers=[], why="no unit trait was marked as one the effect could differ by", target_units=target, by="code")
        _writer()({"heterogeneity": het.model_dump()})
        return Command(goto="threats", update={"ladder": lad.model_copy(update={"heterogeneity": het})})

    def gate(r: Heterogeneity, log: EpisodeLog) -> list[str]:
        ok = _resolver(h, log, lad)
        errs = [f"at most {max_m} modifiers; {len(r.modifiers)} named"] if len(r.modifiers) > max_m else []
        seen: set[str] = set()
        for m in r.modifiers:
            k = _key(m.column)
            if k not in cands:
                errs.append(f"{m.column!r} is not a candidate; the candidates are {', '.join(sorted(cands))}")
            elif k in seen:
                errs.append(f"{m.column!r} named twice")
            seen.add(k)
            if not m.cites:
                errs.append(f"{m.column}: no citation")
            errs += V.cites_resolve(m.cites, h, ok)
        return errs + V.cites_resolve(r.cites, h, ok)

    candidates = "\n\n".join(f"{_card(h, k)}\n  marked as a candidate by [{'], ['.join(c)}]" for k, c in cands.items())
    user = P.HETEROGENEITY_USER.format(question=_question(state), frame=L.frame_text(state), candidates=candidates, max_modifiers=max_m, errors="")
    rec, log, thoughts, errors = run_episode(
        Heterogeneity,
        P.HETEROGENEITY_SYSTEM.replace("{max_modifiers}", str(max_m)),
        user,
        tools=L.data_tools(state, _treated_mask(state)),
        budget=_budget("heterogeneity"),
        gate=gate,
        node="heterogeneity",
    )
    if rec is None:
        return _stop(
            "heterogeneity",
            "the modifiers could not be chosen",
            errors,
            "a story that says where the effect could differ",
            {"debug": thoughts, "episodes": {"heterogeneity": log}},
        )
    het = rec.model_copy(
        update={"modifiers": [m.model_copy(update={"column": _key(m.column)}) for m in rec.modifiers], "target_units": target, "by": "judgement"}
    )
    _writer()({"heterogeneity": het.model_dump()})
    return Command(goto="threats", update={"ladder": lad.model_copy(update={"heterogeneity": het}), "debug": thoughts, "episodes": {"heterogeneity": log}})


# rung 6: the threats (code, from the pack and the comparison rung)


def threats(state: SpecialistState) -> dict:
    """The risks every design carries, from the pack, then the ones the comparison rung named from the story. Each is a flag the
    assessment must answer and the interpretation must cite."""
    h = state["handoff"]
    lad = _ladder(state)
    _, y, _ = _keys(state)
    items = LAD.pack_threats(h, _case(state), y, state.get("columns") or {}, outcome_timing=False)  # the outcome is seen on both sides of the change
    for r in lad.comparison.risks if lad.comparison is not None else []:
        items.append(Threat(name=r.name, level="soft", text=r.reason, cites=[f"ladder:comparison.risk.{r.name}", *r.cites]))
    th = Threats(items=items)
    _writer()({"threats": [f"[{a}] {text}" for a, text in th.lines()]})
    return {"ladder": lad.model_copy(update={"threats": th})}


# ------------------------------------------------------------------ checks (fact)


def check_design(state: SpecialistState) -> dict:
    panel = pd.read_csv(state["panel_path"])
    s: ShapeFacts = state["shape"]
    facts_in = _facts(state)
    robust = any(e.engine != "feols" for e in _applicable(facts_in)) if s.cohorts > 1 else True
    heterogeneity = adapter.cohort_heterogeneity(panel) if s.cohorts > 1 and s.never_treated_exists else None
    results, facts = CK.run_checks(
        panel, s, state["controls"].included, state["contrast"].key, load_checks(), robust_available=robust, heterogeneity=heterogeneity
    )
    W.say(results, load_checks(), state.get("columns") or {})  # the sentence before the number, for the reader
    results += C.as_checks(_case(state))
    more, declines = LAD.checks_and_declines(_ladder(state), _ladder(state).threats, state.get("declines") or [])
    results += more
    _writer()({"checks": [f"{r.level} {r.address} {r.detail}" for r in results]})
    return {"checks": results, "check_facts": facts, "declines": declines}


# ------------------------------------------------------------------ assess (the yaml first, then a judgement, repair loop)


def _design_text(state: SpecialistState) -> str:
    return state["controls"].render() + "\nshape: " + json.dumps(state["shape"].model_dump(exclude={"time_values"}))


def assess(state: SpecialistState) -> Command:
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
    flags, hard = checks.flags, checks.hard
    if not flags:
        return Command(goto="pick_estimator", update={"checks": results})
    candidates = set(_candidates(state))
    absorbed = {x.column for x in state["controls"].dropped_fixed}
    flag_text = "\n".join(f"[{r.address}] {r.level.upper()}: {r.detail}" for r in flags)
    errors: list[str] = []
    debug = []
    for _ in range(MAX_MODEL_RETRIES):
        user = P.ASSESS_USER.format(frame=L.frame_text(state), question=_question(state), design=_design_text(state), flags=flag_text, errors=_rejected(errors))
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
                elif rv.change == "add_control" and rv.column in absorbed:
                    errors.append(f"{rv.column!r} is absorbed by the fixed effects and cannot be a control")
        for c in parsed.cites:
            if not (any(c == r.address for r in results) or h.resolve(c)):
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
                "a comparison group that moved like the treated group before the change, or a design that does not need one",
                {"assessment": parsed, "debug": debug, "checks": results},
            )
        n = state.get("revisions", 0) + 1
        if n > MAX_REVISIONS:
            return _stop(
                "assess",
                "three revisions did not clear the flags",
                [f"{r.address}: {r.detail}" for r in flags],
                "a different design",
                {"assessment": parsed, "debug": debug, "revisions": n, "checks": results},
            )
        return Command(
            goto="merge_controls",
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


def _facts(state: SpecialistState) -> dict[str, Any]:
    """The facts the catalogues apply by: the shape, the adoption pattern, the controls in play."""
    s = state["shape"]
    return {
        "kind": s.kind,
        "units_treated": s.units_treated,
        "units_control": s.units_control,
        "periods_pre": s.periods_pre,
        "periods_post": s.periods_post,
        "cohorts": s.cohorts,
        "never_treated": s.never_treated_exists,
        "clusters": s.clusters,
        "controls": state["controls"].included,
    }


def _applicable(facts: dict[str, Any]) -> list[EstimatorEntry]:
    return [e for e in load_estimators() if e.applies(**facts)]


def pick_estimator(state: SpecialistState) -> Command:
    facts = _facts(state)
    excluded = set(state.get("excluded_estimators") or [])
    allowed = [e for e in _applicable(facts) if e.name not in excluded]
    if not allowed:
        return _stop(
            "pick_estimator",
            "no estimator in the catalogue applies to this design",
            [f"facts: {facts}", f"excluded after failures: {sorted(excluded)}"],
            "an estimator entry for this adoption pattern and period count",
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
    h = state["handoff"]
    s = state["shape"]
    entry = estimator_entry(state["estimator"])
    controls = state["controls"].included
    also = None
    if entry.also_run:
        alt = estimator_entry(entry.also_run)
        if alt.applies(**_facts(state)):
            also = alt
    inf = pick_inference(units_treated=s.units_treated, kind=s.kind)
    vcov: Any = inf.vcov
    declines: list[Decline] = []
    b = _block(h)
    if b is not None and b.cluster_level and _key(b.cluster_level) != state.get("unit_column"):
        panel = pd.read_csv(state["panel_path"])
        clustered = isinstance(inf.vcov, dict) and "CRV1" in inf.vcov
        if "cluster" in panel.columns and clustered and (panel.groupby("unit")["cluster"].nunique() <= 1).all():
            vcov = {"CRV1": "cluster"}
        else:
            why = (
                "the inference for this shape does not cluster"
                if not clustered
                else "the column is not in the table"
                if "cluster" not in panel.columns
                else "a unit sits in more than one of its groups"
            )
            declines.append(
                Decline(
                    stage="freeze_design",
                    kind="declined",
                    about="design.cluster_level",
                    pack_value=b.cluster_level,
                    took=str(inf.vcov),
                    check="inference.cluster_column",
                    reason=f"errors were not clustered at the level the pack names: {why}",
                )
            )
    units = s.units_treated + s.units_control
    # the placebos refit the feols shape; an estimator on another engine gets none until its own falsifications are wired
    placebos = [p.name for p in load_placebos() if p.applies(units=units, periods_pre=s.periods_pre)] if entry.engine == "feols" else []
    lad = _ladder(state)
    level = "the pack's cluster column" if vcov == {"CRV1": "cluster"} else "the unit" if isinstance(vcov, dict) else f"none: {inf.name}"
    cluster = Cluster(level=level, why=f"{inf.name}: {s.units_treated} treated units on a {s.kind} table" + ("; " + declines[0].reason if declines else ""))
    d = Design(
        contrast=state["contrast"],
        groups=state["groups"],
        periods=state["periods"],
        shape=s,
        controls=state["controls"],
        checks=Checks(results=state["checks"]),
        estimator=entry.name,
        engine=entry.engine,
        formula=adapter.describe_spec(entry, controls),
        also_run=also.name if also else None,
        also_formula=adapter.describe_spec(also, controls) if also else None,
        inference=inf.name,
        vcov=vcov,
        placebos=placebos,
        target_units=state["target_units"],
        modifiers=[m.column for m in lad.heterogeneity.modifiers] if lad.heterogeneity is not None else [],
    )
    run_dir = Path(state["run_dir"])
    (run_dir / "design.json").write_text(d.model_dump_json(indent=2))
    (run_dir / "design.md").write_text(d.render())
    _writer()({"design": d.render(), "declines": [x.render() for x in declines]})
    return {"design": d, "declines": declines, "ladder": lad.model_copy(update={"cluster": cluster})}


# ------------------------------------------------------------------ estimate (fact) + placebos (fact, fan-out)


def estimate(state: SpecialistState) -> Command:
    d: Design = state["design"]
    panel = pd.read_csv(state["panel_path"])
    controls = d.controls.included
    fit = adapter.run(estimator_entry(d.estimator), panel, d.vcov, d.contrast.key, d.target_units, controls=controls)
    ests, model = list(fit.estimates), fit.model
    refs: list[Refutation] = []
    dynamic: dict = {str(k): v for k, v in fit.dynamic.items()}
    primary = fit.primary
    if primary.error is None and d.also_run:
        alt = adapter.run(estimator_entry(d.also_run), panel, d.vcov, d.contrast.key, d.target_units, controls=controls, secondary=True)
        ests += alt.estimates
        if alt.dynamic and not dynamic:
            dynamic = {str(k): v for k, v in alt.dynamic.items()}
    if not dynamic:
        dynamic = dict((state.get("check_facts") or {}).get("dynamic") or {})  # the pre-trends fit, when it ran
    if primary.error is None:
        ests += fit.by_cohort  # each cohort's own effect, when the engine reports it, as the effect within a level of "cohort"
        ests += _by_modifier(panel, d)
    inf = pick_inference(units_treated=d.shape.units_treated, kind=d.shape.kind)
    if primary.error is None and inf.resample == "wild_bootstrap" and model is not None:
        p = adapter.wild_bootstrap(model, int(inf.params.get("reps", 999)), int(inf.params.get("seed", 7)))
        refs.append(
            Refutation(
                contrast=d.contrast.key,
                refuter="wild_bootstrap",
                kind="sensitivity",
                p_value=p,
                detail=f"wild cluster bootstrap p-value for the effect: {p}" if p is not None else "wild bootstrap failed",
            )
        )
    _writer()({"estimate": [e.model_dump(exclude_none=True) for e in ests]})
    update: dict[str, Any] = {"estimates": ests, "refutations": refs, "dynamic": dynamic}
    if primary.error:
        if state.get("pick_attempts", 0) < MAX_PICK_ATTEMPTS:
            update["excluded_estimators"] = (state.get("excluded_estimators") or []) + [d.estimator]
            return Command(goto="pick_estimator", update=update)
        return _stop("estimate", "the estimator failed to fit and the re-pick failed too", [primary.error], "a different estimator entry", update)
    sends = [Send("placebo", PlaceboTask(name=n, design=d.model_dump(), panel_path=state["panel_path"], observed=primary.value)) for n in d.placebos]
    return Command(goto=sends or "interpret", update=update)


def _modifier_groups(panel: pd.DataFrame, column: str, max_levels: int) -> list[tuple[str, pd.DataFrame]]:
    """The rows of each level of a unit trait: its values when few, quantile bins over the units when it is a number with many."""
    per_unit = panel.groupby("unit")[column].first()
    if pd.api.types.is_numeric_dtype(per_unit) and per_unit.nunique(dropna=True) > max_levels:
        bins = pd.qcut(per_unit, q=max_levels, duplicates="drop")
        return [(str(level), panel[panel["unit"].map(bins) == level]) for level in bins.cat.categories]
    levels = [v for v in per_unit.astype(str).value_counts().index[:max_levels]]
    return [(str(v), panel[panel["unit"].map(per_unit.astype(str)) == v]) for v in levels]


def _by_modifier(panel: pd.DataFrame, d: Design) -> list[Estimate]:
    """The primary estimator run again within each level of each modifier, without that column among the controls. Too few units
    in a group is recorded as the estimate's error, never skipped in silence."""
    cfg = load_checks().get("modifiers") or {}
    floor = int(cfg.get("min_units_per_group", 2))
    entry = estimator_entry(d.estimator)
    out: list[Estimate] = []
    for col in d.modifiers:
        if col not in panel.columns:
            continue
        controls = [c for c in d.controls.included if c != col]
        for level, part in _modifier_groups(panel, col, int(cfg.get("max_levels", 4))):
            n_t = int(part.loc[part["treated"] == 1, "unit"].nunique())
            n_c = int(part.loc[part["treated"] == 0, "unit"].nunique())
            if min(n_t, n_c) < floor:
                out.append(
                    Estimate(
                        contrast=d.contrast.key,
                        method=d.estimator,
                        n_treated=n_t,
                        n_control=n_c,
                        target_units=d.target_units,
                        modifier=col,
                        level=level,
                        error=f"too few units in a group within this level ({n_t} treated, {n_c} control; floor {floor})",
                    )
                )
                continue
            prim = adapter.run(entry, part, d.vcov, d.contrast.key, d.target_units, controls=controls).primary
            out.append(prim.model_copy(update={"modifier": col, "level": level, "secondary": False}))
    return out


def placebo(task: PlaceboTask) -> dict:
    d = Design.model_validate(task["design"])
    panel = pd.read_csv(task["panel_path"])
    entry = placebo_entry(task["name"])
    base = adapter.formula_for(estimator_entry(d.estimator), [])  # placebos refit the bare shape; controls do not change what a placebo tests
    draws: dict = {}
    if task["name"] == "placebo_group":
        r, effects = adapter.placebo_group(base, panel, task["observed"], entry, d.contrast.key)
        draws = {task["name"]: effects}
    else:
        r = adapter.placebo_timing(base, panel, entry, d.contrast.key)
    _writer()({"placebo": {r.refuter: r.detail}})
    return {"refutations": [r], "placebo_draws": draws}


# ------------------------------------------------------------------ interpret (judgement)


def _addresses(state: SpecialistState) -> list[str]:
    d: Design = state["design"]
    c = d.contrast.key
    out = (
        ["design.assumption", "design.periods", "design.controls"]
        + [b.address for b in state["handoff"].beliefs.values() if b.known() or b.status == "unknown"]
        + [r.address for r in d.checks.results]
    )
    out += [x.address for x in state.get("declines") or []]
    out += sorted(_ladder(state).addresses())
    out += [f.address for log in _episodes(state).values() for f in log.facts]
    for e in state.get("estimates") or []:
        if e.error is None:
            tag = _tag(e, d)
            out += [f"{tag}.value", f"{tag}.ci", f"{tag}.n"]
    for r in state.get("refutations") or []:
        out += [f"placebo:{c}.{r.refuter}.p_value", f"placebo:{c}.{r.refuter}.new_effect", f"placebo:{c}.{r.refuter}.passed"]
    return list(dict.fromkeys(out))


def _tag(e: Estimate, d: Design) -> str:
    """The address stem: the primary is estimate:<contrast>; a secondary carries its method; a level of a modifier its own tail."""
    if e.modifier is not None:
        return e.tag
    return f"estimate:{d.contrast.key}" if e.method == d.estimator else f"estimate:{d.contrast.key}.{e.method}"


def _episodes(state: SpecialistState) -> dict[str, EpisodeLog]:
    return {k: v for k, v in (state.get("episodes") or {}).items() if isinstance(v, EpisodeLog)}


def _required(state: SpecialistState) -> list[str]:
    """What the interpretation must cite: every flagged check (the person's flags among them) and the estimate's interval."""
    d: Design = state["design"]
    out = [r.address for r in d.checks.results if r.level != "pass"]
    if any(e.method == d.estimator and e.error is None and e.modifier is None for e in state.get("estimates") or []):
        out.append(f"estimate:{d.contrast.key}.ci")
    out += [f"{e.tag}.value" for e in state.get("estimates") or [] if e.modifier is not None and e.error is None]
    return out


def _material(state: SpecialistState) -> str:
    d: Design = state["design"]
    names = state.get("columns") or {}
    c = d.contrast.key
    p = d.periods
    lines = [
        f"[design.assumption] the design bets on: without the change, the treated units would have moved like the control units; router's reading: {state['handoff'].chosen_assumption}"
    ]
    lines += [b.render() for b in state["handoff"].beliefs.values() if b.known() or b.status == "unknown"]
    lines += [
        "[design.periods] "
        + (
            f"before {names.get(p.before_column, p.before_column)}, after {names.get(p.after_column, p.after_column)}"
            if p.kind == "wide"
            else f"time {names.get(p.time_column, p.time_column)}, first post {p.first_post}; {d.shape.periods_pre} pre periods, {d.shape.periods_post} post"
        ),
        f"[design.controls] {', '.join(names.get(k, k) for k in d.controls.included) or 'none'}",
        f"comparison: {names.get(d.groups.column, d.groups.column)} = {d.contrast.treated!r} versus {d.contrast.control!r}; outcome: {state['handoff'].outcome}; "
        f"{d.shape.units_treated} treated units, {d.shape.units_control} control",
    ]
    for r in d.checks.results:
        lines.append(f"[{r.address}] {r.level}: {r.detail}")
    lines += [x.render() for x in state.get("declines") or []]
    lines += [f"[{a}] {text}" for a, text in _ladder(state).lines()]
    lines += [f.render() for log in _episodes(state).values() for f in log.facts]
    for e in state.get("estimates") or []:
        tag = _tag(e, d)
        if e.modifier is not None and e.error is not None:
            lines.append(f"[{tag}.error] within {names.get(e.modifier, e.modifier)} = {e.level}: {e.error}")
            continue
        if e.error is None:
            what = (
                f"within {names.get(e.modifier, e.modifier)} = {e.level}"
                if e.modifier is not None
                else "primary"
                if e.method == d.estimator
                else ("with controls added one at a time" if e.method.startswith(d.estimator + "+") else "secondary, mean of post-period coefficients")
            )
            lines.append(f"[{tag}.value] {e.value:.4g} ({what}: {e.method}, target {e.target_units})")
            lines.append(f"[{tag}.ci] 95% interval {e.ci_low:.4g} to {e.ci_high:.4g}" if e.ci_low is not None else f"[{tag}.ci] no interval")
            lines.append(f"[{tag}.n] {e.n_treated} treated units, {e.n_control} control")
    for r in state.get("refutations") or []:
        lines.append(
            f"[placebo:{c}.{r.refuter}.passed] {r.passed}  [placebo:{c}.{r.refuter}.new_effect] {r.new_effect}  [placebo:{c}.{r.refuter}.p_value] {r.p_value}  ({r.detail})"
        )
    return "\n".join(lines)


def interpret(state: SpecialistState) -> dict:
    d: Design = state["design"]
    primary = next((e for e in state.get("estimates") or [] if e.method == d.estimator and e.error is None and e.modifier is None), None)
    allowed = set(_addresses(state))
    required = _required(state)
    tol = float(load_checks().get("effect_tolerance", 0.01))
    errors: list[str] = []
    debug = []
    parsed = None
    for _ in range(MAX_MODEL_RETRIES):
        user = P.INTERPRET_USER.format(
            frame=L.frame_text(state),
            question=_question(state),
            contrast=d.contrast.key,
            material=_material(state),
            addresses="\n".join(sorted(allowed)),
            required="\n".join(required) or "(none)",
            errors=_rejected(errors),
        )
        parsed, th = structured(Interpretation, P.INTERPRET_SYSTEM, user, node="interpret")
        debug.append(th)
        parsed.contrast = d.contrast.key
        errors = [f"{c} is not an address you may cite" for c in parsed.cites if c not in allowed]
        if not parsed.cites:
            errors.append("no citations given")
        missing = [a for a in required if a not in parsed.cites]
        if missing:
            errors.append("these must be cited: " + ", ".join(missing))
        if primary is not None and abs(parsed.effect_stated - primary.value) > max(abs(primary.value), 1e-9) * tol:
            errors.append(f"effect_stated {parsed.effect_stated} does not match the primary estimate {primary.value:.4g}")
        if not errors:
            break
    out: dict[str, Any] = {"interpretations": [parsed], "debug": debug}
    if errors:
        out["interpret_errors"] = {d.contrast.key: errors}
    return out


# ------------------------------------------------------------------ feasibility, figures, assemble (facts)


def figures(state: SpecialistState) -> dict:
    """What this run drew, checked against the addresses it produced: the estimate against its placebos, and the effect by
    period from the dynamic fit or the pre-trends fit, whichever ran."""
    from causal_agent.families.diff_in_diff import postviz as PD

    h = state["handoff"]
    c = state["contrast"].key if state.get("contrast") else None
    dyn = state.get("dynamic") or (state.get("check_facts") or {}).get("dynamic") or {}
    pre = next((r.address for r in state.get("checks") or [] if r.name == "pre_trends"), None)
    ests = [e.model_dump() for e in state.get("estimates") or []]
    refs = [r.model_dump() for r in state.get("refutations") or []]
    primary = next((e for e in ests if not e.get("secondary") and e.get("modifier") is None and e.get("error") is None and e.get("value") is not None), None)
    specs = []
    if c and state.get("panel_path") and Path(state["panel_path"]).exists():
        specs.append(PD.paths_with_counterfactual(pd.read_csv(state["panel_path"]), c, primary.get("value") if primary else None))
    if c:
        specs.append(PV.event_study(dyn, contrast=c, draws_on=[pre] if pre else ["design.dynamic"]))
        group = next((r for r in refs if r.get("refuter") == "placebo_group"), None)
        if group is not None:
            specs.append(
                PD.placebo_distribution(
                    (state.get("placebo_draws") or {}).get("placebo_group") or [], primary.get("value") if primary else None, group.get("p_value"), c
                )
            )
    specs.append(PV.effect_and_refutations([e for e in ests if e.get("modifier") is None], refs, "placebo"))
    if c:
        specs.append(PV.effect_by_modifier(ests, c, state.get("columns") or {}))
    kept, declines = LF.write(state.get("run_dir"), specs, LF.ok_addresses(h, state, "placebo"))
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
            lines += [f"    {x.render()}" for x in log.facts]
            lines += [f"    refused {r.tool}({', '.join(f'{k}={v!r}' for k, v in r.args.items())}): {r.reason}" for r in log.refusals]
        lines.append("")
    if d:
        lines += d.render().splitlines() + ["", "RESULTS"]
        for e in state.get("estimates") or []:
            label = f"{e.method:34}" if e.modifier is None else f"  within {names.get(e.modifier, e.modifier)} = {e.level:<16}"[:34].ljust(34)
            if e.error:
                lines.append(f"    {label} FAILED: {e.error}")
            else:
                ci = f" [{e.ci_low:.3g}, {e.ci_high:.3g}]" if e.ci_low is not None else ""
                kind = "within a level" if e.modifier is not None else "primary" if e.method == d.estimator else "secondary"
                lines.append(f"    {label} {e.value:+.4g}{ci}  units={e.n_treated}/{e.n_control}  {kind}")
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
            lines.append(
                f"PERIODS      {p.kind}: "
                + (f"before {p.before_column}, after {p.after_column}" if p.kind == "wide" else f"time {p.time_column}, first post {p.first_post}")
            )
        if s:
            lines.append(f"SHAPE        {s.units_treated} treated units, {s.units_control} control; {s.periods_pre} pre, {s.periods_post} post")
        for r in state.get("checks") or []:
            lines.append(f"CHECK        {r.level:4} {r.address}  {r.detail}")
    if f:
        lines += ["", f"STOPPED AT   {f.stage}", f"REASON       {f.reason}"] + [f"FACT         {x}" for x in f.facts] + [f"WOULD FIX    {f.what_would_fix}"]
    lines += records.report_tail(state)
    debug = state.get("debug") or []
    if any(t.text for t in debug):
        lines += ["", "MODEL THOUGHTS (debug only)"] + [f"  [{t.node}] {t.text.strip()[:2000]}" for t in debug if t.text]
    report = "\n".join(lines)
    extra = {
        "shape": state.get("shape"),
        "controls": state.get("controls"),
        "dynamic": state.get("dynamic") or {},
        "placebo_draws": state.get("placebo_draws") or {},
    }
    records.write(state.get("run_dir"), records.artifacts(state, extra), report)
    result = records.result(
        state,
        report,
        {
            "shape": state.get("shape"),
            "controls": state.get("controls"),
            "dynamic": state.get("dynamic") or {},
            "ladder": [[a, text] for a, text in lad.lines()],
            "facts": [{"address": x.address, "text": x.render().split("] ", 1)[1], "value": x.value} for log in episodes.values() for x in log.facts],
            "relations": [
                {"column": x.column, "stands_for": None, "redundant_with": None, "nested_in": None, "modifier_candidate": x.modifier_candidate}
                for x in (lad.controls.items if lad.controls is not None else [])
            ],
        },
    )
    _writer()({"report": report})
    return {"report": report, "specialist_result": result}
