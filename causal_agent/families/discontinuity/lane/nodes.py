"""Discontinuity lane nodes. Facts compute; judgements are bounded episodes, gated. The ladder climbs in the order an analyst
reads a cutoff design: the score and the line, the shape of the two sides, whether the line is clean and what could break it,
the candidate covariates all together, where the effect at the cutoff could differ, the threats, the window around the line.
Each rung reads the rungs below it and may look at the data through the read-only tools; every claim cites the pack, a fact it
asked for, or a rung below.

The pack is weighed by code first (the harness's `case`): whether anything else switches at the cutoff, whether a unit
could move its score, and whether the score was fixed before the decision meet the density test and the design by code;
the person's word on a column's timing settles what the model would otherwise be asked; the covariates the pack allows
are honoured; and every place the lane does not take the pack as given is a Decline. Stops are typed: a node that cannot
go on routes to "feasibility" with a Feasibility record; a question for the person is the same with a LaneAsk.
Nothing here names a column, a method, a cutoff, or a dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from langgraph.types import Command, Send

from causal_agent.common.addresses import key as _key
from causal_agent.common.contracts import CheckResult, Checks, Cited, Contrast, Decline, Estimate, Feasibility, Handoff, LaneAsk, Refutation
from causal_agent.common.llm import structured
from causal_agent.families.discontinuity.design import RdDesign
from causal_agent.families.discontinuity.lane import adapter
from causal_agent.families.discontinuity.lane import checks as CK
from causal_agent.families.discontinuity.lane import prompts as P
from causal_agent.families.discontinuity.lane import shape as SH
from causal_agent.families.discontinuity.lane.contracts import (
    BalanceFacts,
    Bandwidths,
    CovariateRelation,
    CovariateRoles,
    Covariates,
    DensityFacts,
    Design,
    DesignAssessment,
    EstimatorPick,
    Excluded,
    Ladder,
    Line,
    RDInterpretation,
    Score,
    ShapeFacts,
    Window,
    WindowPick,
)
from causal_agent.families.discontinuity.lane.knowledge import (
    EstimatorEntry,
    load_beliefs,
    load_checks,
    load_estimators,
    load_placebos,
    pick_inference,
    render_preferences,
)
from causal_agent.families.discontinuity.lane.knowledge import (
    estimator as estimator_entry,
)
from causal_agent.families.discontinuity.lane.knowledge import (
    placebo as placebo_entry,
)
from causal_agent.families.discontinuity.lane.state import PlaceboTask, SpecialistState
from causal_agent.lane import asks, intake, records
from causal_agent.lane import case as C
from causal_agent.lane import figures as LF
from causal_agent.lane import ladder as LAD
from causal_agent.lane import nodes as L
from causal_agent.lane import verify as V
from causal_agent.lane import words as W
from causal_agent.lane.episode import EpisodeLog, run_episode
from causal_agent.lane.ladder import Heterogeneity, Threat, Threats
from causal_agent.lane.nodes import MAX_MODEL_RETRIES, MAX_PICK_ATTEMPTS
from causal_agent.profile.datasets import dataset_entries
from causal_agent.viz.postviz import common as PV

CLAIMS = ("predetermined", "affected_by_treatment", "is_outcome_measure")

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


def _block(h: Handoff) -> RdDesign | None:
    return h.design if isinstance(h.design, RdDesign) else None


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


# ------------------------------------------------------------------ load (fact) and the case (fact)


def load(state: SpecialistState) -> Command:
    h = state["handoff"]
    entry = dataset_entries().get(h.pack_name) or {}
    b = _block(h)
    cluster = _key(b.cluster) if b and b.cluster else (_key(entry["entity"][0]) if entry.get("entity") else None)
    try:
        it = intake.load(h, "rd", extra=[cluster] if cluster else None, dropna=False)
    except intake.IntakeStop as e:
        return Command(goto="feasibility", update={"feasibility": e.feasibility})
    if cluster and cluster not in it.columns:
        cluster = None
    y = _key(h.outcome)
    if not pd.api.types.is_numeric_dtype(it.table[y]):
        return _stop("load", "the outcome is not numeric", [f"outcome {y} is {it.table[y].dtype}"], "a numeric outcome", {"declines": it.declines})
    target = _cfg()["target_units"].get(h.scope.target)
    if target is None:
        return _stop(
            "load",
            f"target '{h.scope.target}' is not supported by this lane",
            [],
            "a question asking for the average effect, or the effect on the treated",
            {"declines": it.declines},
        )
    declines = list(it.declines)
    if h.scope.target == "on_treated":  # a cutoff design has no average over the treated: it answers with the effect at the cutoff, and says so
        declines.append(
            Decline(
                stage="load",
                kind="substituted",
                about="scope.target",
                pack_value="on_treated",
                took=target,
                check="target.effect_at_cutoff",
                reason="a cutoff design estimates the effect for units at the cutoff; it has no average over every treated unit",
            )
        )
    _writer()(
        {
            "load": {
                "rows": len(it.table),
                "columns": list(it.columns),
                "target_units": target,
                "cluster_column": cluster,
                "run_dir": str(it.run_dir),
                "declines": [d.render() for d in declines],
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
            "cluster_column": cluster,
            "declines": declines,
            "check_facts": {"intake": it.facts} if it.facts else {},
            "sampled_by_side": bool(b.sampled_by_side) if b else bool(entry.get("sampled_by_side", False)),
            "target_units": target,
            "ladder": Ladder(),
            "pick_attempts": 0,
            "excluded_estimators": [],
        },
    )


# ------------------------------------------------------------------ the ladder


def _ladder(state: SpecialistState) -> Ladder:
    lad = state.get("ladder")
    return lad if isinstance(lad, Ladder) else Ladder()


def _budget(node: str) -> int:
    return int((_cfg().get("episode_budget") or {}).get(node, 4))


def _treated_mask(state: SpecialistState) -> pd.Series | None:
    """The treated side once the score is set: the rows the rule puts on the treated side of the cutoff."""
    sc = state.get("score")
    if sc is None or not sc.column or sc.cutoff is None:
        return None
    x = pd.to_numeric(_table(state)[sc.column], errors="coerce")
    above, incl, c = sc.treated_side == "above", sc.cutoff_value_treated, float(sc.cutoff)
    return (x >= c) if (above and incl) else (x > c) if above else (x <= c) if incl else (x < c)


def _others(state: SpecialistState, canon: pd.DataFrame) -> pd.DataFrame:
    """The candidate columns the canonical table drops (the categories), on the canonical rows, by their own names."""
    raw = _table(state)
    others = [k for k in state.get("candidates") or [] if k in raw.columns and SH.covcol(k) not in canon.columns]
    return raw.loc[canon["row"].to_numpy(), others].reset_index(drop=True) if others else pd.DataFrame(index=range(len(canon)))


def _canon_tools(state: SpecialistState):
    """The read-only tools over the canonical table, so a rung may look by side and near the line: the recentred score as `x`,
    the outcome as `y`, take-up as `t`, the numeric candidates under their canonical names and the other candidates by their own,
    with aliases from the pack's names. The treated side is `x >= 0`; the outcome by side stays refused before the freeze."""
    sc: Score = state["score"]
    _, y, _ = _keys(state)
    canon = _canon(state)
    numeric = [k for k in state.get("candidates") or [] if SH.covcol(k) in canon.columns]
    others = _others(state, canon)
    df = canon.drop(columns=["row"])
    if len(others.columns):
        df = pd.concat([df, others], axis=1)
    aliases: dict[str, str] = {y: "y"}
    if sc.column:
        aliases[sc.column] = "x"
    if sc.takeup_column and "t" in df.columns:
        aliases[sc.takeup_column] = "t"
    if state.get("cluster_column") and "cluster" in df.columns:
        aliases[str(state["cluster_column"])] = "cluster"
    aliases.update({k: SH.covcol(k) for k in numeric})
    return L.data_tools(state, df["x"] >= 0, table_=df, aliases=aliases)


def _resolver(h: Handoff, log: EpisodeLog, ladder: Ladder):
    """What a rung may cite: the pack, the facts this episode asked for, and the rungs below."""

    def ok(address: str) -> bool:
        return h.resolve(address) or log.resolve(address) or ladder.resolve(address)

    return ok


# rung 0: the score and the line (a fact from the pack, else a judgement, else a question back)


def _block_score(h: Handoff) -> Score | None:
    b = _block(h)
    if not (b and b.score and b.cutoff is not None and b.treated_side):
        return None
    tk = b.takeup or {}
    return Score(
        column=_key(b.score),
        cutoff=float(b.cutoff),
        treated_side=b.treated_side,
        cutoff_value_treated=True if b.cutoff_value_treated is None else bool(b.cutoff_value_treated),
        takeup_column=_key(tk["column"]) if tk.get("column") else None,
        takeup_level=str(tk["level"]) if tk.get("column") and tk.get("level") is not None else None,
        reason="the pack names the score, the cutoff, the treated side, and who took the change up",
        cites=_cites(h, "claim:assignment.score_column", "claim:assignment.cutoff", "claim:assignment.treated_side", "claim:assignment.treatment_column"),
        by="pack",
    )


def _block_about(errors: list[str]) -> str:
    """Which pack field the rejected block failed on."""
    text = " ".join(errors)
    if "not a column" in text or "not numeric" in text or "distinct values" in text:
        return "claim:assignment.score_column"
    if "cutoff" in text:
        return "claim:assignment.cutoff"
    if "take-up" in text or "takeup" in text or "level" in text:
        return "claim:assignment.treatment_column"
    return "claim:assignment.treated_side"


def _ask_score(h: Handoff, state: SpecialistState, errors: list[str], extra: dict[str, Any]) -> Command | None:
    """When the notes say a cutoff rule decided the change but the line or the score is not settled, one question back."""
    a = h.assignment or {}
    if a.get("kind") != "cutoff_rule":
        return None
    names = state.get("columns") or {}
    if a.get("score_column") and a.get("cutoff") is None:
        sc = a["score_column"]
        ask = LaneAsk(
            address="claim:assignment.cutoff",
            question=f"Which value of {names.get(_key(sc), sc)!r} was the line drawn at, in its own units?",
            because="the notes name the score but not the line",
        )
        return asks.ask_back(
            "score", ask, reason="the notes name the score but not the cutoff value that decided who got the change", facts=errors, extra=extra
        )
    if not a.get("score_column"):
        ask = LaneAsk(
            address="claim:assignment.score_column",
            question="Which column holds the score the line was drawn on?",
            because="the notes say a line on a score decided the change but not which column",
        )
        return asks.ask_back(
            "score", ask, reason="the notes say a cutoff rule decided the change but not which column carries the score", facts=errors, extra=extra
        )
    return None


class _NoRule(Exception):
    """The answer says no cutoff rule on a numeric score is stated: a fact that ends the judgement, not an error to retry."""


class _WrongSide(Exception):
    """The notes and the data disagree about which side got the change: a fact that ends the judgement."""

    def __init__(self, facts: list[str]):
        super().__init__("wrong side")
        self.facts = facts


def _score_errors(state: SpecialistState, parsed: Score, table: pd.DataFrame, t: str | None, ok: Any) -> list[str]:
    """The checks a score answer must pass. Normalises the answer in place. Raises _NoRule or _WrongSide for the facts that end
    the judgement rather than retry it."""
    if parsed.column is None or str(parsed.column).strip().lower() in ("", "null", "none"):
        raise _NoRule()
    parsed.column = _key(parsed.column)
    if str(parsed.takeup_column).strip().lower() in ("", "null", "none"):  # the model sometimes writes the word null instead of a null
        parsed.takeup_column, parsed.takeup_level = None, None
    parsed.takeup_column = _key(parsed.takeup_column) if parsed.takeup_column else None
    if parsed.takeup_column and parsed.column and parsed.takeup_column == _key(parsed.column):
        parsed.takeup_column, parsed.takeup_level = None, None  # the score cannot record its own take-up: the rule is the change
    errors = [f"{c} is not an address you may cite" for c in parsed.cites if not ok(c)]
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
            errors.append(
                f"the hand-off names {t!r} as the treatment and it is not a function of the cutoff; name it as the take-up column with its treated level, or name the score differently"
            )
    if errors or not parsed.takeup_column:
        return errors
    # facts that end the judgement rather than retry it
    x = pd.to_numeric(table[parsed.column], errors="coerce")
    tu = (table[parsed.takeup_column].astype(str) == str(parsed.takeup_level)).astype(float)
    tu[table[parsed.takeup_column].isna()] = np.nan
    treated_mask = (x > parsed.cutoff) if parsed.treated_side == "above" else (x < parsed.cutoff)
    other_mask = (x < parsed.cutoff) if parsed.treated_side == "above" else (x > parsed.cutoff)
    s_t, s_o = float(tu[treated_mask].mean()), float(tu[other_mask].mean())
    if not (s_t > s_o):
        raise _WrongSide([f"take-up {s_t:.2f} on the side the notes call treated ({parsed.treated_side} {parsed.cutoff:g}), {s_o:.2f} on the other side"])
    at = tu[x == parsed.cutoff]
    if len(at):
        share_at = float(at.mean())
        if parsed.cutoff_value_treated and share_at < 0.5:
            errors.append(f"cutoff_value_treated is true but only {share_at:.2f} of the {len(at)} units exactly at the cutoff received the change")
        if not parsed.cutoff_value_treated and share_at > 0.5:
            errors.append(f"cutoff_value_treated is false but {share_at:.2f} of the {len(at)} units exactly at the cutoff received the change")
    return errors


def score(state: SpecialistState) -> Command:
    h = state["handoff"]
    t, y, _ = _keys(state)
    table = _table(state)
    lad = _ladder(state)
    declines: list[Decline] = []

    def done(sc: Score, extra: dict[str, Any]) -> Command:
        _writer()({"score": sc.model_dump()})
        return Command(goto="shape_table", update={"score": sc, "ladder": lad.model_copy(update={"score": sc}), "declines": declines, **extra})

    block = _block_score(h)
    if block is not None:  # a fact from the pack; the same checks apply, and the model is asked only if they fail
        try:
            errors = _score_errors(state, block, table, t, h.resolve)
        except _WrongSide as w:
            return _stop(
                "score",
                "the notes and the data disagree about which side got the change",
                w.facts,
                "a note whose account of the rule matches the take-up recorded in the data",
                {"score": block, "declines": declines},
            )
        except _NoRule:
            errors = ["the pack names no score column"]
        if not errors:
            return done(block, {})
        about = "claim:assignment.cutoff_value_treated" if any("cutoff_value_treated" in e for e in errors) else _block_about(errors)
        declines.append(
            Decline(
                stage="score",
                kind="replaced",
                about=about,
                pack_value=f"{block.column} {block.treated_side} {block.cutoff:g}"
                if about != "claim:assignment.cutoff_value_treated"
                else str(block.cutoff_value_treated),
                check="score.cutoff_value_takeup" if about == "claim:assignment.cutoff_value_treated" else "score.rule_in_file",
                reason="; ".join(errors),
            )
        )
        _writer()({"score": {"pack_block_rejected": errors}})
    treatment_card = _card(h, t) if t else "(the hand-off names no treatment column: the change may be the cutoff rule itself)"
    user = P.SCORE_USER.format(
        question=_question(state),
        frame=L.frame_text(state),
        dataset_card=h.render_dataset(),
        changes=h.render_change(),
        treatment_card=treatment_card,
        cards=h.render_columns(),
        errors="",
    )
    first = [f"PREVIOUS ANSWER WAS REJECTED\n- {d.reason}" for d in declines]  # the pack's failure is what the model is told first
    ended: dict[str, Any] = {}

    def gate(r: Score, log: EpisodeLog) -> list[str]:
        try:
            return _score_errors(state, r, table, t, _resolver(h, log, lad))
        except (_NoRule, _WrongSide) as fact:
            ended["fact"], ended["score"] = fact, r
            return []  # a fact ends the episode; the node reads it below

    rec, log, thoughts, errors = run_episode(
        Score,
        P.SCORE_SYSTEM,
        user + ("\n" + "\n".join(first) if first else ""),
        tools=L.data_tools(state, None),
        budget=_budget("score"),
        gate=gate,
        node="score",
    )
    extra: dict[str, Any] = {"debug": thoughts, "episodes": {"score": log}, "declines": declines}
    if "fact" in ended:
        sc = ended["score"]
        if isinstance(ended["fact"], _WrongSide):
            return _stop(
                "score",
                "the notes and the data disagree about which side got the change",
                ended["fact"].facts,
                "a note whose account of the rule matches the take-up recorded in the data",
                {"score": sc, **extra},
            )
        _writer()({"score": {"column": None, "reason": sc.reason}})
        asked = _ask_score(h, state, [f"the model's reading: {sc.reason}"], extra)
        if asked is not None:
            return asked
        return _stop(
            "score",
            "no cutoff rule on a numeric score is stated",
            [f"the model's reading: {sc.reason}"],
            "a note naming the score column and the cutoff value that decided who got the change",
            {"score": sc, **extra},
        )
    if rec is not None:
        return done(rec.model_copy(update={"by": "judgement"}), extra)
    asked = _ask_score(h, state, errors, extra)
    if asked is not None:
        return asked
    return _stop(
        "score", "could not name the score and cutoff", errors, "a note that names the score column, the cutoff value, and which side got the change", extra
    )


# rung 1: the shape (fact)


def _candidates(state: SpecialistState) -> list[str]:
    sc: Score = state["score"]
    _, y, rel = _keys(state)
    return [k for k in rel if k not in (sc.column, y, sc.takeup_column, state.get("cluster_column"))]


def shape_table(state: SpecialistState) -> Command:
    h = state["handoff"]
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
    # the desk's count of rows a side against the lane's, allowing for rows at the cutoff and rows without an outcome
    declines: list[Decline] = []
    for pr in h.probes:
        if pr.family == "discontinuity" and pr.name == "rows_by_side" and pr.value is not None:
            lane = min(facts.n_left, facts.n_right)
            slack = facts.rows_at_cutoff + (facts.rows_score - facts.rows_primary)
            if abs(int(pr.value) - lane) > slack:
                declines.append(
                    Decline(
                        stage="shape_table",
                        kind="replaced",
                        about=pr.address,
                        pack_value=str(int(pr.value)),
                        took=str(lane),
                        check="shape.rows_by_side",
                        reason="the desk counted on the whole file; the lane counts the rows with an outcome after the filter",
                    )
                )
    _writer()({"shape": facts.model_dump(), "declines": [d.render() for d in declines]})
    update = {
        "canon_path": str(canon_path),
        "xall_path": str(xall_path),
        "shape": facts,
        "contrast": contrast,
        "candidates": candidates,
        "declines": declines,
        "ladder": _ladder(state).model_copy(update={"shape": facts}),
    }
    return Command(goto="density", update=update)


# rung 1: the density at the line (evidence, by code, before the line is judged)


def density(state: SpecialistState) -> Command:
    x_all = pd.read_csv(state["xall_path"])["x"]
    facts, raw = CK.density_evidence(x_all, state["shape"], _cfg(), sampled_by_side=bool(state.get("sampled_by_side")))
    _writer()({"density": [f"[{a}] {text}" for a, text in facts.lines()]})
    return Command(goto="line", update={"density_facts": raw, "ladder": _ladder(state).model_copy(update={"density": facts})})


# rung 2: the line (a judgement from the story and the facts)


def line(state: SpecialistState) -> Command:
    h = state["handoff"]
    lad = _ladder(state)
    sc: Score = state["score"]
    s: ShapeFacts = state["shape"]

    dens: DensityFacts | None = lad.density
    bunching = dens is not None and dens.status == "tested" and dens.flagged
    movable = _case(state).beliefs.get("movable") == "confirmed_true"

    def gate(r: Line, log: EpisodeLog) -> list[str]:
        ok = _resolver(h, log, lad)
        errs: list[str] = []
        if not r.why.strip():
            errs.append("say why, from the story and the facts")
        if not r.clean and not r.risks:
            errs.append("a line judged not clean names at least one risk")
        cited = set(r.cites) | {c for k in r.risks for c in k.cites}
        if bunching and r.clean and "ladder:density.test" not in cited:
            errs.append("the score bunches at the line [ladder:density.test]; a clean verdict must answer it, citing that line")
        if bunching and movable and not any(k.name == "manipulation" for k in r.risks):
            errs.append("the person said a unit could move the score and the density jumps [ladder:density.test]; name the manipulation risk")
        for risk in r.risks:
            if not risk.cites:
                errs.append(f"risk {risk.name}: no citation")
            errs += [f"risk {risk.name}: {e}" for e in V.cites_resolve(risk.cites, h, ok)]
        return errs + V.cites_resolve(r.cites, h, ok)

    rule = f"{'at or ' if sc.cutoff_value_treated else ''}{sc.treated_side} {sc.cutoff:g}"
    line_text = (
        f"score {sc.column}, treated when {rule}; {s.kind} design; {s.n_left} rows on the control side, {s.n_right} on the treated side; "
        f"{s.distinct_scores} distinct scores; {s.rows_at_cutoff} rows exactly at the cutoff\n{_card(h, sc.column)}"
    )
    user = P.LINE_USER.format(question=_question(state), frame=L.frame_text(state), line=line_text, errors="")
    rec, log, thoughts, errors = run_episode(Line, P.LINE_SYSTEM, user, tools=_canon_tools(state), budget=_budget("line"), gate=gate, node="line")
    if rec is None:
        return _stop(
            "line",
            "the line could not be judged",
            errors,
            "a story that says how the score was set and what else the line decides",
            {"debug": thoughts, "episodes": {"line": log}},
        )
    _writer()({"line": [f"[{a}] {text}" for a, text in rec.lines()]})
    return Command(goto="balance", update={"ladder": lad.model_copy(update={"line": rec}), "debug": thoughts, "episodes": {"line": log}})


# rung 3's evidence: every candidate's standing at the line (by code, before the covariates are placed)


def balance(state: SpecialistState) -> Command:
    canon = _canon(state)
    shape: ShapeFacts = state["shape"]
    inf = pick_inference(cluster_column=bool(shape.cluster_column), clusters=shape.clusters)
    dens = _ladder(state).density
    window = min(dens.h_left, dens.h_right) if dens is not None and dens.h_left is not None and dens.h_right is not None else None
    facts = CK.balance_evidence(
        canon, _others(state, canon), list(state.get("candidates") or []), shape, _cfg(), cluster=bool(shape.cluster_column), vce=inf.vce, window=window
    )
    _writer()({"balance": [f"[{a}] {text}" for a, text in facts.lines()]})
    return Command(goto="covariates", update={"ladder": _ladder(state).model_copy(update={"balance": facts})})


# rung 3: the covariates, placed together (a judgement only for what the pack leaves open)


def settled_claims(h: Handoff, k: str, case: C.Case) -> tuple[dict[str, bool], dict[str, str]]:
    """What the pack settles about a candidate covariate: fixed before the change means predetermined and not moved; the
    person's word on what the change moved and on what measures the outcome stands."""
    a = f"col:{k}"
    claims: dict[str, bool] = {}
    cites: dict[str, str] = {}
    when = case.fact(f"{a}.when") if case.is_fact(f"{a}.when") else None
    if when == "before":
        claims["predetermined"], cites["predetermined"] = True, f"{a}.when"
        claims.setdefault("affected_by_treatment", False)
        cites.setdefault("affected_by_treatment", f"{a}.when")
    elif when in ("at", "after"):
        claims["predetermined"], cites["predetermined"] = False, f"{a}.when"
    if case.is_fact(f"{a}.moved_by_change"):
        claims["affected_by_treatment"], cites["affected_by_treatment"] = bool(case.fact(f"{a}.moved_by_change")), f"{a}.moved"
        if claims["affected_by_treatment"]:
            claims["predetermined"], cites["predetermined"] = False, f"{a}.moved"
    if case.is_fact(f"{a}.measures_outcome"):
        claims["is_outcome_measure"], cites["is_outcome_measure"] = bool(case.fact(f"{a}.measures_outcome")), f"{a}.measures_outcome"
    return claims, cites


def fact_relation(h: Handoff, k: str, case: C.Case) -> CovariateRelation | None:
    claims, cites = settled_claims(h, k, case)
    name = h.column(k).name if h.column(k) else k
    if claims.get("is_outcome_measure") is True or claims.get("affected_by_treatment") is True or all(c in claims for c in CLAIMS):
        full = {c: bool(claims.get(c, False)) for c in CLAIMS}
        if full["affected_by_treatment"] or full["is_outcome_measure"]:
            full["predetermined"] = False
        return CovariateRelation(
            column=k, reasons=[Cited(reason=f"{name}: {c} settled by the pack", cites=[cites[c]]) for c in CLAIMS if full[c] and c in cites], **full
        )
    return None


_settled_text, apply_settled = L.make_settled(settled_claims)


def _column_block(h: Handoff, k: str, case: C.Case) -> str:
    return f"COLUMN {k!r}\n{_card(h, k)}\n{_settled_text(h, k, case)}"


def _relation_errors(r: CovariateRelation, ok: Any) -> list[str]:
    errs: list[str] = []
    if (r.predetermined or r.affected_by_treatment or r.is_outcome_measure) and not r.reasons:
        errs.append("claims marked true but no reasons given")
    if r.predetermined and r.affected_by_treatment:
        errs.append("cannot be both fixed before the change and changed by it")
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


def covariates(state: SpecialistState) -> Command:
    h = state["handoff"]
    case = _case(state)
    lad = _ladder(state)
    asked = [k for k in state.get("candidates") or [] if fact_relation(h, k, case) is None]
    if not asked:
        return Command(goto="merge_covariates", update={"ladder": lad.model_copy(update={"covariates": CovariateRoles(items=[])})})

    bal: BalanceFacts = lad.balance or BalanceFacts()

    def gate(r: CovariateRoles, log: EpisodeLog) -> list[str]:
        ok = _resolver(h, log, lad)
        errs = _presence_errors(asked, [x.column for x in r.items])
        for x in r.items:
            if x.column in asked:
                errs += [f"{x.column}: {e}" for e in _relation_errors(apply_settled(x, h, case), ok)]
                item = bal.item(x.column)
                if (
                    x.predetermined
                    and item is not None
                    and item.flagged(bal.threshold)
                    and f"ladder:balance.{x.column}" not in {c for rs in x.reasons for c in rs.cites}
                ):
                    errs.append(
                        f"{x.column} jumps at the line [ladder:balance.{x.column}]; a column called fixed before the line must say why it still is, citing that line"
                    )
        return errs

    user = P.COVARIATES_USER.format(
        question=_question(state), frame=L.frame_text(state), count=len(asked), columns="\n\n".join(_column_block(h, k, case) for k in asked), errors=""
    )
    rec, log, thoughts, errors = run_episode(
        CovariateRoles, P.COVARIATES_SYSTEM, user, tools=_canon_tools(state), budget=_budget("covariates"), gate=gate, node="covariates"
    )
    if rec is None:
        return _stop(
            "covariates",
            "the candidate covariates could not be placed",
            errors,
            "clearer column notes about when each column was fixed and what the treatment touches",
            {"debug": thoughts, "episodes": {"covariates": log}},
        )
    out = CovariateRoles(items=[apply_settled(x, h, case) for x in rec.items], unsure=rec.unsure)
    _writer()({"covariates_rung": [f"[{a}] {text}" for a, text in out.lines()]})
    return Command(goto="merge_covariates", update={"ladder": lad.model_copy(update={"covariates": out}), "debug": thoughts, "episodes": {"covariates": log}})


def _latest(state: SpecialistState) -> dict[str, CovariateRelation]:
    h = state["handoff"]
    case = _case(state)
    latest: dict[str, CovariateRelation] = {k: r for k in state.get("candidates") or [] if (r := fact_relation(h, k, case)) is not None}
    lad = _ladder(state)
    for r in lad.covariates.items if lad.covariates is not None else []:
        latest[r.column] = apply_settled(r, h, case)
    return latest


# ------------------------------------------------------------------ merge + verify (facts)


def merge_covariates(state: SpecialistState) -> dict:
    canon = _canon(state)
    numeric = {c for c in canon.columns if pd.api.types.is_numeric_dtype(canon[c])}
    latest = _latest(state)
    tested: list[str] = []
    excluded: list[Excluded] = []
    h = state["handoff"]
    b = _block(h)
    allowed = {_key(c) for c in b.covariates_allowed} if b else set()
    case = _case(state)
    for k in state.get("candidates") or []:
        r = latest.get(k)
        if r is None:
            continue
        card = h.column(k)
        if card is not None and card.facts.kind == "id":
            excluded.append(Excluded(column=k, why="an identifier names a unit; it is not a characteristic that could be continuous or jump at the cutoff"))
            continue
        if r.is_outcome_measure:
            excluded.append(
                Excluded(
                    column=k,
                    why="another measure of the outcome, or a later one; not a covariate"
                    + (f" [col:{k}.measures_outcome]" if case.is_fact(f"col:{k}.measures_outcome") else ""),
                )
            )
        elif r.affected_by_treatment:
            excluded.append(
                Excluded(
                    column=k,
                    why=("the person says the change could have moved it [col:%s.moved]" % k)
                    if case.fact(f"col:{k}.moved_by_change") is True
                    else "could be changed by the treatment; adjusting for it would remove part of the effect",
                )
            )
        elif not r.predetermined:
            excluded.append(Excluded(column=k, why="not judged fixed before the score was set; nothing says it should be continuous at the cutoff"))
        elif SH.covcol(k) not in numeric:
            excluded.append(Excluded(column=k, why="not numeric; needs encoding before it can enter a local fit"))
        elif allowed and k not in allowed:
            excluded.append(Excluded(column=k, why="the pack does not allow it as a covariate [design.covariates_allowed]"))
        else:
            tested.append(k)
    c = Covariates(balance_tested=tested, adjusted=list(tested), excluded=excluded)
    shape = state["shape"].model_copy(update={"rows_covariates": SH.rows_complete_on(canon, tested)})
    _writer()({"covariates": c.render()})
    return {"covariates": c, "shape": shape}


def verify(state: SpecialistState) -> Command:
    """A relation for every candidate; the per-column checks were made in the covariates rung's gate."""
    latest = _latest(state)
    missing = [k for k in state.get("candidates") or [] if k not in latest]
    if missing:
        return _stop(
            "verify", "the covariate relations could not be made to pass verification", ["no relation for " + ", ".join(missing)], "clearer column notes"
        )
    _writer()({"verify": "ok"})
    return Command(goto="heterogeneity")


# rung 4: heterogeneity (a judgement over the candidates)


def _modifier_candidates(state: SpecialistState) -> dict[str, list[str]]:
    """Every predetermined characteristic the covariates rung, or the person, marked as one the effect could differ by, with what
    marked it; it must sit in the canonical table."""
    case = _case(state)
    lad = _ladder(state)
    canon_cols = set(pd.read_csv(state["canon_path"], nrows=0).columns)
    latest = _latest(state)
    out: dict[str, list[str]] = {}

    def usable(k: str) -> bool:
        r = latest.get(k)
        return SH.covcol(k) in canon_cols and r is not None and r.predetermined and not r.affected_by_treatment and not r.is_outcome_measure

    for x in lad.covariates.items if lad.covariates is not None else []:
        if x.modifier_candidate and usable(x.column):
            out.setdefault(x.column, []).append(f"ladder:covariates.{x.column}")
    for k in state.get("candidates") or []:
        if case.fact(f"col:{k}.may_modify") is True and usable(k):
            out.setdefault(k, []).append(f"col:{k}.may_modify")
    return out


def heterogeneity(state: SpecialistState) -> Command:
    h = state["handoff"]
    lad = _ladder(state)
    cands = _modifier_candidates(state)
    target = state["target_units"]
    cfg = _cfg().get("modifiers") or {}
    max_m = int(cfg.get("max", 3))
    if not cands:
        het = Heterogeneity(modifiers=[], why="no predetermined characteristic was marked as one the effect could differ by", target_units=target, by="code")
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
        tools=_canon_tools(state),
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


# rung 5: the threats (code, from the pack and the line rung)


def threats(state: SpecialistState) -> dict:
    """The risks every design carries, from the pack; the ones the line rung named from the story; and this design's own, by code
    from the rungs below: bunching the line rung did not name, a score too coarse for a local fit, take-up that varies on one side
    only, a side thin near the line. Each is a flag the assessment must answer and the interpretation must cite."""
    h = state["handoff"]
    lad = _ladder(state)
    s: ShapeFacts = state["shape"]
    cfg = _cfg()
    _, y, _ = _keys(state)
    items = LAD.pack_threats(h, _case(state), y, state.get("columns") or {})
    named = {r.name for r in (lad.line.risks if lad.line is not None else [])}
    for r in lad.line.risks if lad.line is not None else []:
        items.append(Threat(name=r.name, level="soft", text=r.reason, cites=[f"ladder:line.risk.{r.name}", *r.cites]))
    dens = lad.density
    if dens is not None and dens.status == "tested" and dens.flagged and "manipulation" not in named:
        items.append(
            Threat(
                name="manipulation",
                level="soft",
                text="the rows bunch on one side of the line and the line rung did not name it; units may have moved their score",
                cites=["ladder:density.test", "ladder:line.clean"],
            )
        )
    if s.distinct_scores < int(cfg["support"]["distinct_min"]["soft"]):
        items.append(
            Threat(
                name="discrete_score",
                level="soft",
                text=f"the score takes {s.distinct_scores} distinct values; a local fit cannot pick its own width, and the comparison leans on the units at the few scores nearest the line",
                cites=["ladder:shape.sides"],
            )
        )
    if s.kind == "fuzzy" and (s.takeup_left == 0.0 or s.takeup_right == 1.0):
        items.append(
            Threat(
                name="one_sided_takeup",
                level="soft",
                text="take-up varies on one side of the line only; the effect is for those who crossed it and took up, and nothing is learned about take-up on the other side",
                cites=["ladder:shape.kind"],
            )
        )
    soft = int(cfg["sides"]["min_rows"]["soft"])
    if min(s.n_left, s.n_right) < soft:
        items.append(
            Threat(
                name="thin_side",
                level="soft",
                text=f"one side of the line holds {min(s.n_left, s.n_right)} rows; the fit on that side rests on few units",
                cites=["ladder:shape.sides"],
            )
        )
    th = Threats(items=items)
    _writer()({"threats": [f"[{a}] {text}" for a, text in th.lines()]})
    return {"ladder": lad.model_copy(update={"threats": th})}


# ------------------------------------------------------------------ checks (fact)


def _density_raw(state: SpecialistState) -> dict:
    """The density test as the rung computed it, with the binomial windows the check reads."""
    raw = dict(state.get("density_facts") or {})
    dens = _ladder(state).density
    raw["windows"] = [w.model_dump() for w in dens.windows] if dens is not None else []
    return raw


def check_design(state: SpecialistState) -> dict:
    canon = _canon(state)
    x_all = pd.read_csv(state["xall_path"])["x"]
    shape = state["shape"]
    inf = pick_inference(cluster_column=bool(shape.cluster_column), clusters=shape.clusters)
    results, extra = CK.run_checks(
        canon,
        x_all,
        shape,
        state["covariates"],
        state["contrast"].key,
        _cfg(),
        cluster=bool(shape.cluster_column),
        vce=inf.vce,
        sampled_by_side=bool(state.get("sampled_by_side")),
        density=_density_raw(state),
        balance=_ladder(state).balance or BalanceFacts(),
    )
    W.say(results, _cfg(), state.get("columns") or {})  # the sentence before the number, for the reader
    results += C.as_checks(_case(state))
    more, declines = LAD.checks_and_declines(_ladder(state), _ladder(state).threats, state.get("declines") or [])
    results += more
    _writer()({"checks": [f"{r.level} {r.address} {r.detail}" for r in results]})
    facts = {k: v for k, v in extra.items() if k != "first_stage"}
    fs = extra.get("first_stage")
    if fs is not None:
        facts["first_stage"] = (
            None if fs.error else dict(value=fs.value, ci_low=fs.ci_low, ci_high=fs.ci_high, se=fs.se, n_h_left=fs.n_h_left, n_h_right=fs.n_h_right, h=fs.h)
        )
    return {"checks": results, "check_facts": facts, "declines": declines}


# ------------------------------------------------------------------ assess (the yaml first, then a judgement)


def _design_text(state: SpecialistState) -> str:
    return state["covariates"].render() + "\nshape: " + json.dumps(state["shape"].model_dump())


def assess(state: SpecialistState) -> Command:
    h = state["handoff"]
    sc: Score = state["score"]
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
    argue = set(_cfg().get("argue_from_notes") or [])
    flag_text = "\n".join(f"[{r.address}] {r.level.upper()}: {r.detail}" for r in flags)
    covs: Covariates = state["covariates"]
    cards = "\n\n".join([_card(h, sc.column), h.render_dataset()] + [_card(h, k) for k in covs.balance_tested])
    check_addresses = {r.address for r in results}
    errors: list[str] = []
    debug = []
    for _ in range(MAX_MODEL_RETRIES):
        user = P.ASSESS_USER.format(
            frame=L.frame_text(state), question=_question(state), design=_design_text(state), flags=flag_text, cards=cards, errors=_rejected(errors)
        )
        parsed, th = structured(DesignAssessment, P.ASSESS_SYSTEM, user, node="assess")
        debug.append(th)
        errors = [f"{c} is not a check or pack address" for c in parsed.cites if not (c in check_addresses or h.resolve(c))]
        if parsed.action == "proceed":
            if hard:
                errors.append("proceed is not allowed while a hard flag stands: " + ", ".join(r.address for r in hard))
            missing = [r.address for r in flags if r.address not in parsed.cites]
            if missing:
                errors.append("proceed does not address: " + ", ".join(missing))
            if any(r.name in argue for r in flags) and not any(h.resolve(c) for c in parsed.cites):
                errors.append(
                    "no note cited for the argument about "
                    + ", ".join(sorted({r.name for r in flags if r.name in argue}))
                    + "; cite the card that says how the score was set or that the covariate was fixed before the change"
                )
        if errors:
            continue
        _writer()({"assess": parsed.model_dump()})
        if parsed.action == "proceed":
            return Command(goto="pick_estimator", update={"assessment": parsed, "debug": debug, "checks": results})
        return _stop(
            "assess",
            parsed.reason,
            [f"{r.address}: {r.detail}" for r in flags],
            "units on both sides of the cutoff that are alike in everything the notes call fixed, and a score no unit could move",
            {"assessment": parsed, "debug": debug, "checks": results},
        )
    return _stop("assess", "the design assessment could not be validated", errors, "see the gate errors", {"debug": debug, "checks": results})


# ------------------------------------------------------------------ pick estimator (judgement)


def _facts(state: SpecialistState) -> dict[str, Any]:
    s = state["shape"]
    cf = state.get("check_facts") or {}
    return {
        "kind": s.kind,
        "n_left": s.n_left,
        "n_right": s.n_right,
        "distinct_scores": s.distinct_scores,
        "first_stage": cf.get("first_stage_status"),
        "first_stage_F": cf.get("first_stage_F"),
        "adjusted_covariates": state["covariates"].adjusted,
        "cluster": s.cluster_column,
    }


def _allowed(state: SpecialistState) -> list[EstimatorEntry]:
    s = state["shape"]
    excluded = set(state.get("excluded_estimators") or [])
    status = (state.get("check_facts") or {}).get("first_stage_status")
    return [
        e for e in load_estimators() if e.pickable and e.applies(kind=s.kind, first_stage=status, distinct_scores=s.distinct_scores) and e.name not in excluded
    ]


def pick_estimator(state: SpecialistState) -> Command:
    facts = _facts(state)
    allowed = _allowed(state)
    if not allowed:
        return _stop(
            "pick_estimator",
            "no estimator in the catalogue applies to this design",
            [f"facts: {json.dumps(facts, default=str)}", f"excluded after failures: {sorted(state.get('excluded_estimators') or [])}"],
            "an estimator entry for this kind of design",
        )
    names = [e.name for e in allowed]
    check_text = "\n".join(f"[{r.address}] {r.level}: {r.detail}" for r in state["checks"])
    check_addresses = {r.address for r in state["checks"]}
    errors: list[str] = []
    debug = []
    for _ in range(MAX_MODEL_RETRIES):
        user = P.PICK_USER.format(
            frame=L.frame_text(state),
            facts=json.dumps(facts, default=str),
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
            if not (c in check_addresses or state["handoff"].resolve(c)):
                errors.append(f"{c} is not a check or pack address")
        if not errors:
            _writer()({"estimator": parsed.model_dump()})
            return Command(
                goto="window",
                update={"estimator": parsed.name, "estimator_pick": parsed, "debug": debug, "pick_attempts": state.get("pick_attempts", 0) + 1},
            )
    return _stop("pick_estimator", "the estimator pick could not be validated", errors, "see the gate errors", {"debug": debug})


# ------------------------------------------------------------------ the window (rung 6: a judgement over a table code builds)


def _window_from(name: str, row: dict, why: str, cites: list[str], by: str, unsure=None) -> Window:
    return Window(
        selector=name,
        rule=row["rule"],
        h_left=float(row["h_left"]),
        h_right=float(row["h_right"]),
        b_left=float(row["b_left"]),
        b_right=float(row["b_right"]),
        n_left=int(row["n_left"]),
        n_right=int(row["n_right"]),
        why=why,
        cites=list(cites),
        unsure=list(unsure or []),
        by=by,
    )


def window(state: SpecialistState) -> Command:
    """How far from the line the fit reaches. Code builds the table of every selector with the widths on each side and the rows
    inside; under the support-points rule code sets the window; else a judgement picks a selector, the default unless the ladder
    argues for another, and the gate holds the floor on rows a side and the citation a departure needs."""
    h = state["handoff"]
    s: ShapeFacts = state["shape"]
    lad = _ladder(state)
    entry = estimator_entry(state["estimator"])
    covs: Covariates = state["covariates"]
    inf = pick_inference(cluster_column=bool(s.cluster_column), clusters=s.clusters)
    fuzzy = _is_fuzzy(entry, False)
    canon = _canon(state)
    sharpbw = bool(fuzzy and (s.takeup_left == 0.0 or s.takeup_right == 1.0))
    table = CK.window_table(
        CK.SHARP if entry.engine == "local_randomisation" else entry.params,
        canon,
        s,
        _cfg(),
        fuzzy=fuzzy,
        covs=[SH.covcol(k) for k in covs.adjusted] if entry.covs else None,
        cluster=bool(s.cluster_column),
        vce=inf.vce,
        sharpbw=sharpbw,
    )
    if table.get("error") or not table["rows"]:
        return _stop(
            "window", "no width could be selected for the picked estimator", [str(table.get("error"))], "an estimator whose width can be selected on this data"
        )
    c = state["contrast"].key
    floor = int(_cfg()["window"]["min_rows_each_side"])
    default = table["default"]
    rows: dict[str, dict] = table["rows"]

    def settle(w: Window, extra: dict[str, Any] | None = None) -> Command:
        check = CK.effective_rows_check(w.n_left, w.n_right, w.selector, w.h_left, w.h_right, c, _cfg())
        # the rung climbs after check_design, so what it would not settle becomes a flag here, as the ladder's other unsure items did there
        flags = [
            CheckResult(contrast="all", name=f"unsure.{u.about}", level="soft", detail=f"the window rung would not settle {u.about}: {u.reason}")
            for u in w.unsure
        ]
        _writer()({"window": [f"[{a}] {text}" for a, text in w.lines()], "check": f"{check.level} {check.address} {check.detail}"})
        update = {"ladder": lad.model_copy(update={"window": w}), "window_table": table, "checks": [*state["checks"], check, *flags]}
        update.update(extra or {})
        return Command(goto="freeze_design", update=update)

    if entry.engine == "local_randomisation":
        # the largest window in which the predetermined covariates stay balanced (rdwinselect), else the support-points window
        prm = entry.params
        lr = adapter.local_random_windows(canon, [SH.covcol(k) for k in covs.balance_tested], reps=int(prm.get("reps", 1000)), seed=int(prm.get("seed", 7)))
        lr_rows: dict[str, dict] = {}
        chosen = None
        for r in lr.get("windows") or []:
            name = f"window {r['w_left']:.4g} to {r['w_right']:.4g}"
            lr_rows[name] = dict(
                rule="local_randomisation",
                h_left=-r["w_left"],
                h_right=r["w_right"],
                b_left=-r["w_left"],
                b_right=r["w_right"],
                n_left=r["n_left"],
                n_right=r["n_right"],
            )
            if lr.get("w_left") is not None and r["w_left"] == lr["w_left"] and r["w_right"] == lr["w_right"]:
                chosen = name
        table = dict(rule="local_randomisation", default=chosen or "support_points", rows={**lr_rows, **rows}, notes=lr.get("notes") or [])
        if chosen is not None:
            why = (
                f"the largest window in which the predetermined covariates ({', '.join(covs.balance_tested)}) stay balanced across the line, by the "
                f"library's window selector; {lr_rows[chosen]['n_left']}/{lr_rows[chosen]['n_right']} rows inside"
            )
            return settle(_window_from(chosen, lr_rows[chosen], why, ["ladder:shape.sides", *[f"ladder:balance.{k}" for k in covs.balance_tested]], "code"))
        row = dict(rows["support_points"], rule="local_randomisation")
        return settle(
            _window_from(
                "support_points",
                row,
                "no predetermined covariate to choose the window by balance, so the window keeps the declared support points on each side",
                ["ladder:shape.sides"],
                "code",
            )
        )
    if table["rule"] == "support_points":
        row = rows["support_points"]
        return settle(
            _window_from(
                "support_points",
                row,
                f"the score has few distinct values, so the width keeps {int(_cfg()['support'].get('bandwidth_support_points', 3))} support points on each side; the library's selector is not used",
                ["ladder:shape.sides"],
                "code",
            )
        )

    def gate(r: WindowPick, log: EpisodeLog) -> list[str]:
        ok = _resolver(h, log, lad)
        errs: list[str] = []
        if r.selector not in rows:
            errs.append(f"{r.selector!r} is not one of the selectors on offer: {', '.join(rows)}")
            return errs + V.cites_resolve(r.cites, h, ok)
        row = rows[r.selector]
        if row["n_left"] < floor or row["n_right"] < floor:
            errs.append(
                f"the window {r.selector} leaves {row['n_left']} rows on the control side and {row['n_right']} on the treated side; at least {floor} are needed on each"
            )
        if not r.why.strip():
            errs.append("say why, from the ladder and the pack")
        if r.selector != default and not any(x.startswith(("ladder:", "probe:")) or h.resolve(x) for x in r.cites):
            errs.append("a width other than the default needs a reason from the ladder: cite the density, the balance or the sides")
        return errs + V.cites_resolve(r.cites, h, ok)

    lines = []
    for name, row in rows.items():
        spec = _cfg()["window"]["selectors"][name]
        lines.append(
            f"{name}: {spec['in_words']}; when {spec['when']}; h = {row['h_left']:.4g} on the control side, {row['h_right']:.4g} on the treated side; "
            f"{row['n_left']}/{row['n_right']} rows inside"
        )
    user = P.WINDOW_USER.format(
        question=_question(state), frame=L.frame_text(state), estimator=entry.render(), default=default, table="\n".join(lines), errors=""
    )
    rec, log, thoughts, errors = run_episode(WindowPick, P.WINDOW_SYSTEM, user, tools=_canon_tools(state), budget=_budget("window"), gate=gate, node="window")
    if rec is None:
        # the judgement did not settle; the default width stands, and the flag says so
        row = rows[default]
        w = _window_from(default, row, "the default width; the judgement did not settle on another: " + "; ".join(errors[-3:]), [], "code")
        w = w.model_copy(update={"unsure": [LAD.Unsure(about="window", reason="the width judgement was refused three times; the default stands")]})
        return settle(w, {"debug": thoughts, "episodes": {"window": log}})
    w = _window_from(rec.selector, rows[rec.selector], rec.why, rec.cites, "judgement", rec.unsure)
    return settle(w, {"debug": thoughts, "episodes": {"window": log}})


# ------------------------------------------------------------------ freeze (fact)


def freeze_design(state: SpecialistState) -> Command:
    s = state["shape"]
    entry = estimator_entry(state["estimator"])
    covs: Covariates = state["covariates"]
    inf = pick_inference(cluster_column=bool(s.cluster_column), clusters=s.clusters)
    fuzzy = _is_fuzzy(entry, False)
    canon = _canon(state)
    lad = _ladder(state)
    w: Window = lad.window
    cer = (state.get("window_table") or {}).get("rows", {}).get("cerrd")
    status = (state.get("check_facts") or {}).get("first_stage_status")
    also = [n for n in entry.also_run if estimator_entry(n).applies(kind=s.kind, first_stage=status, distinct_scores=s.distinct_scores)]
    for e in load_estimators():
        if not e.pickable and e.runs_when.get("adjusted_covariates") and covs.adjusted and e.name not in also:
            also.append(e.name)
    d = Design(
        contrast=state["contrast"],
        score=state["score"],
        shape=s,
        covariates=covs,
        checks=Checks(results=state["checks"]),
        estimator=entry.name,
        estimand=entry.estimand,
        spec=dict(entry.params),
        also_run=also,
        inference=inf.name,
        vce=inf.vce,
        cluster=s.cluster_column if inf.cluster else None,
        bandwidths=Bandwidths(
            selector=w.selector,
            rule=w.rule,
            h_left=w.h_left,
            h_right=w.h_right,
            b_left=w.b_left,
            b_right=w.b_right,
            h_cer_left=float(cer["h_left"]) if cer else None,
            h_cer_right=float(cer["h_right"]) if cer else None,
            n_h_left=w.n_left,
            n_h_right=w.n_right,
        ),
        sharp_bandwidth_used=bool(fuzzy and (s.takeup_left == 0.0 or s.takeup_right == 1.0)),
        placebos=[p.name for p in load_placebos() if p.applies(engine=entry.engine, kind=s.kind, rule=w.rule, distinct_scores=s.distinct_scores)],
        target_units=state["target_units"],
        modifiers=[m.column for m in lad.heterogeneity.modifiers] if lad.heterogeneity is not None else [],
    )
    run_dir = Path(state["run_dir"])
    (run_dir / "design.json").write_text(d.model_dump_json(indent=2))
    (run_dir / "design.md").write_text(d.render())
    b = adapter.bins(canon["y"], canon["x"])
    if b is not None:
        b.to_csv(run_dir / "bins.csv", index=False)
    _writer()({"design": d.render()})
    return Command(goto="estimate", update={"design": d})


# ------------------------------------------------------------------ estimate (fact) + placebos (fact, fan-out)


def _window_kw(d: Design) -> dict:
    """The design's window, pinned on both sides: what the primary and every refit that must compare with it use."""
    return {"h": d.bandwidths.h, "b": d.bandwidths.b}


def _reselect_kw(d: Design) -> dict:
    """What a refit on other rows or another spec uses: the design's selector, so the library picks the width again for those rows,
    or the pinned window when the rule has no selector (few distinct scores)."""
    return _window_kw(d) if d.bandwidths.rule in ("support_points", "local_randomisation") else {"bwselect": d.bandwidths.selector}


def _fit_local_random(d: Design, entry: EstimatorEntry, canon: pd.DataFrame, fuzzy: bool, mask: pd.Series | None = None) -> adapter.Fit:
    prm = entry.params
    return adapter.fit_local_random(
        canon,
        -d.bandwidths.h_left,
        d.bandwidths.h_right,
        fuzzy=fuzzy,
        reps=int(prm.get("reps", 1000)),
        seed=int(prm.get("seed", 7)),
        alpha=float(prm.get("alpha", 0.05)),
        mask=mask,
    )


def _fit_entry(d: Design, entry: EstimatorEntry, canon: pd.DataFrame, primary_fuzzy: bool, *, primary: bool) -> adapter.Fit:
    if entry.engine == "local_randomisation":
        return _fit_local_random(d, entry, canon, _is_fuzzy(entry, primary_fuzzy))
    fuzzy = _is_fuzzy(entry, primary_fuzzy)
    return adapter.fit(
        entry.params,
        canon,
        fuzzy=fuzzy,
        covs=[SH.covcol(k) for k in d.covariates.adjusted] if entry.covs else None,
        cluster=bool(d.cluster),
        vce=d.vce,
        sharpbw=bool(fuzzy and d.sharp_bandwidth_used),
        **(_window_kw(d) if primary else _reselect_kw(d)),
    )


def estimate(state: SpecialistState) -> Command:
    d: Design = state["design"]
    canon = _canon(state)
    entry = estimator_entry(d.estimator)
    primary_fuzzy = _is_fuzzy(entry, d.shape.kind == "fuzzy")
    key = d.contrast.key
    f = _fit_entry(d, entry, canon, primary_fuzzy, primary=True)
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
        ests.append(adapter.to_estimate(_fit_entry(d, e2, canon, primary_fuzzy, primary=False), key, name, d.target_units, secondary=True))
    if f.first_stage:
        fs = f.first_stage
        ests.append(
            Estimate(
                contrast=key,
                method="first_stage",
                value=fs["value"],
                ci_low=fs["ci_low"],
                ci_high=fs["ci_high"],
                n_treated=f.n_h_right,
                n_control=f.n_h_left,
                target_units="take_up_jump",
                secondary=True,
            )
        )
    primary = dict(
        value=f.value,
        ci_low=f.ci_low,
        ci_high=f.ci_high,
        p=f.p,
        h=f.h_left,
        h_right=f.h_right,
        b=f.b_left,
        b_right=f.b_right,
        n_h_left=f.n_h_left,
        n_h_right=f.n_h_right,
        n_left=f.n_left,
        n_right=f.n_right,
        vce=f.vce,
        model=f.model,
        notes=f.notes,
    )
    update["primary"] = primary
    ests += _by_modifier(canon, d, primary_fuzzy)
    _writer()({"estimate": [e.model_dump(exclude_none=True) for e in ests]})
    sends = [Send("placebo", PlaceboTask(name=n, design=d.model_dump(), canon_path=state["canon_path"], primary=primary)) for n in d.placebos]
    return Command(goto=sends or "interpret", update=update)


def _modifier_groups(canon: pd.DataFrame, column: str, max_levels: int) -> list[tuple[str, pd.Series]]:
    """The rows of each level of a predetermined characteristic: its values when few, quantile bins when it is a number with many."""
    s = canon[column]
    if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > max_levels:
        bins = pd.qcut(s, q=max_levels, duplicates="drop")
        return [(str(level), bins == level) for level in bins.cat.categories]
    levels = [v for v in s.astype(str).value_counts().index[:max_levels]]
    return [(str(v), s.astype(str) == v) for v in levels]


def _by_modifier(canon: pd.DataFrame, d: Design, primary_fuzzy: bool) -> list[Estimate]:
    """The primary spec fitted again within each level of each modifier, at the design's bandwidth so the levels compare, without
    that column among the covariates. Too few effective rows on a side is recorded as the estimate's error, never skipped in silence."""
    cfg = _cfg()
    entry = estimator_entry(d.estimator)
    floor = int(cfg["effective_rows"]["min"]["soft"])
    out: list[Estimate] = []
    for col in d.modifiers:
        cc = SH.covcol(col)
        if cc not in canon.columns:
            continue
        for level, mask in _modifier_groups(canon, cc, int((cfg.get("modifiers") or {}).get("max_levels", 4))):
            if entry.engine == "local_randomisation":
                f = _fit_local_random(d, entry, canon, primary_fuzzy, mask=mask)
            else:
                f = adapter.fit(d.spec, canon, fuzzy=primary_fuzzy, cluster=bool(d.cluster), vce=d.vce, mask=mask, **_window_kw(d))
            e = adapter.to_estimate(f, d.contrast.key, d.estimator, d.target_units)
            if e.error is None and min(f.n_h_left, f.n_h_right) < floor:
                e = e.model_copy(
                    update={
                        "error": f"too few effective rows on a side within this level ({f.n_h_left} control side, {f.n_h_right} treated side; floor {floor})"
                    }
                )
            out.append(e.model_copy(update={"modifier": col, "level": level, "secondary": False}))
    return out


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


def _points_record(points: list[tuple[str, adapter.Fit | None, bool, str]], at: dict[str, float] | None = None) -> list[dict]:
    """Every refit as data a figure can draw: the label, where it sat (a bandwidth or a cutoff), the estimate and its interval."""
    out = []
    for label, f, informative, note in points:
        rec: dict[str, Any] = {"label": label, "informative": bool(informative), "note": note, "at": (at or {}).get(label)}
        if f is not None and not f.error:
            rec.update(value=f.value, lo=f.ci_low, hi=f.ci_high, n_l=f.n_h_left, n_r=f.n_h_right, h=f.h_left, h_right=f.h_right)
        out.append(rec)
    return out


def placebo(task: PlaceboTask) -> dict:
    d = Design.model_validate(task["design"])
    canon = _canon(task["canon_path"])
    prim = task["primary"]
    entry = placebo_entry(task["name"])
    pentry = estimator_entry(d.estimator)
    primary_fuzzy = _is_fuzzy(pentry, d.shape.kind == "fuzzy")
    floor = int(_cfg()["effective_rows"]["min"]["soft"])
    key = d.contrast.key
    points: list[tuple[str, adapter.Fit | None, bool, str]] = []
    at: dict[str, float] = {}

    def informative(f: adapter.Fit) -> tuple[bool, str]:
        if f.error:
            return False, f.error
        if min(f.n_h_left, f.n_h_right) < floor:
            return False, f"fewer than {floor} effective rows on a side"
        return True, ""

    spec, spec_fuzzy = (CK.SHARP, False) if entry.spec == "sharp_reduced_form" else (d.spec, primary_fuzzy)  # what the entry says to refit
    if entry.name == "placebo_cutoffs":
        for label, mask in (("control side", canon["x"] < 0), ("treated side", canon["x"] >= 0)):
            sub = canon[mask]
            if entry.placement != "side_median":
                points.append((label, None, False, f"placement {entry.placement!r} is not one the lane knows"))
                continue
            c_med = float(sub["x"].median())
            name = f"{label} at {c_med:.4g}"
            at[name] = c_med
            if not (sub["x"].min() < c_med < sub["x"].max()):
                points.append((name, None, False, "the placebo cutoff is not strictly inside that side's scores"))
                continue
            f = adapter.fit(spec, sub, fuzzy=spec_fuzzy, cluster=bool(d.cluster), vce=d.vce, c=c_med, **_reselect_kw(d))
            ok, note = informative(f)
            points.append((name, f, ok, note))
    elif entry.name in ("polynomial_grid", "kernel_grid"):
        held = {"b": d.bandwidths.b} if entry.hold_b else {}
        variants = (
            [(f"p = {o}", dict(spec, p=int(o))) for o in entry.params.get("orders", [1, 2, 3])]
            if entry.name == "polynomial_grid"
            else [(f"kernel {k}", dict(spec, kernel=str(k))) for k in entry.params.get("kernels", ["tri", "epa", "uni"])]
        )
        for i, (label, variant) in enumerate(variants):
            at[label] = float(i)
            f = adapter.fit(
                variant,
                canon,
                fuzzy=spec_fuzzy,
                covs=[SH.covcol(k) for k in d.covariates.adjusted] if pentry.covs else None,
                cluster=bool(d.cluster),
                vce=d.vce,
                h=d.bandwidths.h,
                **held,
            )
            ok, note = informative(f)
            points.append((label, f, ok, note))
    elif entry.name == "bandwidth_grid":
        bws = d.bandwidths
        grid: dict[str, tuple[float, float]] = {"h_mse": (bws.h_left, bws.h_right), "2h_mse": (2 * bws.h_left, 2 * bws.h_right)}
        if bws.h_cer_left is not None and bws.h_cer_right is not None:
            grid.update({"h_cer": (bws.h_cer_left, bws.h_cer_right), "2h_cer": (2 * bws.h_cer_left, 2 * bws.h_cer_right)})
        for name in entry.grid:
            if name not in grid:
                points.append((name, None, False, "no coverage-error bandwidth under the support-points rule"))
                continue
            hl, hr = grid[name]
            label = f"{name} = {hl:.4g}" if abs(hl - hr) < 1e-12 else f"{name} = {hl:.4g}/{hr:.4g}"
            at[label] = hl
            f = adapter.fit(spec, canon, fuzzy=spec_fuzzy, cluster=bool(d.cluster), vce=d.vce, h=[hl, hr], b=bws.b if entry.hold_b else None)
            ok, note = informative(f)
            points.append((label, f, ok, note))
    elif entry.name == "donut":
        for share in entry.radii_share_of_h:
            r = share * min(d.bandwidths.h_left, d.bandwidths.h_right)
            mask = canon["x"].abs() >= r
            dropped = int((~mask).sum())
            if dropped == 0:
                points.append((f"radius {share:.0%} of h", None, False, "no rows lie within the radius"))
                continue
            f = adapter.fit(spec, canon, fuzzy=spec_fuzzy, cluster=bool(d.cluster), vce=d.vce, mask=mask, **_reselect_kw(d))
            ok, note = informative(f)
            label = f"radius {share:.0%} of h ({dropped} rows dropped)"
            at[label] = r
            points.append((label, f, ok, note))
    elif entry.name == "window_sensitivity":
        for share in entry.params.get("window_shares", [0.5, 1.0, 2.0]):
            hl, hr = share * d.bandwidths.h_left, share * d.bandwidths.h_right
            f = _fit_local_random(
                d.model_copy(update={"bandwidths": d.bandwidths.model_copy(update={"h_left": hl, "h_right": hr})}), pentry, canon, primary_fuzzy
            )
            label = f"window ×{share:g} ({hl:.4g}/{hr:.4g})"
            at[label] = hr
            ok, note = informative(f)
            points.append((label, f, ok, note))
    elif entry.name == "rosenbaum_bounds":
        gammas = [float(g) for g in entry.params.get("gammas", [0.1, 0.5, 1.0])]
        b = adapter.rosenbaum_bounds(
            canon, max(d.bandwidths.h_left, d.bandwidths.h_right), gammas, reps=int(entry.params.get("reps", 500)), seed=int(pentry.params.get("seed", 7))
        )
        if "error" in b:
            r = Refutation(contrast=key, refuter=entry.name, kind="sensitivity", passed=None, detail=f"could not run ({b['error']})")
        else:
            parts = [f"gamma {g:g}: p between {lo:.3g} and {hi:.3g}" for g, lo, hi in zip(b["gamma"], b["lower"], b["upper"], strict=True)]
            holds = [g for g, hi in zip(b["gamma"], b["upper"], strict=True) if hi < 0.05]
            r = Refutation(
                contrast=key,
                refuter=entry.name,
                kind="sensitivity",
                p_value=b["p"],
                range_low=float(min(b["lower"])),
                range_high=float(max(b["upper"])),
                passed=None,
                detail=f"randomisation p = {b['p']:.3g} under a coin toss; "
                + "; ".join(parts)
                + (f"; the verdict holds up to gamma {max(holds):g}" if holds else "; the verdict does not survive the smallest departure tried"),
            )
        _writer()({"placebo": {r.refuter: r.detail}})
        return {"refutations": [r]}
    if entry.kind == "sensitivity":
        values = [f.value for _, f, ok, _ in points if f is not None and not f.error and ok]
        _, detail = _refit_rule(points, prim, entry)
        r = Refutation(
            contrast=key,
            refuter=entry.name,
            kind="sensitivity",
            new_effect=float(np.mean(values)) if values else None,
            range_low=float(min(values)) if values else None,
            range_high=float(max(values)) if values else None,
            passed=None,
            detail=detail,
        )
        _writer()({"placebo": {r.refuter: r.detail}})
        return {"refutations": [r], "placebo_points": {entry.name: _points_record(points, at)}}
    passed, detail = _refit_rule(points, prim, entry)
    values = [f.value for _, f, ok, _ in points if f is not None and not f.error and ok]
    r = Refutation(
        contrast=key,
        refuter=entry.name,
        kind="falsification",
        new_effect=float(np.mean(values)) if values else None,
        passed=passed,
        detail=detail + ("" if passed is None else " (pass)" if passed else " (FAIL)"),
    )
    _writer()({"placebo": {r.refuter: r.detail}})
    return {"refutations": [r], "placebo_points": {entry.name: _points_record(points, at)}}


# ------------------------------------------------------------------ interpret (judgement)


def _addresses(state: SpecialistState) -> list[str]:
    d: Design = state["design"]
    c = d.contrast.key
    out = (
        ["design.assumption", "design.score", "design.bandwidth"]
        + [b.address for b in state["handoff"].beliefs.values() if b.known() or b.status == "unknown"]
        + [r.address for r in d.checks.results]
    )
    out += [x.address for x in state.get("declines") or []]
    out += sorted(_ladder(state).addresses())
    out += [f.address for log in _episodes(state).values() for f in log.facts]
    for e in state.get("estimates") or []:
        if e.error is None:
            tag = _tag(e, d)
            out += [f"{tag}.value", f"{tag}.ci", f"{tag}.n"] + ([f"{tag}.p", f"{tag}.bandwidth"] if e.method == d.estimator and e.modifier is None else [])
    for r in state.get("refutations") or []:
        out += [f"placebo:{c}.{r.refuter}.passed", f"placebo:{c}.{r.refuter}.detail", f"placebo:{c}.{r.refuter}.in_words"]
    return list(dict.fromkeys(out))


def _tag(e: Estimate, d: Design) -> str:
    """The address stem: the primary is estimate:<contrast>; a secondary carries its method; a level of a modifier its own tail."""
    if e.modifier is not None:
        return e.tag
    return f"estimate:{d.contrast.key}" if e.method == d.estimator else f"estimate:{d.contrast.key}.{e.method}"


def _episodes(state: SpecialistState) -> dict[str, EpisodeLog]:
    return {k: v for k, v in (state.get("episodes") or {}).items() if isinstance(v, EpisodeLog)}


def _required(state: SpecialistState) -> list[str]:
    d: Design = state["design"]
    c = d.contrast.key
    req = ["design.bandwidth", f"estimate:{c}.ci", f"estimate:{c}.n"]
    req += [r.address for r in d.checks.results if r.level != "pass"]
    req += [f"{e.tag}.value" for e in state.get("estimates") or [] if e.modifier is not None and e.error is None]
    req += [f"placebo:{c}.{r.refuter}.passed" for r in state.get("refutations") or [] if r.passed is False]
    req += [x.address for x in state.get("declines") or [] if x.kind == "substituted"]
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
        *[b.render() for b in state["handoff"].beliefs.values() if b.known() or b.status == "unknown"],
        f"[design.score] score {names.get(sc.column, sc.column)}, treated when {rule}"
        + (f"; take-up recorded in {names.get(sc.takeup_column, sc.takeup_column)}" if sc.takeup_column else "; treatment is the cutoff rule itself")
        + f"; {d.shape.kind} design; scores run from {d.shape.score_min:.4g} to {d.shape.score_max:.4g}",
        f"[design.bandwidth] window {d.bandwidths.selector}: estimation bandwidth h = {d.bandwidths.h_left:.4g} on the control side, {d.bandwidths.h_right:.4g} on the treated side, in the score's units (bias bandwidth b = {d.bandwidths.b_left:.4g}/{d.bandwidths.b_right:.4g}); the effect is estimated from rows within h of the cutoff",
        f"comparison: {d.contrast.treated} versus {d.contrast.control}; outcome: {state['handoff'].outcome}",
    ]
    for r in d.checks.results:
        lines.append(f"[{r.address}] {r.level}: {r.detail}")
    lines += [x.render() for x in state.get("declines") or []]
    lines += [f"[{a}] {text}" for a, text in _ladder(state).lines()]
    lines += [f.render() for log in _episodes(state).values() for f in log.facts]
    for e in state.get("estimates") or []:
        if e.modifier is not None:
            tag = e.tag
            if e.error is not None:
                lines.append(f"[{tag}.error] within {names.get(e.modifier, e.modifier)} = {e.level}: {e.error}")
                continue
            lines.append(f"[{tag}.value] {e.value:.4g} (within {names.get(e.modifier, e.modifier)} = {e.level}: {e.method} at the design's bandwidth)")
            lines.append(f"[{tag}.ci] 95% robust interval {e.ci_low:.4g} to {e.ci_high:.4g}" if e.ci_low is not None else f"[{tag}.ci] no interval")
            lines.append(f"[{tag}.n] {e.n_control} control-side and {e.n_treated} treated-side rows inside the bandwidth")
            continue
        if e.error is not None:
            continue
        if e.method == d.estimator:
            tag = f"estimate:{c}"
            lines.append(f"[{tag}.value] {e.value:.4g} (primary: {e.method}, {d.estimand})")
            lines.append(f"[{tag}.ci] 95% robust interval {e.ci_low:.4g} to {e.ci_high:.4g}")
            lines.append(f"[{tag}.p] {'randomisation' if prim.get('vce') == 'randomisation' else 'robust'} p = {prim.get('p', float('nan')):.3g}")
            lines.append(f"[{tag}.n] {e.n_control} control-side and {e.n_treated} treated-side rows inside the bandwidth")
            lines.append(
                f"[{tag}.bandwidth] h = {prim.get('h', d.bandwidths.h_left):.4g} on the control side, {prim.get('h_right', d.bandwidths.h_right):.4g} on the treated side"
            )
        else:
            tag = f"estimate:{c}.{e.method}"
            what = {
                "first_stage": "the jump in take-up at the cutoff",
                "local_linear_adjusted": "with the adjusted covariates partialled out",
                "local_quadratic": "quadratic on each side",
                "local_linear_itt": "effect of crossing the cutoff, whatever was taken up",
            }.get(e.method, "secondary")
            lines.append(f"[{tag}.value] {e.value:.4g} ({what}: {e.method})")
            lines.append(f"[{tag}.ci] 95% robust interval {e.ci_low:.4g} to {e.ci_high:.4g}" if e.ci_low is not None else f"[{tag}.ci] no interval")
            lines.append(f"[{tag}.n] {e.n_control} control-side and {e.n_treated} treated-side rows")
    for r in state.get("refutations") or []:
        pe = placebo_entry(r.refuter)
        lines.append(f"[placebo:{c}.{r.refuter}.in_words] {pe.kind}: {pe.in_words} (source: {pe.source or 'ours'})")
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
        user = P.INTERPRET_USER.format(
            frame=L.frame_text(state),
            question=_question(state),
            contrast=d.contrast.key,
            material=_material(state),
            addresses="\n".join(sorted(allowed)),
            required="\n".join(required),
            errors=_rejected(errors),
        )
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
            if abs(parsed.bandwidth_left_stated - prim["h"]) > max(abs(prim["h"]), 1e-9) * tol:
                errors.append(f"bandwidth_left_stated {parsed.bandwidth_left_stated} does not match h = {prim['h']:.4g} on the control side")
            if abs(parsed.bandwidth_right_stated - prim["h_right"]) > max(abs(prim["h_right"]), 1e-9) * tol:
                errors.append(f"bandwidth_right_stated {parsed.bandwidth_right_stated} does not match h = {prim['h_right']:.4g} on the treated side")
            if parsed.n_left_stated != prim["n_h_left"] or parsed.n_right_stated != prim["n_h_right"]:
                errors.append(f"effective rows must be {prim['n_h_left']} control-side and {prim['n_h_right']} treated-side")
        if not errors:
            break
    out: dict[str, Any] = {"interpretations": [parsed], "debug": debug}
    if errors:
        out["interpret_errors"] = {d.contrast.key: errors}
    return out


# ------------------------------------------------------------------ feasibility, figures, assemble (facts)


def figures(state: SpecialistState) -> dict:
    """What this run drew, checked against the addresses it produced: the outcome against the score with a fit on each side,
    the score's density either side, each covariate's jump, the estimate across bandwidths and at the placebo cutoffs, and the
    estimate against its placebos."""
    from causal_agent.families.discontinuity import postviz as PR

    h = state["handoff"]
    names = state.get("columns") or {}
    c = state["contrast"].key if state.get("contrast") else None
    d: Design | None = state.get("design")
    cf = state.get("check_facts") or {}
    pts = state.get("placebo_points") or {}
    run_dir = Path(state["run_dir"]) if state.get("run_dir") else None
    specs = []
    if c and d is not None and state.get("canon_path") and Path(state["canon_path"]).exists():
        bins = pd.read_csv(run_dir / "bins.csv") if run_dir and (run_dir / "bins.csv").exists() else None
        specs.append(
            PR.rd_plot(bins, _canon(state), (d.bandwidths.h_left, d.bandwidths.h_right), int(d.spec.get("p", 1)), c, names.get(d.score.column, d.score.column))
        )
    if c and state.get("xall_path") and Path(state["xall_path"]).exists():
        specs.append(PR.density_test(pd.read_csv(state["xall_path"])["x"], cf.get("density"), c, sampled_by_side=bool(state.get("sampled_by_side"))))
    if c:
        specs.append(PR.covariate_continuity(cf.get("continuity"), c, names, float(_cfg()["covariate_continuity"]["p_value"]["soft"])))
        specs.append(PR.bandwidth_curve(pts.get("bandwidth_grid"), d.bandwidths.h_left if d is not None else None, c))
        specs.append(PR.bandwidth_curve(pts.get("window_sensitivity"), d.bandwidths.h_right if d is not None else None, c, name="window_sensitivity"))
        specs.append(PR.spec_sensitivity(pts.get("polynomial_grid"), pts.get("kernel_grid"), state.get("primary"), c))
        specs.append(PR.placebo_cutoffs(pts.get("placebo_cutoffs"), state.get("primary"), c))
    ests = [e.model_dump() for e in state.get("estimates") or []]
    specs.append(PV.effect_and_refutations([e for e in ests if e.get("modifier") is None], [r.model_dump() for r in state.get("refutations") or []], "placebo"))
    if c:
        specs.append(PV.effect_by_modifier(ests, c, names))
    kept, declines = LF.write(state.get("run_dir"), specs, LF.ok_addresses(h, state, "placebo"))
    _writer()({"figures": [s.id for s in kept]})
    return {"figures": [s.model_dump() for s in kept], "declines": declines}


def assemble(state: SpecialistState) -> dict:
    h = state["handoff"]
    names = state.get("columns") or {}
    d: Design | None = state.get("design")
    f: Feasibility | None = state.get("feasibility")
    lines = [
        f"QUESTION     {_question(state)}",
        f"LANE         {h.family} → {h.specialist}",
        f"OUTCOME      {h.outcome}    TREATMENT   {h.treatment or '(the cutoff rule)'}",
        "",
    ]
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
            label = f"{e.method:28}" if e.modifier is None else f"  within {names.get(e.modifier, e.modifier)} = {e.level:<12}"[:28].ljust(28)
            if e.error:
                lines.append(f"    {label} FAILED: {e.error}")
            else:
                ci = f" [{e.ci_low:.3g}, {e.ci_high:.3g}]" if e.ci_low is not None else ""
                kind = "within a level" if e.modifier is not None else "primary" if e.method == d.estimator else "secondary"
                lines.append(f"    {label} {e.value:+.4g}{ci}  rows={e.n_control}/{e.n_treated}  {kind}")
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
            lines.append(
                f"SCORE        {names.get(sc.column, sc.column)} treated when {rule}"
                + (f"; take-up {names.get(sc.takeup_column, sc.takeup_column)} = {sc.takeup_level!r}" if sc.takeup_column else "")
            )
        if s:
            lines.append(f"SHAPE        {s.kind}; {s.n_left} rows on the control side, {s.n_right} on the treated side; {s.distinct_scores} distinct scores")
        for r in state.get("checks") or []:
            lines.append(f"CHECK        {r.level:4} {r.address}  {r.detail}")
        a = state.get("assessment")
        if a:
            lines.append(f"ASSESS       {a.action}: {a.reason}  cites {', '.join(a.cites)}")
    if f:
        lines += ["", f"STOPPED AT   {f.stage}", f"REASON       {f.reason}"] + [f"FACT         {x}" for x in f.facts] + [f"WOULD FIX    {f.what_would_fix}"]
    lines += records.report_tail(state)
    debug = state.get("debug") or []
    if any(t.text for t in debug):
        lines += ["", "MODEL THOUGHTS (debug only)"] + [f"  [{t.node}] {t.text.strip()[:2000]}" for t in debug if t.text]
    report = "\n".join(lines)
    extra = {
        "score": state.get("score"),
        "shape": state.get("shape"),
        "covariates": state.get("covariates"),
        "assessment": state.get("assessment"),
        "primary": state.get("primary"),
        "placebo_points": state.get("placebo_points") or {},
    }
    records.write(state.get("run_dir"), records.artifacts(state, extra), report)
    result = records.result(
        state,
        report,
        {
            "score": state.get("score"),
            "shape": state.get("shape"),
            "covariates": state.get("covariates"),
            "assessment": state.get("assessment"),
            "ladder": [[a, text] for a, text in lad.lines()],
            "facts": [{"address": x.address, "text": x.render().split("] ", 1)[1], "value": x.value} for log in episodes.values() for x in log.facts],
            "relations": [
                {"column": x.column, "stands_for": None, "redundant_with": None, "nested_in": None, "modifier_candidate": x.modifier_candidate}
                for x in (lad.covariates.items if lad.covariates is not None else [])
            ],
        },
    )
    _writer()({"report": report})
    return {"report": report, "specialist_result": result}
