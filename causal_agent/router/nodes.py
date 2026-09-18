"""Router nodes. Deterministic ones never call a model; model ones call structured() once."""

from __future__ import annotations

import os
from typing import Literal

from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime
from langgraph.types import Command, Send

from causal_agent.common.contracts import (
    Candidate,
    FamilyDecision,
    FamilyVerdict,
    Handoff,
    PrefilterVote,
    QuestionFrame,
)
from causal_agent.profile.datasets import load_dataset_pack
from causal_agent.common.llm import structured
from causal_agent.profile.pack import Pack
from causal_agent.knowledge import Family, load_registry, render_preferences
from causal_agent.router import prompts as P
from causal_agent.router.state import Context, FamilyTask, PrefilterTask, RouterState

MAX_DECIDE_ATTEMPTS = 3


# ------------------------------------------------------------------ helpers

_pack_cache: dict[str, Pack] = {}


def _pack(state: RouterState) -> Pack:
    name = state["dataset"]
    if name not in _pack_cache:
        _pack_cache[name] = load_dataset_pack(name)
    return _pack_cache[name]


def _registry(runtime: Runtime[Context] | None) -> list[Family]:
    path = runtime.context.registry_path if runtime and runtime.context else None
    return load_registry(path)


def _width_budget(runtime: Runtime[Context] | None) -> int:
    if runtime and runtime.context and runtime.context.width_budget:
        return runtime.context.width_budget
    return int(os.getenv("ROUTER_WIDTH_BUDGET", "150"))


def _index_cards(pack: Pack) -> list:
    """Columns frame may see: drop ids and constants deterministically."""
    return [c for c in pack.columns if c.profile.kind != "id" and not c.profile.constant]


def _writer():
    try:
        return get_stream_writer()
    except Exception:  # outside a graph run
        return lambda *_: None


def _scope_text(frame: QuestionFrame) -> str:
    s = frame.scope
    return f"filter={s.population_filter or 'none'}; window={s.window or 'none'}; contrast={s.contrast}; target={s.target}"


# ------------------------------------------------------------------ nodes


def load_pack(state: RouterState) -> dict:
    pack = _pack(state)
    if pack.unnoted_columns:
        raise ValueError(f"pack {pack.name!r} has columns without notes: {pack.unnoted_columns}")
    return {"prefilter_votes": [], "family_verdicts": [], "debug": [], "gate_errors": [], "decide_attempts": 0}


def fan_out_prefilter(state: RouterState) -> list[Send] | Literal["frame"]:
    pack = _pack(state)
    cards = _index_cards(pack)
    if len(cards) <= _width_budget(None):
        return "frame"
    changes = pack.render_changes()
    return [
        Send("prefilter", PrefilterTask(question=state["question"], changes=changes, column=c.name, card=c.render()))
        for c in cards
    ]


def prefilter(task: PrefilterTask) -> dict:
    vote, thought = structured(
        PrefilterVote,
        P.PREFILTER_SYSTEM,
        P.PREFILTER_USER.format(question=task["question"], changes=task["changes"], card=task["card"]),
        node=f"prefilter:{task['column']}",
    )
    vote.column = task["column"]
    return {"prefilter_votes": [vote], "debug": [thought]}


def frame(state: RouterState) -> dict:
    pack = _pack(state)
    cards = _index_cards(pack)
    votes = {v.column: v.relevant for v in state.get("prefilter_votes", [])}
    if votes:
        cards = [c for c in cards if votes.get(c.name, True)]
    index = "\n".join(f"[{c.address}] {c.name!r}: {c.first_sentence()}" for c in cards)
    fr, thought = structured(
        QuestionFrame,
        P.FRAME_SYSTEM,
        P.FRAME_USER.format(question=state["question"], digest=pack.digest(), column_index=index),
        node="frame",
    )
    _normalise_columns(fr, pack)
    _writer()({"frame": {"intent": fr.intent, "outcome": fr.outcome, "cause": fr.cause, "relevant": [c.column for c in fr.relevant_columns]}})
    return {"frame": fr, "debug": [thought], "family_verdicts": []}


def _normalise_columns(fr: QuestionFrame, pack: Pack) -> None:
    """The model may name columns by key (math_score). Downstream wants the file's own name (math score)."""
    for group in (fr.outcome_candidates, fr.cause_candidates, fr.relevant_columns):
        for cand in group:
            card = pack.column(cand.column)
            if card is not None:
                cand.column = card.name
    # a relevant "column" that is not in the file (an implicit unit, a concept) is dropped as a fact; the outcome and cause are gated instead
    fr.relevant_columns = [c for c in fr.relevant_columns if pack.column(c.column) is not None]


def fan_out_families(state: RouterState) -> list[Send]:
    pack = _pack(state)
    fr = state["frame"]
    assert fr is not None
    relevant_cards = []
    for cand in fr.relevant_columns:
        card = pack.column(cand.column)
        if card is not None:
            relevant_cards.append(card.render())
    cards_text = "\n\n".join(relevant_cards) or "(frame marked no columns relevant)"
    relevant = "; ".join(f"{c.column}: {c.reason}" for c in fr.relevant_columns) or "none"
    return [
        Send(
            "test_family",
            FamilyTask(
                question=state["question"],
                family=fam.name,
                family_text=fam.render(),
                intent=fr.intent,
                outcome=fr.outcome or "none",
                cause=fr.cause or "none",
                scope=_scope_text(fr),
                relevant=relevant,
                digest=pack.digest(),
                cards=cards_text,
            ),
        )
        for fam in _registry(None)
    ]


def test_family(task: FamilyTask) -> dict:
    verdict, thought = structured(
        FamilyVerdict,
        P.FAMILY_SYSTEM,
        P.FAMILY_USER.format(
            family=task["family_text"],
            question=task["question"],
            intent=task["intent"],
            outcome=task["outcome"],
            cause=task["cause"],
            scope=task["scope"],
            relevant=task["relevant"],
            digest=task["digest"],
            cards=task["cards"],
        ),
        node=f"test_family:{task['family']}",
    )
    verdict.family = task["family"]
    # The rule is stated in the prompt; enforce it so a verdict cannot be admissible with an unmet need.
    if verdict.admissible and not all(n.met for n in verdict.needs):
        verdict.admissible = False
        verdict.concern = (verdict.concern + " | " if verdict.concern else "") + "marked admissible with an unmet need; corrected"
    return {"family_verdicts": [verdict], "debug": [thought]}


def _citable_addresses(state: RouterState, pack: Pack) -> list[str]:
    """Addresses decide may cite: everything the verdicts cited, plus dataset and change addresses."""
    out: set[str] = {"dataset.note"} | {f"{ch.address}.note" for ch in pack.changes}
    for v in state.get("family_verdicts", []):
        for need in v.needs:
            out.update(a for a in need.cites if pack.resolve(a))
    return sorted(out)


def _render_verdicts(verdicts: list[FamilyVerdict]) -> str:
    out = []
    for v in verdicts:
        out.append(f"{v.family}: {'ADMISSIBLE' if v.admissible else 'not admissible'}" + (f"  concern: {v.concern}" if v.concern else ""))
        for n in v.needs:
            out.append(f"    [{'met' if n.met else 'UNMET'}] {n.need}  cites: {', '.join(n.cites) or '-'}  {n.note}")
    return "\n".join(out)


def decide(state: RouterState, runtime: Runtime[Context]) -> dict:
    pack = _pack(state)
    fr = state["frame"]
    assert fr is not None
    errors = state.get("gate_errors") or []
    prev = ("PREVIOUS ATTEMPT FAILED THESE CHECKS; fix them:\n" + "\n".join(f"- {e}" for e in errors)) if errors else ""
    decision, thought = structured(
        FamilyDecision,
        P.DECIDE_SYSTEM,
        P.DECIDE_USER.format(
            question=state["question"],
            intent=fr.intent,
            outcome=fr.outcome or "none",
            cause=fr.cause or "none",
            scope=_scope_text(fr),
            changes=pack.render_changes(),
            preferences=render_preferences(_registry(runtime)),
            verdicts=_render_verdicts(state.get("family_verdicts", [])),
            addresses=", ".join(_citable_addresses(state, pack)),
            previous_errors=prev,
        ),
        node="decide",
    )
    return {"decision": decision, "debug": [thought]}


def gate(state: RouterState, runtime: Runtime[Context]) -> Command[Literal["decide", "handoff", "__end__"]]:
    pack = _pack(state)
    fr = state["frame"]
    d = state["decision"]
    assert fr is not None and d is not None
    verdicts = {v.family: v for v in state.get("family_verdicts", [])}
    registry_names = {f.name for f in _registry(runtime)}
    errors: list[str] = []

    for a in d.cites:
        if not pack.resolve(a):
            errors.append(f"decision cites {a!r}, which is not a pack address; cite only addresses from the list given, or leave cites empty")
    for r in d.rejected:
        for a in r.cites:
            if not pack.resolve(a):
                errors.append(f"rejection of {r.family} cites {a!r}, which is not a pack address; cite only addresses from the list given, or leave cites empty")
    no_family = not d.admissible and str(d.chosen).strip().lower() in {"none", ""}
    if not no_family and d.chosen not in d.admissible:
        errors.append(f"chosen family {d.chosen!r} is not in admissible {d.admissible}; if nothing is admissible, set chosen to 'none'")
    verdict_admissible = {n for n, v in verdicts.items() if v.admissible}
    for fam in d.admissible:
        if fam not in verdict_admissible:
            errors.append(f"{fam!r} listed admissible but its verdict says not admissible")
    covered = set(d.admissible) | {r.family for r in d.rejected}
    missing = registry_names - covered
    if missing:
        errors.append(f"families not accounted for: {sorted(missing)}")
    unknown = covered - registry_names
    if unknown:
        errors.append(f"unknown families named: {sorted(unknown)}")
    relevant_names = {c.column for c in fr.relevant_columns}
    if fr.outcome is None or pack.column(fr.outcome) is None:
        errors.append(f"outcome {fr.outcome!r} is not a column")
    elif fr.outcome not in relevant_names:
        errors.append(f"outcome {fr.outcome!r} missing from relevant columns")
    if fr.intent == "effect_of_change":
        if fr.cause is None or pack.column(fr.cause) is None:
            errors.append(f"cause {fr.cause!r} is not a column")
        elif fr.cause not in relevant_names:
            errors.append(f"cause {fr.cause!r} missing from relevant columns")

    attempts = state.get("decide_attempts", 0) + 1
    _writer()({"gate": {"passed": not errors, "errors": errors, "attempt": attempts}})
    if not errors:
        return Command(update={"gate_errors": [], "decide_attempts": attempts}, goto="handoff")
    if attempts < MAX_DECIDE_ATTEMPTS:
        return Command(update={"gate_errors": errors, "decide_attempts": attempts}, goto="decide")
    return Command(update={"gate_errors": errors, "decide_attempts": attempts, "handoff": None}, goto="__end__")


def handoff(state: RouterState, runtime: Runtime[Context]) -> dict:
    pack = _pack(state)
    fr = state["frame"]
    d = state["decision"]
    assert fr is not None and d is not None
    fam = next((f for f in _registry(runtime) if f.name == d.chosen), None)
    if fam is None:  # no admissible family: honest stop, record kept, no hand-off
        record = _decision_record(state, None)
        _writer()({"handoff": None, "decision_record": record})
        return {"handoff": None, "decision_record": record}
    from causal_agent.desk.handoff import build, claims_for

    table, probes = claims_for(pack.name)
    h = build(question=state["question"], frame=fr, decision=d, family=fam, pack=pack, claims=table, probes=probes)
    record = _decision_record(state, h)
    _writer()({"handoff": h.model_dump(), "decision_record": record})
    return {"handoff": h, "decision_record": record}


def _decision_record(state: RouterState, h: Handoff | None) -> str:
    fr = state["frame"]
    d = state["decision"]
    assert fr is not None and d is not None
    lines = [
        f"QUESTION     {state['question']}",
        f"READ AS      {fr.intent} · outcome: {fr.outcome} · cause: {fr.cause or 'none'} · {_scope_text(fr)}",
        "RELEVANT     " + ", ".join(f"{c.column} ({c.reason})" for c in fr.relevant_columns),
        "",
        "FAMILY CHECKS",
    ]
    for v in sorted(state.get("family_verdicts", []), key=lambda v: (not v.admissible, v.family)):
        met = sum(n.met for n in v.needs)
        lines.append(f"  {v.family:20} {'admissible' if v.admissible else 'not admissible':15} {met}/{len(v.needs)} needs met" + (f"   concern: {v.concern}" if v.concern else ""))
        for n in v.needs:
            lines.append(f"      {'met  ' if n.met else 'UNMET'} {n.need}   [{', '.join(n.cites) or '-'}] {n.note}")
    lines += [
        "",
        (f"CHOSEN       {d.chosen} → {h.specialist}{'' if h.supported_now else ' (not supported yet)'}" if h else "CHOSEN       none: no family is admissible for this question on this data"),
        f"BETS ON      {d.chosen_assumption}",
        f"WHY          {d.why_over_alternatives}",
    ]
    for r in d.rejected:
        lines.append(f"OVER         {r.family}: {r.reason}   [{', '.join(r.cites) or '-'}]")
    debug = state.get("debug") or []
    if any(t.text for t in debug):
        lines += ["", "MODEL THOUGHTS (debug only)"]
        for t in debug:
            if t.text:
                lines.append(f"  [{t.node}] {t.text.strip()[:2000]}")
    return "\n".join(lines)


def route_specialist(state: RouterState) -> str:
    h = state.get("handoff")
    if h is None:
        return "__end__"
    return f"specialist_{h.family}"
