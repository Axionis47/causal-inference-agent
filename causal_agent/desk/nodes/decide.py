"""From the memory to a hand-off: the fit over the memory says which families stand (code), one judgement chooses among them
when more than one does, a gate checks the choice, and the builder projects the pack."""

from __future__ import annotations

from typing import Literal

from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime
from langgraph.types import Command

from causal_agent.common.addresses import norm_address
from causal_agent.common.contracts import FamilyDecision, FamilyVerdict, Handoff, NeedCheck, QuestionFrame, Rejection, render_change_text
from causal_agent.common.llm import structured
from causal_agent.desk import handoff as H
from causal_agent.desk.nodes.frame import memory_of
from causal_agent.desk.prompts import routing as P
from causal_agent.desk.state import Context, RouteState
from causal_agent.families import registry as R
from causal_agent.knowledge import Family, render_preferences
from causal_agent.memory import ops
from causal_agent.memory import views as V
from causal_agent.memory.catalogue import load_catalogue
from causal_agent.memory.claims import ProbeResult, Status
from causal_agent.memory.records import Memory
from causal_agent.memory.views import table_of

MAX_DECIDE_ATTEMPTS = 3


def _writer():
    try:
        return get_stream_writer()
    except Exception:
        return lambda *_: None


def _registry(runtime: Runtime[Context] | None) -> list[Family]:
    return R.knowledge()


def _scope_text(frame: QuestionFrame) -> str:
    s = frame.scope
    return f"filter={s.population_filter or 'none'}; window={s.window or 'none'}; contrast={s.contrast}; target={s.target}"


# ------------------------------------------------------------------ fit (fact)


def verdicts_from(memory: Memory, status: Status, probes: list[ProbeResult], registry: list[Family]) -> list[FamilyVerdict]:
    """One verdict per family, by code, from the fit table. A family stands when nothing struck it and every need the file or the
    person could have settled is settled and fits. A belief not asked yet is listed unmet but does not strike the family: only the
    person can give it, and the routing before the interview cannot wait for it. The assumption bet on must name it."""
    cat = load_catalogue()
    all_needs = R.needs()
    out = []
    for fam in registry:
        needs_spec = all_needs.get(fam.name)
        cells = status.table.get(fam.name, {})
        needs: list[NeedCheck] = []
        blocking = False
        for kind in needs_spec.requires if needs_spec else []:
            cell = cells.get(kind, "not_needed")
            spec = cat.kinds[kind]
            if spec.per_column:
                cites = [f"claim:{c.key}" for c in memory.to_claims(cat).claims.values() if c.kind == kind and c.status != "empty"][:6]
            else:
                cites = [f"claim:{kind}.{n}" for n in memory.values_of(f"claim:{kind}")][:6]
            if cell == "fits":
                needs.append(NeedCheck(need=f"{kind} settled and fits", met=True, cites=cites, note=spec.about))
            elif cell == "does_not_fit":
                needs.append(NeedCheck(need=f"{kind} fits", met=False, cites=cites, note=status.struck.get(fam.name, f"{kind} does not fit")))
                blocking = True
            elif spec.uncheckable:
                needs.append(NeedCheck(need=f"{kind} known", met=False, cites=[], note="not asked yet: a belief only the person can give"))
            else:
                needs.append(NeedCheck(need=f"{kind} settled", met=False, cites=[], note="nothing known yet"))
                blocking = True
        failed = [p for p in probes if p.family == fam.name and p.passed is False]
        for p in failed:
            needs.append(NeedCheck(need=p.name, met=False, cites=[p.address], note=p.detail))
            blocking = True
        concern = status.struck.get(fam.name, "")
        out.append(FamilyVerdict(family=fam.name, admissible=not blocking and fam.name in status.surviving, needs=needs, concern=concern))
    return out


def fit(state: RouteState, runtime: Runtime[Context]) -> dict:
    memory = memory_of(state)
    df = table_of(memory)
    probes = ops.probe(memory, df, R.REGISTRY.values())
    from causal_agent.profile import datasets as DS

    status = ops.fit(memory, probes, R.needs(), columns=V.in_play(memory, state.get("frame"), DS.dataset_entries().get(memory.name) or {}))
    verdicts = verdicts_from(memory, status, probes, _registry(runtime))
    _writer()({"fit": {"surviving": status.surviving, "struck": status.struck, "admissible": [v.family for v in verdicts if v.admissible]}})
    return {"family_verdicts": verdicts, "probes": probes, "fit_status": status.model_dump()}


# ------------------------------------------------------------------ decide (judgement) and gate


def citable(memory: Memory, probes: list[ProbeResult]) -> set[str]:
    out = set(memory.addresses()) | {"dataset.note", "change:1.note"}
    out |= {f"{c.address}.note" for c in memory.columns.values()}
    out |= {p.address for p in probes}
    return out


def resolves(address: str, memory: Memory, probes: list[ProbeResult]) -> bool:
    return norm_address(address) in {norm_address(a) for a in citable(memory, probes)}


def _render_verdicts(verdicts: list[FamilyVerdict]) -> str:
    out = []
    for v in verdicts:
        out.append(f"{v.family}: {'ADMISSIBLE' if v.admissible else 'not admissible'}" + (f"  concern: {v.concern}" if v.concern else ""))
        for n in v.needs:
            out.append(f"    [{'met' if n.met else 'UNMET'}] {n.need}  cites: {', '.join(n.cites) or '-'}  {n.note}")
    return "\n".join(out)


def decide(state: RouteState, runtime: Runtime[Context]) -> dict:
    memory = memory_of(state)
    fr = state["frame"]
    assert fr is not None
    verdicts = state.get("family_verdicts", [])
    admissible = [v.family for v in verdicts if v.admissible]
    if len(admissible) == 1 and not state.get("gate_errors"):  # one family stands: no judgement to make
        fam = admissible[0]
        family = next(f for f in _registry(runtime) if f.name == fam)
        unmet = [n.note for v in verdicts if v.family == fam for n in v.needs if not n.met]
        decision = FamilyDecision(
            admissible=[fam],
            chosen=fam,
            chosen_assumption=family.assumes + ((" Not yet asked: " + "; ".join(unmet)) if unmet else ""),
            why_over_alternatives="only admissible family",
            rejected=[
                Rejection(family=v.family, reason=next((n.note for n in v.needs if not n.met), v.concern or "does not fit"), cites=[])
                for v in verdicts
                if not v.admissible
            ],
            cites=[],
        )
        return {"decision": decision}
    errors = state.get("gate_errors") or []
    prev = ("PREVIOUS ATTEMPT FAILED THESE CHECKS; fix them:\n" + "\n".join(f"- {e}" for e in errors)) if errors else ""
    addresses = sorted(a for v in verdicts for n in v.needs for a in n.cites) + ["dataset.note", "change:1.note"]
    decision, thought = structured(
        FamilyDecision,
        P.DECIDE_SYSTEM,
        P.DECIDE_USER.format(
            question=state["question"],
            intent=fr.intent,
            outcome=fr.outcome or "none",
            cause=fr.cause or "none",
            scope=_scope_text(fr),
            changes=render_change_text(V.fields_of(memory, "change"), V.fields_of(memory, "assignment")),
            preferences=render_preferences(_registry(runtime)),
            verdicts=_render_verdicts(verdicts),
            addresses=", ".join(dict.fromkeys(addresses)),
            previous_errors=prev,
        ),
        node="decide",
    )
    return {"decision": decision, "debug": [thought]}


def gate(state: RouteState, runtime: Runtime[Context]) -> Command[Literal["decide", "handoff", "__end__"]]:
    memory = memory_of(state)
    fr, d = state["frame"], state["decision"]
    assert fr is not None and d is not None
    probes = state.get("probes") or []
    verdicts = {v.family: v for v in state.get("family_verdicts", [])}
    registry_names = {f.name for f in _registry(runtime)}
    errors: list[str] = []
    for a in d.cites:
        if not resolves(a, memory, probes):
            errors.append(f"decision cites {a!r}, which is not a pack address; cite only addresses from the list given, or leave cites empty")
    for r in d.rejected:
        for a in r.cites:
            if not resolves(a, memory, probes):
                errors.append(
                    f"rejection of {r.family} cites {a!r}, which is not a pack address; cite only addresses from the list given, or leave cites empty"
                )
    no_family = not d.admissible and str(d.chosen).strip().lower() in {"none", ""}
    if not no_family and d.chosen not in d.admissible:
        errors.append(f"chosen family {d.chosen!r} is not in admissible {d.admissible}; if nothing is admissible, set chosen to 'none'")
    for fam in d.admissible:
        if not (verdicts.get(fam) and verdicts[fam].admissible):
            errors.append(f"{fam!r} listed admissible but the fit says not admissible")
    covered = set(d.admissible) | {r.family for r in d.rejected}
    if registry_names - covered:
        errors.append(f"families not accounted for: {sorted(registry_names - covered)}")
    if covered - registry_names:
        errors.append(f"unknown families named: {sorted(covered - registry_names)}")
    relevant_names = {c.column for c in fr.relevant_columns}
    if fr.outcome is None or memory.column(fr.outcome) is None:
        errors.append(f"outcome {fr.outcome!r} is not a column")
    elif fr.outcome not in relevant_names:
        errors.append(f"outcome {fr.outcome!r} missing from relevant columns")
    if fr.intent == "effect_of_change":
        if fr.cause is None or memory.column(fr.cause) is None:
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


# ------------------------------------------------------------------ handoff (fact)


def handoff(state: RouteState, runtime: Runtime[Context]) -> dict:
    memory = memory_of(state)
    fr, d = state["frame"], state["decision"]
    assert fr is not None and d is not None
    fam = R.REGISTRY.get(d.chosen)
    if fam is None:  # no admissible family: honest stop, record kept, no hand-off
        record = decision_record(state, None)
        _writer()({"handoff": None, "decision_record": record})
        return {"handoff": None, "decision_record": record}
    h = H.build(question=state["question"], frame=fr, decision=d, family=fam, memory=memory, probes=state.get("probes") or [])
    record = decision_record(state, h)
    _writer()({"handoff": h.model_dump(), "decision_record": record})
    return {"handoff": h, "decision_record": record}


def decision_record(state: RouteState, h: Handoff | None) -> str:
    fr, d = state["frame"], state["decision"]
    assert fr is not None and d is not None
    lines = [
        f"QUESTION     {state['question']}",
        f"READ AS      {fr.intent} · outcome: {fr.outcome} · cause: {fr.cause or 'none'} · {_scope_text(fr)}",
        "RELEVANT     " + ", ".join(f"{c.column} ({c.reason})" for c in fr.relevant_columns),
        "",
        "FAMILY FIT (from the memory)",
    ]
    for v in sorted(state.get("family_verdicts", []), key=lambda v: (not v.admissible, v.family)):
        met = sum(n.met for n in v.needs)
        lines.append(
            f"  {v.family:20} {'admissible' if v.admissible else 'not admissible':15} {met}/{len(v.needs)} needs met"
            + (f"   concern: {v.concern}" if v.concern else "")
        )
        for n in v.needs:
            lines.append(f"      {'met  ' if n.met else 'UNMET'} {n.need}   [{', '.join(n.cites) or '-'}] {n.note}")
    lines += [
        "",
        (
            f"CHOSEN       {d.chosen} → {h.specialist}{'' if h.supported_now else ' (not supported yet)'}"
            if h
            else "CHOSEN       none: no family is admissible for this question on this data"
        ),
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


def route_specialist(state: RouteState) -> str:
    h = state.get("handoff")
    return "__end__" if h is None else f"specialist_{h.family}"
