"""The judge: one reading of the context pack at the ready moment, by a second model call, gated by code and never blocking.
It says ready, risky or not ready and names the fields it would not trust, each with a cite; the page shows it beside the
fields and the ready moment says it in one line. The person still says the word."""

from __future__ import annotations

from langgraph.runtime import Runtime

from causal_agent.common import trace
from causal_agent.common.contracts import FamilyVerdict, Handoff
from causal_agent.common.llm import structured
from causal_agent.common.numbers import grounded, numbers_in
from causal_agent.desk import handoff as H
from causal_agent.desk.contracts import Readiness
from causal_agent.desk.nodes import decide as D
from causal_agent.desk.nodes.frame import memory_of
from causal_agent.desk.prompts import routing as P
from causal_agent.desk.state import Context, DeskState
from causal_agent.families import registry as R
from causal_agent.memory.claims import ProbeResult
from causal_agent.memory.records import Memory

MAX_JUDGE_ATTEMPTS = 3


def check(rec: Readiness, pack: Handoff, memory: Memory, probes: list[ProbeResult]) -> list[str]:
    """What code holds the judge to: a verdict with the gaps it needs, every gap on a field the pack carries, every cite an address
    that resolves, and no number the pack does not hold."""
    errors: list[str] = []
    if rec.verdict == "ready" and rec.gaps:
        errors.append("verdict is ready but gaps are named; a named gap makes it risky or not_ready")
    if rec.verdict != "ready" and not rec.gaps:
        errors.append(f"verdict is {rec.verdict} but no gap is named; name at least one field, with a cite")
    if not rec.summary.strip():
        errors.append("summary is empty")
    carried = set(pack.facts) | {b.address for b in pack.columns}
    for g in rec.gaps:
        if g.address not in carried:
            errors.append(f"gap names {g.address!r}, which is not a field the pack carries")
        for a in g.cites:
            if not D.resolves(a, memory, probes):
                errors.append(f"gap on {g.address} cites {a!r}, which is not an address in the pack")
        if not g.cites:
            errors.append(f"gap on {g.address} cites nothing")
    known = [float(f.value) for f in pack.facts.values() if isinstance(f.value, (int, float)) and not isinstance(f.value, bool)]
    known += [float(p.value) for p in probes if isinstance(p.value, (int, float)) and not isinstance(p.value, bool)]
    for text in [rec.summary] + [g.reason for g in rec.gaps]:
        loose = [n for n in numbers_in(text) if not grounded(n, known)]
        if loose:
            errors.append(f"{text[:60]!r} states {loose}, which the pack does not hold")
    return errors


def judge(state: DeskState, runtime: Runtime[Context] | None = None) -> dict:
    """The node: the pack the ready moment reads back is built here and kept as the preview; the judge reads it; three tries,
    then no verdict, and the ready moment goes on without one."""
    memory = memory_of(state)
    fr, d = state.get("frame"), state.get("decision")
    registry = {f.name: f for f in R.knowledge()}
    if fr is None or d is None or d.chosen not in registry or state.get("gate_errors"):
        return {"readiness": None, "preview": None}
    fam = registry[d.chosen]
    probes: list[ProbeResult] = state.get("probes") or []
    brief = state.get("brief")
    pack = H.build(question=state["question"], frame=fr, decision=d, family=R.REGISTRY[fam.name], memory=memory, probes=probes, brief=brief)
    verdict: FamilyVerdict | None = next((v for v in state.get("family_verdicts") or [] if v.family == fam.name), None)
    addresses = sorted(D.citable(memory, probes) | set(pack.facts))
    errors: list[str] = []
    thoughts = []
    for attempt in range(1, MAX_JUDGE_ATTEMPTS + 1):
        prev = ("PREVIOUS ATTEMPT FAILED THESE CHECKS; fix them:\n" + "\n".join(f"- {e}" for e in errors)) if errors else ""
        try:
            rec, thought = structured(
                Readiness,
                P.JUDGE_SYSTEM,
                P.JUDGE_USER.format(
                    question=state.get("question") or "",
                    family=fam.name,
                    specialist=pack.specialist,
                    bets_on=brief.bets_on if brief is not None else fam.assumes,
                    context=pack.render_context(),
                    verdict=_render_verdict(verdict),
                    addresses=", ".join(addresses),
                    previous_errors=prev,
                ),
                node=f"judge:{attempt}",
            )
        except ValueError as e:
            errors = trace.gate([str(e)], name="gate:judge")
            D._writer()({"judge": {"attempt": attempt, "errors": errors}})
            continue
        thoughts.append(thought)
        errors = trace.gate(check(rec, pack, memory, probes), name="gate:judge")
        D._writer()({"judge": {"attempt": attempt, "errors": errors, "verdict": rec.verdict if not errors else None}})
        if not errors:
            return {"readiness": rec, "preview": pack, "debug": thoughts}
    return {"readiness": None, "preview": pack, "debug": thoughts}


def _render_verdict(v: FamilyVerdict | None) -> str:
    if v is None:
        return "(no verdict)"
    lines = [f"{v.family}: {'admissible' if v.admissible else 'not admissible'}" + (f"; concern: {v.concern}" if v.concern else "")]
    lines += [f"  {'met' if n.met else 'NOT MET'}: {n.note}" + (f" [{', '.join(n.cites)}]" if n.cites else "") for n in v.needs]
    return "\n".join(lines)
