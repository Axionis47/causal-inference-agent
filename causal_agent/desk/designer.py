"""The Designer: one gated judgement that writes what the design rests on, in the dataset's terms. The desk node `design`
sits between the gate and the hand-off; the ready moment and the routing call it the same way. Three tries, then an honest
brief built from the fields."""

from __future__ import annotations

from langgraph.runtime import Runtime

from causal_agent.common.addresses import norm_address
from causal_agent.common.contracts import DecisionMade, DesignBrief, FamilyVerdict
from causal_agent.common.llm import structured
from causal_agent.desk.nodes import decide as D
from causal_agent.desk.nodes.frame import memory_of
from causal_agent.desk.prompts import routing as P
from causal_agent.desk.state import Context, RouteState
from causal_agent.families import registry as R
from causal_agent.families.base import Family
from causal_agent.memory.claims import ProbeResult
from causal_agent.memory.records import Memory

MAX_DESIGN_ATTEMPTS = 3


def _plain(c: str) -> str:
    return norm_address(str(c).strip().strip("[]").strip())


def check(brief: DesignBrief, family: Family, chosen: str, memory: Memory, probes: list[ProbeResult]) -> list[str]:
    """What the gate refuses: the wrong family, a decision the family does not list or lists once and the brief fills twice or not
    at all, a cite that resolves nowhere, a road the family has no decision for, or no road where it has one."""
    errors: list[str] = []
    if brief.family != chosen:
        errors.append(f"family {brief.family!r} is not the chosen family {chosen!r}")
    listed = [d.name for d in family.decisions]
    names = [d.name for d in brief.decisions]
    for n in names:
        if n not in listed:
            errors.append(f"decision {n!r} is not one the family lists; the names are: {', '.join(listed)}")
    for n in listed:
        if names.count(n) != 1:
            errors.append(f"decision {n!r} must appear exactly once; it appears {names.count(n)} times")
    for d in brief.decisions:
        if not d.rests_on:
            errors.append(f"decision {d.name!r} rests on nothing; cite at least one address")
        for a in d.rests_on:
            if not D.resolves(_plain(a), memory, probes):
                errors.append(f"decision {d.name!r} rests on {a!r}, which is not an address in the memory or the probes")
    for i, t in enumerate(brief.threats, start=1):
        for a in t.cites:
            if not D.resolves(_plain(a), memory, probes):
                errors.append(f"threat {i} cites {a!r}, which is not an address in the memory or the probes")
    has_road = family.decision("road") is not None
    if brief.road is not None and not has_road:
        errors.append("road must be null: this family lists no decision named road")
    if brief.road is None and has_road:
        errors.append("road is required for this family: one of backdoor, frontdoor, iv")
    return errors


def fallback(family: Family, errors: list[str]) -> DesignBrief:
    """The honest brief when three tries failed: every decision not decided, resting on the change card, betting on the family's
    stated assumption; no road named."""
    why = "the design brief failed its checks: " + "; ".join(errors[:3])
    return DesignBrief(
        family=family.name,
        road=None,
        target="average",
        decisions=[DecisionMade(name=d.name, choice="not decided", rests_on=["change:1.note"], reason=why) for d in family.decisions],
        threats=[],
        checks=[],
        bets_on=family.assumes,
    )


def _canonical(brief: DesignBrief) -> DesignBrief:
    """The cites as the memory spells them, once the gate passed."""
    for d in brief.decisions:
        d.rests_on = list(dict.fromkeys(_plain(a) for a in d.rests_on))
    for t in brief.threats:
        t.cites = list(dict.fromkeys(_plain(a) for a in t.cites))
    return brief


def _previous(state: RouteState) -> DesignBrief | None:
    """The brief of the last design that ran, so a revise says what it keeps and changes."""
    runs = state.get("runs") or []
    raw = (runs[-1].decision or {}).get("brief") if runs else None
    return DesignBrief.model_validate(raw) if raw else None


def design_brief(state: RouteState, runtime: Runtime[Context] | None = None) -> dict:
    memory = memory_of(state)
    fr, d = state.get("frame"), state.get("decision")
    registry = {f.name: f for f in R.knowledge()}
    if fr is None or d is None or d.chosen not in registry or state.get("gate_errors"):  # no family stands: nothing to design
        return {"brief": None}
    family = registry[d.chosen]
    probes: list[ProbeResult] = state.get("probes") or []
    verdict = next((v for v in state.get("family_verdicts") or [] if v.family == family.name), None)
    previous = _previous(state)
    addresses = sorted(D.citable(memory, probes))
    said = "\n".join(f"[user:turn:{s.turn}] {s.text}" for s in memory.said) or "(nothing said yet)"
    errors: list[str] = []
    thoughts = []
    brief: DesignBrief | None = None
    for attempt in range(1, MAX_DESIGN_ATTEMPTS + 1):
        prev = ("PREVIOUS ATTEMPT FAILED THESE CHECKS; fix them:\n" + "\n".join(f"- {e}" for e in errors)) if errors else ""
        brief, thought = structured(
            DesignBrief,
            P.DESIGN_SYSTEM,
            P.DESIGN_USER.format(
                question=state.get("question") or "",
                intent=fr.intent,
                outcome=fr.outcome or "none",
                cause=fr.cause or "none",
                scope=D._scope_text(fr),
                family=family.render(),
                memory=memory.render() or "(nothing known yet)",
                probes="\n".join(p.render() for p in probes) or "(no probe applies)",
                verdict=_render_verdict(verdict),
                said=said,
                previous=previous.render() if previous else "(none: this is the first design)",
                addresses=", ".join(addresses),
                previous_errors=prev,
            ),
            node=f"design:{attempt}",
        )
        thoughts.append(thought)
        errors = check(brief, family, d.chosen, memory, probes)
        D._writer()({"design": {"attempt": attempt, "errors": errors}})
        if not errors:
            return {"brief": _canonical(brief), "debug": thoughts}
    return {"brief": fallback(family, errors), "debug": thoughts}


def _render_verdict(v: FamilyVerdict | None) -> str:
    if v is None:
        return "(no verdict)"
    lines = [f"{v.family}: {'ADMISSIBLE' if v.admissible else 'not admissible'}" + (f"  concern: {v.concern}" if v.concern else "")]
    for n in v.needs:
        lines.append(f"    [{'met' if n.met else 'UNMET'}] {n.need}  cites: {', '.join(n.cites) or '-'}  {n.note}")
    return "\n".join(lines)
