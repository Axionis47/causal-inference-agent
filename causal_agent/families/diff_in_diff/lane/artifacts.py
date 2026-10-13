"""Design-qualified evidence, interpretations and result artifacts."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

from langgraph.types import Command
from pydantic import BaseModel, Field

from causal_agent.common import trace
from causal_agent.common.contracts import Interpretation
from causal_agent.families.diff_in_diff.lane.history import event
from causal_agent.families.diff_in_diff.lane.knowledge import load_checks
from causal_agent.families.diff_in_diff.lane.state import SpecialistState
from causal_agent.families.diff_in_diff.postviz import design_comparison
from causal_agent.lane import figures as F
from causal_agent.lane import records
from causal_agent.lane.episode import run_episode
from causal_agent.lane.ledger import Ledger
from causal_agent.lane.nodes import stop


class Interpretations(BaseModel):
    notes: str
    answers: list[Interpretation] = Field(description="one cited interpretation per successful design/component; preserve different populations and weights")


def comparisons(state: SpecialistState) -> list[dict]:
    out = []
    good = [e for e in state.get("estimates", []) if e.error is None and e.value is not None]
    for a, b in combinations(good, 2):
        keys = (
            "population",
            "aggregation",
            "component",
            "outcome",
            "contrast",
            "table_version",
            "horizons",
            "cohorts",
            "support",
            "included_units",
            "weighting",
            "weights",
        )
        comparable = all(a.target.get(k) == b.target.get(k) for k in keys)
        out.append(
            {
                "left": a.tag + ".value",
                "right": b.tag + ".value",
                "comparable": comparable,
                "numerical_difference": a.value - b.value if comparable else None,
                "same_direction": (a.value > 0) == (b.value > 0) if comparable else None,
                "reason": "same frozen population, support and weights; no universal agreement threshold"
                if comparable
                else "target population, temporal support or weights differ",
            }
        )
    return out


def material(state: SpecialistState) -> tuple[str, set[str]]:
    chunks, addresses = [], set()
    for a, text in state.get("ledger", Ledger()).lines():
        chunks.append(f"[{a}] {text}")
        addresses.add(a)
    for d in state.get("designs", {}).values():
        a = f"design:{d.id}"
        addresses.add(a)
        chunks.append(f"[{a}] {d.render()}")
    for e in state.get("estimates", []):
        if e.error:
            chunks.append(f"[{e.tag}.error] {e.error}")
            addresses.add(e.tag + ".error")
        else:
            brief = {
                k: e.target.get(k)
                for k in ("aggregation", "component", "population", "outcome", "horizons", "cohorts", "cohort_calendar", "support", "weighting")
            }
            brief["target_units"] = len(e.target.get("included_units", []))
            brief["comparison_units"] = e.n_control
            chunks.append(f"[{e.tag}.value] {e.value}; interval [{e.ci_low}, {e.ci_high}]; target {brief}")
            addresses.add(e.tag + ".value")
    for c in state.get("checks", []):
        chunks.append(f"[{c.address}] {c.level}: {c.detail}")
        addresses.add(c.address)
    for e in [*state.get("falsifications", []), *state.get("sensitivities", [])]:
        shown = e.model_dump(exclude={"target"})
        shown["target"] = {
            k: e.target.get(k)
            for k in (
                "aggregation",
                "component",
                "population",
                "outcome",
                "horizons",
                "cohorts",
                "cohort_calendar",
                "support",
                "weighting",
                "different_estimand",
            )
        }
        shown["target"]["target_units"] = len(e.target.get("included_units", []))
        chunks.append(f"[{e.address}] {shown}")
        addresses.add(e.address)
    chunks.append("DESIGN COMPARISONS\n" + json.dumps(comparisons(state)))
    return "\n".join(chunks), addresses


def interpret(state: SpecialistState) -> Command:
    text, addresses = material(state)
    good = [e for e in state.get("estimates", []) if e.error is None and e.value is not None]

    def gate(record: Interpretations, log):
        errors = []
        if len(record.answers) != len(good):
            errors.append("one interpretation per successful target component")
        used = set()
        for answer in record.answers:
            matched = [e for e in good if e.design_id == answer.design_id and e.tag + ".value" in answer.cites]
            if len(matched) != 1:
                errors.append("cite exactly one successful design-qualified estimate for each answer")
                continue
            e = matched[0]
            if e.tag in used:
                errors.append("a target component cannot be interpreted twice")
            used.add(e.tag)
            if abs(answer.effect_stated - e.value) > max(abs(e.value) * load_checks()["interpretation"]["relative_effect_tolerance"], 1e-8):
                errors.append("effect_stated differs from its cited estimate")
            if any(a not in addresses for a in answer.cites):
                errors.append("unrendered evidence address")
            d = state["designs"][e.design_id]
            if not answer.caveats:
                errors.append("state the identifying assumption and uncertainty")
            if d.inference == "inference_unavailable" and not any("inference" in c.lower() for c in answer.caveats):
                errors.append("unavailable inference must be explicit")
        return errors

    record, log, thoughts, errors = run_episode(
        Interpretations,
        "Explain each successful target using only the supplied records. A passing pretrend diagnostic does not prove parallel trends. Cite the design-qualified estimate. Distinguish imputation, projection, horizon and cohort targets and changing populations. Cohort IDs are internal period indices; use cohort_calendar for the dataset calendar label in prose. Do not declare agreement for different weights or support. Failed falsifications are caveats, sensitivities have no verdict. No causal certainty from a person's confirmation.",
        text,
        tools=None,
        budget=0,
        gate=gate,
        node="interpret",
    )
    extra = {"episodes": {"interpret:1": log}, "debug": thoughts, "reasoning_history": [event(state, "interpret", episode=log.model_dump())]}
    if record is None:
        return stop(
            "interpret",
            "interpretation failed its evidence gate",
            errors,
            "faithful interpretation of the recorded targets",
            extra,
            terminal_kind="exhausted_reasoning",
        )
    return Command(goto="figures", update={"interpretations": record.answers, **extra})


def figures(state: SpecialistState) -> dict:
    specs = [design_comparison(state.get("estimates", []))]
    _, ok = material(state)
    kept, declines = F.write(state.get("run_dir"), specs, ok)
    return {"figures": [x.model_dump() for x in kept], "declines": declines}


def assemble(state: SpecialistState) -> dict:
    h = state["handoff"]
    ledger = state.get("ledger")
    text, _ = material(state) if ledger is not None else ("", set())
    parts = [f"QUESTION {state.get('question') or h.question}", "DIFFERENCE-IN-DIFFERENCES", text]
    if ledger is not None:
        parts.append("LEDGER HISTORY\n" + json.dumps([x.model_dump() for x in ledger.history], indent=2))
    parts.append("CANDIDATE REVISIONS\n" + json.dumps(state.get("candidate_history", []), indent=2, default=str))
    parts.append("CANDIDATES\n" + json.dumps([x.model_dump() for x in state.get("candidates", [])], indent=2, default=str))
    parts.append("LEAD SELECTION\n" + json.dumps(state.get("lead_selection", {}), indent=2))
    parts += ["ANSWER " + x.answer + "\n" + "\n".join("CAVEAT " + c for c in x.caveats) for x in state.get("interpretations", [])]
    if state.get("feasibility"):
        parts.append("STOP " + state["feasibility"].model_dump_json())
    report = "\n\n".join(parts)
    extra = {
        "shape": state.get("shape"),
        "ledger": ledger,
        "candidates": state.get("candidates", []),
        "reasoning_history": state.get("reasoning_history", []),
        "candidate_history": state.get("candidate_history", []),
        "comparisons": comparisons(state),
        "ladder": ledger.lines() if ledger else [],
        "facts": [{"address": f.address, "text": f.text, "value": f.value} for log in state.get("episodes", {}).values() for f in log.facts],
    }
    arts = records.artifacts(state, extra)
    records.write(state.get("run_dir"), arts, report)
    result = records.result(state, report, extra)
    if state.get("run_dir"):
        trace.write(Path(state["run_dir"]) / "result.json", json.dumps(result, indent=2, default=str))
    return {"report": report, "specialist_result": result}
