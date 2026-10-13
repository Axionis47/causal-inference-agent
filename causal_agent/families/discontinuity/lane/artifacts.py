"""Explain and publish recorded RD reasoning and frozen-design results."""

from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path

from causal_agent.common import trace
from causal_agent.common.addresses import norm_address
from causal_agent.families.discontinuity.lane.contracts import RDInterpretation
from causal_agent.lane import figures as LF
from causal_agent.lane import records
from causal_agent.lane.episode import run_episode
from causal_agent.viz.spec import FigureSpec, Series

INTERPRET = (
    """Explain one frozen design's results. Preserve the endpoint and target, its local population and conditional assumptions. Describe eligibility endpoints as eligibility or assignment effects (ITT), never effects of actual receipt. Use the confidence-set status exactly: bounded means no accepted searched boundary; unresolved_tail means an accepted boundary remains. Cite the estimate and any confidence set, every flagged check and failed falsification. Sensitivity is a range, never a passing test. An unresolved-tail or disconnected confidence set must be described honestly, without inventing a finite interval. Set the effect_stated field to the primary estimate.value exactly; null is allowed only when no point estimate exists. Discuss a confidence set only when a confidence_set record is shown. All stated numeric effect values must match the records. """
    + "Copy the exact bracketed citation addresses. Cite only the recorded results shown to you. Provenance references inside assumption text are not separate available evidence: cite the shown design assumptions address instead."
)


def _material(state, d):
    lines = [
        (f"design:{d.id}.identity", f"design_id={d.id}; endpoint_id={d.endpoint_id}; contrast={d.contrast.key}"),
        (f"design:{d.id}.target", json.dumps(d.target)),
        (f"design:{d.id}.bandwidths", d.bandwidths.model_dump_json()),
        (f"design:{d.id}.assumptions", json.dumps(d.assumptions)),
    ]
    required = []
    for e in state.get("estimates", []):
        if e.design_id != d.id:
            continue
        lines.extend(
            [
                (f"{e.tag}.value", str(e.value)),
                (f"{e.tag}.ci", f"{e.ci_low}, {e.ci_high}"),
                (f"{e.tag}.n", f"{e.n_control}, {e.n_treated}"),
                (f"{e.tag}.p", str(e.p_value)),
            ]
        )
        if not e.secondary:
            required += [f"{e.tag}.value", f"{e.tag}.ci"]
        if e.confidence_set:
            lines.append((f"{e.tag}.confidence_set", e.confidence_set.model_dump_json()))
            if not e.secondary:
                required.append(f"{e.tag}.confidence_set")
        if e.error:
            lines.append((f"{e.tag}.error", e.error))
    for c in d.checks.results:
        lines.append((c.address, f"{c.level}: {c.detail}"))
        if c.level != "pass":
            required.append(c.address)
    for f in state.get("falsifications", []):
        if f.design_id != d.id:
            continue
        tag = f"falsification:{d.id}.{f.contrast}.{f.refuter}"
        lines.append((tag, f.model_dump_json()))
        if f.passed is False:
            required.append(tag)
    for s in state.get("sensitivities", []):
        if s.design_id == d.id:
            lines.append((f"sensitivity:{d.id}.{s.contrast}.{s.method}", s.model_dump_json()))
    return lines, required


def interpret(state):
    records_out = []
    visits = []
    thoughts = []
    failures = {}
    episodes = {}
    for d in state.get("designs", {}).values():
        lines, required = _material(state, d)
        material = f"QUESTION: {state['handoff'].question}\n" + "\n".join(f"[{a}] {v}" for a, v in lines)
        addresses = {a for a, _ in lines}
        primary = next(e for e in state["estimates"] if e.design_id == d.id and not e.secondary)
        material += "\n\nALLOWED CITATION ADDRESSES:\n" + "\n".join(sorted(addresses))
        material += "\n\nREQUIRED CITATIONS (copy these addresses exactly):\n" + "\n".join(required)

        def gate(rec, log, d=d, addresses=addresses, required=required, primary=primary):
            errors = []
            rec.cites = [norm_address(a) for a in rec.cites]
            if rec.design_id != d.id or rec.endpoint_id != d.endpoint_id:
                errors.append("copy the frozen design and endpoint identifiers")
            if rec.contrast != d.contrast.key:
                errors.append("copy the recorded contrast")
            if rec.target_contrast != d.target["contrast"]:
                errors.append("copy target_contrast from the frozen target, preserving assignment, receipt or eligibility")
            if d.endpoint_id.startswith("eligibility") and not any(word in rec.answer.lower() for word in ("eligibility", "assignment", "itt")):
                errors.append("eligibility target: state the eligibility/assignment ITT effect, do not claim the effect of actual receipt")
            if not primary.confidence_set and "confidence set" in " ".join([rec.answer, *rec.caveats]).lower():
                errors.append("no confidence_set record exists for this design; describe its confidence interval without inventing an inversion status")
            if primary.confidence_set and primary.confidence_set.status == "bounded" and "unresolved" in " ".join([rec.answer, *rec.caveats]).lower():
                errors.append("the recorded confidence set is bounded; its searched boundaries were rejected, so do not claim unresolved tails")
            inline = [
                norm_address(a)
                for a in re.findall(
                    r"\[((?:design|estimate|check|falsification|sensitivity|claim|col|ladder|probe|artifact|user|eval):[^\]\n]+)\]",
                    "\n".join([rec.answer, *rec.caveats]),
                )
            ]
            errors += [f"{a}: unresolved citation in explanation text" for a in inline if a not in addresses]
            errors += [f"{a}: unresolved citation" for a in rec.cites if a not in addresses]
            errors += [f"cite required evidence {a}" for a in required if a not in rec.cites]
            if primary.value is None:
                if rec.effect_stated is not None:
                    errors.append("there is no identified point estimate to state")
            elif rec.effect_stated is None or abs(rec.effect_stated - primary.value) > max(abs(primary.value), 1e-9) * 0.01:
                errors.append(f"set effect_stated to the primary point estimate {primary.value}; it must match within one percent")
            return errors

        rec, log, debug, errors = run_episode(RDInterpretation, INTERPRET, material, tools=None, budget=0, gate=gate, node=f"interpret:{d.id}")
        episodes[f"interpret:{d.id}"] = log
        visits.append(
            {"node": "interpret", "design_id": d.id, "view": material, "record": rec.model_dump() if rec else None, "log": log.model_dump(), "errors": errors}
        )
        thoughts += debug
        if rec is None:
            failures[d.id] = errors
            rec = RDInterpretation(
                target_contrast=d.target["contrast"],
                design_id=d.id,
                endpoint_id=d.endpoint_id,
                contrast=d.contrast.key,
                effect_stated=primary.value,
                answer="The result is recorded for this local target. The explanation could not pass its evidence gate.",
                caveats=d.assumptions + [f"{c.name}: {c.detail}" for c in d.checks.flags],
                cites=required,
            )
        records_out.append(rec)
    return {
        "interpretations": records_out,
        "episodes": episodes,
        "reasoning_history": [
            {"node": "interpret", "design_ids": list(state.get("designs", {})), "episode_visits": [log.visit for log in episodes.values()], "errors": failures}
        ],
        "episode_visits": visits,
        "debug": thoughts,
        "interpretation_errors": failures,
    }


def comparison_groups(state):
    """Compare only the same contrast, local population, scale and analysis rows."""
    groups = {}
    for d in state.get("designs", {}).values():
        target = {k: v for k, v in d.target.items() if k != "assumptions"}
        identity = json.dumps({"target": target, "rows": d.rows}, sort_keys=True)
        groups.setdefault(identity, []).append(d.id)
    return list(groups.values())


def comparisons(state):
    records_out = []
    groups = comparison_groups(state)
    estimates = {e.design_id: e for e in state.get("estimates", []) if not e.secondary}
    for left, right in combinations(state.get("designs", {}).values(), 2):
        comparable = any(left.id in group and right.id in group for group in groups)
        a, b = estimates.get(left.id), estimates.get(right.id)
        records_out.append(
            {
                "left_design_id": left.id,
                "right_design_id": right.id,
                "comparable": comparable,
                "reason": "same contrast, target, outcome and analysis rows" if comparable else "different contrast, target, local population or analysis rows",
                "left_target": left.target,
                "right_target": right.target,
                "difference": a.value - b.value
                if comparable and a and b and a.value is not None and b.value is not None and not a.error and not b.error
                else None,
                "estimate_addresses": [e.tag for e in (a, b) if e],
                "conditional_on_recorded_assumptions": True,
            }
        )
    return records_out


def figures(state):
    specs = []
    ok = set(state["handoff"].addresses())
    for d in state.get("designs", {}).values():
        lines, _ = _material(state, d)
        ok.update(a for a, _ in lines)
    for group in comparison_groups(state):
        estimates = [e for e in state.get("estimates", []) if e.design_id in group and not e.secondary and e.value is not None and not e.error]
        if not estimates:
            continue
        d = state["designs"][group[0]]
        specs.append(
            FigureSpec(
                id=f"rd_estimates_{group[0]}",
                kind="interval",
                title=f"{d.estimand}: recorded design estimates",
                x_label="design",
                y_label=state["handoff"].outcome,
                series=[
                    Series(
                        name="effect",
                        x=[e.design_id for e in estimates],
                        y=[e.value for e in estimates],
                        lo=[e.ci_low for e in estimates],
                        hi=[e.ci_high for e in estimates],
                    )
                ],
                note="Only results with the same target and analysis population share this panel. Intervals follow their recorded inference.",
                draws_on=[f"{e.tag}.value" for e in estimates],
            )
        )
    kept, declines = LF.write(state.get("run_dir"), specs, ok)
    return {"figures": [f.model_dump() for f in kept], "declines": declines}


def _dump(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [_dump(v) for v in value]
    if isinstance(value, dict):
        return {k: _dump(v) for k, v in value.items()}
    return value


def assemble(state):
    h = state["handoff"]
    ledger = state.get("ledger")
    lines = [f"QUESTION: {h.question}", "EVIDENCE AND ASSUMPTION LEDGER"]
    if ledger:
        lines += [f"[{a}] {v}" for a, v in ledger.lines()]
        lines += ["LEDGER HISTORY", json.dumps(_dump(ledger.history), indent=2)]
    lines += [
        "CANDIDATES",
        json.dumps(_dump(state.get("candidates", [])), indent=2),
        "FROZEN DESIGNS",
        json.dumps(_dump(state.get("designs", {})), indent=2),
        "LEAD SELECTION",
        json.dumps(_dump(state.get("lead_selection", {})), indent=2),
    ]
    lines += [
        "RESULTS",
        json.dumps(_dump(state.get("estimates", [])), indent=2),
        "FALSIFICATIONS",
        json.dumps(_dump(state.get("falsifications", [])), indent=2),
        "SENSITIVITY",
        json.dumps(_dump(state.get("sensitivities", [])), indent=2),
        "INTERPRETATIONS",
        json.dumps(_dump(state.get("interpretations", [])), indent=2),
    ]
    f = state.get("feasibility")
    if f:
        lines += ["STOP", f"{f.stage}: {f.reason}", *f.facts]
    report = "\n".join(lines)
    facts = [{"address": a, "text": v, "value": None} for a, v in state["ladder"].lines()] if state.get("ladder") else []
    if ledger:
        facts.extend({"address": a, "text": v, "value": None} for a, v in ledger.lines())
    facts.extend({"address": fact.address, "text": fact.text, "value": fact.value} for log in state.get("episodes", {}).values() for fact in log.facts)
    compare = comparisons(state)
    state = {**state, "comparisons": compare}
    extra = {
        "episode_visits": state.get("episode_visits", []),
        "interpretation_errors": state.get("interpretation_errors", {}),
        "facts": facts,
        "ladder": state["ladder"].lines() if state.get("ladder") else [],
    }
    records.write(state.get("run_dir"), records.artifacts(state, extra), report)
    result = records.result(state, report, extra)
    if state.get("run_dir"):
        trace.write(Path(state["run_dir"]) / "result.json", json.dumps(result, indent=2, default=str))
    return {"report": report, "comparisons": compare, "specialist_result": result}
