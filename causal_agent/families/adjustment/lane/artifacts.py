"""Reviewable adjustment evidence, reports and target-aware conversation material."""

from __future__ import annotations

import json
import math
import re
from itertools import combinations
from pathlib import Path
from typing import Literal

from pydantic import create_model

from causal_agent.common import trace
from causal_agent.families.adjustment import postviz
from causal_agent.families.adjustment.lane import prompts as P
from causal_agent.families.adjustment.lane.contracts import ComparisonInterpretation
from causal_agent.families.adjustment.lane.state import SpecialistState
from causal_agent.families.adjustment.lane.support import primary_summaries, signature, summary_key
from causal_agent.lane import figures as LF
from causal_agent.lane import records
from causal_agent.lane.episode import run_episode


def _model_material(design):
    estimand = design.estimand
    record = {
        "estimator": design.estimator.name,
        "identification_adjustment_set": estimand.adjustment_set,
        "provided_baseline_modifiers": design.modifiers,
        "modifier_groups": design.modifier_groups,
    }
    if estimand.kind in {"frontdoor", "mediation"}:
        record["stage_adjustment_sets"] = {"first": estimand.first_stage_set, "second": estimand.second_stage_set, "total": estimand.total_set}
        record["mediator_variables"] = estimand.frontdoor_set if estimand.kind == "frontdoor" else [estimand.mediator]
    else:
        record["model_covariates"] = estimand.model_covariates if design.modifiers else estimand.adjustment_set
        if estimand.kind == "iv":
            record["excluded_instruments"] = estimand.instruments
    if design.modifiers:
        record["conditional_model_policy"] = (
            "Estimation conditions on every model_covariate and includes treatment interactions with every provided baseline modifier. "
            "An empty identification_adjustment_set means causal identification does not require covariate adjustment, not that the fitted model omits these covariates."
        )
    return record


def _material(state):
    lines = []
    numbers = {}
    addresses = set()

    def add(address, text, value=None, quantitative=False):
        lines.append(f"[{address}] {text}")
        addresses.add(address)
        if quantitative:
            numbers[address] = [float(raw) for raw in re.findall(r"(?<![\w:.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)]
        if value is not None:
            numbers.setdefault(address, []).append(float(value))

    for d in (state.get("designs") or {}).values():
        add(f"design:{d.id}.target", json.dumps(d.target))
        add(f"design:{d.id}.contrasts", json.dumps([c.model_dump(mode="json") for c in d.contrasts]), quantitative=True)
        add(f"design:{d.id}.confidence_level", "95 percent confidence intervals when an interval is available", 95)
        add(f"design:{d.id}.assumptions", d.estimand.symbolic + "; " + ", ".join(d.estimand.assumptions))
        add(f"design:{d.id}.model", json.dumps(_model_material(d)))
        add(
            f"design:{d.id}.support",
            json.dumps(
                {
                    "missingness_policy": d.support.get("missingness_policy"),
                    "contrasts": {
                        k: {"analysis_population_id": v["analysis_population_id"], "rows": len(v["source_row_ids"])}
                        for k, v in d.support.get("by_contrast", {}).items()
                    },
                }
            ),
        )
        numbers[f"design:{d.id}.support"] = [len(v["source_row_ids"]) for v in d.support.get("by_contrast", {}).values()]
        for r in d.checks.results:
            context = ""
            if r.name == "stratification_support" and r.level != "pass":
                context = "Unused stratification estimator EXCLUDED before freeze; this is a hypothetical support screen, not rows removed from the selected analysis. "
            elif r.name.startswith("balance."):
                context = "The flag grades unadjusted baseline imbalance. Diagnostic score weighting is hypothetical; its after-value is not final estimator balance or proof of remaining imbalance. "
            elif r.name == "separation":
                context = "AUC measures arm predictability, not common support. Larger AUC means more distinguishable arms. A pass means the hard separation threshold was not crossed; overlap must be read from its separate check. "
            add(r.address, r.level + ": " + context + r.detail + (f"; value={r.value}" if r.value is not None else ""), r.value, quantitative=True)

    def target_numbers(value):
        if isinstance(value, dict):
            return [n for v in value.values() for n in target_numbers(v)]
        if isinstance(value, list):
            return [n for v in value for n in target_numbers(v)]
        return [float(value)] if isinstance(value, (int, float)) and not isinstance(value, bool) else []

    for e in state.get("estimates") or []:
        design = (state.get("designs") or {}).get(e.design_id)
        contrast = next((c for c in design.contrasts if c.key == e.contrast), None) if design else None
        definition = {"treated": contrast.treated, "control": contrast.control} if contrast else {}
        label_numbers = []
        for label in definition.values():
            try:
                value = float(label)
                if math.isfinite(value):
                    label_numbers.append(value)
            except ValueError:
                pass
        add(e.tag + ".target", json.dumps({**e.target, "contrast_definition": definition}))
        numbers[e.tag + ".target"] = target_numbers(e.target) + label_numbers
        add(e.tag + ".role", "secondary reference or superseded attempt" if e.secondary else "primary requested target summary")
        add(e.tag + ".counts", f"treated={e.n_treated}; control={e.n_control}", quantitative=True)
        add(
            e.tag + ".value",
            str(e.value)
            + ("; secondary reference or superseded attempt" if e.secondary else "; primary requested target summary")
            + "; target "
            + json.dumps(e.target)
            + "; contrast "
            + json.dumps(definition)
            + "; method "
            + e.method,
            e.value,
        )
        numbers[e.tag + ".value"] = ([e.value] if e.value is not None else []) + target_numbers(e.target) + label_numbers
        if e.ci_low is not None:
            add(e.tag + ".ci", f"95 percent confidence interval: {e.ci_low}, {e.ci_high}")
            numbers[e.tag + ".ci"] = [95, e.ci_low, e.ci_high]
        add(e.tag + ".error", e.error or "none")
    for f in state.get("falsifications") or []:
        add(f.address + ".detail", f.detail, quantitative=True)
        add(f.address + ".passed", str(f.passed))
        if f.p_value is not None:
            add(f.address + ".p_value", str(f.p_value), f.p_value)
        if f.new_effect is not None:
            add(f.address + ".new_effect", str(f.new_effect), f.new_effect)
    for s in state.get("sensitivities") or []:
        if s.points:
            add(s.address + ".points", json.dumps(s.points), quantitative=True)
        add(s.address + ".detail", s.detail or s.error or "", quantitative=True)
        if s.range_low is not None:
            add(s.address + ".range", f"{s.range_low}, {s.range_high}")
            numbers[s.address + ".range"] = [s.range_low, s.range_high]
    ledger = state.get("ledger")
    if ledger:
        for a, text in ledger.lines():
            add(a, text)
    add("run:design_count", str(len(state.get("designs") or {})), len(state.get("designs") or {}))
    return "\n".join(lines), addresses, numbers


def comparisons(state):
    good = [e for e in state.get("estimates") or [] if e.error is None and e.value is not None]
    return [
        {
            "left": a.tag + ".value",
            "right": b.tag + ".value",
            "comparable": signature(a) == signature(b),
            "numerical_difference": a.value - b.value if signature(a) == signature(b) else None,
            "same_direction": (a.value > 0) == (b.value > 0) if signature(a) == signature(b) else None,
            "reason": "same complete target, contrast and frozen analysis population; no universal agreement threshold"
            if signature(a) == signature(b)
            else "full targets or frozen analysis populations differ",
        }
        for a, b in combinations(good, 2)
    ]


def _interpret_errors(record, addresses, numbers, compatible):
    errors = [f"not displayed: {a}" for a in record.cites if a not in addresses]
    allowed = [v for a in record.cites for v in numbers.get(a, [])]
    for raw in re.findall(r"(?<![\w:.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", record.answer + " " + " ".join(record.comparisons + record.caveats)):
        v = float(raw)
        if not any(abs(v - n) <= max(abs(n) * 0.01, 1e-8) for n in allowed):
            errors.append(f"number {raw} does not match cited quantitative evidence within one percent")
    for comparison in record.comparisons:
        ids = set(re.findall(r"\bd[a-f0-9]{12}\b", comparison))
        if len(ids) > 1 and not any(ids <= group for group in compatible):
            errors.append("comparison crosses different targets or frozen analysis populations; explain their differences as caveats instead")
    if not record.cites:
        errors.append("at least one evidence citation required")
    return errors


def interpret(state: SpecialistState) -> dict:
    material, addresses, numbers = _material(state)
    question = state["handoff"].question
    groups = {}
    for e in state.get("estimates") or []:
        if e.error is None:
            groups.setdefault(signature(e), set()).add(e.design_id)
    compatible = list(groups.values())
    material += "\nNUMERICAL COMPARISON GROUPS (same contrast, full target and frozen analysis population):\n" + json.dumps([sorted(g) for g in compatible])

    def gate(record, log):
        return _interpret_errors(record, addresses, numbers, compatible)

    AllowedCitation = Literal[tuple(sorted(addresses))]
    Record = create_model("AddressedComparisonInterpretation", __base__=ComparisonInterpretation, cites=(list[AllowedCitation], ...))
    rec, log, thoughts, errors = run_episode(
        Record,
        P.INTERPRET,
        f"QUESTION: {question}\n\n{material}\n\nADDRESSES YOU MAY CITE\n" + "\n".join(sorted(addresses)),
        tools=None,
        budget=0,
        gate=gate,
        node="interpret_1",
        notes=True,
    )
    return {
        "interpretations": [{"contrast": "all", **rec.model_dump()}]
        if rec
        else [{"contrast": "all", "answer": "Interpretation could not pass its evidence gate.", "caveats": errors, "cites": [], "comparisons": []}],
        "episodes": {"interpret_1": log},
        "debug": thoughts,
        "comparisons": comparisons(state),
        "reasoning_history": [
            {"node": "interpret", "visit": log.visit, "view": material, "record": rec.model_dump() if rec else None, "log": log.model_dump(), "errors": errors}
        ],
    }


def figures(state: SpecialistState) -> dict:
    _, addresses, _ = _material(state)
    specs = [postviz.causal_graph(d.graph.model_dump(), state.get("columns") or {}, design_id=d.id) for d in (state.get("designs") or {}).values()]
    specs += postviz.comparisons(state.get("estimates") or [])
    addresses.update(f"design:{d}.graph" for d in state.get("designs") or {})
    kept, declines = LF.write(state.get("run_dir"), specs, addresses)
    return {"figures": [s.model_dump() for s in kept], "declines": declines}


def _dump(x):
    if hasattr(x, "model_dump"):
        return x.model_dump(mode="json")
    if isinstance(x, dict):
        return {k: _dump(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_dump(v) for v in x]
    return x


def _execution_summary(state):
    summary = {}
    for design_id, design in (state.get("designs") or {}).items():
        obligations = primary_summaries(design.contrasts, design.modifier_groups if design.endpoint_id == "adjustment.linear_conditional" else {})
        expected = {summary_key(**item) for item in obligations}
        primary = [record for record in state.get("estimates") or [] if record.design_id == design_id and not record.secondary]
        executed = {
            summary_key(record.contrast, record.modifier, record.level) for record in primary if record.error is None and record.value is not None
        } & expected
        failed = {
            summary_key(record.contrast, record.modifier, record.level) for record in primary if record.error is not None or record.value is None
        } & expected
        summary[design_id] = {
            "expected": sorted(expected),
            "executed": sorted(executed),
            "failed": sorted(failed),
            "missing": sorted(expected - executed - failed),
            "complete": executed == expected and not failed,
        }
    return summary


def assemble(state: SpecialistState) -> dict:
    h = state["handoff"]
    ledger = state.get("ledger")
    episodes = state.get("episodes") or {}
    report = [f"QUESTION {h.question}", f"CONTEXT {h.context_id}", "RELATIONSHIP LEDGER", ledger.render() if ledger else "none", "LEDGER HISTORY"]
    report.extend(json.dumps(_dump(change), sort_keys=True) for change in (ledger.history if ledger else []))
    report += [
        "CANDIDATES",
        json.dumps(_dump(state.get("candidates") or []), indent=2),
        "IDENTIFICATION DIAGNOSTICS",
        json.dumps(state.get("diagnostic_candidates") or [], indent=2),
    ]
    report.extend(d.render() for d in (state.get("designs") or {}).values())
    material, _, _ = _material(state)
    report += ["RESULTS BY DESIGN", material, "INTERPRETATION", json.dumps(state.get("interpretations") or [], indent=2)]
    if state.get("feasibility"):
        report += ["STOP", state["feasibility"].model_dump_json()]
    report += ["EPISODE HISTORY", json.dumps(_dump(episodes), indent=2)]
    text = "\n\n".join(report)
    facts = []
    for log in episodes.values():
        facts += [{"address": f.address, "text": f.text, "value": f.value} for f in log.facts]
    history = (
        (state.get("reasoning_history") or []) + [{"node": change.by, "kind": "ledger_change", **change.model_dump(mode="json")} for change in ledger.history]
        if ledger
        else (state.get("reasoning_history") or [])
    )
    history += [{"node": c["stage"], "kind": "candidate_transition", **c} for c in state.get("candidate_history") or []]
    extra = {
        "candidates": _dump(state.get("candidates") or []) + (state.get("diagnostic_candidates") or []),
        "reasoning_history": history,
        "comparisons": comparisons(state),
        "ladder": state["ladder"].lines() if state.get("ladder") else [],
        "facts": facts,
        "lead_selection": state.get("lead_selection") or {},
    }
    result = records.result(state, text, extra)
    lead = state.get("lead_design_id")
    completion = _execution_summary(state)
    successes = {design_id for design_id, item in completion.items() if item["complete"]}
    failures = set(completion) - successes
    result["terminal"].update(
        {
            "stage": state["feasibility"].stage if state.get("feasibility") else "assemble",
            "lead_design_id": lead,
            "lead_executed": lead in successes,
            "failed_design_ids": sorted(failures),
            "executed_design_ids": sorted(successes),
            "contrast_completion": completion,
        }
    )
    if lead and lead not in successes:
        result["terminal"]["kind"] = "execution_failure"
        result["status"] = "partial" if any(item["executed"] for item in completion.values()) else "execution_failed"
        result["terminal"]["status"] = result["status"]
    artifacts = records.artifacts(
        state,
        {
            **extra,
            "status": result["status"],
            "terminal": result["terminal"],
            "context_id": h.context_id,
            "dataset_identity": h.dataset_identity,
            "context_admission": h.admission,
            "context_facts": h.facts,
        },
    )
    records.write(state.get("run_dir"), artifacts, text)
    if state.get("run_dir"):
        trace.write(Path(state["run_dir"]) / "result.json", json.dumps(result, indent=2))
    return {"report": text, "specialist_result": result}
