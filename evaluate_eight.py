#!/usr/bin/env python3
"""End-to-end evaluation: one real dataset for each supported analysis lane."""
from __future__ import annotations

import json
import math
import time
import uuid
import warnings
from pathlib import Path

from langgraph.types import Command

from causal.preparation_agent import run as prepare
from causal.studio_prep import build_data_version, quality_summary
from causal.studio_protocols import (
    build_contract,
    contract_hash,
    get_protocol,
    protocol_as_dict,
    run_preflight,
)
from causal.studio_workflow import build_in_memory
from fixtures import cases


DATA = Path(__file__).parent / "data"
OUTPUT = Path(__file__).parent / ".studio_evals"

SPECS = {
    "observational": {
        "file": "ihdp.csv",
        "question": "What is the effect of the intervention on the child's test score?",
        "vague": "IHDP child development data. y_factual is the observed test score.",
        "semantic": {"outcome": "y_factual", "treatment": "treatment"},
        "context": {
            "unit": "one child", "assignment": "observational / unknown",
            "timing": "intervention precedes the observed child test score",
            "population": "children represented by the IHDP benchmark",
            "intended_use": "method evaluation only", "high_impact": False,
        },
        "lane_check": "balance",
        "scientific_basis": "semi-synthetic ground truth",
    },
    "matching": {
        "file": "lalonde.csv",
        "question": "Did the job training program raise earnings in 1978?",
        "vague": "NSW training participants combined with a PSID comparison group.",
        "semantic": {"outcome": "re78", "treatment": "treat"},
        "context": {
            "unit": "one person", "assignment": "self-selected / observational",
            "timing": "1974/1975 earnings precede training; re78 follows training",
            "population": "NSW treated units and PSID comparison units",
            "intended_use": "method evaluation only", "high_impact": False,
        },
        "lane_check": "overlap",
        "scientific_basis": "literature-calibrated range",
    },
    "iv": {
        "file": "card.csv",
        "question": "What is the effect of years of schooling on log hourly wages using college proximity as an instrument?",
        "vague": "Card schooling data. nearc4 is proximity to a four-year college; use lwage.",
        "semantic": {"outcome": "lwage", "treatment": "educ"},
        "context": {
            "unit": "one person", "assignment": "observational / unknown",
            "timing": "college proximity precedes schooling; schooling precedes wage",
            "population": "men represented in the Card benchmark",
            "intended_use": "method evaluation only", "high_impact": False,
        },
        "lane_check": "subsample_stability",
        "scientific_basis": "published reference estimate",
    },
    "survival": {
        "file": "heart_failure.csv",
        "question": "Does high blood pressure affect how long patients survive?",
        "vague": "Clinical records. DEATH_EVENT marks death and time is follow-up duration.",
        # The generic context outcome is the event. The lane itself still needs
        # separate duration/event roles, which the preparation schema cannot yet
        # persist and is therefore called out as a product limitation below.
        "semantic": {"outcome": "DEATH_EVENT", "treatment": "high_blood_pressure"},
        "context": {
            "unit": "one patient", "assignment": "observational / unknown",
            "timing": "blood-pressure status is measured before/during follow-up",
            "population": "patients represented in the heart-failure dataset",
            "intended_use": "method evaluation only", "high_impact": False,
        },
        "lane_check": "proportional_hazards",
        "scientific_basis": "cross-implementation regression pin",
        "preparation_limitations": [
            "preparation context cannot persist separate survival duration and event roles",
        ],
    },
    "did": {
        "file": "card_krueger.csv",
        "question": "Did New Jersey's minimum wage rise change fast-food employment relative to Pennsylvania?",
        "vague": "NJ and PA stores observed before and after the policy. period is before/after.",
        "semantic": {"outcome": "fte", "group": "state", "period": "period"},
        "context": {
            "unit": "one store-period", "assignment": "policy at a date",
            "timing": "period 0 precedes the minimum-wage rise and period 1 follows it",
            "population": "sampled fast-food stores in New Jersey and Pennsylvania",
            "intended_use": "method evaluation only", "high_impact": False,
        },
        "lane_check": "pre_trend",
        "scientific_basis": "published reference estimate",
    },
    "rdd": {
        "file": "bank.csv",
        "question": "Does crossing the expected recovery cutoff of 1000 increase actual recovery?",
        "vague": "A more intensive recovery strategy starts at expected recovery 1000.",
        "semantic": {"outcome": "actual_recovery_amount", "running_variable": "expected_recovery_amount"},
        "context": {
            "unit": "one recovery case", "assignment": "rule / threshold",
            "timing": "expected recovery and cutoff assignment precede actual recovery",
            "population": "recovery cases near the policy cutoff",
            "intended_use": "method evaluation only", "high_impact": False,
        },
        "lane_check": "placebo_cutoff",
        "scientific_basis": "sanity only",
    },
    "mediation": {
        "file": "student.csv",
        "question": "Does study time affect final grades through past failures?",
        "vague": "Portuguese secondary-school records. G3 is final grade; failures is proposed as mediator.",
        "semantic": {"outcome": "G3", "treatment": "studytime"},
        "context": {
            "unit": "one student", "assignment": "observational / unknown",
            "timing": "the claimed ordering is study time, failures, then final grade; this needs expert confirmation",
            "population": "students represented in the dataset",
            "intended_use": "method evaluation only", "high_impact": False,
        },
        "lane_check": "confounder_strength",
        "scientific_basis": "sanity only",
    },
    "time_series": {
        "file": "visitors.csv",
        "question": "Did site traffic change at the start of 2018?",
        "vague": "Daily site statistics. Date is time and Unique.Visits is the outcome.",
        "semantic": {"outcome": "Unique.Visits", "time_column": "Date"},
        "context": {
            "unit": "one calendar day", "assignment": "policy at a date",
            "timing": "the candidate interruption is 2018-01-01",
            "population": "days represented in the site-traffic series",
            "intended_use": "negative-control method evaluation only", "high_impact": False,
        },
        "lane_check": "placebo_date",
        "scientific_basis": "negative control",
    },
}


def benchmark(case, estimate: dict, basis: str) -> tuple[bool, str]:
    value = estimate.get("value", float("nan"))
    if not math.isfinite(value):
        return False, "estimate is not finite"
    if basis == "negative control":
        low, high = estimate.get("ci_low"), estimate.get("ci_high")
        covers_zero = low is not None and high is not None and low <= 0 <= high
        return covers_zero, f"arbitrary intervention date; interval {low:.4g}..{high:.4g} must cover zero"
    if case.ranges:
        _, low, high = case.ranges[0]
        return low <= value <= high, f"expected {low:g}..{high:g}"
    if case.checks:
        _, expected, band = case.checks[0]
        relative = abs(value - expected) / abs(expected)
        return relative <= band, f"expected {expected:.6g} ±{band:.0%}; error {relative:.2%}"
    return True, "finite-result sanity check only; no external effect benchmark"


def semantic_score(draft: dict, expected: dict) -> tuple[bool, list[str]]:
    misses = [f"{key}: expected {value!r}, got {draft.get(key)!r}" for key, value in expected.items() if draft.get(key) != value]
    return not misses, misses


def main() -> int:
    graph = build_in_memory()
    results = []
    case_by_name = {case.name: case for case in cases()}

    for lane, spec in SPECS.items():
        case = case_by_name[lane]
        df = case.frame
        started = time.perf_counter()
        prep = prepare({spec["file"]: df}, spec["question"], spec["vague"]).to_dict()
        prep_seconds = time.perf_counter() - started
        semantic_ok, semantic_misses = semantic_score(prep["context_draft"], spec["semantic"])
        recommendation_ok = prep.get("recommended_lane") == lane

        context = {
            "question": spec["question"],
            "description": spec["vague"],
            "outcome": spec["semantic"]["outcome"],
            "treatment": spec["semantic"].get("treatment", ""),
            **spec["context"],
        }
        if lane == "did":
            context["exposure"] = "state × period"
        elif lane == "rdd":
            context["exposure"] = "expected_recovery_amount crossing 1000"
        elif lane == "time_series":
            context["exposure"] = "interruption at 2018-01-01"

        run_id = f"eval-{lane}-{uuid.uuid4().hex[:6]}"
        config = {"configurable": {"thread_id": run_id}}
        run_path = OUTPUT / f"{run_id}.csv"
        OUTPUT.mkdir(parents=True, exist_ok=True)
        df.to_csv(run_path, index=False)
        protocol = get_protocol(lane)
        preflight = run_preflight(df, lane, case.kwargs, context)
        data_version = build_data_version(df, df, cohort=None, repairs=[])
        data_version["revision"] = 1
        base_contract = build_contract(
            dataset_id=spec["file"], lane=lane, kwargs=case.kwargs,
            context=context, cohort=None, data_version=data_version,
        )
        design_contract = build_contract(
            dataset_id=spec["file"], lane=lane, kwargs=case.kwargs,
            context=context, cohort=None, data_version=data_version,
            answers={
                question: "Evaluation fixture assumption; requires real domain review in production."
                for question in protocol.review_questions
            },
        )
        design_contract["configuration_hash"] = contract_hash(base_contract)
        design_contract["contract_hash"] = contract_hash(design_contract)
        design_contract["revision"] = 1
        design_contract["revision_timing"] = "pre_estimation"
        design_contract["approval"] = {
            "approved": True,
            "reviewer": "automated evaluation harness",
            "role_ledger_and_map_confirmed": True,
        }
        analysis_started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            graph.invoke({
                "run_id": run_id,
                "csv_path": str(run_path),
                "source": spec["file"],
                "lane": lane,
                "kwargs": case.kwargs,
                "context": context,
                "repairs": [],
                "data_version": data_version,
                "data_quality": quality_summary(df),
                "preparation": prep,
                "prompt_versions": {"preparation-agent": prep["prompt"]},
                "design_contract": design_contract,
                "design_approval": design_contract["approval"],
                "preflight": preflight,
                "protocol": protocol_as_dict(lane),
                "events": [],
            }, config=config)
        analysis_seconds = time.perf_counter() - analysis_started
        snap = graph.get_state(config)
        before_review = dict(snap.values)
        initial_policy = (before_review.get("policy") or {}).get("decision", "")
        paused = snap.next == ("human_gate",)
        if paused:
            graph.invoke(Command(resume={
                "approved": True,
                "reviewer": "automated evaluation harness",
                "note": "Evaluation-only approval; not a production scientific review.",
            }), config=config)
            snap = graph.get_state(config)
        final = dict(snap.values)
        estimate = final.get("estimate") or {}
        scientific_basis = spec["scientific_basis"]
        math_ok, benchmark_note = benchmark(case, estimate, scientific_basis)
        diagnostics = final.get("diagnostics") or []
        diagnostic_names = {d["check"] for d in diagnostics}
        lane_check_ok = spec["lane_check"] in diagnostic_names
        end_to_end = bool(estimate and final.get("report") and not final.get("error"))
        data_version_bound = (
            (final.get("data_version") or {}).get("version_id")
            == ((final.get("design_contract") or {}).get("data_version") or {}).get("version_id")
        )
        execution_guard_passed = bool(
            estimate
            and not [
                event for event in final.get("events", [])
                if event.get("stage") == "execution_guard" and event.get("status") == "blocked"
            ]
        )
        results.append({
            "lane": lane,
            "dataset": spec["file"],
            "preparation_mode": prep["mode"],
            "preparation_seconds": round(prep_seconds, 2),
            "tool_calls": len(prep["trace"]),
            "semantic_mapping_ok": semantic_ok,
            "semantic_misses": semantic_misses,
            "preparation_context": prep.get("context_draft", {}),
            "preparation_limitations": spec.get("preparation_limitations", []),
            "unresolved_questions": prep.get("unresolved_questions", []),
            "escalation_reasons": prep.get("escalation_reasons", []),
            "recommended_lane": prep.get("recommended_lane"),
            "recommendation_ok": recommendation_ok,
            "eligible_lanes": prep.get("eligible_lanes", []),
            "analysis_seconds": round(analysis_seconds, 2),
            "estimate": estimate,
            "benchmark_ok": math_ok,
            "benchmark_note": benchmark_note,
            "scientific_basis": scientific_basis,
            "protocol_version": design_contract["protocol_version"],
            "protocol_pre_checks": list(protocol.pre_checks),
            "protocol_post_checks": list(protocol.post_checks),
            "preflight": preflight,
            "preflight_failures": [item["check"] for item in preflight if item["verdict"] == "fail"],
            "preflight_reviews": [item["check"] for item in preflight if item["verdict"] in {"review", "warn", "untestable"}],
            "contract_hash": design_contract["contract_hash"],
            "contract_revision": design_contract["revision"],
            "data_version_id": data_version["version_id"],
            "prepared_fingerprint": data_version["prepared_fingerprint"],
            "data_version_bound": data_version_bound,
            "execution_guard_passed": execution_guard_passed,
            "diagnostics": diagnostics,
            "lane_specific_check_present": lane_check_ok,
            "diagnostic_failures": [d["check"] for d in diagnostics if d["verdict"] == "fail"],
            "diagnostic_warnings": [d["check"] for d in diagnostics if d["verdict"] == "warn"],
            "diagnostic_untestable": [d["check"] for d in diagnostics if d["verdict"] == "untestable"],
            "runtime_warnings": len(caught),
            "runtime_warning_types": sorted({
                f"{item.category.__name__}: {item.message}" for item in caught
            }),
            "initial_policy": initial_policy,
            "paused_for_review": paused,
            "final_policy": (final.get("policy") or {}).get("decision", ""),
            "report_complete": bool(final.get("report")),
            "end_to_end_ok": end_to_end,
        })
        print(
            f"{lane:<14} prep={'PASS' if semantic_ok and recommendation_ok else 'FAIL':<4} "
            f"math={'PASS' if math_ok else 'FAIL':<4} check={'FOUND' if lane_check_ok else 'MISS':<5} "
            f"policy={initial_policy:<6} report={'PASS' if end_to_end else 'FAIL'}"
        )

    summary = {
        "cases": len(results),
        "preparation_semantics_passed": sum(r["semantic_mapping_ok"] for r in results),
        "lane_recommendations_passed": sum(r["recommendation_ok"] for r in results),
        "benchmarks_passed": sum(r["benchmark_ok"] for r in results),
        "lane_checks_present": sum(r["lane_specific_check_present"] for r in results),
        "preflight_protocols_complete": sum(
            {item["check"] for item in r["preflight"]} == set(r["protocol_pre_checks"])
            for r in results
        ),
        "preflight_failures": sum(bool(r["preflight_failures"]) for r in results),
        "contracts_frozen": sum(bool(r["contract_hash"]) for r in results),
        "data_versions_bound": sum(r["data_version_bound"] for r in results),
        "execution_guards_passed": sum(r["execution_guard_passed"] for r in results),
        "end_to_end_passed": sum(r["end_to_end_ok"] for r in results),
        "independent_benchmarks": sum(r["scientific_basis"] in {
            "semi-synthetic ground truth", "literature-calibrated range", "published reference estimate",
        } for r in results),
        "regression_pins": sum(r["scientific_basis"] == "cross-implementation regression pin" for r in results),
        "negative_controls": sum(r["scientific_basis"] == "negative control" for r in results),
        "sanity_only": sum(r["scientific_basis"] == "sanity only" for r in results),
    }
    output = OUTPUT / "eight_lane_latest.json"
    output.write_text(json.dumps({"summary": summary, "results": results}, indent=2, default=str))
    print(json.dumps(summary, indent=2))
    print(f"wrote {output}")
    return 0 if (
        summary["benchmarks_passed"] == 8
        and summary["data_versions_bound"] == 8
        and summary["execution_guards_passed"] == 8
        and summary["end_to_end_passed"] == 8
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
