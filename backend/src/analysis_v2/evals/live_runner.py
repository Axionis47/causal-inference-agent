"""Live full-spine runner: one representative case, real LLM, no stubs.

    LOCAL_STORAGE_PATH=/tmp/live/<case> python -m src.analysis_v2.evals.live_runner <case_id> --out <dir>

Stages the case's dataset exactly like the hermetic harness, drives
S1..S12 against the configured provider, auto-confirms the plan gate,
and writes a scorecard JSON: lane match, truth-band check, claim
language, dossier facts, per-agent tokens and timing. Run cases in
separate processes for parallelism; heavy estimation blocks the loop.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path


def load_cases() -> dict:
    from .live_inputs import EXTRA_CASES
    from .workflow_evals import test_representative_types as types_mod
    from .workflow_evals import test_representative_workflows as base_mod

    cases = {}
    for entry in [*base_mod.CASES, *types_mod.CASES, *EXTRA_CASES]:
        case = entry.values[0] if hasattr(entry, "values") else entry
        cases[case.case_id] = case
    return cases


def _effects(run) -> list[dict]:
    if run.estimate_result is None:
        return []
    return [
        {
            "estimand": e.estimand,
            "estimate": e.estimate,
            "std_error": e.std_error,
            "ci": [e.ci_lower, e.ci_upper],
            "p_value": e.p_value,
        }
        for e in run.estimate_result.effects
    ]


async def run_case(case, out_dir: Path) -> dict:
    from .live_inputs import CASE_CONTEXTS
    from .workflow_evals.harness import drive

    started = time.monotonic()
    user_context = CASE_CONTEXTS.get(case.case_id)
    record: dict = {"case_id": case.case_id, "expected_lane": case.expected_lane.value,
                    "estimand": case.estimand, "truth_band": case.truth_band,
                    "user_context": user_context}
    try:
        run, state = await drive(
            f"live-{case.case_id[:24]}", case.frame_fn(), case.question,
            user_context=user_context,
        )
        record["status"] = run.status.value
        record["error"] = run.error_message
        if run.estimate_result is not None:
            record["lane"] = run.estimate_result.lane.value
            record["estimator"] = run.estimate_result.estimator
            record["lane_match"] = run.estimate_result.lane == case.expected_lane
            record["effects"] = _effects(run)
            target = next(
                (e for e in run.estimate_result.effects if e.estimand == case.estimand),
                None,
            )
            if target is not None and case.truth_band is not None:
                lo, hi = case.truth_band
                record["estimate"] = target.estimate
                record["in_band"] = bool(lo <= target.estimate <= hi)
        if run.claim_critique is not None:
            record["claim"] = {
                "strength": run.claim_critique.strength.value,
                "limitations": run.claim_critique.limitations,
                "rationale": run.claim_critique.rationale,
            }
        if run.dataset_dossier is not None:
            d = run.dataset_dossier
            record["dossier"] = {
                "investigated": d.investigated,
                "summary": d.summary,
                "n_roles": len(d.roles),
                "roles": [
                    {"column": r.column, "role": r.role.value, "reason": r.reason}
                    for r in d.roles
                ],
                "open_questions": d.open_questions,
                "ledger": [
                    {"assumption": i.assumption, "status": i.status.value}
                    for i in d.context_ledger
                ],
            }
        if run.sensitivity_result is not None:
            record["robustness"] = run.sensitivity_result.robustness.value
        record["notebook"] = (
            run.notebook_verification.notebook_status.value
            if run.notebook_verification
            else None
        )
        record["agents"] = [
            {
                "agent": r.agent,
                "status": r.status.value,
                "tokens": r.tokens.total,
                "tool_calls": len(r.tool_calls),
                "elapsed_s": r.elapsed_seconds,
            }
            for r in run.agent_runs
        ]
    except Exception as exc:  # a crashed run is a finding, not a crash of the audit
        record["status"] = "runner_error"
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["wall_seconds"] = round(time.monotonic() - started, 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{case.case_id}.json").write_text(json.dumps(record, indent=2, default=str))
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_id")
    parser.add_argument("--out", default="/tmp/live-audit/results")
    args = parser.parse_args()
    cases = load_cases()
    if args.case_id not in cases:
        raise SystemExit(f"unknown case '{args.case_id}'; known: {sorted(cases)}")
    record = asyncio.run(run_case(cases[args.case_id], Path(args.out)))
    print(json.dumps({k: record.get(k) for k in
                      ("case_id", "status", "lane", "estimate", "in_band", "wall_seconds")}))


if __name__ == "__main__":
    main()
