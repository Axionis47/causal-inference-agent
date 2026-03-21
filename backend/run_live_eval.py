#!/usr/bin/env python3
"""Live pipeline evaluation — run 3 datasets, collect every issue.

Usage:
    python run_live_eval.py
    python run_live_eval.py --dataset lalonde
    python run_live_eval.py --dataset ihdp
    python run_live_eval.py --dataset synthetic
"""

import asyncio
import json
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.agents.base.state import AnalysisState, DatasetInfo, JobStatus
from src.agents.specialists.data_profiler import DataProfilerAgent
from src.agents.specialists.data_repair import DataRepairAgent
from src.agents.specialists.eda_agent import EDAAgent
from src.agents.specialists.causal_discovery import CausalDiscoveryAgent
from src.agents.specialists.confounder_discovery import ConfounderDiscoveryAgent
from src.agents.specialists.ps_diagnostics import PSDiagnosticsAgent
from src.agents.specialists.effect_estimation.agent import EffectEstimatorAgent
from src.agents.specialists.sensitivity_analyst import SensitivityAnalystAgent
from src.agents.critique.critique_agent import CritiqueAgent
from src.agents.specialists.notebook.agent import NotebookGeneratorAgent
from src.benchmarks.datasets import get_benchmark, get_all_benchmarks
from src.logging_config.structured import setup_logging

setup_logging()


@dataclass
class AgentResult:
    agent_name: str
    status: str  # success, failed, skipped, timeout
    duration_seconds: float
    output_fields: list[str]  # which state fields were populated
    error: str | None = None
    issues: list[str] = field(default_factory=list)


@dataclass
class DatasetEvalResult:
    dataset_name: str
    n_rows: int
    n_cols: int
    treatment: str
    outcome: str
    ground_truth: dict
    total_duration: float
    agent_results: list[AgentResult] = field(default_factory=list)
    treatment_effects: list[dict] = field(default_factory=list)
    sensitivity_results: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    # Scorecard
    pipeline_completed: bool = False
    agents_completed: int = 0
    agents_total: int = 0
    methods_run: int = 0
    methods_agree_direction: bool = False
    sensitivity_ran: bool = False
    notebook_generated: bool = False
    estimate_bias: float | None = None
    ci_covers_truth: bool | None = None


PIPELINE_AGENTS = [
    ("data_profiler", DataProfilerAgent, ["data_profile", "dataframe_path"]),
    ("data_repair", DataRepairAgent, ["data_repairs"]),
    ("eda_agent", EDAAgent, ["eda_result"]),
    ("causal_discovery", CausalDiscoveryAgent, ["proposed_dag"]),
    ("confounder_discovery", ConfounderDiscoveryAgent, ["confounder_discovery"]),
    ("ps_diagnostics", PSDiagnosticsAgent, ["ps_diagnostics"]),
    ("effect_estimator", EffectEstimatorAgent, ["treatment_effects"]),
    ("sensitivity_analyst", SensitivityAnalystAgent, ["sensitivity_results"]),
    ("critique", CritiqueAgent, ["critique_history"]),
    ("notebook_generator", NotebookGeneratorAgent, ["notebook_path"]),
]


def generate_dataset(name: str) -> tuple[pd.DataFrame, dict, str, str]:
    """Generate a benchmark dataset, return (df, ground_truth, treatment_col, outcome_col)."""
    benchmarks = get_all_benchmarks()
    # Find first matching benchmark
    for b in benchmarks:
        if b.name.lower() == name.lower():
            df = b.generate()
            gt = b.get_ground_truth()
            return df, gt, b.treatment_col, b.outcome_col
    raise ValueError(f"Unknown dataset: {name}. Available: {[b.name for b in benchmarks]}")


async def run_agent(agent_name: str, agent_cls, expected_fields: list[str],
                    state: AnalysisState) -> tuple[AnalysisState, AgentResult]:
    """Run a single agent, capture result and issues."""
    start = time.time()
    issues = []

    try:
        agent = agent_cls()
        state = await asyncio.wait_for(
            agent.execute_with_tracing(state),
            timeout=300,
        )
        elapsed = time.time() - start

        # Check if agent failed
        if state.status == JobStatus.FAILED:
            return state, AgentResult(
                agent_name=agent_name,
                status="failed",
                duration_seconds=elapsed,
                output_fields=[],
                error=state.error_message,
                issues=[f"{agent_name} failed: {state.error_message}"],
            )

        # Check which fields were populated
        populated = []
        for f in expected_fields:
            val = getattr(state, f, None)
            if val is not None:
                populated.append(f)
            else:
                issues.append(f"{agent_name} did not populate expected field: {f}")

        # Reset failed status for next agent (some agents set status but don't fail)
        if state.status == JobStatus.FAILED:
            state.status = JobStatus.PROFILING  # reset

        return state, AgentResult(
            agent_name=agent_name,
            status="success",
            duration_seconds=elapsed,
            output_fields=populated,
            issues=issues,
        )

    except asyncio.TimeoutError:
        elapsed = time.time() - start
        return state, AgentResult(
            agent_name=agent_name,
            status="timeout",
            duration_seconds=elapsed,
            output_fields=[],
            error=f"Timed out after 300s",
            issues=[f"{agent_name} timed out after 300s"],
        )

    except Exception as e:
        elapsed = time.time() - start
        tb = traceback.format_exc()
        return state, AgentResult(
            agent_name=agent_name,
            status="failed",
            duration_seconds=elapsed,
            output_fields=[],
            error=f"{type(e).__name__}: {str(e)}",
            issues=[f"{agent_name} crashed: {type(e).__name__}: {str(e)}\n{tb}"],
        )


def check_statistical_sanity(result: DatasetEvalResult, state: AnalysisState):
    """Run Tier 2 statistical sanity checks."""
    if not state.treatment_effects:
        result.issues.append("NO treatment effects produced — pipeline produced no estimates")
        return

    result.methods_run = len(state.treatment_effects)

    for e in state.treatment_effects:
        # NaN/Inf checks
        if np.isnan(e.estimate) or np.isinf(e.estimate):
            result.issues.append(f"{e.method}: estimate is NaN/Inf")
        if np.isnan(e.std_error) or np.isinf(e.std_error):
            result.issues.append(f"{e.method}: std_error is NaN/Inf")
        if e.p_value is not None and (e.p_value < 0 or e.p_value > 1):
            result.issues.append(f"{e.method}: p-value {e.p_value} outside [0,1]")
        if e.ci_lower is not None and e.ci_upper is not None:
            if not np.isnan(e.ci_lower) and not np.isnan(e.ci_upper):
                if e.ci_lower > e.ci_upper:
                    result.issues.append(f"{e.method}: CI inverted [{e.ci_lower}, {e.ci_upper}]")
                if not np.isnan(e.estimate) and (e.estimate < e.ci_lower or e.estimate > e.ci_upper):
                    result.issues.append(f"{e.method}: estimate {e.estimate} outside CI [{e.ci_lower}, {e.ci_upper}]")
        if e.std_error is not None and not np.isnan(e.std_error) and e.std_error <= 0:
            result.issues.append(f"{e.method}: std_error is non-positive ({e.std_error})")

        result.treatment_effects.append({
            "method": e.method,
            "estimand": e.estimand,
            "estimate": float(e.estimate) if not np.isnan(e.estimate) else None,
            "std_error": float(e.std_error) if e.std_error and not np.isnan(e.std_error) else None,
            "ci": [float(e.ci_lower) if e.ci_lower else None, float(e.ci_upper) if e.ci_upper else None],
            "p_value": float(e.p_value) if e.p_value and not np.isnan(e.p_value) else None,
        })

    # Direction agreement
    valid_estimates = [e.estimate for e in state.treatment_effects if not np.isnan(e.estimate)]
    if len(valid_estimates) >= 2:
        signs = [np.sign(e) for e in valid_estimates]
        result.methods_agree_direction = len(set(signs)) == 1
        if not result.methods_agree_direction:
            result.issues.append(f"Methods DISAGREE on direction: estimates = {[f'{e:.2f}' for e in valid_estimates]}")

    # Ground truth comparison
    gt_ate = result.ground_truth.get("ate") or result.ground_truth.get("att")
    if gt_ate is not None and valid_estimates:
        best = np.median(valid_estimates)
        result.estimate_bias = abs(best - gt_ate) / abs(gt_ate) if gt_ate != 0 else abs(best)
        # CI coverage
        for e in state.treatment_effects:
            if e.ci_lower is not None and e.ci_upper is not None:
                if not np.isnan(e.ci_lower) and not np.isnan(e.ci_upper):
                    if e.ci_lower <= gt_ate <= e.ci_upper:
                        result.ci_covers_truth = True
                        break
        if result.ci_covers_truth is None:
            result.ci_covers_truth = False
            result.issues.append(f"No method's CI covers ground truth ({gt_ate})")

    # Sensitivity
    if state.sensitivity_results:
        result.sensitivity_ran = True
        for s in state.sensitivity_results:
            result.sensitivity_results.append({
                "method": s.method,
                "robustness_value": s.robustness_value,
                "interpretation": s.interpretation[:100] if s.interpretation else None,
            })
    else:
        result.issues.append("No sensitivity analysis results")

    # Decisions
    if hasattr(state, "decisions") and state.decisions:
        for d in state.decisions:
            result.decisions.append({
                "agent": d.agent,
                "type": d.decision_type,
                "choice": d.choice,
                "reason": d.reason[:120],
            })


async def run_dataset_eval(dataset_name: str) -> DatasetEvalResult:
    """Run full pipeline on one dataset, collect everything."""
    print(f"\n{'='*70}")
    print(f"  DATASET: {dataset_name}")
    print(f"{'='*70}")

    # Generate dataset
    df, gt, treatment_col, outcome_col = generate_dataset(dataset_name)
    print(f"  Rows: {len(df)}, Cols: {len(df.columns)}")
    print(f"  Treatment: {treatment_col}, Outcome: {outcome_col}")
    print(f"  Ground truth: {gt}")

    result = DatasetEvalResult(
        dataset_name=dataset_name,
        n_rows=len(df),
        n_cols=len(df.columns),
        treatment=treatment_col,
        outcome=outcome_col,
        ground_truth=gt,
        total_duration=0,
        agents_total=len(PIPELINE_AGENTS),
    )

    # Save as parquet
    tmp_dir = Path(tempfile.gettempdir()) / "causal_live_eval"
    tmp_dir.mkdir(exist_ok=True)
    parquet_path = tmp_dir / f"{dataset_name}_data.parquet"
    df.to_parquet(parquet_path)

    # Create state
    state = AnalysisState(
        job_id=f"eval-{dataset_name}-{datetime.now().strftime('%H%M%S')}",
        dataset_info=DatasetInfo(
            url=f"benchmark://{dataset_name}",
            name=dataset_name,
        ),
        treatment_variable=treatment_col,
        outcome_variable=outcome_col,
    )
    state.dataframe_path = str(parquet_path)

    # Run each agent sequentially
    total_start = time.time()

    for agent_name, agent_cls, expected_fields in PIPELINE_AGENTS:
        print(f"\n  [{agent_name}] Running...", end=" ", flush=True)

        # Skip agents that depend on missing upstream
        if agent_name == "data_repair" and not state.data_profile:
            print("SKIPPED (no data_profile)")
            result.agent_results.append(AgentResult(agent_name, "skipped", 0, [], issues=["Skipped: upstream data_profile missing"]))
            continue
        if agent_name == "sensitivity_analyst" and not state.treatment_effects:
            print("SKIPPED (no treatment_effects)")
            result.agent_results.append(AgentResult(agent_name, "skipped", 0, [], issues=["Skipped: upstream treatment_effects missing"]))
            continue
        if agent_name == "critique" and not state.treatment_effects:
            print("SKIPPED (no treatment_effects)")
            result.agent_results.append(AgentResult(agent_name, "skipped", 0, [], issues=["Skipped: upstream treatment_effects missing"]))
            continue

        state, agent_result = await run_agent(agent_name, agent_cls, expected_fields, state)
        print(f"{agent_result.status.upper()} ({agent_result.duration_seconds:.1f}s)")

        if agent_result.issues:
            for issue in agent_result.issues:
                print(f"    ISSUE: {issue[:120]}")

        result.agent_results.append(agent_result)

        if agent_result.status == "success":
            result.agents_completed += 1

        # If profiler failed, everything downstream fails
        if agent_name == "data_profiler" and agent_result.status != "success":
            result.issues.append("CRITICAL: Data profiler failed — pipeline cannot continue")
            break

        # If effect estimator failed, skip sensitivity
        if agent_name == "effect_estimator" and agent_result.status != "success":
            result.issues.append("Effect estimation failed — skipping downstream agents")

    result.total_duration = time.time() - total_start

    # Check statistical sanity
    check_statistical_sanity(result, state)

    # Notebook check
    if state.notebook_path:
        result.notebook_generated = True

    # Pipeline completion
    result.pipeline_completed = state.status == JobStatus.COMPLETED or result.agents_completed >= 7

    return result


def print_scorecard(result: DatasetEvalResult):
    """Print a scorecard for one dataset."""
    print(f"\n{'─'*70}")
    print(f"  SCORECARD: {result.dataset_name}")
    print(f"{'─'*70}")

    # Process quality
    print(f"\n  Process Quality:")
    print(f"    Pipeline completed:    {'YES' if result.pipeline_completed else 'NO'}")
    print(f"    Agents completed:      {result.agents_completed}/{result.agents_total}")
    print(f"    Methods run:           {result.methods_run}")
    print(f"    Direction agreement:   {'YES' if result.methods_agree_direction else 'NO' if result.methods_run >= 2 else 'N/A'}")
    print(f"    Sensitivity ran:       {'YES' if result.sensitivity_ran else 'NO'}")
    print(f"    Notebook generated:    {'YES' if result.notebook_generated else 'NO'}")
    print(f"    Decisions logged:      {len(result.decisions)}")
    print(f"    Total duration:        {result.total_duration:.1f}s")

    # Accuracy (if ground truth available)
    if result.estimate_bias is not None:
        print(f"\n  Accuracy (vs ground truth):")
        print(f"    Estimate bias:         {result.estimate_bias:.1%}")
        print(f"    CI covers truth:       {'YES' if result.ci_covers_truth else 'NO'}")

    # Per-agent breakdown
    print(f"\n  Agent Breakdown:")
    for ar in result.agent_results:
        status_icon = {"success": "✓", "failed": "✗", "timeout": "⏱", "skipped": "⊘"}.get(ar.status, "?")
        fields = ", ".join(ar.output_fields) if ar.output_fields else "none"
        print(f"    {status_icon} {ar.agent_name:25s} {ar.status:8s} {ar.duration_seconds:6.1f}s  [{fields}]")

    # Treatment effects
    if result.treatment_effects:
        print(f"\n  Treatment Effects:")
        for e in result.treatment_effects:
            sig = "***" if e["p_value"] and e["p_value"] < 0.01 else "**" if e["p_value"] and e["p_value"] < 0.05 else ""
            ci_str = f"[{e['ci'][0]:.2f}, {e['ci'][1]:.2f}]" if e["ci"][0] is not None else "N/A"
            est_str = f"{e['estimate']:.4f}" if e["estimate"] is not None else "NaN"
            print(f"    {e['method']:12s} {e['estimand']:5s} {est_str:>10s} {ci_str:>24s}  p={e['p_value']:.4f}{sig}" if e["p_value"] else f"    {e['method']:12s} {est_str:>10s}")

    # Issues
    if result.issues:
        print(f"\n  Issues ({len(result.issues)}):")
        for issue in result.issues:
            print(f"    ▸ {issue[:120]}")

    # Decisions
    if result.decisions:
        print(f"\n  Key Decisions ({len(result.decisions)}):")
        for d in result.decisions[:8]:
            print(f"    [{d['agent']}] {d['type']}: {d['choice']}")

    print(f"\n{'─'*70}")


async def main():
    datasets = ["lalonde", "ihdp", "synthetic"]

    # Check for CLI arg
    if len(sys.argv) > 1 and sys.argv[1] == "--dataset":
        datasets = [sys.argv[2]]

    all_results = []

    for ds_name in datasets:
        try:
            result = await run_dataset_eval(ds_name)
            all_results.append(result)
            print_scorecard(result)
        except Exception as e:
            print(f"\n  FATAL ERROR on {ds_name}: {type(e).__name__}: {e}")
            traceback.print_exc()

    # Summary
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"  SUMMARY: {len(all_results)} datasets evaluated")
        print(f"{'='*70}")
        total_issues = sum(len(r.issues) for r in all_results)
        total_agents = sum(r.agents_completed for r in all_results)
        total_expected = sum(r.agents_total for r in all_results)
        print(f"  Agents completed: {total_agents}/{total_expected}")
        print(f"  Total issues:     {total_issues}")
        for r in all_results:
            status = "PASS" if r.pipeline_completed and len(r.issues) <= 3 else "WARN" if r.pipeline_completed else "FAIL"
            print(f"    {r.dataset_name:15s} {status:5s}  {r.agents_completed}/{r.agents_total} agents  {len(r.issues)} issues  {r.total_duration:.0f}s")

    # Save results as JSON
    output_path = Path(tempfile.gettempdir()) / "causal_live_eval" / "results.json"
    with open(output_path, "w") as f:
        json.dump([{
            "dataset": r.dataset_name,
            "pipeline_completed": r.pipeline_completed,
            "agents_completed": r.agents_completed,
            "methods_run": r.methods_run,
            "issues": r.issues,
            "treatment_effects": r.treatment_effects,
            "decisions": r.decisions,
            "duration": r.total_duration,
        } for r in all_results], f, indent=2, default=str)
    print(f"\n  Results saved: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
