#!/usr/bin/env python3
"""End-to-end test runner for causal inference pipeline with benchmark datasets.

This script simulates actual user usage by:
1. Loading benchmark datasets
2. Running the full pipeline (profiler -> EDA -> discovery -> estimator -> sensitivity)
3. Validating results against known ground truth
4. Reporting quality metrics
"""

import asyncio
import pickle
import sys
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from benchmarks.datasets.loader import BenchmarkDataset, BenchmarkDatasetLoader
from src.agents import (
    AnalysisState,
    CausalDiscoveryAgent,
    ConfounderDiscoveryAgent,
    CritiqueAgent,
    DataProfile,
    DataProfilerAgent,
    DataRepairAgent,
    DatasetInfo,
    EDAAgent,
    EffectEstimatorAgent,
    JobStatus,
    NotebookGeneratorAgent,
    OrchestratorAgent,
    PSDiagnosticsAgent,
    SensitivityAnalystAgent,
)
from src.logging_config.structured import setup_logging, get_logger

# Initialize logging
setup_logging()
logger = get_logger(__name__)


@dataclass
class TestResult:
    """Result of a single test run."""
    dataset_name: str
    status: str  # PASS, FAIL, ERROR
    duration_seconds: float
    ate_estimate: float | None = None
    ate_ground_truth: float | None = None
    ate_error: float | None = None
    ate_within_ci: bool | None = None
    num_methods_run: int = 0
    error_message: str | None = None
    stage_reached: str = "unknown"
    recommendations: list[str] | None = None


class E2ETestRunner:
    """End-to-end test runner for the causal inference pipeline."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: list[TestResult] = []
        self.temp_dir = Path(tempfile.gettempdir()) / "causal_e2e_tests"
        self.temp_dir.mkdir(exist_ok=True)

    def log(self, message: str, level: str = "INFO"):
        """Log a message."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")

    async def run_single_test(
        self,
        dataset: BenchmarkDataset,
        use_orchestrator: bool = False,
    ) -> TestResult:
        """Run a single end-to-end test on a benchmark dataset.

        Args:
            dataset: The benchmark dataset to test
            use_orchestrator: If True, use full orchestrator. If False, run agents sequentially.

        Returns:
            TestResult with metrics
        """
        start_time = datetime.now()
        self.log(f"\n{'='*60}")
        self.log(f"Testing: {dataset.name}")
        self.log(f"Description: {dataset.description}")
        self.log(f"Treatment: {dataset.treatment_variable}, Outcome: {dataset.outcome_variable}")
        self.log(f"Ground Truth ATE: {dataset.ground_truth.get('ATE', 'N/A') if dataset.ground_truth else 'N/A'}")
        self.log(f"{'='*60}")

        try:
            # Save dataset to temp file
            df_path = self.temp_dir / f"{dataset.name}_data.pkl"
            with open(df_path, "wb") as f:
                pickle.dump(dataset.data, f)

            # Create analysis state
            state = AnalysisState(
                job_id=f"e2e-test-{dataset.name.lower()}-{datetime.now().strftime('%H%M%S')}",
                dataset_info=DatasetInfo(
                    url=f"benchmark://{dataset.name}",
                    name=dataset.name,
                ),
                treatment_variable=dataset.treatment_variable,
                outcome_variable=dataset.outcome_variable,
            )
            state.dataframe_path = str(df_path)

            if use_orchestrator:
                state = await self._run_with_orchestrator(state)
            else:
                state = await self._run_sequential_pipeline(state, dataset)

            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()

            # Analyze results
            return self._analyze_results(dataset, state, duration)

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.log(f"ERROR: {str(e)}", "ERROR")
            traceback.print_exc()
            return TestResult(
                dataset_name=dataset.name,
                status="ERROR",
                duration_seconds=duration,
                error_message=str(e),
                stage_reached="exception",
            )

    async def _run_sequential_pipeline(
        self,
        state: AnalysisState,
        dataset: BenchmarkDataset,
    ) -> AnalysisState:
        """Run the pipeline sequentially without the full orchestrator."""

        # Step 1: Data Profiling
        self.log("Step 1: Data Profiling...")
        profiler = DataProfilerAgent()
        state = await profiler.execute_with_tracing(state)

        if state.status == JobStatus.FAILED:
            self.log(f"Data profiling failed: {state.error_message}", "ERROR")
            return state

        self.log(f"  Samples: {state.data_profile.n_samples}")
        self.log(f"  Features: {state.data_profile.n_features}")
        self.log(f"  Treatment candidates: {state.data_profile.treatment_candidates}")
        self.log(f"  Outcome candidates: {state.data_profile.outcome_candidates}")

        # Override treatment/outcome if profiler didn't pick the right ones
        if dataset.treatment_variable not in state.data_profile.treatment_candidates:
            state.data_profile.treatment_candidates.insert(0, dataset.treatment_variable)
        if dataset.outcome_variable not in state.data_profile.outcome_candidates:
            state.data_profile.outcome_candidates.insert(0, dataset.outcome_variable)

        # Step 1b: Data Repair (autonomous)
        self.log("Step 1b: Autonomous Data Repair...")
        try:
            repair_agent = DataRepairAgent()
            state = await repair_agent.execute_with_tracing(state)
            if hasattr(state, 'data_repairs') and state.data_repairs:
                self.log(f"  Repairs applied: {len(state.data_repairs)}")
                for repair in state.data_repairs[:3]:
                    self.log(f"    - {repair['type']}: {repair['strategy']} on {len(repair.get('columns', []))} columns")
            else:
                self.log("  No repairs needed")
        except Exception as e:
            self.log(f"  Data repair skipped: {e}", "WARN")

        # Step 2: EDA
        self.log("Step 2: Exploratory Data Analysis...")
        eda_agent = EDAAgent()
        state = await eda_agent.execute_with_tracing(state)

        if state.status == JobStatus.FAILED:
            self.log(f"EDA failed: {state.error_message}", "ERROR")
            return state

        if state.eda_result:
            self.log(f"  Data Quality Score: {state.eda_result.data_quality_score:.1f}/100")
            self.log(f"  High Correlations: {len(state.eda_result.high_correlations)}")
            self.log(f"  Quality Issues: {state.eda_result.data_quality_issues[:3] if state.eda_result.data_quality_issues else 'None'}")

        # Step 2b: Confounder Discovery (agentic)
        self.log("Step 2b: Confounder Discovery...")
        try:
            confounder_agent = ConfounderDiscoveryAgent()
            state = await confounder_agent.execute_with_tracing(state)
            if hasattr(state, 'confounder_discovery') and state.confounder_discovery:
                ranked = state.confounder_discovery.get('ranked_confounders', [])
                self.log(f"  Confounders identified: {len(ranked)}")
                self.log(f"  Top confounders: {ranked[:5]}")
                self.log(f"  Strategy: {state.confounder_discovery.get('adjustment_strategy', 'N/A')[:60]}")
        except Exception as e:
            self.log(f"  Confounder discovery skipped: {e}", "WARN")

        # Step 2c: PS Diagnostics (agentic)
        self.log("Step 2c: Propensity Score Diagnostics...")
        try:
            ps_agent = PSDiagnosticsAgent()
            state = await ps_agent.execute_with_tracing(state)
            if hasattr(state, 'ps_diagnostics') and state.ps_diagnostics:
                overlap = state.ps_diagnostics.get('overlap', {})
                balance = state.ps_diagnostics.get('balance', {})
                self.log(f"  Overlap quality: {overlap.get('quality', 'N/A')}")
                self.log(f"  Balance quality: {balance.get('quality', 'N/A')}")
                self.log(f"  Mean SMD: {balance.get('mean_smd', 0):.3f}")
                recs = state.ps_diagnostics.get('recommendations', {})
                if recs:
                    self.log(f"  Recommended method: {recs.get('recommended_method', 'N/A')}")
        except Exception as e:
            self.log(f"  PS diagnostics skipped: {e}", "WARN")

        # Step 3: Causal Discovery (optional but useful)
        self.log("Step 3: Causal Discovery...")
        try:
            discovery_agent = CausalDiscoveryAgent()
            state = await discovery_agent.execute_with_tracing(state)
            if state.proposed_dag:
                self.log(f"  DAG nodes: {len(state.proposed_dag.nodes)}")
                self.log(f"  DAG edges: {len(state.proposed_dag.edges)}")
        except Exception as e:
            self.log(f"  Causal discovery skipped: {e}", "WARN")

        # Step 4: Effect Estimation
        self.log("Step 4: Effect Estimation...")
        estimator = EffectEstimatorAgent()
        state = await estimator.execute_with_tracing(state)

        if state.status == JobStatus.FAILED:
            self.log(f"Effect estimation failed: {state.error_message}", "ERROR")
            return state

        if state.treatment_effects:
            self.log(f"  Methods run: {len(state.treatment_effects)}")
            for effect in state.treatment_effects:
                self.log(f"    {effect.method}: {effect.estimate:.4f} [{effect.ci_lower:.4f}, {effect.ci_upper:.4f}]")

        # Step 5: Sensitivity Analysis
        self.log("Step 5: Sensitivity Analysis...")
        try:
            sensitivity_agent = SensitivityAnalystAgent()
            state = await sensitivity_agent.execute_with_tracing(state)

            if state.sensitivity_results:
                self.log(f"  Sensitivity checks: {len(state.sensitivity_results)}")
                for sens in state.sensitivity_results[:2]:
                    self.log(f"    {sens.method}: {sens.interpretation[:60]}...")
        except Exception as e:
            self.log(f"  Sensitivity analysis skipped: {e}", "WARN")

        # Step 6: Critique (optional)
        self.log("Step 6: Critique Review...")
        try:
            critique_agent = CritiqueAgent()
            state = await critique_agent.execute_with_tracing(state)

            if state.critique_history:
                latest = state.get_latest_critique()
                if latest:
                    self.log(f"  Decision: {latest.decision.value}")
                    self.log(f"  Score: {sum(latest.scores.values()) / len(latest.scores):.1f}/5")
        except Exception as e:
            self.log(f"  Critique skipped: {e}", "WARN")

        # Step 7: Generate Notebook
        self.log("Step 7: Generating Jupyter Notebook...")
        try:
            notebook_agent = NotebookGeneratorAgent()
            state = await notebook_agent.execute_with_tracing(state)

            if state.notebook_path:
                self.log(f"  Notebook saved: {state.notebook_path}")
        except Exception as e:
            self.log(f"  Notebook generation skipped: {e}", "WARN")

        state.mark_completed()
        return state

    async def _run_with_orchestrator(self, state: AnalysisState) -> AnalysisState:
        """Run the full pipeline using the orchestrator."""
        self.log("Running with full orchestrator coordination...")

        orchestrator = OrchestratorAgent()
        orchestrator.register_specialist("data_profiler", DataProfilerAgent())
        orchestrator.register_specialist("eda_agent", EDAAgent())
        orchestrator.register_specialist("causal_discovery", CausalDiscoveryAgent())
        orchestrator.register_specialist("effect_estimator", EffectEstimatorAgent())
        orchestrator.register_specialist("sensitivity_analyst", SensitivityAnalystAgent())
        orchestrator.register_specialist("notebook_generator", NotebookGeneratorAgent())
        orchestrator.register_specialist("critique", CritiqueAgent())

        return await orchestrator.execute_with_tracing(state)

    def _analyze_results(
        self,
        dataset: BenchmarkDataset,
        state: AnalysisState,
        duration: float,
    ) -> TestResult:
        """Analyze the results of a test run."""

        # Determine stage reached
        if state.status == JobStatus.COMPLETED:
            stage = "completed"
        elif state.status == JobStatus.FAILED:
            stage = state.error_agent or "failed"
        elif state.treatment_effects:
            stage = "effects_estimated"
        elif state.eda_result:
            stage = "eda_complete"
        elif state.data_profile:
            stage = "profiled"
        else:
            stage = "not_started"

        # Get ATE estimates
        ate_estimate = None
        ate_error = None
        ate_within_ci = None
        ground_truth_ate = dataset.ground_truth.get("ATE") if dataset.ground_truth else None

        if state.treatment_effects:
            # Use average of all ATE estimates
            ate_estimates = [
                e.estimate for e in state.treatment_effects
                if e.estimand == "ATE"
            ]
            if ate_estimates:
                ate_estimate = np.mean(ate_estimates)

            # Check if any CI contains ground truth
            if ground_truth_ate is not None and state.treatment_effects:
                for effect in state.treatment_effects:
                    if effect.ci_lower <= ground_truth_ate <= effect.ci_upper:
                        ate_within_ci = True
                        break
                else:
                    ate_within_ci = False

                ate_error = abs(ate_estimate - ground_truth_ate) if ate_estimate else None

        # Determine status
        if state.status == JobStatus.FAILED:
            status = "FAIL"
        elif len(state.treatment_effects) == 0:
            status = "FAIL"
        elif ground_truth_ate is not None and ate_within_ci:
            status = "PASS"
        elif ground_truth_ate is not None and ate_error and ate_error < abs(ground_truth_ate) * 0.5:
            status = "PASS"  # Within 50% of ground truth
        elif ground_truth_ate is None:
            status = "PASS"  # No ground truth to compare
        else:
            status = "WARN"

        result = TestResult(
            dataset_name=dataset.name,
            status=status,
            duration_seconds=duration,
            ate_estimate=ate_estimate,
            ate_ground_truth=ground_truth_ate,
            ate_error=ate_error,
            ate_within_ci=ate_within_ci,
            num_methods_run=len(state.treatment_effects),
            error_message=state.error_message,
            stage_reached=stage,
            recommendations=state.recommendations if state.recommendations else None,
        )

        self.results.append(result)
        return result

    def print_summary(self):
        """Print a summary of all test results."""
        print("\n" + "=" * 80)
        print("E2E TEST SUMMARY")
        print("=" * 80)

        passed = sum(1 for r in self.results if r.status == "PASS")
        warned = sum(1 for r in self.results if r.status == "WARN")
        failed = sum(1 for r in self.results if r.status in ["FAIL", "ERROR"])

        print(f"\nTotal Tests: {len(self.results)}")
        print(f"  Passed:  {passed} ✓")
        print(f"  Warning: {warned} ⚠")
        print(f"  Failed:  {failed} ✗")

        print("\n" + "-" * 80)
        print(f"{'Dataset':<20} {'Status':<8} {'ATE Est':<12} {'Ground Truth':<12} {'Error':<10} {'CI':<5} {'Methods':<8} {'Time':<8}")
        print("-" * 80)

        for r in self.results:
            ate_str = f"{r.ate_estimate:.4f}" if r.ate_estimate else "N/A"
            gt_str = f"{r.ate_ground_truth:.4f}" if r.ate_ground_truth else "N/A"
            err_str = f"{r.ate_error:.4f}" if r.ate_error else "N/A"
            ci_str = "✓" if r.ate_within_ci else ("✗" if r.ate_within_ci is False else "-")
            status_icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗", "ERROR": "✗"}[r.status]

            print(f"{r.dataset_name:<20} {status_icon} {r.status:<6} {ate_str:<12} {gt_str:<12} {err_str:<10} {ci_str:<5} {r.num_methods_run:<8} {r.duration_seconds:.1f}s")

        if any(r.error_message for r in self.results):
            print("\n" + "-" * 80)
            print("ERRORS:")
            for r in self.results:
                if r.error_message:
                    print(f"  {r.dataset_name}: {r.error_message[:100]}")

        print("=" * 80)

        return passed, warned, failed


async def run_all_tests():
    """Run all benchmark tests."""
    runner = E2ETestRunner(verbose=True)

    # Load all benchmark datasets
    datasets = [
        BenchmarkDatasetLoader.load_ihdp(),
        BenchmarkDatasetLoader.load_lalonde(),
        BenchmarkDatasetLoader.load_card_iv(),
        BenchmarkDatasetLoader.load_twins(),
        BenchmarkDatasetLoader.load_acic(),
        BenchmarkDatasetLoader.load_minimum_wage_did(),
    ]

    # Run tests
    for dataset in datasets:
        try:
            await runner.run_single_test(dataset, use_orchestrator=False)
        except Exception as e:
            print(f"ERROR testing {dataset.name}: {e}")
            traceback.print_exc()

    # Print summary
    runner.print_summary()


async def run_single_dataset_test(dataset_name: str):
    """Run test on a single dataset."""
    runner = E2ETestRunner(verbose=True)

    loader_map = {
        "ihdp": BenchmarkDatasetLoader.load_ihdp,
        "lalonde": BenchmarkDatasetLoader.load_lalonde,
        "card_iv": BenchmarkDatasetLoader.load_card_iv,
        "twins": BenchmarkDatasetLoader.load_twins,
        "acic": BenchmarkDatasetLoader.load_acic,
        "minimum_wage_did": BenchmarkDatasetLoader.load_minimum_wage_did,
        "time_series": BenchmarkDatasetLoader.load_time_series,
        "news": BenchmarkDatasetLoader.load_news,
    }

    if dataset_name not in loader_map:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available: {list(loader_map.keys())}")
        return

    dataset = loader_map[dataset_name]()
    await runner.run_single_test(dataset, use_orchestrator=False)
    runner.print_summary()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run E2E tests for causal inference pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Run test on a specific dataset (ihdp, lalonde, card_iv, twins, acic, minimum_wage_did)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmark tests",
    )

    args = parser.parse_args()

    if args.dataset:
        asyncio.run(run_single_dataset_test(args.dataset))
    else:
        asyncio.run(run_all_tests())
