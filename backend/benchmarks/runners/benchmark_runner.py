"""Benchmark runner for evaluating the causal inference system."""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.agents import (
    AnalysisState,
    CritiqueAgent,
    DataProfilerAgent,
    DatasetInfo,
    EffectEstimatorAgent,
    OrchestratorAgent,
    SensitivityAnalystAgent,
)
from src.logging_config.structured import get_logger

from ..datasets.loader import BenchmarkDataset, BenchmarkDatasetLoader

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    dataset_name: str
    analysis_type: str
    ground_truth: dict[str, float] | None
    estimated_effects: list[dict[str, Any]]
    metrics: dict[str, float]
    agent_evaluation: dict[str, float]
    execution_time_seconds: float
    success: bool
    error: str | None = None


@dataclass
class BenchmarkReport:
    """Complete benchmark report across all datasets."""

    timestamp: str
    datasets_tested: int
    datasets_passed: int
    overall_metrics: dict[str, float]
    results: list[BenchmarkResult] = field(default_factory=list)


class BenchmarkRunner:
    """Runner for benchmark evaluation.

    Runs the causal inference system against all 8 benchmark datasets
    and evaluates both statistical accuracy AND agentic reasoning quality.

    The system must work on ANY dataset - these benchmarks are for stress testing,
    not for hardcoding specific logic.
    """

    def __init__(self) -> None:
        """Initialize the benchmark runner."""
        self.loader = BenchmarkDatasetLoader()

        # Initialize agents
        self.orchestrator = OrchestratorAgent()
        self.data_profiler = DataProfilerAgent()
        self.effect_estimator = EffectEstimatorAgent()
        self.sensitivity_analyst = SensitivityAnalystAgent()
        self.critique = CritiqueAgent()

        # Register specialists with orchestrator
        self.orchestrator.register_specialist("data_profiler", self.data_profiler)
        self.orchestrator.register_specialist("effect_estimator", self.effect_estimator)
        self.orchestrator.register_specialist("sensitivity_analyst", self.sensitivity_analyst)
        self.orchestrator.register_specialist("critique", self.critique)

    async def run_all(self) -> BenchmarkReport:
        """Run benchmarks on all 8 datasets.

        Returns:
            Complete benchmark report
        """
        logger.info("benchmark_run_all_start")

        datasets = self.loader.load_all()
        results = []

        for dataset in datasets:
            try:
                result = await self.run_single(dataset)
                results.append(result)
            except Exception as e:
                logger.error(
                    "benchmark_dataset_failed",
                    dataset=dataset.name,
                    error=str(e),
                )
                results.append(BenchmarkResult(
                    dataset_name=dataset.name,
                    analysis_type=dataset.analysis_type,
                    ground_truth=dataset.ground_truth,
                    estimated_effects=[],
                    metrics={},
                    agent_evaluation={},
                    execution_time_seconds=0,
                    success=False,
                    error=str(e),
                ))

        # Calculate overall metrics
        passed = sum(1 for r in results if r.success)
        overall_metrics = self._calculate_overall_metrics(results)

        report = BenchmarkReport(
            timestamp=datetime.utcnow().isoformat(),
            datasets_tested=len(datasets),
            datasets_passed=passed,
            overall_metrics=overall_metrics,
            results=results,
        )

        logger.info(
            "benchmark_run_all_complete",
            tested=len(datasets),
            passed=passed,
        )

        return report

    async def run_single(self, dataset: BenchmarkDataset) -> BenchmarkResult:
        """Run benchmark on a single dataset.

        Args:
            dataset: The benchmark dataset

        Returns:
            Benchmark result
        """
        logger.info(
            "benchmark_single_start",
            dataset=dataset.name,
            analysis_type=dataset.analysis_type,
        )

        start_time = time.time()

        try:
            # Create analysis state
            state = AnalysisState(
                job_id=f"benchmark_{dataset.name}_{int(time.time())}",
                dataset_info=DatasetInfo(
                    url=f"benchmark://{dataset.name}",
                    name=dataset.name,
                ),
                treatment_variable=dataset.treatment_variable,
                outcome_variable=dataset.outcome_variable,
            )

            # Save dataset to temp file for agents
            import pickle
            import tempfile
            temp_path = Path(tempfile.gettempdir()) / "causal_orchestrator" / f"{state.job_id}_data.pkl"
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_path, "wb") as f:
                pickle.dump(dataset.data, f)
            state.dataframe_path = str(temp_path)

            # Run data profiler
            state = await self.data_profiler.execute_with_tracing(state)

            # Run effect estimator
            state = await self.effect_estimator.execute_with_tracing(state)

            # Run sensitivity analyst
            state = await self.sensitivity_analyst.execute_with_tracing(state)

            # Calculate metrics
            metrics = self._calculate_metrics(dataset, state)

            # Evaluate agent reasoning (simplified - in production use LLM)
            agent_eval = self._evaluate_agents(dataset, state)

            execution_time = time.time() - start_time

            result = BenchmarkResult(
                dataset_name=dataset.name,
                analysis_type=dataset.analysis_type,
                ground_truth=dataset.ground_truth,
                estimated_effects=[
                    {
                        "method": e.method,
                        "estimand": e.estimand,
                        "estimate": e.estimate,
                        "std_error": e.std_error,
                    }
                    for e in state.treatment_effects
                ],
                metrics=metrics,
                agent_evaluation=agent_eval,
                execution_time_seconds=execution_time,
                success=True,
            )

            logger.info(
                "benchmark_single_complete",
                dataset=dataset.name,
                n_effects=len(state.treatment_effects),
                execution_time=execution_time,
            )

            return result

        except Exception as e:
            logger.error(
                "benchmark_single_failed",
                dataset=dataset.name,
                error=str(e),
            )
            return BenchmarkResult(
                dataset_name=dataset.name,
                analysis_type=dataset.analysis_type,
                ground_truth=dataset.ground_truth,
                estimated_effects=[],
                metrics={},
                agent_evaluation={},
                execution_time_seconds=time.time() - start_time,
                success=False,
                error=str(e),
            )

    def _calculate_metrics(
        self,
        dataset: BenchmarkDataset,
        state: AnalysisState,
    ) -> dict[str, float]:
        """Calculate statistical metrics comparing estimates to ground truth.

        Args:
            dataset: Benchmark dataset with ground truth
            state: Analysis state with estimates

        Returns:
            Dict of metric names to values
        """
        metrics = {}

        if not dataset.ground_truth or not state.treatment_effects:
            return metrics

        # Get estimates
        estimates = [e.estimate for e in state.treatment_effects]

        # Get ground truth ATE
        true_ate = dataset.ground_truth.get("ATE") or dataset.ground_truth.get("LATE") or dataset.ground_truth.get("DiD")

        if true_ate is not None and estimates:
            # Bias (average estimate - true)
            avg_estimate = np.mean(estimates)
            metrics["bias"] = avg_estimate - true_ate

            # Absolute bias
            metrics["abs_bias"] = abs(avg_estimate - true_ate)

            # RMSE
            metrics["rmse"] = np.sqrt(np.mean([(e - true_ate) ** 2 for e in estimates]))

            # Coverage (how many CIs contain true value)
            covered = sum(
                1 for e in state.treatment_effects
                if e.ci_lower <= true_ate <= e.ci_upper
            )
            metrics["coverage"] = covered / len(state.treatment_effects)

            # Relative bias
            if true_ate != 0:
                metrics["relative_bias"] = (avg_estimate - true_ate) / abs(true_ate)

        return metrics

    def _evaluate_agents(
        self,
        dataset: BenchmarkDataset,
        state: AnalysisState,
    ) -> dict[str, float]:
        """Evaluate agent reasoning quality.

        In production, this would use an LLM to evaluate reasoning.
        Here we use heuristics as a proxy.

        Args:
            dataset: Benchmark dataset
            state: Analysis state

        Returns:
            Dict of evaluation dimensions to scores (1-5)
        """
        scores = {}

        # Method selection score
        expected_methods = set(dataset.expected_methods)
        used_methods = set(e.method.lower().replace(" ", "_").replace("(", "").replace(")", "")
                          for e in state.treatment_effects)

        # Simplified matching
        method_overlap = 0
        for expected in expected_methods:
            for used in used_methods:
                if expected in used or used in expected:
                    method_overlap += 1
                    break

        if expected_methods:
            scores["method_selection"] = min(5, 1 + 4 * method_overlap / len(expected_methods))
        else:
            scores["method_selection"] = 3

        # Completeness score (based on number of methods used)
        n_methods = len(state.treatment_effects)
        scores["completeness"] = min(5, 1 + n_methods)

        # Sensitivity analysis score
        n_sensitivity = len(state.sensitivity_results)
        scores["sensitivity_coverage"] = min(5, 1 + n_sensitivity)

        # Data profiling score
        if state.data_profile:
            scores["data_profiling"] = 4 if state.data_profile.treatment_candidates else 2
        else:
            scores["data_profiling"] = 1

        return scores

    def _calculate_overall_metrics(
        self,
        results: list[BenchmarkResult],
    ) -> dict[str, float]:
        """Calculate overall metrics across all benchmarks.

        Args:
            results: List of benchmark results

        Returns:
            Dict of overall metrics
        """
        successful = [r for r in results if r.success]

        if not successful:
            return {"success_rate": 0}

        metrics = {
            "success_rate": len(successful) / len(results),
            "avg_execution_time": np.mean([r.execution_time_seconds for r in successful]),
        }

        # Average bias across datasets with ground truth
        biases = [r.metrics.get("abs_bias") for r in successful if r.metrics.get("abs_bias") is not None]
        if biases:
            metrics["avg_abs_bias"] = np.mean(biases)

        # Average coverage
        coverages = [r.metrics.get("coverage") for r in successful if r.metrics.get("coverage") is not None]
        if coverages:
            metrics["avg_coverage"] = np.mean(coverages)

        # Average agent scores
        for dim in ["method_selection", "completeness", "sensitivity_coverage", "data_profiling"]:
            scores = [r.agent_evaluation.get(dim) for r in successful if r.agent_evaluation.get(dim) is not None]
            if scores:
                metrics[f"avg_{dim}_score"] = np.mean(scores)

        return metrics

    def save_report(self, report: BenchmarkReport, path: str) -> None:
        """Save benchmark report to JSON file.

        Args:
            report: The benchmark report
            path: Output file path
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict
        report_dict = {
            "timestamp": report.timestamp,
            "datasets_tested": report.datasets_tested,
            "datasets_passed": report.datasets_passed,
            "overall_metrics": report.overall_metrics,
            "results": [
                {
                    "dataset_name": r.dataset_name,
                    "analysis_type": r.analysis_type,
                    "ground_truth": r.ground_truth,
                    "estimated_effects": r.estimated_effects,
                    "metrics": r.metrics,
                    "agent_evaluation": r.agent_evaluation,
                    "execution_time_seconds": r.execution_time_seconds,
                    "success": r.success,
                    "error": r.error,
                }
                for r in report.results
            ],
        }

        with open(output_path, "w") as f:
            json.dump(report_dict, f, indent=2)

        logger.info("benchmark_report_saved", path=str(output_path))


async def run_benchmarks():
    """CLI entry point for running benchmarks."""
    runner = BenchmarkRunner()
    report = await runner.run_all()

    # Save report
    output_path = Path("benchmarks/results") / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    runner.save_report(report, str(output_path))

    # Print summary
    print(f"\n{'=' * 60}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print(f"Datasets tested: {report.datasets_tested}")
    print(f"Datasets passed: {report.datasets_passed}")
    print(f"Success rate: {report.overall_metrics.get('success_rate', 0):.1%}")

    if report.overall_metrics.get("avg_abs_bias") is not None:
        print(f"Average absolute bias: {report.overall_metrics['avg_abs_bias']:.4f}")

    if report.overall_metrics.get("avg_coverage") is not None:
        print(f"Average CI coverage: {report.overall_metrics['avg_coverage']:.1%}")

    print(f"\nResults saved to: {output_path}")

    return report


if __name__ == "__main__":
    asyncio.run(run_benchmarks())
