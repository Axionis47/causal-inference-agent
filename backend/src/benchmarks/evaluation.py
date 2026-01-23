"""Evaluation framework for causal inference methods.

This module provides comprehensive evaluation of treatment effect estimators
against benchmark datasets with known ground truth.

Metrics:
- ATE/ATT bias and RMSE
- CATE prediction error (PEHE)
- Coverage of confidence intervals
- Calibration of uncertainty estimates
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
import numpy as np
import pandas as pd
from scipy import stats

from src.logging_config.structured import get_logger
from .datasets import BenchmarkDataset, get_all_benchmarks, get_benchmark

logger = get_logger(__name__)


@dataclass
class EvaluationMetrics:
    """Metrics for evaluating causal inference quality."""

    # ATE metrics
    ate_bias: float | None = None
    ate_rmse: float | None = None
    ate_coverage: float | None = None  # CI coverage

    # ATT metrics
    att_bias: float | None = None
    att_rmse: float | None = None
    att_coverage: float | None = None

    # CATE metrics (if heterogeneous effects)
    pehe: float | None = None  # Precision in Estimation of Heterogeneous Effects
    cate_rmse: float | None = None
    cate_corr: float | None = None  # Correlation with true CATE

    # Calibration metrics
    coverage_90: float | None = None  # 90% CI coverage
    coverage_95: float | None = None  # 95% CI coverage

    # Additional diagnostics
    propensity_auc: float | None = None  # If propensity model evaluated
    balance_improvement: float | None = None  # Covariate balance after matching/weighting

    def to_dict(self) -> dict[str, float | None]:
        """Convert metrics to dictionary."""
        return {
            "ate_bias": self.ate_bias,
            "ate_rmse": self.ate_rmse,
            "ate_coverage": self.ate_coverage,
            "att_bias": self.att_bias,
            "att_rmse": self.att_rmse,
            "att_coverage": self.att_coverage,
            "pehe": self.pehe,
            "cate_rmse": self.cate_rmse,
            "cate_corr": self.cate_corr,
            "coverage_90": self.coverage_90,
            "coverage_95": self.coverage_95,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = []
        if self.ate_bias is not None:
            lines.append(f"ATE Bias: {self.ate_bias:.4f}")
        if self.ate_rmse is not None:
            lines.append(f"ATE RMSE: {self.ate_rmse:.4f}")
        if self.pehe is not None:
            lines.append(f"PEHE: {self.pehe:.4f}")
        if self.coverage_95 is not None:
            lines.append(f"95% CI Coverage: {self.coverage_95:.2%}")
        return " | ".join(lines) if lines else "No metrics computed"


@dataclass
class EvaluationResult:
    """Complete evaluation result for a single run."""

    benchmark_name: str
    method_name: str
    metrics: EvaluationMetrics
    estimated_ate: float | None = None
    estimated_att: float | None = None
    estimated_cate: np.ndarray | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    runtime_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_successful(self) -> bool:
        """Check if evaluation completed successfully."""
        return self.metrics.ate_bias is not None or self.metrics.att_bias is not None

    def passes_threshold(
        self,
        max_bias: float = 0.5,
        max_rmse: float = 1.0,
        min_coverage: float = 0.85,
    ) -> bool:
        """Check if results pass quality thresholds."""
        if self.metrics.ate_bias is not None:
            if abs(self.metrics.ate_bias) > max_bias:
                return False
        if self.metrics.ate_rmse is not None:
            if self.metrics.ate_rmse > max_rmse:
                return False
        if self.metrics.coverage_95 is not None:
            if self.metrics.coverage_95 < min_coverage:
                return False
        return True


class CausalEvaluator:
    """Evaluator for causal inference methods against benchmarks."""

    def __init__(self):
        """Initialize evaluator."""
        self.logger = get_logger("causal_evaluator")

    def evaluate_ate(
        self,
        estimated: float,
        true_value: float,
        std_error: float | None = None,
        ci_lower: float | None = None,
        ci_upper: float | None = None,
    ) -> dict[str, float]:
        """Evaluate ATE/ATT estimation.

        Args:
            estimated: Estimated treatment effect
            true_value: True treatment effect (ground truth)
            std_error: Standard error of estimate
            ci_lower: Lower bound of confidence interval
            ci_upper: Upper bound of confidence interval

        Returns:
            Dictionary of evaluation metrics
        """
        bias = estimated - true_value
        rmse = np.sqrt(bias ** 2)

        metrics = {
            "bias": bias,
            "abs_bias": abs(bias),
            "rmse": rmse,
            "relative_bias": bias / abs(true_value) if true_value != 0 else np.nan,
        }

        # CI coverage
        if ci_lower is not None and ci_upper is not None:
            covered = ci_lower <= true_value <= ci_upper
            metrics["covered"] = float(covered)
            metrics["ci_width"] = ci_upper - ci_lower

        return metrics

    def evaluate_cate(
        self,
        estimated_cate: np.ndarray,
        true_cate: np.ndarray,
    ) -> dict[str, float]:
        """Evaluate CATE (heterogeneous effects) estimation.

        Args:
            estimated_cate: Estimated individual treatment effects
            true_cate: True individual treatment effects

        Returns:
            Dictionary of CATE evaluation metrics
        """
        # Align arrays
        estimated_cate = np.asarray(estimated_cate).flatten()
        true_cate = np.asarray(true_cate).flatten()

        if len(estimated_cate) != len(true_cate):
            raise ValueError("CATE arrays must have same length")

        # PEHE: Precision in Estimation of Heterogeneous Effects
        pehe = np.sqrt(np.mean((estimated_cate - true_cate) ** 2))

        # RMSE
        rmse = np.sqrt(np.mean((estimated_cate - true_cate) ** 2))

        # Correlation
        if np.std(estimated_cate) > 0 and np.std(true_cate) > 0:
            corr = np.corrcoef(estimated_cate, true_cate)[0, 1]
        else:
            corr = np.nan

        # Mean absolute error
        mae = np.mean(np.abs(estimated_cate - true_cate))

        # Bias
        bias = np.mean(estimated_cate - true_cate)

        return {
            "pehe": pehe,
            "rmse": rmse,
            "correlation": corr,
            "mae": mae,
            "bias": bias,
        }

    async def evaluate_method(
        self,
        method_fn: Callable[[pd.DataFrame, str, str], dict[str, Any]],
        benchmark: BenchmarkDataset,
        n_samples: int | None = None,
        n_repetitions: int = 10,
        seed: int = 42,
    ) -> EvaluationResult:
        """Evaluate a causal inference method on a benchmark.

        Args:
            method_fn: Function that takes (df, treatment_col, outcome_col)
                      and returns dict with 'ate', 'std_error', 'ci_lower', 'ci_upper'
            benchmark: Benchmark dataset
            n_samples: Number of samples per repetition
            n_repetitions: Number of Monte Carlo repetitions
            seed: Base random seed

        Returns:
            EvaluationResult with aggregated metrics
        """
        import time

        self.logger.info(
            "evaluating_method",
            benchmark=benchmark.name,
            n_samples=n_samples,
            n_repetitions=n_repetitions,
        )

        start_time = time.time()
        results = []

        for rep in range(n_repetitions):
            rep_seed = seed + rep

            # Generate data
            df = benchmark.generate(n_samples=n_samples, seed=rep_seed)

            try:
                # Run method
                estimate = await self._run_method(
                    method_fn,
                    df,
                    benchmark.treatment_col,
                    benchmark.outcome_col,
                )
                results.append(estimate)

            except Exception as e:
                self.logger.warning(
                    "method_failed",
                    benchmark=benchmark.name,
                    repetition=rep,
                    error=str(e),
                )

        if not results:
            return EvaluationResult(
                benchmark_name=benchmark.name,
                method_name="unknown",
                metrics=EvaluationMetrics(),
                runtime_seconds=time.time() - start_time,
                metadata={"error": "All repetitions failed"},
            )

        # Aggregate results
        metrics = self._aggregate_results(results, benchmark)

        # Mean estimates
        mean_ate = np.mean([r.get("ate", np.nan) for r in results if "ate" in r])
        mean_ci_lower = np.mean([r.get("ci_lower", np.nan) for r in results if "ci_lower" in r])
        mean_ci_upper = np.mean([r.get("ci_upper", np.nan) for r in results if "ci_upper" in r])

        return EvaluationResult(
            benchmark_name=benchmark.name,
            method_name=results[0].get("method", "unknown"),
            metrics=metrics,
            estimated_ate=float(mean_ate) if not np.isnan(mean_ate) else None,
            ci_lower=float(mean_ci_lower) if not np.isnan(mean_ci_lower) else None,
            ci_upper=float(mean_ci_upper) if not np.isnan(mean_ci_upper) else None,
            runtime_seconds=time.time() - start_time,
            metadata={"n_successful": len(results), "n_repetitions": n_repetitions},
        )

    async def _run_method(
        self,
        method_fn: Callable,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
    ) -> dict[str, Any]:
        """Run a causal method (sync or async)."""
        if asyncio.iscoroutinefunction(method_fn):
            return await method_fn(df, treatment_col, outcome_col)
        else:
            return method_fn(df, treatment_col, outcome_col)

    def _aggregate_results(
        self,
        results: list[dict[str, Any]],
        benchmark: BenchmarkDataset,
    ) -> EvaluationMetrics:
        """Aggregate results across repetitions."""
        ate_estimates = [r["ate"] for r in results if "ate" in r and r["ate"] is not None]
        att_estimates = [r.get("att") for r in results if r.get("att") is not None]

        metrics = EvaluationMetrics()

        # ATE metrics
        if ate_estimates and benchmark.true_ate is not None:
            ate_array = np.array(ate_estimates)
            metrics.ate_bias = float(np.mean(ate_array) - benchmark.true_ate)
            metrics.ate_rmse = float(np.sqrt(np.mean((ate_array - benchmark.true_ate) ** 2)))

            # Coverage
            coverages = []
            for r in results:
                if "ci_lower" in r and "ci_upper" in r:
                    covered = r["ci_lower"] <= benchmark.true_ate <= r["ci_upper"]
                    coverages.append(float(covered))
            if coverages:
                metrics.coverage_95 = float(np.mean(coverages))

        # ATT metrics
        if att_estimates and benchmark.true_att is not None:
            att_array = np.array(att_estimates)
            metrics.att_bias = float(np.mean(att_array) - benchmark.true_att)
            metrics.att_rmse = float(np.sqrt(np.mean((att_array - benchmark.true_att) ** 2)))

        # CATE metrics
        cate_estimates = [r.get("cate") for r in results if r.get("cate") is not None]
        if cate_estimates and benchmark.true_cate is not None:
            # Use first repetition's CATE for PEHE
            cate_eval = self.evaluate_cate(cate_estimates[0], benchmark.true_cate)
            metrics.pehe = cate_eval["pehe"]
            metrics.cate_rmse = cate_eval["rmse"]
            metrics.cate_corr = cate_eval["correlation"]

        return metrics


async def run_benchmark_suite(
    method_fns: dict[str, Callable],
    benchmarks: list[str] | None = None,
    n_samples: int = 1000,
    n_repetitions: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Run full benchmark suite across methods and datasets.

    Args:
        method_fns: Dictionary of method name -> method function
        benchmarks: List of benchmark names (or None for all)
        n_samples: Samples per dataset
        n_repetitions: Monte Carlo repetitions
        seed: Random seed

    Returns:
        DataFrame with all results
    """
    evaluator = CausalEvaluator()

    # Get benchmarks
    if benchmarks:
        benchmark_list = [get_benchmark(name) for name in benchmarks]
    else:
        benchmark_list = get_all_benchmarks()

    results = []

    for benchmark in benchmark_list:
        for method_name, method_fn in method_fns.items():
            logger.info(
                "running_benchmark",
                benchmark=benchmark.name,
                method=method_name,
            )

            try:
                result = await evaluator.evaluate_method(
                    method_fn=method_fn,
                    benchmark=benchmark,
                    n_samples=n_samples,
                    n_repetitions=n_repetitions,
                    seed=seed,
                )

                results.append({
                    "benchmark": benchmark.name,
                    "method": method_name,
                    "true_ate": benchmark.true_ate,
                    "true_att": benchmark.true_att,
                    "estimated_ate": result.estimated_ate,
                    "ate_bias": result.metrics.ate_bias,
                    "ate_rmse": result.metrics.ate_rmse,
                    "coverage_95": result.metrics.coverage_95,
                    "pehe": result.metrics.pehe,
                    "runtime_seconds": result.runtime_seconds,
                    "passed": result.passes_threshold(),
                })

            except Exception as e:
                logger.error(
                    "benchmark_failed",
                    benchmark=benchmark.name,
                    method=method_name,
                    error=str(e),
                )
                results.append({
                    "benchmark": benchmark.name,
                    "method": method_name,
                    "error": str(e),
                    "passed": False,
                })

    return pd.DataFrame(results)


def create_method_wrapper(
    causal_methods_module,
    method_name: str,
    **kwargs,
) -> Callable:
    """Create a wrapper function for a causal method.

    This adapts the existing causal methods to the evaluation interface.

    Args:
        causal_methods_module: The src.causal.methods module
        method_name: Name of method (e.g., "ols", "ipw", "aipw")
        **kwargs: Additional arguments to pass to the method

    Returns:
        Callable matching evaluation interface
    """
    def method_fn(
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
    ) -> dict[str, Any]:
        """Wrapper that calls the causal method."""
        # Get covariates (all columns except treatment and outcome)
        covariate_cols = [
            c for c in df.columns
            if c not in [treatment_col, outcome_col]
            and not c.endswith("_true")  # Exclude ground truth columns
        ]

        T = df[treatment_col].values
        Y = df[outcome_col].values
        X = df[covariate_cols].values

        # Call the appropriate method
        if method_name == "ols":
            result = causal_methods_module.estimate_ate_ols(Y, T, X)
        elif method_name == "ipw":
            result = causal_methods_module.estimate_ate_ipw(Y, T, X)
        elif method_name == "aipw":
            result = causal_methods_module.estimate_ate_aipw(Y, T, X)
        elif method_name == "psm":
            result = causal_methods_module.estimate_ate_psm(Y, T, X, **kwargs)
        elif method_name == "did":
            # DiD needs pre/post structure
            result = {"ate": None, "error": "DiD requires panel data"}
        else:
            raise ValueError(f"Unknown method: {method_name}")

        return {
            "method": method_name,
            "ate": result.get("ate"),
            "att": result.get("att"),
            "std_error": result.get("std_error"),
            "ci_lower": result.get("ci_lower"),
            "ci_upper": result.get("ci_upper"),
        }

    return method_fn
