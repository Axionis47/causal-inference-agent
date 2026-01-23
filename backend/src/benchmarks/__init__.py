"""Benchmark datasets and evaluation framework for causal inference."""

from .datasets import (
    BenchmarkDataset,
    LaLondeDataset,
    IHDPDataset,
    TwinsDataset,
    SyntheticBenchmark,
    get_all_benchmarks,
    get_benchmark,
)

from .evaluation import (
    CausalEvaluator,
    EvaluationResult,
    EvaluationMetrics,
    run_benchmark_suite,
)

__all__ = [
    # Datasets
    "BenchmarkDataset",
    "LaLondeDataset",
    "IHDPDataset",
    "TwinsDataset",
    "SyntheticBenchmark",
    "get_all_benchmarks",
    "get_benchmark",
    # Evaluation
    "CausalEvaluator",
    "EvaluationResult",
    "EvaluationMetrics",
    "run_benchmark_suite",
]
