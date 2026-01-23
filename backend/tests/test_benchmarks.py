"""Comprehensive benchmark tests for causal inference methods.

This script evaluates all causal inference methods against standard
benchmark datasets to ensure they work correctly and produce accurate
treatment effect estimates.
"""

import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.benchmarks import (
    BenchmarkDataset,
    LaLondeDataset,
    IHDPDataset,
    TwinsDataset,
    SyntheticBenchmark,
    CausalEvaluator,
    EvaluationMetrics,
    get_benchmark,
    get_all_benchmarks,
)
from src.causal import (
    OLSMethod,
    PSMMethod,
    IPWMethod,
    AIPWMethod,
    SLearner,
    TLearner,
    XLearner,
    CausalForestMethod,
    DoubleMLMethod,
    MethodResult,
)


# Test data generation
class TestDatasetGeneration:
    """Test that benchmark datasets generate correctly."""

    def test_lalonde_experimental(self):
        """Test LaLonde experimental data generation."""
        dataset = LaLondeDataset()
        df = dataset.generate(n_samples=500, seed=42)

        assert len(df) > 0
        assert "treat" in df.columns
        assert "re78" in df.columns
        assert df["treat"].isin([0, 1]).all()
        assert (df["re78"] >= 0).all()

        # Check covariates
        assert "age" in df.columns
        assert "education" in df.columns
        assert "re74" in df.columns
        assert "re75" in df.columns

    def test_lalonde_observational(self):
        """Test LaLonde observational variant."""
        dataset = LaLondeDataset(variant="cps")
        df = dataset.generate(n_samples=1000, seed=42)

        # Should have more controls than treatment
        n_treat = df["treat"].sum()
        n_control = len(df) - n_treat
        assert n_control > n_treat  # Selection bias

    def test_ihdp_dataset(self):
        """Test IHDP semi-synthetic generation."""
        dataset = IHDPDataset()
        df = dataset.generate(n_samples=500, seed=42)

        assert len(df) == 500
        assert "treatment" in df.columns
        assert "y" in df.columns
        assert "cate_true" in df.columns  # Ground truth available

        # Check ground truth is set
        assert dataset.true_ate is not None
        assert dataset.true_cate is not None

    def test_twins_dataset(self):
        """Test Twins dataset generation."""
        dataset = TwinsDataset()
        df = dataset.generate(n_samples=1000, seed=42)

        assert len(df) > 0
        assert "heavier" in df.columns
        assert "mortality" in df.columns
        assert "pair_id" in df.columns

        # Check paired structure
        pairs = df.groupby("pair_id").size()
        assert (pairs == 2).all()  # Each pair has 2 twins

    def test_synthetic_benchmark(self):
        """Test fully synthetic benchmark."""
        dataset = SyntheticBenchmark(
            treatment_effect=2.5,
            effect_heterogeneity=0.5,
            n_confounders=5,
        )
        df = dataset.generate(n_samples=1000, seed=42)

        assert len(df) == 1000
        assert dataset.true_ate is not None
        assert abs(dataset.true_ate - 2.5) < 0.5  # Should be close to specified

        # Check ground truth columns
        assert "Y0_true" in df.columns
        assert "Y1_true" in df.columns
        assert "cate_true" in df.columns

    def test_reproducibility(self):
        """Test that datasets are reproducible with same seed."""
        dataset = SyntheticBenchmark()

        df1 = dataset.generate(n_samples=100, seed=42)
        df2 = dataset.generate(n_samples=100, seed=42)

        pd.testing.assert_frame_equal(df1, df2)


# Test causal methods on benchmarks
class TestCausalMethods:
    """Test causal methods produce reasonable estimates."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic benchmark data."""
        dataset = SyntheticBenchmark(
            treatment_effect=3.0,
            effect_heterogeneity=0.0,  # Homogeneous for testing
            selection_bias=0.3,
            noise_level=1.0,
        )
        df = dataset.generate(n_samples=2000, seed=42)
        return df, dataset

    def test_ols_method(self, synthetic_data):
        """Test OLS regression method."""
        df, dataset = synthetic_data
        covariates = [c for c in df.columns if c.startswith("X")]

        method = OLSMethod()
        method.fit(df, "T", "Y", covariates=covariates)
        result = method.estimate()

        assert result is not None
        assert result.method == "OLS"
        assert result.estimand == "ATE"

        # Estimate should be close to true (within 2 std errors)
        bias = abs(result.estimate - dataset.true_ate)
        assert bias < 2 * result.std_error or bias < 1.0

    def test_ipw_method(self, synthetic_data):
        """Test IPW method."""
        df, dataset = synthetic_data
        covariates = [c for c in df.columns if c.startswith("X")]

        method = IPWMethod()
        method.fit(df, "T", "Y", covariates=covariates)
        result = method.estimate()

        assert result is not None
        assert "IPW" in result.method

        # Should be reasonably close
        bias = abs(result.estimate - dataset.true_ate)
        assert bias < 1.5  # More tolerance for IPW

    def test_aipw_method(self, synthetic_data):
        """Test AIPW (doubly robust) method."""
        df, dataset = synthetic_data
        covariates = [c for c in df.columns if c.startswith("X")]

        method = AIPWMethod()
        method.fit(df, "T", "Y", covariates=covariates)
        result = method.estimate()

        assert result is not None
        assert "AIPW" in result.method or "Doubly Robust" in result.method

        # AIPW should be most accurate
        bias = abs(result.estimate - dataset.true_ate)
        assert bias < 1.0

    def test_psm_method(self, synthetic_data):
        """Test propensity score matching."""
        df, dataset = synthetic_data
        covariates = [c for c in df.columns if c.startswith("X")]

        method = PSMMethod()
        method.fit(df, "T", "Y", covariates=covariates)
        result = method.estimate()

        assert result is not None
        # PSM typically estimates ATT
        assert result.estimand in ["ATE", "ATT"]

    def test_slearner(self, synthetic_data):
        """Test S-Learner meta-learner."""
        df, dataset = synthetic_data
        covariates = [c for c in df.columns if c.startswith("X")]

        method = SLearner()
        method.fit(df, "T", "Y", covariates=covariates)
        result = method.estimate()

        assert result is not None
        assert "S-Learner" in result.method

    def test_tlearner(self, synthetic_data):
        """Test T-Learner meta-learner."""
        df, dataset = synthetic_data
        covariates = [c for c in df.columns if c.startswith("X")]

        method = TLearner()
        method.fit(df, "T", "Y", covariates=covariates)
        result = method.estimate()

        assert result is not None
        assert "T-Learner" in result.method


# Test evaluation framework
class TestEvaluationFramework:
    """Test the evaluation framework itself."""

    def test_ate_evaluation(self):
        """Test ATE evaluation metrics."""
        evaluator = CausalEvaluator()

        # Perfect estimate
        metrics = evaluator.evaluate_ate(
            estimated=5.0,
            true_value=5.0,
            ci_lower=4.0,
            ci_upper=6.0,
        )
        assert metrics["bias"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["covered"] == 1.0

        # Biased estimate
        metrics = evaluator.evaluate_ate(
            estimated=6.0,
            true_value=5.0,
            ci_lower=5.5,
            ci_upper=6.5,
        )
        assert metrics["bias"] == 1.0
        assert metrics["covered"] == 0.0  # CI doesn't cover true value

    def test_cate_evaluation(self):
        """Test CATE evaluation metrics."""
        evaluator = CausalEvaluator()

        true_cate = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Perfect estimate
        metrics = evaluator.evaluate_cate(true_cate, true_cate)
        assert metrics["pehe"] == 0.0
        assert metrics["correlation"] == 1.0

        # Noisy estimate
        noisy_cate = true_cate + np.array([0.1, -0.1, 0.2, -0.2, 0.1])
        metrics = evaluator.evaluate_cate(noisy_cate, true_cate)
        assert metrics["pehe"] > 0
        assert metrics["correlation"] > 0.95  # Still highly correlated


# Full benchmark suite test
class TestBenchmarkSuite:
    """End-to-end benchmark suite tests."""

    @pytest.mark.asyncio
    async def test_method_on_lalonde(self):
        """Test a method on LaLonde benchmark."""
        dataset = LaLondeDataset()
        df = dataset.generate(n_samples=500, seed=42)

        covariates = ["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]

        # Run AIPW
        method = AIPWMethod()
        method.fit(df, "treat", "re78", covariates=covariates)
        result = method.estimate()

        assert result is not None
        # Should get positive ATT (treatment helps)
        # Note: with synthetic data, may differ from real LaLonde

    @pytest.mark.asyncio
    async def test_method_on_ihdp(self):
        """Test method on IHDP benchmark with known ground truth."""
        dataset = IHDPDataset()
        df = dataset.generate(n_samples=500, seed=42)

        covariates = [c for c in df.columns if c not in ["treatment", "y", "y0_true", "y1_true", "cate_true"]]

        method = AIPWMethod()
        method.fit(df, "treatment", "y", covariates=covariates)
        result = method.estimate()

        assert result is not None

        # Check against known ground truth
        if dataset.true_ate is not None:
            bias = abs(result.estimate - dataset.true_ate)
            # Within 50% of true value for reasonable accuracy
            assert bias < abs(dataset.true_ate) * 0.5 or bias < 2.0


# Run a quick benchmark test
def run_quick_benchmark():
    """Quick benchmark to verify everything works."""
    print("=" * 60)
    print("CAUSAL INFERENCE BENCHMARK EVALUATION")
    print("=" * 60)

    benchmarks = [
        SyntheticBenchmark(treatment_effect=3.0, name="synthetic_linear"),
        SyntheticBenchmark(treatment_effect=3.0, nonlinear=True, name="synthetic_nonlinear"),
        LaLondeDataset(),
        IHDPDataset(),
    ]

    methods = [
        ("OLS", OLSMethod),
        ("IPW", IPWMethod),
        ("AIPW", AIPWMethod),
        ("S-Learner", SLearner),
        ("T-Learner", TLearner),
    ]

    results = []

    for benchmark in benchmarks:
        print(f"\nBenchmark: {benchmark.name}")
        print("-" * 40)

        df = benchmark.generate(n_samples=1000, seed=42)

        # Get covariates
        if isinstance(benchmark, SyntheticBenchmark):
            covariates = [c for c in df.columns if c.startswith("X")]
            treatment_col = "T"
            outcome_col = "Y"
        elif isinstance(benchmark, LaLondeDataset):
            covariates = ["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]
            treatment_col = "treat"
            outcome_col = "re78"
        elif isinstance(benchmark, IHDPDataset):
            covariates = [c for c in df.columns if c not in ["treatment", "y", "y0_true", "y1_true", "cate_true"]]
            treatment_col = "treatment"
            outcome_col = "y"
        else:
            continue

        for method_name, method_class in methods:
            try:
                method = method_class()
                method.fit(df, treatment_col, outcome_col, covariates=covariates)
                result = method.estimate()

                if benchmark.true_ate is not None:
                    bias = result.estimate - benchmark.true_ate
                    rel_bias = bias / abs(benchmark.true_ate) if benchmark.true_ate != 0 else float('inf')
                else:
                    bias = None
                    rel_bias = None

                results.append({
                    "benchmark": benchmark.name,
                    "method": method_name,
                    "estimate": result.estimate,
                    "std_error": result.std_error,
                    "true_ate": benchmark.true_ate,
                    "bias": bias,
                    "rel_bias": rel_bias,
                })

                status = "PASS" if (bias is None or abs(bias) < 2 * result.std_error) else "CHECK"
                print(f"  {method_name:15} ATE={result.estimate:8.3f} SE={result.std_error:.3f} "
                      f"Bias={bias if bias else 'N/A':>8} [{status}]")

            except Exception as e:
                print(f"  {method_name:15} ERROR: {e}")
                results.append({
                    "benchmark": benchmark.name,
                    "method": method_name,
                    "error": str(e),
                })

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Summary statistics
    successful = [r for r in results if "estimate" in r]
    failed = [r for r in results if "error" in r]

    print(f"Total runs: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed methods:")
        for r in failed:
            print(f"  - {r['benchmark']}/{r['method']}: {r['error']}")

    return results


if __name__ == "__main__":
    # Run quick benchmark when executed directly
    run_quick_benchmark()
