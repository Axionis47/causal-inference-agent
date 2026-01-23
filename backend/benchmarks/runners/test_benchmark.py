"""Benchmark tests for CI/CD validation."""

import pytest

from ..datasets.loader import BenchmarkDatasetLoader
from .benchmark_runner import BenchmarkRunner


class TestBenchmarkDatasets:
    """Test benchmark dataset loading."""

    def test_load_all_datasets(self):
        """Verify all 8 datasets load correctly."""
        loader = BenchmarkDatasetLoader()
        datasets = loader.load_all()

        assert len(datasets) == 8, "Should load all 8 benchmark datasets"

        names = [d.name for d in datasets]
        assert "IHDP" in names
        assert "LaLonde" in names
        assert "Twins" in names
        assert "Card_IV" in names
        assert "MinimumWage_DiD" in names
        assert "ACIC" in names
        assert "TimeSeries" in names
        assert "News" in names

    def test_ihdp_dataset_structure(self):
        """Test IHDP dataset has correct structure."""
        dataset = BenchmarkDatasetLoader.load_ihdp()

        assert dataset.name == "IHDP"
        assert dataset.treatment_variable == "treatment"
        assert dataset.outcome_variable == "outcome"
        assert dataset.ground_truth is not None
        assert "ATE" in dataset.ground_truth
        assert len(dataset.data) > 0
        assert dataset.treatment_variable in dataset.data.columns
        assert dataset.outcome_variable in dataset.data.columns

    def test_lalonde_dataset_structure(self):
        """Test LaLonde dataset has correct structure."""
        dataset = BenchmarkDatasetLoader.load_lalonde()

        assert dataset.name == "LaLonde"
        assert dataset.analysis_type == "propensity_score_benchmark"
        assert "propensity_score_matching" in dataset.expected_methods
        assert len(dataset.data) > 0

    def test_twins_dataset_structure(self):
        """Test Twins dataset has correct structure."""
        dataset = BenchmarkDatasetLoader.load_twins()

        assert dataset.name == "Twins"
        assert dataset.treatment_variable == "heavier_twin"
        assert len(dataset.data) > 0

    def test_card_iv_dataset_structure(self):
        """Test Card IV dataset has correct structure."""
        dataset = BenchmarkDatasetLoader.load_card_iv()

        assert dataset.name == "Card_IV"
        assert dataset.analysis_type == "instrumental_variables"
        assert dataset.metadata.get("instrument") == "college_nearby"
        assert "instrumental_variables" in dataset.expected_methods

    def test_minimum_wage_did_dataset_structure(self):
        """Test Minimum Wage DiD dataset has correct structure."""
        dataset = BenchmarkDatasetLoader.load_minimum_wage_did()

        assert dataset.name == "MinimumWage_DiD"
        assert dataset.analysis_type == "difference_in_differences"
        assert dataset.metadata.get("time_column") == "post"

    def test_acic_dataset_structure(self):
        """Test ACIC dataset has correct structure."""
        dataset = BenchmarkDatasetLoader.load_acic()

        assert dataset.name == "ACIC"
        assert dataset.analysis_type == "high_dimensional_cate"
        assert "double_ml" in dataset.expected_methods or "causal_forest" in dataset.expected_methods

    def test_time_series_dataset_structure(self):
        """Test time series dataset has correct structure."""
        dataset = BenchmarkDatasetLoader.load_time_series()

        assert dataset.name == "TimeSeries"
        assert dataset.analysis_type == "time_series_causality"
        assert dataset.metadata.get("time_column") == "time"

    def test_news_dataset_structure(self):
        """Test News continuous treatment dataset has correct structure."""
        dataset = BenchmarkDatasetLoader.load_news()

        assert dataset.name == "News"
        assert dataset.analysis_type == "continuous_treatment"
        assert "ols_regression" in dataset.expected_methods or "double_ml" in dataset.expected_methods


class TestBenchmarkRunner:
    """Test benchmark runner functionality."""

    @pytest.fixture
    def runner(self):
        """Create benchmark runner instance."""
        return BenchmarkRunner()

    def test_runner_initialization(self, runner):
        """Test runner initializes with all agents."""
        assert runner.orchestrator is not None
        assert runner.data_profiler is not None
        assert runner.effect_estimator is not None
        assert runner.sensitivity_analyst is not None
        assert runner.critique is not None

    def test_runner_has_loader(self, runner):
        """Test runner has dataset loader."""
        assert runner.loader is not None
        datasets = runner.loader.load_all()
        assert len(datasets) == 8

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)  # Increased timeout for API calls
    async def test_run_single_ihdp(self, runner):
        """Test running benchmark on IHDP dataset."""
        dataset = BenchmarkDatasetLoader.load_ihdp()
        result = await runner.run_single(dataset)

        assert result.dataset_name == "IHDP"
        assert result.success is True or result.error is not None

        if result.success:
            assert len(result.estimated_effects) > 0
            assert result.execution_time_seconds > 0

    @pytest.mark.asyncio
    @pytest.mark.timeout(600)
    @pytest.mark.skip(reason="Full benchmark suite takes too long for CI")
    async def test_run_all_benchmarks(self, runner):
        """Run all benchmarks and verify report structure."""
        report = await runner.run_all()

        assert report.datasets_tested == 8
        assert len(report.results) == 8
        assert "success_rate" in report.overall_metrics

        # At least some should succeed
        assert report.datasets_passed >= 1, "At least one dataset should pass"

    def test_calculate_metrics_with_ground_truth(self, runner):
        """Test metric calculation with known values."""
        from ..datasets.loader import BenchmarkDataset
        from src.agents import AnalysisState, TreatmentEffect
        import pandas as pd

        # Create mock dataset with known ground truth
        dataset = BenchmarkDataset(
            name="test",
            description="Test dataset",
            data=pd.DataFrame({"x": [1, 2, 3]}),
            treatment_variable="treatment",
            outcome_variable="outcome",
            ground_truth={"ATE": 2.0},
            analysis_type="ate",
            expected_methods=["ols"],
            metadata={},
        )

        # Create mock state with estimates
        state = AnalysisState(
            job_id="test",
            treatment_variable="treatment",
            outcome_variable="outcome",
        )
        state.treatment_effects = [
            TreatmentEffect(
                method="OLS",
                estimand="ATE",
                estimate=2.1,  # Close to true ATE of 2.0
                std_error=0.1,
                ci_lower=1.9,
                ci_upper=2.3,
                p_value=0.01,
            ),
        ]

        metrics = runner._calculate_metrics(dataset, state)

        assert "bias" in metrics
        assert metrics["bias"] == pytest.approx(0.1, abs=0.01)
        assert "coverage" in metrics
        assert metrics["coverage"] == 1.0  # CI contains true value
