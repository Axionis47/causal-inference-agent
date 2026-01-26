"""Agentic evaluation for EDAAgent.

Tests the agent's ability to:
1. Detect data quality issues (outliers, missing data)
2. Identify multicollinearity
3. Assess covariate balance
4. Give appropriate causal readiness assessment
5. Handle different data scenarios
"""

import numpy as np
import pandas as pd

from src.agents.base import DataProfile
from src.agents.specialists.eda_agent import EDAAgent
from src.logging_config.structured import get_logger

from .base import (
    AgenticEvaluator,
    EvalCase,
    EvalCategory,
    EvalDifficulty,
    EvalSuite,
    AnalysisState,
)

logger = get_logger(__name__)


class EDAAgentEvaluator(AgenticEvaluator):
    """Evaluator for EDAAgent."""

    @property
    def agent_name(self) -> str:
        return "eda_agent"

    def create_agent(self) -> EDAAgent:
        return EDAAgent()

    def get_eval_suite(self) -> EvalSuite:
        """Create comprehensive evaluation suite."""
        suite = EvalSuite(
            name="EDA Agent Evaluation Suite",
            agent_name=self.agent_name,
            description="Tests EDA agent data quality assessment capabilities",
        )

        # EASY cases
        suite.add_case(self._case_clean_balanced_data())
        suite.add_case(self._case_obvious_outliers())
        suite.add_case(self._case_missing_data_pattern())

        # MEDIUM cases
        suite.add_case(self._case_covariate_imbalance())
        suite.add_case(self._case_high_multicollinearity())
        suite.add_case(self._case_skewed_distributions())
        suite.add_case(self._case_mixed_quality_issues())

        # HARD cases
        suite.add_case(self._case_subtle_outliers())
        suite.add_case(self._case_confounding_pattern())
        suite.add_case(self._case_selection_bias())
        suite.add_case(self._case_complex_interactions())

        return suite

    def check_task_correctness(
        self,
        case: EvalCase,
        final_state: AnalysisState,
    ) -> tuple[bool, dict[str, float]]:
        """Check if EDA output is correct."""
        metrics = {}
        expected = case.expected_outputs

        eda_result = final_state.eda_result
        if eda_result is None:
            return False, {"eda_completed": 0.0}

        metrics["eda_completed"] = 1.0

        # Check data quality score accuracy
        expected_score_range = expected.get("quality_score_range", (0, 100))
        if eda_result.data_quality_score is not None:
            in_range = expected_score_range[0] <= eda_result.data_quality_score <= expected_score_range[1]
            metrics["quality_score_accurate"] = 1.0 if in_range else 0.0
        else:
            metrics["quality_score_accurate"] = 0.0

        # Check issue detection
        expected_issues = expected.get("expected_issues", [])
        if expected_issues:
            detected_issues = eda_result.data_quality_issues or []
            # Check if expected issues are detected (fuzzy match)
            detected_count = 0
            for expected_issue in expected_issues:
                for detected in detected_issues:
                    if expected_issue.lower() in detected.lower():
                        detected_count += 1
                        break
            metrics["issue_detection_rate"] = detected_count / len(expected_issues)
        else:
            metrics["issue_detection_rate"] = 1.0

        # Check outlier detection
        expected_outlier_cols = expected.get("outlier_columns", [])
        if expected_outlier_cols and eda_result.outliers:
            detected_cols = set(eda_result.outliers.keys())
            overlap = len(detected_cols & set(expected_outlier_cols))
            metrics["outlier_detection_recall"] = overlap / len(expected_outlier_cols)
        else:
            metrics["outlier_detection_recall"] = 1.0 if not expected_outlier_cols else 0.0

        # Check balance assessment
        expected_imbalanced = expected.get("imbalanced_covariates", [])
        if expected_imbalanced and eda_result.covariate_balance:
            detected_imbalanced = [
                col for col, info in eda_result.covariate_balance.items()
                if not info.get("is_balanced", True)
            ]
            overlap = len(set(detected_imbalanced) & set(expected_imbalanced))
            metrics["balance_detection_recall"] = overlap / len(expected_imbalanced)
        else:
            metrics["balance_detection_recall"] = 1.0 if not expected_imbalanced else 0.0

        # Check causal readiness
        expected_readiness = expected.get("causal_readiness")
        if expected_readiness:
            # Allow partial match (e.g., "needs_attention" matches "needs attention")
            actual_readiness = getattr(eda_result, 'causal_readiness', None)
            if actual_readiness:
                match = expected_readiness.lower().replace("_", " ") in actual_readiness.lower().replace("_", " ")
                metrics["readiness_correct"] = 1.0 if match else 0.0
            else:
                metrics["readiness_correct"] = 0.0
        else:
            metrics["readiness_correct"] = 1.0

        # Determine overall correctness
        is_correct = (
            metrics.get("eda_completed", 0) >= 1.0 and
            metrics.get("quality_score_accurate", 0) >= 0.5 and
            metrics.get("issue_detection_rate", 0) >= 0.5
        )

        return is_correct, metrics

    def _create_data_profile(self, df: pd.DataFrame, treatment_col: str, outcome_col: str) -> DataProfile:
        """Create a DataProfile for the dataframe."""
        feature_types = {}
        for col in df.columns:
            n_unique = df[col].nunique()
            if n_unique == 2:
                feature_types[col] = "binary"
            elif df[col].dtype in ['object', 'category']:
                feature_types[col] = "categorical"
            else:
                feature_types[col] = "numeric"

        return DataProfile(
            n_samples=len(df),
            n_features=len(df.columns),
            feature_names=list(df.columns),
            feature_types=feature_types,
            missing_values={col: int(df[col].isna().sum()) for col in df.columns},
            numeric_stats={},
            categorical_stats={},
            treatment_candidates=[treatment_col],
            outcome_candidates=[outcome_col],
            potential_confounders=[c for c in df.columns if c not in [treatment_col, outcome_col]],
        )

    # ================== EASY CASES ==================

    def _case_clean_balanced_data(self) -> EvalCase:
        """Easy: Clean, well-balanced data with no issues."""
        np.random.seed(100)
        n = 500

        treatment = np.random.binomial(1, 0.5, n)
        age = np.random.normal(40, 10, n)
        income = np.random.normal(50000, 15000, n)
        outcome = 100 + 10 * treatment + 0.5 * age + np.random.normal(0, 10, n)

        df = pd.DataFrame({
            "treatment": treatment,
            "age": age,
            "income": income,
            "outcome": outcome,
        })

        profile = self._create_data_profile(df, "treatment", "outcome")

        return EvalCase(
            name="clean_balanced_data",
            description="Clean, well-balanced RCT data with no quality issues",
            difficulty=EvalDifficulty.EASY,
            category=EvalCategory.TASK_COMPLETION,
            dataframe=df,
            dataset_info={"name": "clean_data"},
            initial_state_overrides={
                "data_profile": profile,
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "quality_score_range": (75, 100),
                "expected_issues": [],
                "causal_readiness": "ready",
            },
            max_expected_steps=10,
        )

    def _case_obvious_outliers(self) -> EvalCase:
        """Easy: Data with obvious outliers."""
        np.random.seed(101)
        n = 500

        treatment = np.random.binomial(1, 0.5, n)
        age = np.random.normal(40, 10, n)
        income = np.random.normal(50000, 15000, n)

        # Add obvious outliers
        income[0:10] = 1000000  # 10 extreme outliers
        age[10:15] = 200  # Impossible ages

        outcome = 100 + 10 * treatment + np.random.normal(0, 10, n)

        df = pd.DataFrame({
            "treatment": treatment,
            "age": age,
            "income": income,
            "outcome": outcome,
        })

        profile = self._create_data_profile(df, "treatment", "outcome")

        return EvalCase(
            name="obvious_outliers",
            description="Data with obvious outliers in age and income",
            difficulty=EvalDifficulty.EASY,
            category=EvalCategory.TASK_COMPLETION,
            dataframe=df,
            dataset_info={"name": "outlier_data"},
            initial_state_overrides={
                "data_profile": profile,
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "quality_score_range": (40, 75),
                "expected_issues": ["outlier"],
                "outlier_columns": ["age", "income"],
                "causal_readiness": "needs_attention",
            },
            expected_tool_calls=["detect_outliers"],
            max_expected_steps=12,
        )

    def _case_missing_data_pattern(self) -> EvalCase:
        """Easy: Data with significant missing values."""
        np.random.seed(102)
        n = 500

        treatment = np.random.binomial(1, 0.5, n).astype(float)
        age = np.random.normal(40, 10, n)
        income = np.random.normal(50000, 15000, n)
        outcome = np.random.normal(100, 20, n)

        # Add missing values
        age[np.random.choice(n, 100, replace=False)] = np.nan  # 20% missing
        income[np.random.choice(n, 75, replace=False)] = np.nan  # 15% missing

        df = pd.DataFrame({
            "treatment": treatment,
            "age": age,
            "income": income,
            "outcome": outcome,
        })

        profile = self._create_data_profile(df, "treatment", "outcome")

        return EvalCase(
            name="missing_data_pattern",
            description="Data with 15-20% missing values",
            difficulty=EvalDifficulty.EASY,
            category=EvalCategory.TASK_COMPLETION,
            dataframe=df,
            dataset_info={"name": "missing_data"},
            initial_state_overrides={
                "data_profile": profile,
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "quality_score_range": (50, 80),
                "expected_issues": ["missing"],
            },
            expected_tool_calls=["check_missing_patterns"],
            max_expected_steps=10,
        )

    # ================== MEDIUM CASES ==================

    def _case_covariate_imbalance(self) -> EvalCase:
        """Medium: Significant covariate imbalance between groups."""
        np.random.seed(103)
        n = 600

        # Create imbalanced groups
        treatment = np.zeros(n, dtype=int)
        treatment[:200] = 1  # Treated group

        # Covariates different between groups
        age = np.concatenate([
            np.random.normal(55, 8, 200),  # Treated: older
            np.random.normal(35, 8, 400),  # Control: younger
        ])

        income = np.concatenate([
            np.random.normal(70000, 10000, 200),  # Treated: higher income
            np.random.normal(40000, 10000, 400),  # Control: lower income
        ])

        outcome = 100 + 10 * treatment + 0.5 * age + np.random.normal(0, 10, n)

        df = pd.DataFrame({
            "treatment": treatment,
            "age": age,
            "income": income,
            "outcome": outcome,
        })

        profile = self._create_data_profile(df, "treatment", "outcome")

        return EvalCase(
            name="covariate_imbalance",
            description="Significant covariate imbalance between treatment groups",
            difficulty=EvalDifficulty.MEDIUM,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "imbalanced_data"},
            initial_state_overrides={
                "data_profile": profile,
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "quality_score_range": (50, 80),
                "expected_issues": ["imbalance", "balance"],
                "imbalanced_covariates": ["age", "income"],
                "causal_readiness": "needs_attention",
            },
            expected_tool_calls=["check_covariate_balance"],
            max_expected_steps=12,
        )

    def _case_high_multicollinearity(self) -> EvalCase:
        """Medium: High multicollinearity among covariates."""
        np.random.seed(104)
        n = 500

        treatment = np.random.binomial(1, 0.5, n)

        # Create multicollinear features
        x1 = np.random.normal(0, 1, n)
        x2 = x1 * 0.95 + np.random.normal(0, 0.1, n)  # r = 0.95 with x1
        x3 = x1 * 0.9 + x2 * 0.1 + np.random.normal(0, 0.1, n)  # Linear combo
        x4 = np.random.normal(0, 1, n)  # Independent

        outcome = 10 + 2 * treatment + x1 + np.random.normal(0, 1, n)

        df = pd.DataFrame({
            "treatment": treatment,
            "feature_1": x1,
            "feature_2": x2,
            "feature_3": x3,
            "feature_4": x4,
            "outcome": outcome,
        })

        profile = self._create_data_profile(df, "treatment", "outcome")

        return EvalCase(
            name="high_multicollinearity",
            description="High multicollinearity among covariates (VIF > 10)",
            difficulty=EvalDifficulty.MEDIUM,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "multicollinear_data"},
            initial_state_overrides={
                "data_profile": profile,
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "quality_score_range": (50, 80),
                "expected_issues": ["multicollinearity", "correlation", "vif"],
            },
            expected_tool_calls=["compute_vif", "compute_correlations"],
            max_expected_steps=12,
        )

    def _case_skewed_distributions(self) -> EvalCase:
        """Medium: Highly skewed distributions."""
        np.random.seed(105)
        n = 500

        treatment = np.random.binomial(1, 0.5, n)

        # Skewed distributions
        income = np.random.lognormal(10, 1, n)  # Highly right-skewed
        age = np.random.beta(1, 5, n) * 80  # Left-skewed

        outcome = np.random.normal(100, 20, n)

        df = pd.DataFrame({
            "treatment": treatment,
            "income": income,
            "age": age,
            "outcome": outcome,
        })

        profile = self._create_data_profile(df, "treatment", "outcome")

        return EvalCase(
            name="skewed_distributions",
            description="Highly skewed distributions requiring transformation",
            difficulty=EvalDifficulty.MEDIUM,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "skewed_data"},
            initial_state_overrides={
                "data_profile": profile,
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "quality_score_range": (60, 85),
                "expected_issues": ["skew", "distribution", "normality"],
            },
            expected_tool_calls=["analyze_variable"],
            max_expected_steps=12,
        )

    def _case_mixed_quality_issues(self) -> EvalCase:
        """Medium: Multiple simultaneous quality issues."""
        np.random.seed(106)
        n = 500

        treatment = np.random.binomial(1, 0.4, n).astype(float)

        age = np.random.normal(40, 10, n)
        age[np.random.choice(n, 30, replace=False)] = np.nan  # Missing
        age[0:5] = 150  # Outliers

        income = np.random.normal(50000, 15000, n)
        income[np.random.choice(n, 20, replace=False)] = np.nan  # Missing

        # Correlated feature
        related = income * 0.8 + np.random.normal(0, 5000, n)

        outcome = np.random.normal(100, 20, n)

        df = pd.DataFrame({
            "treatment": treatment,
            "age": age,
            "income": income,
            "related_income": related,
            "outcome": outcome,
        })

        profile = self._create_data_profile(df, "treatment", "outcome")

        return EvalCase(
            name="mixed_quality_issues",
            description="Multiple issues: missing data, outliers, and correlation",
            difficulty=EvalDifficulty.MEDIUM,
            category=EvalCategory.TASK_COMPLETION,
            dataframe=df,
            dataset_info={"name": "mixed_issues"},
            initial_state_overrides={
                "data_profile": profile,
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "quality_score_range": (40, 70),
                "expected_issues": ["missing", "outlier"],
                "causal_readiness": "needs_attention",
            },
            max_expected_steps=15,
        )

    # ================== HARD CASES ==================

    def _case_subtle_outliers(self) -> EvalCase:
        """Hard: Subtle outliers that are not immediately obvious."""
        np.random.seed(107)
        n = 1000

        treatment = np.random.binomial(1, 0.5, n)

        # Subtle outliers - values are plausible but affect results
        age = np.random.normal(40, 10, n)
        # Add influential observations (high leverage points)
        age[0:20] = np.random.normal(75, 2, 20)  # Old cluster

        income = np.random.normal(50000, 15000, n)
        # Outliers in 2D space (not univariate outliers)
        income[0:20] = np.random.normal(20000, 1000, 20)  # Low income for old people

        outcome = 100 + 10 * treatment + 0.5 * age + 0.0001 * income + np.random.normal(0, 10, n)

        df = pd.DataFrame({
            "treatment": treatment,
            "age": age,
            "income": income,
            "outcome": outcome,
        })

        profile = self._create_data_profile(df, "treatment", "outcome")

        return EvalCase(
            name="subtle_outliers",
            description="Subtle influential observations (high leverage points)",
            difficulty=EvalDifficulty.HARD,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "subtle_outliers"},
            initial_state_overrides={
                "data_profile": profile,
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "quality_score_range": (60, 85),
                "expected_issues": [],  # May or may not detect
            },
            max_expected_steps=15,
        )

    def _case_confounding_pattern(self) -> EvalCase:
        """Hard: Clear confounding pattern in data."""
        np.random.seed(108)
        n = 800

        # Confounder
        socioeconomic = np.random.normal(0, 1, n)

        # Treatment depends on confounder
        p_treatment = 1 / (1 + np.exp(-socioeconomic))
        treatment = np.random.binomial(1, p_treatment)

        # Outcome depends on both
        outcome = 50 + 5 * treatment + 10 * socioeconomic + np.random.normal(0, 5, n)

        df = pd.DataFrame({
            "treatment": treatment,
            "socioeconomic_index": socioeconomic,
            "outcome": outcome,
        })

        profile = self._create_data_profile(df, "treatment", "outcome")

        return EvalCase(
            name="confounding_pattern",
            description="Clear confounding pattern - treatment correlated with outcome predictor",
            difficulty=EvalDifficulty.HARD,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "confounded_data"},
            initial_state_overrides={
                "data_profile": profile,
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "quality_score_range": (50, 80),
                "expected_issues": ["balance", "confound"],
                "imbalanced_covariates": ["socioeconomic_index"],
            },
            max_expected_steps=15,
        )

    def _case_selection_bias(self) -> EvalCase:
        """Hard: Selection bias pattern."""
        np.random.seed(109)
        n = 600

        # Underlying population
        age_pop = np.random.normal(45, 15, n * 2)
        treatment_pop = np.random.binomial(1, 0.5, n * 2)

        # Selection into sample depends on treatment and age
        selection_prob = 1 / (1 + np.exp(-(treatment_pop * 0.5 + (age_pop - 45) * 0.05)))
        selected = np.random.binomial(1, selection_prob).astype(bool)

        age = age_pop[selected][:n]
        treatment = treatment_pop[selected][:n]
        outcome = 100 + 10 * treatment + np.random.normal(0, 10, n)

        df = pd.DataFrame({
            "treatment": treatment,
            "age": age,
            "outcome": outcome,
        })

        profile = self._create_data_profile(df, "treatment", "outcome")

        return EvalCase(
            name="selection_bias",
            description="Sample shows selection bias (treatment affects selection)",
            difficulty=EvalDifficulty.HARD,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "selection_bias"},
            initial_state_overrides={
                "data_profile": profile,
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "quality_score_range": (60, 90),
                "expected_issues": [],  # Hard to detect
            },
            max_expected_steps=15,
        )

    def _case_complex_interactions(self) -> EvalCase:
        """Hard: Complex interactions between variables."""
        np.random.seed(110)
        n = 800

        treatment = np.random.binomial(1, 0.5, n)
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)

        # Complex interaction
        outcome = (
            50 +
            5 * treatment +
            3 * x1 +
            2 * x2 +
            4 * treatment * x1 +  # Treatment effect modifier
            2 * x1 * x2 +  # Interaction
            np.random.normal(0, 5, n)
        )

        df = pd.DataFrame({
            "treatment": treatment,
            "moderator": x1,
            "covariate": x2,
            "outcome": outcome,
        })

        profile = self._create_data_profile(df, "treatment", "outcome")

        return EvalCase(
            name="complex_interactions",
            description="Complex interactions including treatment effect heterogeneity",
            difficulty=EvalDifficulty.HARD,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "interaction_data"},
            initial_state_overrides={
                "data_profile": profile,
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "quality_score_range": (70, 100),
                "expected_issues": [],
                "causal_readiness": "ready",
            },
            max_expected_steps=15,
        )
