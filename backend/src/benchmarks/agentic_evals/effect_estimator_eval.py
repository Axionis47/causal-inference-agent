"""Agentic evaluation for EffectEstimatorAgent.

Tests the agent's ability to:
1. Select appropriate estimation methods for the data
2. Produce accurate treatment effect estimates
3. Check method assumptions
4. Handle edge cases and method failures
5. Compare multiple methods appropriately
"""

import numpy as np
import pandas as pd

from src.agents.base import DataProfile, EDAResult
from src.agents.specialists.effect_estimator import EffectEstimatorAgent
from src.logging_config.structured import get_logger

from .base import (
    AgenticEvaluator,
    AnalysisState,
    EvalCase,
    EvalCategory,
    EvalDifficulty,
    EvalSuite,
)

logger = get_logger(__name__)


class EffectEstimatorEvaluator(AgenticEvaluator):
    """Evaluator for EffectEstimatorAgent."""

    @property
    def agent_name(self) -> str:
        return "effect_estimator"

    def create_agent(self) -> EffectEstimatorAgent:
        return EffectEstimatorAgent()

    def get_eval_suite(self) -> EvalSuite:
        """Create comprehensive evaluation suite."""
        suite = EvalSuite(
            name="Effect Estimator Evaluation Suite",
            agent_name=self.agent_name,
            description="Tests treatment effect estimation capabilities",
        )

        # EASY cases
        suite.add_case(self._case_randomized_experiment())
        suite.add_case(self._case_large_sample_balanced())
        suite.add_case(self._case_strong_effect())

        # MEDIUM cases
        suite.add_case(self._case_confounded_observational())
        suite.add_case(self._case_small_sample())
        suite.add_case(self._case_weak_overlap())
        suite.add_case(self._case_heterogeneous_effects())

        # HARD cases
        suite.add_case(self._case_severe_imbalance())
        suite.add_case(self._case_nonlinear_confounding())
        suite.add_case(self._case_near_zero_effect())
        suite.add_case(self._case_method_failure_recovery())

        return suite

    def check_task_correctness(
        self,
        case: EvalCase,
        final_state: AnalysisState,
    ) -> tuple[bool, dict[str, float]]:
        """Check if estimation output is correct."""
        metrics = {}
        expected = case.expected_outputs

        effects = final_state.treatment_effects
        if not effects:
            return False, {"effects_estimated": 0.0}

        metrics["effects_estimated"] = 1.0
        metrics["n_methods_used"] = len(effects)

        # Check if any estimate is close to true ATE
        true_ate = expected.get("true_ate")
        if true_ate is not None:
            estimates = [e.estimate for e in effects]
            best_estimate = min(estimates, key=lambda x: abs(x - true_ate))
            mean_estimate = np.mean(estimates)

            bias = mean_estimate - true_ate
            rel_bias = abs(bias) / abs(true_ate) if true_ate != 0 else abs(bias)

            metrics["best_estimate"] = best_estimate
            metrics["mean_estimate"] = mean_estimate
            metrics["bias"] = bias
            metrics["relative_bias"] = rel_bias

            # Check if any CI covers true value
            ci_covers = any(
                e.ci_lower <= true_ate <= e.ci_upper
                for e in effects
            )
            metrics["ci_coverage"] = 1.0 if ci_covers else 0.0

        # Check method selection appropriateness
        expected_methods = expected.get("expected_methods", [])
        if expected_methods:
            used_methods = [e.method.lower() for e in effects]
            overlap = sum(
                1 for em in expected_methods
                if any(em.lower() in um for um in used_methods)
            )
            metrics["method_selection_score"] = overlap / len(expected_methods)
        else:
            metrics["method_selection_score"] = 1.0

        # Check methods to avoid
        avoid_methods = expected.get("avoid_methods", [])
        if avoid_methods:
            used_methods = [e.method.lower() for e in effects]
            avoided = sum(
                1 for am in avoid_methods
                if not any(am.lower() in um for um in used_methods)
            )
            metrics["avoided_bad_methods"] = avoided / len(avoid_methods)
        else:
            metrics["avoided_bad_methods"] = 1.0

        # Determine correctness
        is_correct = (
            metrics.get("effects_estimated", 0) >= 1.0 and
            metrics.get("relative_bias", 1.0) < 0.5 and  # Within 50% of true
            metrics.get("method_selection_score", 0) >= 0.5
        )

        return is_correct, metrics

    def _create_state_components(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
    ) -> tuple[DataProfile, EDAResult]:
        """Create DataProfile and EDAResult for state."""
        feature_types = {}
        for col in df.columns:
            n_unique = df[col].nunique()
            if n_unique == 2:
                feature_types[col] = "binary"
            elif df[col].dtype in ['object', 'category']:
                feature_types[col] = "categorical"
            else:
                feature_types[col] = "numeric"

        profile = DataProfile(
            n_samples=len(df),
            n_features=len(df.columns),
            feature_names=list(df.columns),
            feature_types=feature_types,
            missing_values=dict.fromkeys(df.columns, 0),
            numeric_stats={},
            categorical_stats={},
            treatment_candidates=[treatment_col],
            outcome_candidates=[outcome_col],
            potential_confounders=[c for c in df.columns if c not in [treatment_col, outcome_col]],
        )

        eda_result = EDAResult(
            data_quality_score=80.0,
            data_quality_issues=[],
            key_findings=[],
            recommendations=[],
        )

        return profile, eda_result

    # ================== EASY CASES ==================

    def _case_randomized_experiment(self) -> EvalCase:
        """Easy: Randomized experiment with clear effect."""
        np.random.seed(300)
        n = 1000

        # Randomized treatment
        treatment = np.random.binomial(1, 0.5, n)
        age = np.random.normal(40, 10, n)
        outcome = 100 + 15 * treatment + 0.5 * age + np.random.normal(0, 10, n)

        TRUE_ATE = 15.0

        df = pd.DataFrame({
            "treatment": treatment,
            "age": age,
            "outcome": outcome,
        })

        profile, eda = self._create_state_components(df, "treatment", "outcome")

        return EvalCase(
            name="randomized_experiment",
            description="Randomized experiment with ATE=15",
            difficulty=EvalDifficulty.EASY,
            category=EvalCategory.TASK_COMPLETION,
            dataframe=df,
            dataset_info={"name": "rct", "description": "Randomized controlled trial"},
            initial_state_overrides={
                "data_profile": profile,
                "eda_result": eda,
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "true_ate": TRUE_ATE,
                "expected_methods": ["ols", "regression"],
            },
            max_expected_steps=12,
        )

    def _case_large_sample_balanced(self) -> EvalCase:
        """Easy: Large sample with balanced groups."""
        np.random.seed(301)
        n = 2000

        treatment = np.random.binomial(1, 0.5, n)
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        outcome = 50 + 10 * treatment + 2 * x1 + 3 * x2 + np.random.normal(0, 5, n)

        TRUE_ATE = 10.0

        df = pd.DataFrame({
            "treatment": treatment,
            "x1": x1,
            "x2": x2,
            "outcome": outcome,
        })

        profile, eda = self._create_state_components(df, "treatment", "outcome")

        return EvalCase(
            name="large_sample_balanced",
            description="Large balanced sample with ATE=10",
            difficulty=EvalDifficulty.EASY,
            category=EvalCategory.TASK_COMPLETION,
            dataframe=df,
            dataset_info={"name": "large_balanced"},
            initial_state_overrides={
                "data_profile": profile,
                "eda_result": eda,
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "true_ate": TRUE_ATE,
                "expected_methods": ["ols", "ipw", "aipw"],
            },
            max_expected_steps=12,
        )

    def _case_strong_effect(self) -> EvalCase:
        """Easy: Strong treatment effect easy to detect."""
        np.random.seed(302)
        n = 500

        treatment = np.random.binomial(1, 0.5, n)
        covariate = np.random.normal(0, 1, n)
        outcome = 50 + 50 * treatment + 2 * covariate + np.random.normal(0, 5, n)

        TRUE_ATE = 50.0

        df = pd.DataFrame({
            "treatment": treatment,
            "covariate": covariate,
            "outcome": outcome,
        })

        profile, eda = self._create_state_components(df, "treatment", "outcome")

        return EvalCase(
            name="strong_effect",
            description="Very strong effect (ATE=50) easy to detect",
            difficulty=EvalDifficulty.EASY,
            category=EvalCategory.TASK_COMPLETION,
            dataframe=df,
            dataset_info={"name": "strong_effect"},
            initial_state_overrides={
                "data_profile": profile,
                "eda_result": eda,
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "true_ate": TRUE_ATE,
            },
            max_expected_steps=10,
        )

    # ================== MEDIUM CASES ==================

    def _case_confounded_observational(self) -> EvalCase:
        """Medium: Observational data with confounding."""
        np.random.seed(303)
        n = 800

        # Confounder affects both treatment and outcome
        confounder = np.random.normal(0, 1, n)
        p_treatment = 1 / (1 + np.exp(-confounder))
        treatment = np.random.binomial(1, p_treatment)
        outcome = 50 + 8 * treatment + 5 * confounder + np.random.normal(0, 5, n)

        TRUE_ATE = 8.0  # But naive estimate will be biased

        df = pd.DataFrame({
            "confounder": confounder,
            "treatment": treatment,
            "outcome": outcome,
        })

        profile, eda = self._create_state_components(df, "treatment", "outcome")
        eda.covariate_balance = {
            "confounder": {"smd": 0.8, "is_balanced": False}
        }

        return EvalCase(
            name="confounded_observational",
            description="Confounded observational data requiring adjustment",
            difficulty=EvalDifficulty.MEDIUM,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "confounded"},
            initial_state_overrides={
                "data_profile": profile,
                "eda_result": eda,
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "true_ate": TRUE_ATE,
                "expected_methods": ["ipw", "aipw", "matching", "regression"],
                "avoid_methods": [],  # OLS with adjustment is fine
            },
            max_expected_steps=15,
        )

    def _case_small_sample(self) -> EvalCase:
        """Medium: Small sample requiring careful method selection."""
        np.random.seed(304)
        n = 100

        treatment = np.random.binomial(1, 0.4, n)
        covariate = np.random.normal(0, 1, n)
        outcome = 50 + 12 * treatment + 3 * covariate + np.random.normal(0, 8, n)

        TRUE_ATE = 12.0

        df = pd.DataFrame({
            "treatment": treatment,
            "covariate": covariate,
            "outcome": outcome,
        })

        profile, eda = self._create_state_components(df, "treatment", "outcome")

        return EvalCase(
            name="small_sample",
            description="Small sample (n=100) requiring simpler methods",
            difficulty=EvalDifficulty.MEDIUM,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "small_sample"},
            initial_state_overrides={
                "data_profile": profile,
                "eda_result": eda,
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "true_ate": TRUE_ATE,
                "expected_methods": ["ols", "regression"],
                "avoid_methods": ["causal_forest", "double_ml"],  # Need larger samples
            },
            max_expected_steps=12,
        )

    def _case_weak_overlap(self) -> EvalCase:
        """Medium: Weak propensity score overlap."""
        np.random.seed(305)
        n = 600

        confounder = np.random.normal(0, 1, n)
        # Strong selection -> weak overlap
        p_treatment = 1 / (1 + np.exp(-3 * confounder))
        treatment = np.random.binomial(1, p_treatment)
        outcome = 50 + 10 * treatment + 4 * confounder + np.random.normal(0, 5, n)

        TRUE_ATE = 10.0

        df = pd.DataFrame({
            "confounder": confounder,
            "treatment": treatment,
            "outcome": outcome,
        })

        profile, eda = self._create_state_components(df, "treatment", "outcome")

        return EvalCase(
            name="weak_overlap",
            description="Weak propensity score overlap requiring matching/trimming",
            difficulty=EvalDifficulty.MEDIUM,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "weak_overlap"},
            initial_state_overrides={
                "data_profile": profile,
                "eda_result": eda,
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "true_ate": TRUE_ATE,
                "expected_methods": ["matching", "aipw"],  # Robust to overlap
            },
            max_expected_steps=15,
        )

    def _case_heterogeneous_effects(self) -> EvalCase:
        """Medium: Heterogeneous treatment effects."""
        np.random.seed(306)
        n = 1000

        treatment = np.random.binomial(1, 0.5, n)
        modifier = np.random.normal(0, 1, n)

        # Treatment effect varies with modifier
        individual_effect = 10 + 5 * modifier
        outcome = 50 + treatment * individual_effect + 2 * modifier + np.random.normal(0, 5, n)

        TRUE_ATE = 10.0  # Average effect

        df = pd.DataFrame({
            "treatment": treatment,
            "effect_modifier": modifier,
            "outcome": outcome,
        })

        profile, eda = self._create_state_components(df, "treatment", "outcome")

        return EvalCase(
            name="heterogeneous_effects",
            description="Heterogeneous treatment effects (CATE varies)",
            difficulty=EvalDifficulty.MEDIUM,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "heterogeneous"},
            initial_state_overrides={
                "data_profile": profile,
                "eda_result": eda,
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "true_ate": TRUE_ATE,
                "expected_methods": ["x_learner", "t_learner", "causal_forest"],
            },
            max_expected_steps=15,
        )

    # ================== HARD CASES ==================

    def _case_severe_imbalance(self) -> EvalCase:
        """Hard: Severely imbalanced treatment groups."""
        np.random.seed(307)
        n = 1000

        confounder = np.random.normal(0, 1, n)
        # Very strong selection
        p_treatment = 1 / (1 + np.exp(-5 * confounder))
        treatment = np.random.binomial(1, p_treatment)
        outcome = 50 + 8 * treatment + 6 * confounder + np.random.normal(0, 5, n)

        TRUE_ATE = 8.0

        df = pd.DataFrame({
            "confounder": confounder,
            "treatment": treatment,
            "outcome": outcome,
        })

        profile, eda = self._create_state_components(df, "treatment", "outcome")
        eda.covariate_balance = {
            "confounder": {"smd": 2.0, "is_balanced": False}
        }

        return EvalCase(
            name="severe_imbalance",
            description="Severely imbalanced groups with extreme selection",
            difficulty=EvalDifficulty.HARD,
            category=EvalCategory.ERROR_RECOVERY,
            dataframe=df,
            dataset_info={"name": "severe_imbalance"},
            initial_state_overrides={
                "data_profile": profile,
                "eda_result": eda,
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "true_ate": TRUE_ATE,
                "expected_methods": ["matching", "aipw"],
            },
            max_expected_steps=18,
        )

    def _case_nonlinear_confounding(self) -> EvalCase:
        """Hard: Nonlinear confounding relationship."""
        np.random.seed(308)
        n = 1000

        confounder = np.random.normal(0, 1, n)
        # Nonlinear effect on treatment
        p_treatment = 1 / (1 + np.exp(-(confounder ** 2 - 1)))
        treatment = np.random.binomial(1, p_treatment)
        # Nonlinear effect on outcome
        outcome = 50 + 10 * treatment + 3 * confounder ** 2 + np.random.normal(0, 5, n)

        TRUE_ATE = 10.0

        df = pd.DataFrame({
            "confounder": confounder,
            "treatment": treatment,
            "outcome": outcome,
        })

        profile, eda = self._create_state_components(df, "treatment", "outcome")

        return EvalCase(
            name="nonlinear_confounding",
            description="Nonlinear confounding requiring flexible methods",
            difficulty=EvalDifficulty.HARD,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "nonlinear"},
            initial_state_overrides={
                "data_profile": profile,
                "eda_result": eda,
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "true_ate": TRUE_ATE,
                "expected_methods": ["double_ml", "causal_forest", "aipw"],
            },
            max_expected_steps=18,
        )

    def _case_near_zero_effect(self) -> EvalCase:
        """Hard: Near-zero treatment effect (null result)."""
        np.random.seed(309)
        n = 800

        treatment = np.random.binomial(1, 0.5, n)
        covariate = np.random.normal(0, 1, n)
        outcome = 50 + 0.5 * treatment + 5 * covariate + np.random.normal(0, 10, n)

        TRUE_ATE = 0.5  # Very small effect

        df = pd.DataFrame({
            "treatment": treatment,
            "covariate": covariate,
            "outcome": outcome,
        })

        profile, eda = self._create_state_components(df, "treatment", "outcome")

        return EvalCase(
            name="near_zero_effect",
            description="Near-zero effect that might not be significant",
            difficulty=EvalDifficulty.HARD,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "null_effect"},
            initial_state_overrides={
                "data_profile": profile,
                "eda_result": eda,
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "true_ate": TRUE_ATE,
            },
            max_expected_steps=15,
        )

    def _case_method_failure_recovery(self) -> EvalCase:
        """Hard: Data that causes some methods to fail."""
        np.random.seed(310)
        n = 200

        # Extreme propensity scores
        confounder = np.random.normal(0, 1, n)
        p_treatment = np.clip(1 / (1 + np.exp(-4 * confounder)), 0.01, 0.99)
        treatment = np.random.binomial(1, p_treatment)
        outcome = 50 + 8 * treatment + 3 * confounder + np.random.normal(0, 5, n)

        TRUE_ATE = 8.0

        df = pd.DataFrame({
            "confounder": confounder,
            "treatment": treatment,
            "outcome": outcome,
        })

        profile, eda = self._create_state_components(df, "treatment", "outcome")

        return EvalCase(
            name="method_failure_recovery",
            description="Data causing IPW to have extreme weights",
            difficulty=EvalDifficulty.HARD,
            category=EvalCategory.ERROR_RECOVERY,
            dataframe=df,
            dataset_info={"name": "extreme_weights"},
            initial_state_overrides={
                "data_profile": profile,
                "eda_result": eda,
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "true_ate": TRUE_ATE,
                # Agent should recover when IPW fails
            },
            max_expected_steps=18,
        )
