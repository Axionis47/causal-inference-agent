"""Agentic evaluation for SensitivityAnalystAgent.

Tests the agent's ability to:
1. Perform appropriate robustness checks
2. Correctly interpret sensitivity results
3. Identify when estimates are fragile
4. Handle different estimation scenarios
"""

import numpy as np
import pandas as pd

from src.agents.base import DataProfile, EDAResult, TreatmentEffectResult
from src.agents.specialists.sensitivity_analyst import SensitivityAnalystAgent
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


class SensitivityAnalystEvaluator(AgenticEvaluator):
    """Evaluator for SensitivityAnalystAgent."""

    @property
    def agent_name(self) -> str:
        return "sensitivity_analyst"

    def create_agent(self) -> SensitivityAnalystAgent:
        return SensitivityAnalystAgent()

    def get_eval_suite(self) -> EvalSuite:
        """Create comprehensive evaluation suite."""
        suite = EvalSuite(
            name="Sensitivity Analyst Evaluation Suite",
            agent_name=self.agent_name,
            description="Tests sensitivity analysis capabilities",
        )

        # EASY cases
        suite.add_case(self._case_robust_estimate())
        suite.add_case(self._case_clear_fragility())

        # MEDIUM cases
        suite.add_case(self._case_moderate_sensitivity())
        suite.add_case(self._case_matching_based_estimate())
        suite.add_case(self._case_multiple_methods())

        # HARD cases
        suite.add_case(self._case_borderline_sensitivity())
        suite.add_case(self._case_conflicting_methods())
        suite.add_case(self._case_hidden_bias_concern())

        return suite

    def check_task_correctness(
        self,
        case: EvalCase,
        final_state: AnalysisState,
    ) -> tuple[bool, dict[str, float]]:
        """Check if sensitivity analysis is correct."""
        metrics = {}
        expected = case.expected_outputs

        sens_results = final_state.sensitivity_results
        if not sens_results:
            return False, {"analysis_completed": 0.0}

        metrics["analysis_completed"] = 1.0
        metrics["n_analyses"] = len(sens_results)

        # Check robustness assessment
        expected_robustness = expected.get("robustness")  # "robust", "fragile", "moderate"
        if expected_robustness:
            # Infer robustness from interpretations
            all_interps = " ".join([r.interpretation.lower() for r in sens_results])

            if expected_robustness == "robust":
                is_correct = "robust" in all_interps or "insensitive" in all_interps
            elif expected_robustness == "fragile":
                is_correct = "fragile" in all_interps or "sensitive" in all_interps
            else:  # moderate
                is_correct = True  # Hard to judge

            metrics["robustness_assessment_correct"] = 1.0 if is_correct else 0.0
        else:
            metrics["robustness_assessment_correct"] = 1.0

        # Check e-value computation
        expected_min_evalue = expected.get("min_e_value")
        if expected_min_evalue is not None:
            e_value_results = [
                r for r in sens_results
                if "e-value" in r.method.lower() or "e_value" in r.method.lower()
            ]
            if e_value_results:
                # Check if computed e-value is reasonable
                metrics["e_value_computed"] = 1.0
            else:
                metrics["e_value_computed"] = 0.0
        else:
            metrics["e_value_computed"] = 1.0

        # Check if correct methods used
        expected_methods = expected.get("expected_methods", [])
        if expected_methods:
            used_methods = [r.method.lower() for r in sens_results]
            overlap = sum(
                1 for em in expected_methods
                if any(em.lower() in um for um in used_methods)
            )
            metrics["method_selection"] = overlap / len(expected_methods)
        else:
            metrics["method_selection"] = 1.0

        # Determine correctness
        is_correct = (
            metrics.get("analysis_completed", 0) >= 1.0 and
            metrics.get("robustness_assessment_correct", 0) >= 0.5
        )

        return is_correct, metrics

    def _create_state_with_effects(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        effects: list[TreatmentEffectResult],
    ) -> dict:
        """Create state components including treatment effects."""
        feature_types = {}
        for col in df.columns:
            n_unique = df[col].nunique()
            if n_unique == 2:
                feature_types[col] = "binary"
            else:
                feature_types[col] = "numeric"

        profile = DataProfile(
            n_samples=len(df),
            n_features=len(df.columns),
            feature_names=list(df.columns),
            feature_types=feature_types,
            missing_values={col: 0 for col in df.columns},
            numeric_stats={},
            categorical_stats={},
            treatment_candidates=[treatment_col],
            outcome_candidates=[outcome_col],
            potential_confounders=[c for c in df.columns if c not in [treatment_col, outcome_col]],
        )

        eda = EDAResult(
            data_quality_score=80.0,
            data_quality_issues=[],
            key_findings=[],
            recommendations=[],
        )

        return {
            "data_profile": profile,
            "eda_result": eda,
            "treatment_variable": treatment_col,
            "outcome_variable": outcome_col,
            "treatment_effects": effects,
        }

    # ================== EASY CASES ==================

    def _case_robust_estimate(self) -> EvalCase:
        """Easy: Robust estimate with strong effect."""
        np.random.seed(400)
        n = 1000

        treatment = np.random.binomial(1, 0.5, n)
        covariate = np.random.normal(0, 1, n)
        outcome = 50 + 20 * treatment + 2 * covariate + np.random.normal(0, 5, n)

        df = pd.DataFrame({
            "treatment": treatment,
            "covariate": covariate,
            "outcome": outcome,
        })

        effects = [
            TreatmentEffectResult(
                method="OLS Regression",
                estimand="ATE",
                estimate=20.0,
                std_error=0.5,
                ci_lower=19.0,
                ci_upper=21.0,
                p_value=0.0001,
            )
        ]

        state_overrides = self._create_state_with_effects(df, "treatment", "outcome", effects)

        return EvalCase(
            name="robust_estimate",
            description="Strong effect that should be robust to confounding",
            difficulty=EvalDifficulty.EASY,
            category=EvalCategory.TASK_COMPLETION,
            dataframe=df,
            dataset_info={"name": "robust"},
            initial_state_overrides=state_overrides,
            expected_outputs={
                "robustness": "robust",
                "expected_methods": ["e-value", "e_value"],
            },
            max_expected_steps=10,
        )

    def _case_clear_fragility(self) -> EvalCase:
        """Easy: Clearly fragile estimate."""
        np.random.seed(401)
        n = 500

        treatment = np.random.binomial(1, 0.5, n)
        covariate = np.random.normal(0, 1, n)
        outcome = 50 + 1.5 * treatment + 5 * covariate + np.random.normal(0, 10, n)

        df = pd.DataFrame({
            "treatment": treatment,
            "covariate": covariate,
            "outcome": outcome,
        })

        effects = [
            TreatmentEffectResult(
                method="OLS Regression",
                estimand="ATE",
                estimate=1.5,
                std_error=1.2,
                ci_lower=-0.8,
                ci_upper=3.8,
                p_value=0.15,
            )
        ]

        state_overrides = self._create_state_with_effects(df, "treatment", "outcome", effects)

        return EvalCase(
            name="clear_fragility",
            description="Weak, non-significant effect that is fragile",
            difficulty=EvalDifficulty.EASY,
            category=EvalCategory.TASK_COMPLETION,
            dataframe=df,
            dataset_info={"name": "fragile"},
            initial_state_overrides=state_overrides,
            expected_outputs={
                "robustness": "fragile",
            },
            max_expected_steps=10,
        )

    # ================== MEDIUM CASES ==================

    def _case_moderate_sensitivity(self) -> EvalCase:
        """Medium: Moderate sensitivity to unmeasured confounding."""
        np.random.seed(402)
        n = 800

        treatment = np.random.binomial(1, 0.5, n)
        covariate = np.random.normal(0, 1, n)
        outcome = 50 + 5 * treatment + 3 * covariate + np.random.normal(0, 5, n)

        df = pd.DataFrame({
            "treatment": treatment,
            "covariate": covariate,
            "outcome": outcome,
        })

        effects = [
            TreatmentEffectResult(
                method="OLS Regression",
                estimand="ATE",
                estimate=5.0,
                std_error=0.6,
                ci_lower=3.8,
                ci_upper=6.2,
                p_value=0.001,
            )
        ]

        state_overrides = self._create_state_with_effects(df, "treatment", "outcome", effects)

        return EvalCase(
            name="moderate_sensitivity",
            description="Moderate effect with some sensitivity to confounding",
            difficulty=EvalDifficulty.MEDIUM,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "moderate"},
            initial_state_overrides=state_overrides,
            expected_outputs={
                "robustness": "moderate",
                "expected_methods": ["e-value", "rosenbaum"],
            },
            max_expected_steps=12,
        )

    def _case_matching_based_estimate(self) -> EvalCase:
        """Medium: Matching-based estimate requiring Rosenbaum bounds."""
        np.random.seed(403)
        n = 600

        confounder = np.random.normal(0, 1, n)
        p_treat = 1 / (1 + np.exp(-confounder))
        treatment = np.random.binomial(1, p_treat)
        outcome = 50 + 8 * treatment + 4 * confounder + np.random.normal(0, 5, n)

        df = pd.DataFrame({
            "confounder": confounder,
            "treatment": treatment,
            "outcome": outcome,
        })

        effects = [
            TreatmentEffectResult(
                method="Propensity Score Matching",
                estimand="ATT",
                estimate=8.0,
                std_error=0.8,
                ci_lower=6.4,
                ci_upper=9.6,
                p_value=0.001,
            )
        ]

        state_overrides = self._create_state_with_effects(df, "treatment", "outcome", effects)

        return EvalCase(
            name="matching_based_estimate",
            description="PSM estimate appropriate for Rosenbaum bounds",
            difficulty=EvalDifficulty.MEDIUM,
            category=EvalCategory.TOOL_SELECTION,
            dataframe=df,
            dataset_info={"name": "matching"},
            initial_state_overrides=state_overrides,
            expected_outputs={
                "robustness": "moderate",
                "expected_methods": ["rosenbaum"],
            },
            max_expected_steps=12,
        )

    def _case_multiple_methods(self) -> EvalCase:
        """Medium: Multiple estimation methods to analyze."""
        np.random.seed(404)
        n = 800

        treatment = np.random.binomial(1, 0.5, n)
        covariate = np.random.normal(0, 1, n)
        outcome = 50 + 10 * treatment + 3 * covariate + np.random.normal(0, 5, n)

        df = pd.DataFrame({
            "treatment": treatment,
            "covariate": covariate,
            "outcome": outcome,
        })

        effects = [
            TreatmentEffectResult(
                method="OLS Regression",
                estimand="ATE",
                estimate=10.2,
                std_error=0.5,
                ci_lower=9.2,
                ci_upper=11.2,
                p_value=0.0001,
            ),
            TreatmentEffectResult(
                method="IPW",
                estimand="ATE",
                estimate=9.8,
                std_error=0.6,
                ci_lower=8.6,
                ci_upper=11.0,
                p_value=0.0001,
            ),
            TreatmentEffectResult(
                method="AIPW",
                estimand="ATE",
                estimate=10.0,
                std_error=0.5,
                ci_lower=9.0,
                ci_upper=11.0,
                p_value=0.0001,
            ),
        ]

        state_overrides = self._create_state_with_effects(df, "treatment", "outcome", effects)

        return EvalCase(
            name="multiple_methods",
            description="Multiple consistent estimates to analyze",
            difficulty=EvalDifficulty.MEDIUM,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "multiple_methods"},
            initial_state_overrides=state_overrides,
            expected_outputs={
                "robustness": "robust",
            },
            max_expected_steps=15,
        )

    # ================== HARD CASES ==================

    def _case_borderline_sensitivity(self) -> EvalCase:
        """Hard: Borderline sensitivity that requires careful interpretation."""
        np.random.seed(405)
        n = 600

        treatment = np.random.binomial(1, 0.5, n)
        covariate = np.random.normal(0, 1, n)
        outcome = 50 + 3 * treatment + 4 * covariate + np.random.normal(0, 6, n)

        df = pd.DataFrame({
            "treatment": treatment,
            "covariate": covariate,
            "outcome": outcome,
        })

        effects = [
            TreatmentEffectResult(
                method="OLS Regression",
                estimand="ATE",
                estimate=3.0,
                std_error=0.8,
                ci_lower=1.4,
                ci_upper=4.6,
                p_value=0.01,
            )
        ]

        state_overrides = self._create_state_with_effects(df, "treatment", "outcome", effects)

        return EvalCase(
            name="borderline_sensitivity",
            description="Borderline significant effect with moderate e-value",
            difficulty=EvalDifficulty.HARD,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "borderline"},
            initial_state_overrides=state_overrides,
            expected_outputs={
                "robustness": "moderate",
            },
            max_expected_steps=15,
        )

    def _case_conflicting_methods(self) -> EvalCase:
        """Hard: Conflicting results from different methods."""
        np.random.seed(406)
        n = 500

        confounder = np.random.normal(0, 1, n)
        p_treat = 1 / (1 + np.exp(-2 * confounder))
        treatment = np.random.binomial(1, p_treat)
        outcome = 50 + 8 * treatment + 6 * confounder + np.random.normal(0, 5, n)

        df = pd.DataFrame({
            "confounder": confounder,
            "treatment": treatment,
            "outcome": outcome,
        })

        effects = [
            TreatmentEffectResult(
                method="OLS (unadjusted)",
                estimand="ATE",
                estimate=12.0,  # Biased
                std_error=0.7,
                ci_lower=10.6,
                ci_upper=13.4,
                p_value=0.0001,
            ),
            TreatmentEffectResult(
                method="OLS (adjusted)",
                estimand="ATE",
                estimate=8.0,  # Correct
                std_error=0.6,
                ci_lower=6.8,
                ci_upper=9.2,
                p_value=0.0001,
            ),
            TreatmentEffectResult(
                method="IPW",
                estimand="ATE",
                estimate=7.5,  # Close to correct
                std_error=0.8,
                ci_lower=5.9,
                ci_upper=9.1,
                p_value=0.0001,
            ),
        ]

        state_overrides = self._create_state_with_effects(df, "treatment", "outcome", effects)

        return EvalCase(
            name="conflicting_methods",
            description="Conflicting estimates suggesting confounding",
            difficulty=EvalDifficulty.HARD,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "conflicting"},
            initial_state_overrides=state_overrides,
            expected_outputs={
                "robustness": "fragile",  # Due to sensitivity to adjustment
            },
            max_expected_steps=15,
        )

    def _case_hidden_bias_concern(self) -> EvalCase:
        """Hard: Strong concern about hidden bias."""
        np.random.seed(407)
        n = 800

        # Observed confounder
        observed = np.random.normal(0, 1, n)
        # Hidden confounder (not in data)
        hidden = np.random.normal(0, 1, n)

        p_treat = 1 / (1 + np.exp(-(observed + hidden)))
        treatment = np.random.binomial(1, p_treat)
        outcome = 50 + 5 * treatment + 3 * observed + 4 * hidden + np.random.normal(0, 5, n)

        # Only include observed confounder
        df = pd.DataFrame({
            "observed_confounder": observed,
            "treatment": treatment,
            "outcome": outcome,
        })

        effects = [
            TreatmentEffectResult(
                method="OLS (partial adjustment)",
                estimand="ATE",
                estimate=7.0,  # Biased due to hidden confounder
                std_error=0.5,
                ci_lower=6.0,
                ci_upper=8.0,
                p_value=0.0001,
            )
        ]

        state_overrides = self._create_state_with_effects(df, "treatment", "outcome", effects)

        return EvalCase(
            name="hidden_bias_concern",
            description="Potential hidden bias from unmeasured confounding",
            difficulty=EvalDifficulty.HARD,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "hidden_bias"},
            initial_state_overrides=state_overrides,
            expected_outputs={
                "robustness": "moderate",
                "expected_methods": ["e-value"],
            },
            max_expected_steps=15,
        )
