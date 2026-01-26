"""Agentic evaluation for CritiqueAgent.

Tests the agent's ability to:
1. Identify quality issues in analyses
2. Make appropriate approve/reject/iterate decisions
3. Provide actionable feedback
4. Catch methodological problems
"""

import numpy as np
import pandas as pd

from src.agents.base import (
    DataProfile,
    EDAResult,
    SensitivityResult,
    TreatmentEffectResult,
)
from src.agents.critique.critique_agent import CritiqueAgent
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


class CritiqueAgentEvaluator(AgenticEvaluator):
    """Evaluator for CritiqueAgent."""

    @property
    def agent_name(self) -> str:
        return "critique"

    def create_agent(self) -> CritiqueAgent:
        return CritiqueAgent()

    def get_eval_suite(self) -> EvalSuite:
        """Create comprehensive evaluation suite."""
        suite = EvalSuite(
            name="Critique Agent Evaluation Suite",
            agent_name=self.agent_name,
            description="Tests critique and quality control capabilities",
        )

        # EASY cases
        suite.add_case(self._case_high_quality_analysis())
        suite.add_case(self._case_obvious_problems())

        # MEDIUM cases
        suite.add_case(self._case_moderate_issues())
        suite.add_case(self._case_conflicting_estimates())
        suite.add_case(self._case_missing_sensitivity())

        # HARD cases
        suite.add_case(self._case_subtle_methodology_issue())
        suite.add_case(self._case_borderline_quality())
        suite.add_case(self._case_false_precision())

        return suite

    def check_task_correctness(
        self,
        case: EvalCase,
        final_state: AnalysisState,
    ) -> tuple[bool, dict[str, float]]:
        """Check if critique output is correct."""
        metrics = {}
        expected = case.expected_outputs

        critique = final_state.get_latest_critique() if final_state.critique_history else None
        if critique is None:
            return False, {"critique_completed": 0.0}

        metrics["critique_completed"] = 1.0

        # Check decision correctness
        expected_decision = expected.get("expected_decision")
        if expected_decision:
            decision_correct = critique.decision.value == expected_decision
            metrics["decision_correct"] = 1.0 if decision_correct else 0.0
        else:
            metrics["decision_correct"] = 1.0

        # Check if issues were identified
        expected_issues = expected.get("expected_issues", [])
        if expected_issues:
            critique_text = critique.summary.lower() + " " + " ".join(critique.suggestions).lower()
            issues_found = sum(
                1 for issue in expected_issues
                if issue.lower() in critique_text
            )
            metrics["issue_detection_rate"] = issues_found / len(expected_issues)
        else:
            metrics["issue_detection_rate"] = 1.0

        # Check score ranges
        expected_score_range = expected.get("score_range")
        if expected_score_range and critique.scores:
            avg_score = sum(critique.scores.values()) / len(critique.scores)
            in_range = expected_score_range[0] <= avg_score <= expected_score_range[1]
            metrics["score_in_range"] = 1.0 if in_range else 0.0
        else:
            metrics["score_in_range"] = 1.0

        # Check if suggestions were actionable
        if critique.suggestions:
            metrics["has_suggestions"] = 1.0
            metrics["n_suggestions"] = len(critique.suggestions)
        else:
            metrics["has_suggestions"] = 0.5  # May be OK if approving

        # Determine correctness
        is_correct = (
            metrics.get("critique_completed", 0) >= 1.0 and
            metrics.get("decision_correct", 0) >= 1.0
        )

        return is_correct, metrics

    def _create_full_state(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        effects: list[TreatmentEffectResult],
        eda_issues: list[str] = None,
        sensitivity_results: list[SensitivityResult] = None,
        quality_score: float = 80.0,
    ) -> dict:
        """Create complete state for critique evaluation."""
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
            missing_values=dict.fromkeys(df.columns, 0),
            numeric_stats={},
            categorical_stats={},
            treatment_candidates=[treatment_col],
            outcome_candidates=[outcome_col],
            potential_confounders=[c for c in df.columns if c not in [treatment_col, outcome_col]],
        )

        eda = EDAResult(
            data_quality_score=quality_score,
            data_quality_issues=eda_issues or [],
            key_findings=["Treatment-outcome relationship observed"],
            recommendations=[],
        )

        return {
            "data_profile": profile,
            "eda_result": eda,
            "treatment_variable": treatment_col,
            "outcome_variable": outcome_col,
            "treatment_effects": effects,
            "sensitivity_results": sensitivity_results or [],
        }

    # ================== EASY CASES ==================

    def _case_high_quality_analysis(self) -> EvalCase:
        """Easy: High quality analysis that should be approved."""
        np.random.seed(500)
        n = 1000

        treatment = np.random.binomial(1, 0.5, n)
        covariate = np.random.normal(0, 1, n)
        outcome = 50 + 10 * treatment + 2 * covariate + np.random.normal(0, 5, n)

        df = pd.DataFrame({
            "treatment": treatment,
            "covariate": covariate,
            "outcome": outcome,
        })

        effects = [
            TreatmentEffectResult(
                method="OLS Regression",
                estimand="ATE",
                estimate=10.0,
                std_error=0.3,
                ci_lower=9.4,
                ci_upper=10.6,
                p_value=0.0001,
            ),
            TreatmentEffectResult(
                method="AIPW",
                estimand="ATE",
                estimate=10.1,
                std_error=0.3,
                ci_lower=9.5,
                ci_upper=10.7,
                p_value=0.0001,
            ),
        ]

        sensitivity = [
            SensitivityResult(
                method="E-value",
                result={"e_value": 5.2, "e_value_ci": 4.8},
                interpretation="Effect is robust to moderate unmeasured confounding",
            )
        ]

        state_overrides = self._create_full_state(
            df, "treatment", "outcome", effects,
            eda_issues=[],
            sensitivity_results=sensitivity,
            quality_score=90.0,
        )

        return EvalCase(
            name="high_quality_analysis",
            description="Well-executed analysis that should be approved",
            difficulty=EvalDifficulty.EASY,
            category=EvalCategory.TASK_COMPLETION,
            dataframe=df,
            dataset_info={"name": "high_quality"},
            initial_state_overrides=state_overrides,
            expected_outputs={
                "expected_decision": "approve",
                "score_range": (3.5, 5.0),
            },
            max_expected_steps=12,
        )

    def _case_obvious_problems(self) -> EvalCase:
        """Easy: Obvious problems that should be rejected/iterated."""
        np.random.seed(501)
        n = 100

        treatment = np.random.binomial(1, 0.1, n)  # Very imbalanced
        covariate = np.random.normal(0, 1, n)
        outcome = 50 + 5 * treatment + 10 * covariate + np.random.normal(0, 20, n)

        df = pd.DataFrame({
            "treatment": treatment,
            "covariate": covariate,
            "outcome": outcome,
        })

        effects = [
            TreatmentEffectResult(
                method="OLS (unadjusted)",
                estimand="ATE",
                estimate=15.0,  # Likely biased
                std_error=5.0,
                ci_lower=5.0,
                ci_upper=25.0,
                p_value=0.05,
            ),
        ]

        state_overrides = self._create_full_state(
            df, "treatment", "outcome", effects,
            eda_issues=["Severe covariate imbalance", "Small sample size"],
            sensitivity_results=[],
            quality_score=45.0,
        )

        return EvalCase(
            name="obvious_problems",
            description="Analysis with obvious problems requiring iteration",
            difficulty=EvalDifficulty.EASY,
            category=EvalCategory.TASK_COMPLETION,
            dataframe=df,
            dataset_info={"name": "obvious_problems"},
            initial_state_overrides=state_overrides,
            expected_outputs={
                "expected_decision": "iterate",
                "expected_issues": ["imbalance", "sample"],
                "score_range": (1.0, 3.0),
            },
            max_expected_steps=12,
        )

    # ================== MEDIUM CASES ==================

    def _case_moderate_issues(self) -> EvalCase:
        """Medium: Moderate issues that may need iteration."""
        np.random.seed(502)
        n = 500

        confounder = np.random.normal(0, 1, n)
        p_treat = 1 / (1 + np.exp(-confounder))
        treatment = np.random.binomial(1, p_treat)
        outcome = 50 + 8 * treatment + 4 * confounder + np.random.normal(0, 8, n)

        df = pd.DataFrame({
            "confounder": confounder,
            "treatment": treatment,
            "outcome": outcome,
        })

        effects = [
            TreatmentEffectResult(
                method="OLS Regression",
                estimand="ATE",
                estimate=8.5,
                std_error=0.8,
                ci_lower=6.9,
                ci_upper=10.1,
                p_value=0.001,
            ),
        ]

        state_overrides = self._create_full_state(
            df, "treatment", "outcome", effects,
            eda_issues=["Some covariate imbalance detected"],
            sensitivity_results=[],
            quality_score=65.0,
        )

        return EvalCase(
            name="moderate_issues",
            description="Analysis with moderate issues",
            difficulty=EvalDifficulty.MEDIUM,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "moderate_issues"},
            initial_state_overrides=state_overrides,
            expected_outputs={
                "expected_decision": "iterate",  # Should suggest improvements
                "expected_issues": ["sensitivity", "method"],
            },
            max_expected_steps=15,
        )

    def _case_conflicting_estimates(self) -> EvalCase:
        """Medium: Conflicting estimates from different methods."""
        np.random.seed(503)
        n = 600

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
                estimate=14.0,  # Biased
                std_error=0.6,
                ci_lower=12.8,
                ci_upper=15.2,
                p_value=0.0001,
            ),
            TreatmentEffectResult(
                method="OLS (adjusted)",
                estimand="ATE",
                estimate=8.0,  # Correct
                std_error=0.5,
                ci_lower=7.0,
                ci_upper=9.0,
                p_value=0.0001,
            ),
            TreatmentEffectResult(
                method="IPW",
                estimand="ATE",
                estimate=7.5,
                std_error=0.7,
                ci_lower=6.1,
                ci_upper=8.9,
                p_value=0.0001,
            ),
        ]

        state_overrides = self._create_full_state(
            df, "treatment", "outcome", effects,
            eda_issues=["Covariate imbalance between groups"],
            quality_score=70.0,
        )

        return EvalCase(
            name="conflicting_estimates",
            description="Conflicting estimates suggesting confounding",
            difficulty=EvalDifficulty.MEDIUM,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "conflicting"},
            initial_state_overrides=state_overrides,
            expected_outputs={
                "expected_decision": "iterate",
                "expected_issues": ["conflict", "confound"],
            },
            max_expected_steps=15,
        )

    def _case_missing_sensitivity(self) -> EvalCase:
        """Medium: Good analysis but missing sensitivity analysis."""
        np.random.seed(504)
        n = 800

        treatment = np.random.binomial(1, 0.5, n)
        covariate = np.random.normal(0, 1, n)
        outcome = 50 + 10 * treatment + 2 * covariate + np.random.normal(0, 5, n)

        df = pd.DataFrame({
            "treatment": treatment,
            "covariate": covariate,
            "outcome": outcome,
        })

        effects = [
            TreatmentEffectResult(
                method="OLS Regression",
                estimand="ATE",
                estimate=10.0,
                std_error=0.4,
                ci_lower=9.2,
                ci_upper=10.8,
                p_value=0.0001,
            ),
            TreatmentEffectResult(
                method="AIPW",
                estimand="ATE",
                estimate=10.1,
                std_error=0.4,
                ci_lower=9.3,
                ci_upper=10.9,
                p_value=0.0001,
            ),
        ]

        state_overrides = self._create_full_state(
            df, "treatment", "outcome", effects,
            eda_issues=[],
            sensitivity_results=[],  # Missing!
            quality_score=80.0,
        )

        return EvalCase(
            name="missing_sensitivity",
            description="Good estimates but no sensitivity analysis",
            difficulty=EvalDifficulty.MEDIUM,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "missing_sensitivity"},
            initial_state_overrides=state_overrides,
            expected_outputs={
                "expected_decision": "iterate",
                "expected_issues": ["sensitivity"],
            },
            max_expected_steps=12,
        )

    # ================== HARD CASES ==================

    def _case_subtle_methodology_issue(self) -> EvalCase:
        """Hard: Subtle methodological issue."""
        np.random.seed(505)
        n = 500

        # Collider structure (shouldn't condition on collider)
        treatment = np.random.binomial(1, 0.5, n)
        outcome = 50 + 10 * treatment + np.random.normal(0, 5, n)
        collider = treatment + outcome / 10 + np.random.normal(0, 1, n)

        df = pd.DataFrame({
            "treatment": treatment,
            "collider": collider,  # Wrongly included as confounder
            "outcome": outcome,
        })

        effects = [
            TreatmentEffectResult(
                method="OLS (adjusted for collider)",  # Wrong!
                estimand="ATE",
                estimate=8.0,  # Biased
                std_error=0.5,
                ci_lower=7.0,
                ci_upper=9.0,
                p_value=0.0001,
            ),
        ]

        state_overrides = self._create_full_state(
            df, "treatment", "outcome", effects,
            eda_issues=[],
            quality_score=75.0,
        )

        return EvalCase(
            name="subtle_methodology_issue",
            description="Adjusted for collider (methodological error)",
            difficulty=EvalDifficulty.HARD,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "collider_bias"},
            initial_state_overrides=state_overrides,
            expected_outputs={
                "expected_decision": "iterate",
                "expected_issues": ["collider", "bias", "methodology"],
            },
            max_expected_steps=18,
        )

    def _case_borderline_quality(self) -> EvalCase:
        """Hard: Borderline quality that requires careful judgment."""
        np.random.seed(506)
        n = 400

        confounder = np.random.normal(0, 1, n)
        p_treat = 1 / (1 + np.exp(-0.8 * confounder))
        treatment = np.random.binomial(1, p_treat)
        outcome = 50 + 6 * treatment + 3 * confounder + np.random.normal(0, 7, n)

        df = pd.DataFrame({
            "confounder": confounder,
            "treatment": treatment,
            "outcome": outcome,
        })

        effects = [
            TreatmentEffectResult(
                method="OLS Regression",
                estimand="ATE",
                estimate=6.2,
                std_error=0.7,
                ci_lower=4.8,
                ci_upper=7.6,
                p_value=0.001,
            ),
            TreatmentEffectResult(
                method="IPW",
                estimand="ATE",
                estimate=5.8,
                std_error=0.8,
                ci_lower=4.2,
                ci_upper=7.4,
                p_value=0.001,
            ),
        ]

        sensitivity = [
            SensitivityResult(
                method="E-value",
                result={"e_value": 2.1, "e_value_ci": 1.6},
                interpretation="Moderate sensitivity to unmeasured confounding",
            )
        ]

        state_overrides = self._create_full_state(
            df, "treatment", "outcome", effects,
            eda_issues=["Minor covariate imbalance"],
            sensitivity_results=sensitivity,
            quality_score=70.0,
        )

        return EvalCase(
            name="borderline_quality",
            description="Borderline quality requiring careful judgment",
            difficulty=EvalDifficulty.HARD,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "borderline"},
            initial_state_overrides=state_overrides,
            expected_outputs={
                # Could be approve or iterate depending on interpretation
                "score_range": (2.5, 4.0),
            },
            max_expected_steps=18,
        )

    def _case_false_precision(self) -> EvalCase:
        """Hard: Overly precise estimate from problematic analysis."""
        np.random.seed(507)
        n = 2000

        # Selection bias
        confounder = np.random.normal(0, 1, n)
        p_treat = 1 / (1 + np.exp(-3 * confounder))
        treatment = np.random.binomial(1, p_treat)

        # Hidden confounder
        hidden = np.random.normal(0, 1, n)
        outcome = 50 + 5 * treatment + 4 * confounder + 3 * hidden + np.random.normal(0, 3, n)

        df = pd.DataFrame({
            "confounder": confounder,
            "treatment": treatment,
            "outcome": outcome,
        })

        effects = [
            TreatmentEffectResult(
                method="OLS Regression",
                estimand="ATE",
                estimate=7.5,  # Biased but precise
                std_error=0.15,  # Very small SE
                ci_lower=7.2,
                ci_upper=7.8,
                p_value=0.0001,
            ),
        ]

        state_overrides = self._create_full_state(
            df, "treatment", "outcome", effects,
            eda_issues=["Significant covariate imbalance"],
            quality_score=60.0,
        )

        return EvalCase(
            name="false_precision",
            description="Overly precise estimate from biased analysis",
            difficulty=EvalDifficulty.HARD,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "false_precision"},
            initial_state_overrides=state_overrides,
            expected_outputs={
                "expected_decision": "iterate",
                "expected_issues": ["bias", "precision", "confound"],
            },
            max_expected_steps=18,
        )
