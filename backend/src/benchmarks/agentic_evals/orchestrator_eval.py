"""Agentic evaluation for ReActOrchestrator.

Tests the orchestrator's ability to:
1. Coordinate agents effectively
2. Handle failures gracefully
3. Know when to iterate based on critique
4. Complete full analysis pipelines
5. Make appropriate workflow decisions
"""

import numpy as np
import pandas as pd

from src.agents.base import AnalysisState, DatasetInfo, JobStatus
from src.agents.orchestrator.react_orchestrator import ReActOrchestrator
from src.logging_config.structured import get_logger

from .base import (
    AgenticEvaluator,
    EvalCase,
    EvalCategory,
    EvalDifficulty,
    EvalSuite,
)

logger = get_logger(__name__)


class OrchestratorEvaluator(AgenticEvaluator):
    """Evaluator for ReActOrchestrator."""

    @property
    def agent_name(self) -> str:
        return "orchestrator"

    def create_agent(self) -> ReActOrchestrator:
        return ReActOrchestrator()

    def get_eval_suite(self) -> EvalSuite:
        """Create comprehensive evaluation suite."""
        suite = EvalSuite(
            name="Orchestrator Evaluation Suite",
            agent_name=self.agent_name,
            description="Tests pipeline coordination capabilities",
        )

        # EASY cases
        suite.add_case(self._case_simple_pipeline())
        suite.add_case(self._case_clean_data_flow())

        # MEDIUM cases
        suite.add_case(self._case_data_quality_issues())
        suite.add_case(self._case_iteration_required())
        suite.add_case(self._case_method_selection_challenge())

        # HARD cases
        suite.add_case(self._case_complex_coordination())
        suite.add_case(self._case_failure_recovery())
        suite.add_case(self._case_early_termination())

        return suite

    def check_task_correctness(
        self,
        case: EvalCase,
        final_state: AnalysisState,
    ) -> tuple[bool, dict[str, float]]:
        """Check if orchestration is correct."""
        metrics = {}
        expected = case.expected_outputs

        # Check completion status
        metrics["completed"] = 1.0 if final_state.status == JobStatus.COMPLETED else 0.0

        # Check if required stages were reached
        required_stages = expected.get("required_stages", [])
        stages_completed = []

        if final_state.data_profile:
            stages_completed.append("profiling")
        if final_state.eda_result:
            stages_completed.append("eda")
        if final_state.proposed_dag:
            stages_completed.append("discovery")
        if final_state.treatment_effects:
            stages_completed.append("estimation")
        if final_state.sensitivity_results:
            stages_completed.append("sensitivity")
        if final_state.critique_history:
            stages_completed.append("critique")
        if final_state.notebook_path:
            stages_completed.append("notebook")

        if required_stages:
            completed = sum(1 for s in required_stages if s in stages_completed)
            metrics["stages_completed"] = completed / len(required_stages)
        else:
            metrics["stages_completed"] = 1.0

        metrics["n_stages"] = len(stages_completed)

        # Check if treatment effects were estimated
        if final_state.treatment_effects:
            metrics["effects_estimated"] = 1.0
            metrics["n_methods"] = len(final_state.treatment_effects)
        else:
            metrics["effects_estimated"] = 0.0
            metrics["n_methods"] = 0

        # Check expected behavior
        expected_behavior = expected.get("expected_behavior")
        if expected_behavior == "iterate":
            did_iterate = final_state.iteration_count > 0
            metrics["iterated_as_expected"] = 1.0 if did_iterate else 0.0
        elif expected_behavior == "no_iterate":
            did_not_iterate = final_state.iteration_count == 0
            metrics["iterated_as_expected"] = 1.0 if did_not_iterate else 0.0
        else:
            metrics["iterated_as_expected"] = 1.0

        # Check final decision if expected
        expected_decision = expected.get("expected_final_decision")
        if expected_decision and final_state.critique_history:
            latest = final_state.get_latest_critique()
            if latest:
                decision_match = latest.decision.value == expected_decision
                metrics["final_decision_correct"] = 1.0 if decision_match else 0.0

        # Determine correctness
        min_stages = expected.get("min_stages", 3)
        is_correct = (
            metrics.get("completed", 0) >= 1.0 or
            (metrics.get("n_stages", 0) >= min_stages and metrics.get("effects_estimated", 0) >= 1.0)
        )

        return is_correct, metrics

    # ================== EASY CASES ==================

    def _case_simple_pipeline(self) -> EvalCase:
        """Easy: Simple, clean data for straightforward pipeline."""
        np.random.seed(600)
        n = 500

        treatment = np.random.binomial(1, 0.5, n)
        age = np.random.normal(40, 10, n)
        outcome = 100 + 15 * treatment + 0.5 * age + np.random.normal(0, 10, n)

        df = pd.DataFrame({
            "treatment": treatment,
            "age": age,
            "outcome": outcome,
        })

        return EvalCase(
            name="simple_pipeline",
            description="Simple pipeline with clean data",
            difficulty=EvalDifficulty.EASY,
            category=EvalCategory.TASK_COMPLETION,
            dataframe=df,
            dataset_info={
                "name": "simple_study",
                "description": "Clean RCT-like data",
            },
            initial_state_overrides={
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "required_stages": ["profiling", "eda", "estimation"],
                "min_stages": 3,
                "expected_behavior": "no_iterate",
            },
            max_expected_steps=25,
            timeout_seconds=180,
        )

    def _case_clean_data_flow(self) -> EvalCase:
        """Easy: Clean data with clear treatment effect."""
        np.random.seed(601)
        n = 800

        treatment = np.random.binomial(1, 0.5, n)
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        outcome = 50 + 12 * treatment + 3 * x1 + 2 * x2 + np.random.normal(0, 5, n)

        df = pd.DataFrame({
            "intervention": treatment,
            "covariate_1": x1,
            "covariate_2": x2,
            "response": outcome,
        })

        return EvalCase(
            name="clean_data_flow",
            description="Clean data with obvious variable names",
            difficulty=EvalDifficulty.EASY,
            category=EvalCategory.TASK_COMPLETION,
            dataframe=df,
            dataset_info={
                "name": "clean_study",
            },
            initial_state_overrides={
                "treatment_variable": "intervention",
                "outcome_variable": "response",
            },
            expected_outputs={
                "required_stages": ["profiling", "estimation"],
                "min_stages": 2,
            },
            max_expected_steps=25,
            timeout_seconds=180,
        )

    # ================== MEDIUM CASES ==================

    def _case_data_quality_issues(self) -> EvalCase:
        """Medium: Data with quality issues requiring EDA attention."""
        np.random.seed(602)
        n = 500

        treatment = np.random.binomial(1, 0.4, n).astype(float)
        age = np.random.normal(40, 10, n)
        income = np.random.normal(50000, 15000, n)

        # Add outliers and missing
        age[0:10] = 150
        income[np.random.choice(n, 30, replace=False)] = np.nan

        outcome = 100 + 10 * treatment + np.random.normal(0, 15, n)

        df = pd.DataFrame({
            "treatment": treatment,
            "age": age,
            "income": income,
            "outcome": outcome,
        })

        return EvalCase(
            name="data_quality_issues",
            description="Data with outliers and missing values",
            difficulty=EvalDifficulty.MEDIUM,
            category=EvalCategory.ERROR_RECOVERY,
            dataframe=df,
            dataset_info={"name": "quality_issues"},
            initial_state_overrides={
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "required_stages": ["profiling", "eda", "estimation"],
                "min_stages": 3,
            },
            max_expected_steps=30,
            timeout_seconds=240,
        )

    def _case_iteration_required(self) -> EvalCase:
        """Medium: Analysis requiring iteration after critique."""
        np.random.seed(603)
        n = 400

        confounder = np.random.normal(0, 1, n)
        p_treat = 1 / (1 + np.exp(-1.5 * confounder))
        treatment = np.random.binomial(1, p_treat)
        outcome = 50 + 8 * treatment + 5 * confounder + np.random.normal(0, 5, n)

        df = pd.DataFrame({
            "confounder": confounder,
            "treatment": treatment,
            "outcome": outcome,
        })

        return EvalCase(
            name="iteration_required",
            description="Confounded data that should trigger critique iteration",
            difficulty=EvalDifficulty.MEDIUM,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "confounded"},
            initial_state_overrides={
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "required_stages": ["profiling", "eda", "estimation", "critique"],
                "expected_behavior": "iterate",
                "min_stages": 4,
            },
            max_expected_steps=35,
            timeout_seconds=300,
        )

    def _case_method_selection_challenge(self) -> EvalCase:
        """Medium: Data requiring careful method selection."""
        np.random.seed(604)
        n = 200

        # Small sample, some confounding
        confounder = np.random.normal(0, 1, n)
        p_treat = 1 / (1 + np.exp(-0.8 * confounder))
        treatment = np.random.binomial(1, p_treat)
        outcome = 50 + 10 * treatment + 4 * confounder + np.random.normal(0, 8, n)

        df = pd.DataFrame({
            "confounder": confounder,
            "treatment": treatment,
            "outcome": outcome,
        })

        return EvalCase(
            name="method_selection_challenge",
            description="Small sample requiring simpler methods",
            difficulty=EvalDifficulty.MEDIUM,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "small_sample"},
            initial_state_overrides={
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "required_stages": ["profiling", "estimation"],
                "min_stages": 2,
            },
            max_expected_steps=30,
            timeout_seconds=240,
        )

    # ================== HARD CASES ==================

    def _case_complex_coordination(self) -> EvalCase:
        """Hard: Complex scenario requiring full pipeline."""
        np.random.seed(605)
        n = 800

        # Multiple confounders
        age = np.random.normal(45, 12, n)
        income = np.random.normal(50000, 20000, n)
        education = np.random.normal(14, 3, n)

        # Treatment depends on confounders
        logit = -2 + 0.02 * age + 0.00001 * income + 0.1 * education
        p_treat = 1 / (1 + np.exp(-logit))
        treatment = np.random.binomial(1, p_treat)

        # Outcome with heterogeneous effects
        base_effect = 8
        het_effect = 0.5 * (age - 45) / 12  # Effect varies with age
        outcome = (
            100 +
            treatment * (base_effect + het_effect) +
            0.3 * age +
            0.0001 * income +
            education +
            np.random.normal(0, 10, n)
        )

        df = pd.DataFrame({
            "age": age,
            "income": income,
            "education": education,
            "treatment": treatment,
            "outcome": outcome,
        })

        return EvalCase(
            name="complex_coordination",
            description="Complex scenario with multiple confounders and heterogeneity",
            difficulty=EvalDifficulty.HARD,
            category=EvalCategory.TASK_COMPLETION,
            dataframe=df,
            dataset_info={
                "name": "complex_study",
                "description": "Observational study with multiple confounders",
            },
            initial_state_overrides={
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "required_stages": ["profiling", "eda", "discovery", "estimation", "sensitivity"],
                "min_stages": 4,
            },
            max_expected_steps=40,
            timeout_seconds=360,
        )

    def _case_failure_recovery(self) -> EvalCase:
        """Hard: Scenario with potential method failures."""
        np.random.seed(606)
        n = 300

        # Extreme propensity scores can cause failures
        confounder = np.random.normal(0, 1, n)
        p_treat = np.clip(1 / (1 + np.exp(-4 * confounder)), 0.02, 0.98)
        treatment = np.random.binomial(1, p_treat)
        outcome = 50 + 10 * treatment + 5 * confounder + np.random.normal(0, 5, n)

        df = pd.DataFrame({
            "confounder": confounder,
            "treatment": treatment,
            "outcome": outcome,
        })

        return EvalCase(
            name="failure_recovery",
            description="Data that may cause some methods to fail",
            difficulty=EvalDifficulty.HARD,
            category=EvalCategory.ERROR_RECOVERY,
            dataframe=df,
            dataset_info={"name": "extreme_ps"},
            initial_state_overrides={
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "required_stages": ["profiling", "estimation"],
                "min_stages": 2,
            },
            max_expected_steps=35,
            timeout_seconds=300,
        )

    def _case_early_termination(self) -> EvalCase:
        """Hard: Scenario where analysis should be rejected early."""
        np.random.seed(607)
        n = 50

        # Very small sample
        treatment = np.random.binomial(1, 0.2, n)  # Only ~10 treated
        outcome = np.random.normal(50, 20, n)

        df = pd.DataFrame({
            "treatment": treatment,
            "outcome": outcome,
        })

        return EvalCase(
            name="early_termination",
            description="Very small sample that may require early termination/warning",
            difficulty=EvalDifficulty.HARD,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "tiny_sample"},
            initial_state_overrides={
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "min_stages": 1,  # May terminate early
            },
            max_expected_steps=25,
            timeout_seconds=180,
        )
