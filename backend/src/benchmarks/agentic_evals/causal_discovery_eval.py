"""Agentic evaluation for CausalDiscoveryAgent.

Tests the agent's ability to:
1. Run appropriate discovery algorithms
2. Identify causal structure
3. Detect confounders from DAG
4. Handle different data structures
5. Validate graph consistency
"""

import numpy as np
import pandas as pd

from src.agents.base import DataProfile
from src.agents.specialists.causal_discovery import CausalDiscoveryAgent
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


class CausalDiscoveryEvaluator(AgenticEvaluator):
    """Evaluator for CausalDiscoveryAgent."""

    @property
    def agent_name(self) -> str:
        return "causal_discovery"

    def create_agent(self) -> CausalDiscoveryAgent:
        return CausalDiscoveryAgent()

    def get_eval_suite(self) -> EvalSuite:
        """Create comprehensive evaluation suite."""
        suite = EvalSuite(
            name="Causal Discovery Evaluation Suite",
            agent_name=self.agent_name,
            description="Tests causal structure learning capabilities",
        )

        # EASY cases
        suite.add_case(self._case_simple_chain())
        suite.add_case(self._case_fork_structure())
        suite.add_case(self._case_collider_structure())

        # MEDIUM cases
        suite.add_case(self._case_confounded_treatment())
        suite.add_case(self._case_mediated_effect())
        suite.add_case(self._case_multiple_confounders())
        suite.add_case(self._case_noisy_data())

        # HARD cases
        suite.add_case(self._case_complex_dag())
        suite.add_case(self._case_hidden_confounder())
        suite.add_case(self._case_bidirectional_relationship())

        return suite

    def check_task_correctness(
        self,
        case: EvalCase,
        final_state: AnalysisState,
    ) -> tuple[bool, dict[str, float]]:
        """Check if discovery output is correct."""
        metrics = {}
        expected = case.expected_outputs

        dag = final_state.proposed_dag
        if dag is None:
            return False, {"dag_created": 0.0}

        metrics["dag_created"] = 1.0

        # Check if treatment -> outcome edge exists
        expected_treatment = expected.get("treatment")
        expected_outcome = expected.get("outcome")

        if expected_treatment and expected_outcome:
            has_treatment_effect = any(
                e.source == expected_treatment and e.target == expected_outcome
                for e in dag.edges
            )
            metrics["treatment_outcome_edge"] = 1.0 if has_treatment_effect else 0.0

        # Check confounder identification
        expected_confounders = expected.get("confounders", [])
        if expected_confounders and dag.nodes:
            # Confounders should have edges to both treatment and outcome
            identified_confounders = []
            for node in dag.nodes:
                if node in [expected_treatment, expected_outcome]:
                    continue
                edges_to_treatment = any(
                    e.source == node and e.target == expected_treatment for e in dag.edges
                )
                edges_to_outcome = any(
                    e.source == node and e.target == expected_outcome for e in dag.edges
                )
                if edges_to_treatment and edges_to_outcome:
                    identified_confounders.append(node)

            if expected_confounders:
                overlap = len(set(identified_confounders) & set(expected_confounders))
                metrics["confounder_recall"] = overlap / len(expected_confounders)
            else:
                metrics["confounder_recall"] = 1.0
        else:
            metrics["confounder_recall"] = 1.0

        # Check mediator identification
        expected_mediators = expected.get("mediators", [])
        if expected_mediators:
            # Mediators: treatment -> mediator -> outcome
            identified_mediators = []
            for node in dag.nodes:
                if node in [expected_treatment, expected_outcome]:
                    continue
                treatment_to_node = any(
                    e.source == expected_treatment and e.target == node for e in dag.edges
                )
                node_to_outcome = any(
                    e.source == node and e.target == expected_outcome for e in dag.edges
                )
                if treatment_to_node and node_to_outcome:
                    identified_mediators.append(node)

            overlap = len(set(identified_mediators) & set(expected_mediators))
            metrics["mediator_recall"] = overlap / len(expected_mediators)
        else:
            metrics["mediator_recall"] = 1.0

        # Check for spurious edges
        expected_no_edge = expected.get("should_not_have_edges", [])
        if expected_no_edge:
            spurious = sum(
                1 for src, tgt in expected_no_edge
                if any(e.source == src and e.target == tgt for e in dag.edges)
            )
            metrics["spurious_edge_rate"] = spurious / len(expected_no_edge)
        else:
            metrics["spurious_edge_rate"] = 0.0

        # Determine overall correctness
        is_correct = (
            metrics.get("dag_created", 0) >= 1.0 and
            metrics.get("treatment_outcome_edge", 0) >= 0.5 and
            metrics.get("confounder_recall", 0) >= 0.5
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
            missing_values=dict.fromkeys(df.columns, 0),
            numeric_stats={},
            categorical_stats={},
            treatment_candidates=[treatment_col],
            outcome_candidates=[outcome_col],
            potential_confounders=[c for c in df.columns if c not in [treatment_col, outcome_col]],
        )

    # ================== EASY CASES ==================

    def _case_simple_chain(self) -> EvalCase:
        """Easy: Simple chain structure X -> T -> Y."""
        np.random.seed(200)
        n = 1000

        x = np.random.normal(0, 1, n)
        treatment = (x > 0).astype(int)
        outcome = 50 + 10 * treatment + np.random.normal(0, 5, n)

        df = pd.DataFrame({
            "x": x,
            "treatment": treatment,
            "outcome": outcome,
        })

        profile = self._create_data_profile(df, "treatment", "outcome")

        return EvalCase(
            name="simple_chain",
            description="Simple chain: X -> T -> Y",
            difficulty=EvalDifficulty.EASY,
            category=EvalCategory.TASK_COMPLETION,
            dataframe=df,
            dataset_info={"name": "chain_structure"},
            initial_state_overrides={
                "data_profile": profile,
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "treatment": "treatment",
                "outcome": "outcome",
                "confounders": [],
            },
            max_expected_steps=10,
        )

    def _case_fork_structure(self) -> EvalCase:
        """Easy: Fork structure (confounder) C -> T, C -> Y."""
        np.random.seed(201)
        n = 1000

        confounder = np.random.normal(0, 1, n)
        treatment = (confounder + np.random.normal(0, 0.5, n) > 0).astype(int)
        outcome = 50 + 10 * treatment + 5 * confounder + np.random.normal(0, 5, n)

        df = pd.DataFrame({
            "confounder": confounder,
            "treatment": treatment,
            "outcome": outcome,
        })

        profile = self._create_data_profile(df, "treatment", "outcome")

        return EvalCase(
            name="fork_structure",
            description="Fork structure with confounder: C -> T, C -> Y, T -> Y",
            difficulty=EvalDifficulty.EASY,
            category=EvalCategory.TASK_COMPLETION,
            dataframe=df,
            dataset_info={"name": "fork_structure"},
            initial_state_overrides={
                "data_profile": profile,
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "treatment": "treatment",
                "outcome": "outcome",
                "confounders": ["confounder"],
            },
            max_expected_steps=10,
        )

    def _case_collider_structure(self) -> EvalCase:
        """Easy: Collider structure T -> C <- Y (should NOT condition on C)."""
        np.random.seed(202)
        n = 1000

        treatment = np.random.binomial(1, 0.5, n)
        outcome = 50 + 10 * treatment + np.random.normal(0, 5, n)
        collider = treatment + outcome / 10 + np.random.normal(0, 1, n)

        df = pd.DataFrame({
            "treatment": treatment,
            "collider": collider,
            "outcome": outcome,
        })

        profile = self._create_data_profile(df, "treatment", "outcome")

        return EvalCase(
            name="collider_structure",
            description="Collider: T -> C <- Y (should identify collider, not confounder)",
            difficulty=EvalDifficulty.EASY,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "collider_structure"},
            initial_state_overrides={
                "data_profile": profile,
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "treatment": "treatment",
                "outcome": "outcome",
                "confounders": [],  # Collider is NOT a confounder
            },
            max_expected_steps=12,
        )

    # ================== MEDIUM CASES ==================

    def _case_confounded_treatment(self) -> EvalCase:
        """Medium: Clear confounding with single confounder."""
        np.random.seed(203)
        n = 800

        age = np.random.normal(45, 12, n)
        p_treatment = 1 / (1 + np.exp(-(age - 45) / 10))
        treatment = np.random.binomial(1, p_treatment)
        outcome = 100 + 15 * treatment + 2 * age + np.random.normal(0, 10, n)

        df = pd.DataFrame({
            "age": age,
            "treatment": treatment,
            "outcome": outcome,
        })

        profile = self._create_data_profile(df, "treatment", "outcome")

        return EvalCase(
            name="confounded_treatment",
            description="Treatment confounded by age",
            difficulty=EvalDifficulty.MEDIUM,
            category=EvalCategory.TASK_COMPLETION,
            dataframe=df,
            dataset_info={"name": "confounded"},
            initial_state_overrides={
                "data_profile": profile,
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "treatment": "treatment",
                "outcome": "outcome",
                "confounders": ["age"],
            },
            max_expected_steps=12,
        )

    def _case_mediated_effect(self) -> EvalCase:
        """Medium: Effect mediated through intermediate variable."""
        np.random.seed(204)
        n = 800

        treatment = np.random.binomial(1, 0.5, n)
        mediator = 20 + 10 * treatment + np.random.normal(0, 3, n)
        outcome = 50 + 2 * mediator + np.random.normal(0, 5, n)

        df = pd.DataFrame({
            "treatment": treatment,
            "mediator": mediator,
            "outcome": outcome,
        })

        profile = self._create_data_profile(df, "treatment", "outcome")

        return EvalCase(
            name="mediated_effect",
            description="Effect mediated: T -> M -> Y",
            difficulty=EvalDifficulty.MEDIUM,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "mediation"},
            initial_state_overrides={
                "data_profile": profile,
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "treatment": "treatment",
                "outcome": "outcome",
                "mediators": ["mediator"],
                "confounders": [],
            },
            max_expected_steps=12,
        )

    def _case_multiple_confounders(self) -> EvalCase:
        """Medium: Multiple confounding variables."""
        np.random.seed(205)
        n = 1000

        age = np.random.normal(45, 10, n)
        income = np.random.normal(50000, 15000, n)
        education = np.random.normal(14, 3, n)

        # Treatment depends on all confounders
        logit = -2 + 0.03 * age + 0.00002 * income + 0.1 * education
        p_treatment = 1 / (1 + np.exp(-logit))
        treatment = np.random.binomial(1, p_treatment)

        # Outcome depends on treatment and confounders
        outcome = (
            50 +
            10 * treatment +
            0.5 * age +
            0.0002 * income +
            2 * education +
            np.random.normal(0, 10, n)
        )

        df = pd.DataFrame({
            "age": age,
            "income": income,
            "education": education,
            "treatment": treatment,
            "outcome": outcome,
        })

        profile = self._create_data_profile(df, "treatment", "outcome")

        return EvalCase(
            name="multiple_confounders",
            description="Multiple confounders affecting both treatment and outcome",
            difficulty=EvalDifficulty.MEDIUM,
            category=EvalCategory.TASK_COMPLETION,
            dataframe=df,
            dataset_info={"name": "multi_confounder"},
            initial_state_overrides={
                "data_profile": profile,
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "treatment": "treatment",
                "outcome": "outcome",
                "confounders": ["age", "income", "education"],
            },
            max_expected_steps=15,
        )

    def _case_noisy_data(self) -> EvalCase:
        """Medium: Noisy data making structure harder to detect."""
        np.random.seed(206)
        n = 500

        confounder = np.random.normal(0, 1, n)
        treatment = (confounder + np.random.normal(0, 2, n) > 0).astype(int)  # Noisy
        outcome = 50 + 5 * treatment + 3 * confounder + np.random.normal(0, 15, n)  # High noise

        df = pd.DataFrame({
            "confounder": confounder,
            "treatment": treatment,
            "outcome": outcome,
        })

        profile = self._create_data_profile(df, "treatment", "outcome")

        return EvalCase(
            name="noisy_data",
            description="Noisy data with weak signal-to-noise ratio",
            difficulty=EvalDifficulty.MEDIUM,
            category=EvalCategory.ERROR_RECOVERY,
            dataframe=df,
            dataset_info={"name": "noisy_data"},
            initial_state_overrides={
                "data_profile": profile,
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "treatment": "treatment",
                "outcome": "outcome",
                "confounders": ["confounder"],
            },
            max_expected_steps=15,
        )

    # ================== HARD CASES ==================

    def _case_complex_dag(self) -> EvalCase:
        """Hard: Complex DAG with multiple paths."""
        np.random.seed(207)
        n = 1000

        # Complex structure
        u1 = np.random.normal(0, 1, n)  # Confounder 1
        u2 = np.random.normal(0, 1, n)  # Confounder 2

        x1 = u1 + np.random.normal(0, 0.5, n)
        x2 = u2 + np.random.normal(0, 0.5, n)

        treatment = (u1 + x1 + np.random.normal(0, 0.5, n) > 0).astype(int)
        mediator = 2 * treatment + u2 + np.random.normal(0, 0.5, n)
        outcome = 50 + 5 * treatment + 3 * mediator + 2 * u1 + np.random.normal(0, 5, n)

        df = pd.DataFrame({
            "u1": u1,
            "u2": u2,
            "x1": x1,
            "x2": x2,
            "treatment": treatment,
            "mediator": mediator,
            "outcome": outcome,
        })

        profile = self._create_data_profile(df, "treatment", "outcome")

        return EvalCase(
            name="complex_dag",
            description="Complex DAG with confounders, mediators, and multiple paths",
            difficulty=EvalDifficulty.HARD,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "complex_dag"},
            initial_state_overrides={
                "data_profile": profile,
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "treatment": "treatment",
                "outcome": "outcome",
                "confounders": ["u1"],
                "mediators": ["mediator"],
            },
            max_expected_steps=15,
        )

    def _case_hidden_confounder(self) -> EvalCase:
        """Hard: Hidden confounder (not in data)."""
        np.random.seed(208)
        n = 800

        # Hidden confounder
        hidden = np.random.normal(0, 1, n)

        # Observed variables
        proxy = hidden + np.random.normal(0, 0.5, n)  # Proxy for hidden
        treatment = (hidden + np.random.normal(0, 0.5, n) > 0).astype(int)
        outcome = 50 + 10 * treatment + 5 * hidden + np.random.normal(0, 5, n)

        # Don't include hidden in the data
        df = pd.DataFrame({
            "proxy": proxy,
            "treatment": treatment,
            "outcome": outcome,
        })

        profile = self._create_data_profile(df, "treatment", "outcome")

        return EvalCase(
            name="hidden_confounder",
            description="Hidden confounder not in data (only proxy available)",
            difficulty=EvalDifficulty.HARD,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "hidden_confounder"},
            initial_state_overrides={
                "data_profile": profile,
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "treatment": "treatment",
                "outcome": "outcome",
                # Proxy might be identified as confounder (acceptable)
                "confounders": ["proxy"],
            },
            max_expected_steps=15,
        )

    def _case_bidirectional_relationship(self) -> EvalCase:
        """Hard: Bidirectional/feedback relationship (hard for PC algorithm)."""
        np.random.seed(209)
        n = 1000

        # Simulate equilibrium of feedback system
        treatment = np.random.binomial(1, 0.5, n)
        x = np.random.normal(0, 1, n)

        # Outcome and x have feedback relationship
        outcome = 50 + 10 * treatment + 2 * x + np.random.normal(0, 5, n)

        df = pd.DataFrame({
            "x": x,
            "treatment": treatment,
            "outcome": outcome,
        })

        profile = self._create_data_profile(df, "treatment", "outcome")

        return EvalCase(
            name="bidirectional_relationship",
            description="Variables with potential feedback relationship",
            difficulty=EvalDifficulty.HARD,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "feedback"},
            initial_state_overrides={
                "data_profile": profile,
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
            expected_outputs={
                "treatment": "treatment",
                "outcome": "outcome",
            },
            max_expected_steps=15,
        )
