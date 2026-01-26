"""Agentic evaluation for DataProfilerAgent.

Tests the agent's ability to:
1. Identify treatment and outcome variables
2. Classify feature types correctly
3. Detect potential confounders
4. Handle edge cases (missing data, unusual types)
5. Make appropriate tool selections
"""

import numpy as np
import pandas as pd

from src.agents.specialists.data_profiler import DataProfilerAgent
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


class DataProfilerEvaluator(AgenticEvaluator):
    """Evaluator for DataProfilerAgent."""

    @property
    def agent_name(self) -> str:
        return "data_profiler"

    def create_agent(self) -> DataProfilerAgent:
        return DataProfilerAgent()

    def get_eval_suite(self) -> EvalSuite:
        """Create comprehensive evaluation suite."""
        suite = EvalSuite(
            name="DataProfiler Evaluation Suite",
            agent_name=self.agent_name,
            description="Tests data profiling agent capabilities",
        )

        # EASY cases - clear-cut scenarios
        suite.add_case(self._case_binary_treatment_numeric_outcome())
        suite.add_case(self._case_explicit_column_names())
        suite.add_case(self._case_standard_rct_data())

        # MEDIUM cases - some ambiguity
        suite.add_case(self._case_multiple_treatment_candidates())
        suite.add_case(self._case_continuous_treatment())
        suite.add_case(self._case_categorical_outcome())
        suite.add_case(self._case_missing_data())

        # HARD cases - edge cases and ambiguity
        suite.add_case(self._case_high_cardinality_features())
        suite.add_case(self._case_multicollinear_features())
        suite.add_case(self._case_time_series_structure())
        suite.add_case(self._case_instrument_candidates())

        return suite

    def check_task_correctness(
        self,
        case: EvalCase,
        final_state: AnalysisState,
    ) -> tuple[bool, dict[str, float]]:
        """Check if profiler output is correct."""
        metrics = {}
        expected = case.expected_outputs

        profile = final_state.data_profile
        if profile is None:
            return False, {"profile_created": 0.0}

        metrics["profile_created"] = 1.0

        # Check treatment identification
        expected_treatments = expected.get("treatment_candidates", [])
        if expected_treatments and profile.treatment_candidates:
            overlap = len(set(profile.treatment_candidates) & set(expected_treatments))
            metrics["treatment_recall"] = overlap / len(expected_treatments)
            # Check if top candidate is correct
            if profile.treatment_candidates[0] in expected_treatments:
                metrics["treatment_top1_correct"] = 1.0
            else:
                metrics["treatment_top1_correct"] = 0.0
        else:
            metrics["treatment_recall"] = 0.0 if expected_treatments else 1.0
            metrics["treatment_top1_correct"] = 0.0 if expected_treatments else 1.0

        # Check outcome identification
        expected_outcomes = expected.get("outcome_candidates", [])
        if expected_outcomes and profile.outcome_candidates:
            overlap = len(set(profile.outcome_candidates) & set(expected_outcomes))
            metrics["outcome_recall"] = overlap / len(expected_outcomes)
            if profile.outcome_candidates[0] in expected_outcomes:
                metrics["outcome_top1_correct"] = 1.0
            else:
                metrics["outcome_top1_correct"] = 0.0
        else:
            metrics["outcome_recall"] = 0.0 if expected_outcomes else 1.0
            metrics["outcome_top1_correct"] = 0.0 if expected_outcomes else 1.0

        # Check confounder identification
        expected_confounders = expected.get("confounders", [])
        if expected_confounders and profile.potential_confounders:
            overlap = len(set(profile.potential_confounders) & set(expected_confounders))
            metrics["confounder_recall"] = overlap / len(expected_confounders)
        else:
            metrics["confounder_recall"] = 0.0 if expected_confounders else 1.0

        # Check feature type classification
        expected_types = expected.get("feature_types", {})
        if expected_types and profile.feature_types:
            correct = sum(
                1 for col, expected_type in expected_types.items()
                if profile.feature_types.get(col) == expected_type
            )
            metrics["type_accuracy"] = correct / len(expected_types)
        else:
            metrics["type_accuracy"] = 1.0

        # Determine overall correctness
        is_correct = (
            metrics.get("treatment_top1_correct", 0) >= 0.5 and
            metrics.get("outcome_top1_correct", 0) >= 0.5 and
            metrics.get("type_accuracy", 0) >= 0.8
        )

        return is_correct, metrics

    # ================== EASY CASES ==================

    def _case_binary_treatment_numeric_outcome(self) -> EvalCase:
        """Easy: Clear binary treatment and numeric outcome."""
        np.random.seed(42)
        n = 500

        df = pd.DataFrame({
            "treatment": np.random.binomial(1, 0.4, n),
            "age": np.random.normal(40, 10, n),
            "income": np.random.normal(50000, 15000, n),
            "outcome": np.random.normal(100, 20, n),
        })

        return EvalCase(
            name="binary_treatment_numeric_outcome",
            description="Clear binary treatment (0/1) and numeric outcome",
            difficulty=EvalDifficulty.EASY,
            category=EvalCategory.TASK_COMPLETION,
            dataframe=df,
            dataset_info={"name": "simple_rct", "description": "Simple RCT data"},
            expected_outputs={
                "treatment_candidates": ["treatment"],
                "outcome_candidates": ["outcome", "income"],
                "confounders": ["age", "income"],
                "feature_types": {
                    "treatment": "binary",
                    "age": "numeric",
                    "income": "numeric",
                    "outcome": "numeric",
                },
            },
            expected_tool_calls=["get_dataset_overview", "analyze_column", "finalize_profile"],
            max_expected_steps=10,
        )

    def _case_explicit_column_names(self) -> EvalCase:
        """Easy: Columns named 'treat' and 'y'."""
        np.random.seed(43)
        n = 300

        df = pd.DataFrame({
            "treat": np.random.binomial(1, 0.5, n),
            "x1": np.random.normal(0, 1, n),
            "x2": np.random.normal(0, 1, n),
            "y": np.random.normal(10, 5, n),
        })

        return EvalCase(
            name="explicit_column_names",
            description="Columns with obvious names: treat, y",
            difficulty=EvalDifficulty.EASY,
            category=EvalCategory.TASK_COMPLETION,
            dataframe=df,
            dataset_info={"name": "explicit_names"},
            expected_outputs={
                "treatment_candidates": ["treat"],
                "outcome_candidates": ["y"],
                "feature_types": {"treat": "binary", "y": "numeric"},
            },
            max_expected_steps=8,
        )

    def _case_standard_rct_data(self) -> EvalCase:
        """Easy: Standard RCT structure."""
        np.random.seed(44)
        n = 1000

        # Confounders
        age = np.random.normal(45, 12, n)
        gender = np.random.binomial(1, 0.48, n)
        baseline = np.random.normal(100, 15, n)

        # Random treatment
        treatment = np.random.binomial(1, 0.5, n)

        # Outcome
        outcome = baseline + 10 * treatment + 0.5 * age + np.random.normal(0, 10, n)

        df = pd.DataFrame({
            "patient_id": range(n),
            "age": age,
            "gender": gender,
            "baseline_score": baseline,
            "randomized_treatment": treatment,
            "outcome_score": outcome,
        })

        return EvalCase(
            name="standard_rct",
            description="Standard RCT with clear structure",
            difficulty=EvalDifficulty.EASY,
            category=EvalCategory.TASK_COMPLETION,
            dataframe=df,
            dataset_info={"name": "rct_study", "description": "Randomized controlled trial"},
            expected_outputs={
                "treatment_candidates": ["randomized_treatment"],
                "outcome_candidates": ["outcome_score"],
                "confounders": ["age", "gender", "baseline_score"],
            },
            max_expected_steps=10,
        )

    # ================== MEDIUM CASES ==================

    def _case_multiple_treatment_candidates(self) -> EvalCase:
        """Medium: Multiple binary columns could be treatment."""
        np.random.seed(45)
        n = 500

        df = pd.DataFrame({
            "intervention": np.random.binomial(1, 0.3, n),
            "placebo": np.random.binomial(1, 0.3, n),
            "control": np.random.binomial(1, 0.4, n),
            "age": np.random.normal(40, 10, n),
            "health_score": np.random.normal(70, 15, n),
        })

        return EvalCase(
            name="multiple_treatment_candidates",
            description="Multiple binary columns - agent must reason about which is treatment",
            difficulty=EvalDifficulty.MEDIUM,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "multi_treatment", "description": "Study with multiple arms"},
            expected_outputs={
                "treatment_candidates": ["intervention", "placebo", "control"],
                "outcome_candidates": ["health_score"],
            },
            max_expected_steps=12,
        )

    def _case_continuous_treatment(self) -> EvalCase:
        """Medium: Continuous treatment (dose-response)."""
        np.random.seed(46)
        n = 400

        dose = np.random.uniform(0, 100, n)  # Continuous dose
        age = np.random.normal(50, 10, n)
        response = 50 + 0.5 * dose + 0.2 * age + np.random.normal(0, 5, n)

        df = pd.DataFrame({
            "drug_dose_mg": dose,
            "patient_age": age,
            "clinical_response": response,
        })

        return EvalCase(
            name="continuous_treatment",
            description="Continuous treatment variable (dose-response)",
            difficulty=EvalDifficulty.MEDIUM,
            category=EvalCategory.TASK_COMPLETION,
            dataframe=df,
            dataset_info={"name": "dose_response", "description": "Dose-response study"},
            expected_outputs={
                "treatment_candidates": ["drug_dose_mg"],
                "outcome_candidates": ["clinical_response"],
                "feature_types": {"drug_dose_mg": "numeric"},
            },
            max_expected_steps=10,
        )

    def _case_categorical_outcome(self) -> EvalCase:
        """Medium: Categorical outcome variable."""
        np.random.seed(47)
        n = 600

        treatment = np.random.binomial(1, 0.5, n)
        age = np.random.normal(40, 10, n)
        outcome = np.random.choice(["improved", "stable", "worsened"], n, p=[0.4, 0.4, 0.2])

        df = pd.DataFrame({
            "treatment_arm": treatment,
            "patient_age": age,
            "health_status": outcome,
        })

        return EvalCase(
            name="categorical_outcome",
            description="Categorical outcome with 3 levels",
            difficulty=EvalDifficulty.MEDIUM,
            category=EvalCategory.TASK_COMPLETION,
            dataframe=df,
            dataset_info={"name": "categorical_outcome_study"},
            expected_outputs={
                "treatment_candidates": ["treatment_arm"],
                "outcome_candidates": ["health_status"],
                "feature_types": {"health_status": "categorical"},
            },
            max_expected_steps=10,
        )

    def _case_missing_data(self) -> EvalCase:
        """Medium: Dataset with missing values."""
        np.random.seed(48)
        n = 500

        treatment = np.random.binomial(1, 0.4, n).astype(float)
        treatment[np.random.choice(n, 20, replace=False)] = np.nan

        age = np.random.normal(40, 10, n)
        age[np.random.choice(n, 50, replace=False)] = np.nan

        income = np.random.normal(50000, 15000, n)
        income[np.random.choice(n, 30, replace=False)] = np.nan

        outcome = np.random.normal(100, 20, n)

        df = pd.DataFrame({
            "treated": treatment,
            "age": age,
            "income": income,
            "score": outcome,
        })

        return EvalCase(
            name="missing_data",
            description="Dataset with missing values in treatment and covariates",
            difficulty=EvalDifficulty.MEDIUM,
            category=EvalCategory.ERROR_RECOVERY,
            dataframe=df,
            dataset_info={"name": "missing_data_study"},
            expected_outputs={
                "treatment_candidates": ["treated"],
                "outcome_candidates": ["score"],
            },
            max_expected_steps=12,
        )

    # ================== HARD CASES ==================

    def _case_high_cardinality_features(self) -> EvalCase:
        """Hard: High cardinality categorical features."""
        np.random.seed(49)
        n = 1000

        treatment = np.random.binomial(1, 0.3, n)
        zip_code = np.random.randint(10000, 99999, n)  # High cardinality
        hospital_id = np.random.randint(1, 500, n)  # High cardinality
        outcome = np.random.normal(100, 20, n)

        df = pd.DataFrame({
            "intervention": treatment,
            "zip_code": zip_code,
            "hospital_id": hospital_id,
            "patient_outcome": outcome,
        })

        return EvalCase(
            name="high_cardinality",
            description="High cardinality categorical features",
            difficulty=EvalDifficulty.HARD,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "high_cardinality_study"},
            expected_outputs={
                "treatment_candidates": ["intervention"],
                "outcome_candidates": ["patient_outcome"],
            },
            max_expected_steps=15,
        )

    def _case_multicollinear_features(self) -> EvalCase:
        """Hard: Multicollinear features that could confuse profiler."""
        np.random.seed(50)
        n = 500

        treatment = np.random.binomial(1, 0.4, n)
        x1 = np.random.normal(0, 1, n)
        x2 = x1 * 0.9 + np.random.normal(0, 0.1, n)  # Nearly identical to x1
        x3 = x1 + x2  # Linear combination
        outcome = 10 + 2 * treatment + x1 + np.random.normal(0, 1, n)

        df = pd.DataFrame({
            "treat": treatment,
            "feature_a": x1,
            "feature_b": x2,
            "feature_c": x3,
            "result": outcome,
        })

        return EvalCase(
            name="multicollinear_features",
            description="Multicollinear features - agent should identify relationship",
            difficulty=EvalDifficulty.HARD,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "multicollinear_study"},
            expected_outputs={
                "treatment_candidates": ["treat"],
                "outcome_candidates": ["result"],
            },
            max_expected_steps=15,
        )

    def _case_time_series_structure(self) -> EvalCase:
        """Hard: Data with time structure (panel data)."""
        np.random.seed(51)
        n_units = 100
        n_periods = 5

        data = []
        for unit in range(n_units):
            treatment_period = np.random.randint(2, 5)
            for period in range(n_periods):
                treated = 1 if period >= treatment_period else 0
                outcome = 50 + 5 * treated + 2 * period + np.random.normal(0, 3)
                data.append({
                    "unit_id": unit,
                    "time_period": period,
                    "post_treatment": treated,
                    "measurement": outcome,
                })

        df = pd.DataFrame(data)

        return EvalCase(
            name="time_series_structure",
            description="Panel data with time structure - should detect DiD opportunity",
            difficulty=EvalDifficulty.HARD,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "panel_data", "description": "Panel data for DiD"},
            expected_outputs={
                "treatment_candidates": ["post_treatment"],
                "outcome_candidates": ["measurement"],
            },
            expected_tool_calls=["check_time_dimension"],
            max_expected_steps=15,
        )

    def _case_instrument_candidates(self) -> EvalCase:
        """Hard: Data with potential instrumental variables."""
        np.random.seed(52)
        n = 800

        # Instrument: distance to college
        distance = np.random.exponential(50, n)

        # Unobserved confounder: ability
        ability = np.random.normal(0, 1, n)

        # Treatment: college attendance (affected by distance and ability)
        p_college = 1 / (1 + np.exp(0.02 * distance - ability))
        college = np.random.binomial(1, p_college)

        # Outcome: earnings (affected by college and ability)
        earnings = 30000 + 10000 * college + 5000 * ability + np.random.normal(0, 5000, n)

        df = pd.DataFrame({
            "miles_to_college": distance,
            "attended_college": college,
            "annual_earnings": earnings,
        })

        return EvalCase(
            name="instrument_candidates",
            description="Data with potential IV (distance as instrument for college)",
            difficulty=EvalDifficulty.HARD,
            category=EvalCategory.REASONING_QUALITY,
            dataframe=df,
            dataset_info={"name": "iv_study", "description": "IV estimation scenario"},
            expected_outputs={
                "treatment_candidates": ["attended_college"],
                "outcome_candidates": ["annual_earnings"],
            },
            max_expected_steps=15,
        )
