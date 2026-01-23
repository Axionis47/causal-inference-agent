"""Benchmark dataset loader for the 8 diverse causal inference datasets."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.logging_config.structured import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkDataset:
    """A benchmark dataset for testing causal inference methods."""

    name: str
    description: str
    analysis_type: str
    treatment_variable: str
    outcome_variable: str
    expected_methods: list[str]
    ground_truth: dict[str, float] | None
    data: pd.DataFrame
    metadata: dict[str, Any]


class BenchmarkDatasetLoader:
    """Loader for benchmark datasets.

    Supports 8 diverse causal inference benchmark datasets:
    1. IHDP - Binary treatment, heterogeneous effects
    2. LaLonde/NSW - Propensity score matching benchmark
    3. Twins - Binary treatment with counterfactuals
    4. Card College - Instrumental variables
    5. Card-Krueger Min Wage - Difference-in-differences
    6. ACIC 2016 - High-dimensional CATE
    7. Synthetic Time Series - Time series causality
    8. News - Continuous/text treatment

    These are for stress testing - the system must work on ANY dataset.
    """

    @staticmethod
    def load_ihdp() -> BenchmarkDataset:
        """Load IHDP (Infant Health and Development Program) dataset.

        Binary treatment (home visits), continuous outcome (cognitive scores).
        Strong selection bias, heterogeneous treatment effects.
        """
        # Generate synthetic IHDP-like data
        np.random.seed(42)
        n = 747

        # Covariates
        X = np.random.randn(n, 25)
        X[:, :6] = np.random.randn(n, 6)  # Continuous
        X[:, 6:] = np.random.binomial(1, 0.5, (n, 19))  # Binary

        # Treatment assignment (biased towards certain covariate values)
        propensity = 1 / (1 + np.exp(-(X[:, 0] + 0.5 * X[:, 1] - 0.3)))
        T = np.random.binomial(1, propensity)

        # Outcome (treatment effect varies with covariates)
        tau = 4 + 2 * X[:, 0]  # Heterogeneous treatment effect
        Y0 = X[:, 0] + X[:, 1] + np.random.randn(n)
        Y1 = Y0 + tau
        Y = T * Y1 + (1 - T) * Y0

        # Create DataFrame
        cols = [f"x{i}" for i in range(25)]
        df = pd.DataFrame(X, columns=cols)
        df["treatment"] = T
        df["outcome"] = Y

        true_ate = np.mean(tau)

        return BenchmarkDataset(
            name="IHDP",
            description="Infant Health Development Program - binary treatment with heterogeneous effects",
            analysis_type="binary_treatment_heterogeneous",
            treatment_variable="treatment",
            outcome_variable="outcome",
            expected_methods=["propensity_score_matching", "doubly_robust", "t_learner", "x_learner"],
            ground_truth={"ATE": true_ate, "ATT": np.mean(tau[T == 1])},
            data=df,
            metadata={"n_samples": n, "n_covariates": 25, "treatment_rate": T.mean()},
        )

    @staticmethod
    def load_lalonde() -> BenchmarkDataset:
        """Load LaLonde/NSW Jobs Training dataset.

        Classic propensity score matching benchmark with RCT comparison.
        """
        np.random.seed(43)
        n = 445

        # Demographics
        age = np.random.randint(18, 55, n)
        education = np.random.randint(0, 16, n)
        black = np.random.binomial(1, 0.8, n)
        hispanic = np.random.binomial(1, 0.1, n)
        married = np.random.binomial(1, 0.2, n)
        nodegree = (education < 12).astype(int)

        # Prior earnings
        re74 = np.maximum(0, np.random.normal(2000, 3000, n))
        re75 = np.maximum(0, np.random.normal(2500, 3500, n))

        # Treatment (job training program)
        T = np.random.binomial(1, 0.4, n)

        # Outcome (1978 earnings)
        # True treatment effect ~ $1,794
        base_earnings = 1000 + 500 * education + 100 * age - 2000 * nodegree
        treatment_effect = 1794
        Y = base_earnings + treatment_effect * T + np.random.normal(0, 2000, n)
        Y = np.maximum(0, Y)

        df = pd.DataFrame({
            "age": age,
            "education": education,
            "black": black,
            "hispanic": hispanic,
            "married": married,
            "nodegree": nodegree,
            "re74": re74,
            "re75": re75,
            "treatment": T,
            "re78": Y,
        })

        return BenchmarkDataset(
            name="LaLonde",
            description="NSW Jobs Training - propensity score benchmark with RCT comparison",
            analysis_type="propensity_score_benchmark",
            treatment_variable="treatment",
            outcome_variable="re78",
            expected_methods=["propensity_score_matching", "inverse_probability_weighting", "doubly_robust"],
            ground_truth={"ATE": treatment_effect},
            data=df,
            metadata={"n_samples": n, "experimental_ate": 1794},
        )

    @staticmethod
    def load_twins() -> BenchmarkDataset:
        """Load Twins mortality dataset.

        Binary treatment (heavier twin), binary outcome (mortality).
        Both potential outcomes observed for twin pairs.
        """
        np.random.seed(44)
        n_pairs = 5000
        n = n_pairs * 2

        # Parent/pregnancy characteristics
        mother_age = np.repeat(np.random.randint(18, 45, n_pairs), 2)
        gestation = np.repeat(np.random.randint(25, 42, n_pairs), 2)
        prenatal_care = np.repeat(np.random.binomial(1, 0.7, n_pairs), 2)

        # Twin-specific
        birth_weight = np.random.normal(2500, 500, n)
        pair_id = np.repeat(np.arange(n_pairs), 2)

        # Treatment: being heavier twin (within pair)
        T = np.zeros(n)
        for i in range(n_pairs):
            if birth_weight[2 * i] > birth_weight[2 * i + 1]:
                T[2 * i] = 1
            else:
                T[2 * i + 1] = 1

        # Outcome: mortality (lower for heavier twin)
        base_mortality = 0.035
        mortality_reduction = 0.01  # True treatment effect
        prob = base_mortality - mortality_reduction * T + 0.001 * (42 - gestation)
        prob = np.clip(prob, 0.001, 0.1)
        Y = np.random.binomial(1, prob)

        df = pd.DataFrame({
            "pair_id": pair_id,
            "mother_age": mother_age,
            "gestation": gestation,
            "prenatal_care": prenatal_care,
            "birth_weight": birth_weight,
            "heavier_twin": T.astype(int),
            "mortality": Y,
        })

        return BenchmarkDataset(
            name="Twins",
            description="Twin births mortality - binary treatment with counterfactual pairs",
            analysis_type="twin_pairs",
            treatment_variable="heavier_twin",
            outcome_variable="mortality",
            expected_methods=["propensity_score_matching", "doubly_robust"],
            ground_truth={"ATE": -mortality_reduction},
            data=df,
            metadata={"n_pairs": n_pairs, "mortality_rate": Y.mean()},
        )

    @staticmethod
    def load_card_iv() -> BenchmarkDataset:
        """Load Card College Proximity dataset.

        Instrumental variables benchmark - college proximity as instrument.
        """
        np.random.seed(45)
        n = 3010

        # Instrument: proximity to 4-year college
        college_nearby = np.random.binomial(1, 0.4, n)

        # Covariates
        family_income = np.random.lognormal(10, 1, n)
        parents_education = np.random.randint(0, 20, n)
        ability = np.random.randn(n)  # Unobserved confounder

        # Treatment: years of education (endogenous)
        education = 12 + 2 * college_nearby + 0.5 * ability + np.random.randn(n)
        education = np.clip(education, 8, 20)

        # Outcome: log wages (affected by both education and ability)
        true_return = 0.08  # True return to education
        log_wage = 2 + true_return * education + 0.3 * ability + np.random.randn(n) * 0.5

        df = pd.DataFrame({
            "college_nearby": college_nearby,
            "family_income": family_income,
            "parents_education": parents_education,
            "education": education,
            "log_wage": log_wage,
        })

        return BenchmarkDataset(
            name="Card_IV",
            description="Card College Proximity - instrumental variables for education returns",
            analysis_type="instrumental_variables",
            treatment_variable="education",
            outcome_variable="log_wage",
            expected_methods=["instrumental_variables", "ols_regression"],
            ground_truth={"LATE": true_return},
            data=df,
            metadata={"instrument": "college_nearby", "n_samples": n},
        )

    @staticmethod
    def load_minimum_wage_did() -> BenchmarkDataset:
        """Load Card-Krueger Minimum Wage dataset.

        Difference-in-differences benchmark.
        """
        np.random.seed(46)
        n_restaurants = 400

        # Restaurant characteristics
        chain = np.random.choice(["BurgerKing", "KFC", "Wendys", "RoyRogers"], n_restaurants)
        company_owned = np.random.binomial(1, 0.3, n_restaurants)

        # State (treatment vs control)
        state = np.random.choice(["NJ", "PA"], n_restaurants, p=[0.6, 0.4])
        treated = (state == "NJ").astype(int)

        # Time periods
        periods = ["before", "after"]
        data = []

        true_effect = 0.5  # True DiD effect on FTE employment

        for i in range(n_restaurants):
            for period in periods:
                is_after = 1 if period == "after" else 0

                # Employment
                base_fte = 20 + np.random.randn() * 5
                time_trend = 2 * is_after  # Common trend
                treatment_effect = true_effect * treated[i] * is_after
                fte = base_fte + time_trend + treatment_effect + np.random.randn() * 2

                # Wages
                starting_wage = 4.25 + np.random.randn() * 0.3
                if treated[i] and is_after:
                    starting_wage = max(5.05, starting_wage)  # NJ minimum wage increase

                data.append({
                    "restaurant_id": i,
                    "chain": chain[i],
                    "company_owned": company_owned[i],
                    "state": state[i],
                    "treated": treated[i],
                    "period": period,
                    "post": is_after,
                    "fte_employment": max(0, fte),
                    "starting_wage": starting_wage,
                })

        df = pd.DataFrame(data)

        return BenchmarkDataset(
            name="MinimumWage_DiD",
            description="Card-Krueger Minimum Wage - difference-in-differences",
            analysis_type="difference_in_differences",
            treatment_variable="treated",
            outcome_variable="fte_employment",
            expected_methods=["difference_in_differences"],
            ground_truth={"DiD": true_effect},
            data=df,
            metadata={"time_column": "post", "n_restaurants": n_restaurants},
        )

    @staticmethod
    def load_acic() -> BenchmarkDataset:
        """Load ACIC-style high-dimensional dataset.

        High-dimensional confounders with heterogeneous treatment effects.
        """
        np.random.seed(47)
        n = 4000

        # High-dimensional covariates
        n_features = 58
        X = np.random.randn(n, n_features)

        # Some covariates are important
        important_idx = [0, 1, 2, 5, 10, 15, 20]

        # Treatment assignment (targeted selection)
        logit = sum(0.3 * X[:, i] for i in important_idx)
        propensity = 1 / (1 + np.exp(-logit))
        T = np.random.binomial(1, propensity)

        # Heterogeneous treatment effect
        tau = 1 + 0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2]

        # Outcome
        Y0 = sum(0.5 * X[:, i] for i in important_idx) + np.random.randn(n)
        Y1 = Y0 + tau
        Y = T * Y1 + (1 - T) * Y0

        # Create DataFrame
        cols = [f"x{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=cols)
        df["treatment"] = T
        df["outcome"] = Y

        return BenchmarkDataset(
            name="ACIC",
            description="ACIC-style high-dimensional with targeted selection",
            analysis_type="high_dimensional_cate",
            treatment_variable="treatment",
            outcome_variable="outcome",
            expected_methods=["causal_forest", "double_ml", "x_learner"],
            ground_truth={"ATE": np.mean(tau), "ATT": np.mean(tau[T == 1])},
            data=df,
            metadata={"n_features": n_features, "true_cate_std": np.std(tau)},
        )

    @staticmethod
    def load_time_series() -> BenchmarkDataset:
        """Load synthetic time series causal dataset.

        For testing Granger causality and time series methods.
        """
        np.random.seed(48)
        n = 500

        # Generate time series with known causal structure
        # X1 -> X2 -> X3 (chain)
        # X1 -> X3 (direct effect)

        X1 = np.zeros(n)
        X2 = np.zeros(n)
        X3 = np.zeros(n)

        X1[0] = np.random.randn()
        X2[0] = np.random.randn()
        X3[0] = np.random.randn()

        for t in range(1, n):
            X1[t] = 0.8 * X1[t - 1] + np.random.randn() * 0.5
            X2[t] = 0.5 * X2[t - 1] + 0.4 * X1[t - 1] + np.random.randn() * 0.5
            X3[t] = 0.3 * X3[t - 1] + 0.3 * X2[t - 1] + 0.2 * X1[t - 1] + np.random.randn() * 0.5

        df = pd.DataFrame({
            "time": np.arange(n),
            "X1": X1,
            "X2": X2,
            "X3": X3,
        })

        return BenchmarkDataset(
            name="TimeSeries",
            description="Synthetic time series with known causal structure",
            analysis_type="time_series_causality",
            treatment_variable="X1",
            outcome_variable="X3",
            expected_methods=["ols_regression"],  # Granger methods if available
            ground_truth={"X1_to_X3_direct": 0.2, "X1_to_X2": 0.4, "X2_to_X3": 0.3},
            data=df,
            metadata={"time_column": "time", "causal_structure": "X1->X2->X3, X1->X3"},
        )

    @staticmethod
    def load_news() -> BenchmarkDataset:
        """Load synthetic news/text treatment dataset.

        Continuous treatment with high-dimensional features.
        """
        np.random.seed(49)
        n = 2000

        # Simulate "bag of words" style features
        n_words = 50
        word_features = np.random.poisson(2, (n, n_words))

        # Treatment: continuous (e.g., exposure intensity)
        treatment = np.random.exponential(1, n)

        # Some words are confounders
        confounder_words = [0, 5, 10, 15]
        for w in confounder_words:
            treatment += 0.1 * word_features[:, w]

        # Outcome
        true_effect = 0.5
        Y = true_effect * treatment
        for w in confounder_words:
            Y += 0.2 * word_features[:, w]
        Y += np.random.randn(n) * 0.5

        # Create DataFrame
        cols = [f"word_{i}" for i in range(n_words)]
        df = pd.DataFrame(word_features, columns=cols)
        df["exposure"] = treatment
        df["response"] = Y

        return BenchmarkDataset(
            name="News",
            description="Continuous treatment with text-like features",
            analysis_type="continuous_treatment",
            treatment_variable="exposure",
            outcome_variable="response",
            expected_methods=["ols_regression", "double_ml"],
            ground_truth={"ATE": true_effect},
            data=df,
            metadata={"n_words": n_words, "treatment_type": "continuous"},
        )

    @classmethod
    def load_all(cls) -> list[BenchmarkDataset]:
        """Load all 8 benchmark datasets."""
        return [
            cls.load_ihdp(),
            cls.load_lalonde(),
            cls.load_twins(),
            cls.load_card_iv(),
            cls.load_minimum_wage_did(),
            cls.load_acic(),
            cls.load_time_series(),
            cls.load_news(),
        ]

    @classmethod
    def load_by_name(cls, name: str) -> BenchmarkDataset:
        """Load a benchmark dataset by name."""
        loaders = {
            "ihdp": cls.load_ihdp,
            "lalonde": cls.load_lalonde,
            "twins": cls.load_twins,
            "card_iv": cls.load_card_iv,
            "minimum_wage_did": cls.load_minimum_wage_did,
            "acic": cls.load_acic,
            "time_series": cls.load_time_series,
            "news": cls.load_news,
        }

        name_lower = name.lower()
        if name_lower not in loaders:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(loaders.keys())}")

        return loaders[name_lower]()
