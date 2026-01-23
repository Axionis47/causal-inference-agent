"""Benchmark datasets for causal inference evaluation.

This module provides standard benchmark datasets used in the causal inference
literature for evaluating treatment effect estimation methods.

Datasets included:
- LaLonde (1986): National Supported Work job training program
- IHDP: Infant Health and Development Program (semi-synthetic)
- Twins: Twin births mortality data
- Synthetic: Fully controlled synthetic benchmarks

References:
- LaLonde, R. (1986). Evaluating the Econometric Evaluations of Training Programs
- Hill, J. (2011). Bayesian Nonparametric Modeling for Causal Inference
- Louizos et al. (2017). Causal Effect Inference with Deep Latent-Variable Models
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class BenchmarkDataset(ABC):
    """Abstract base class for benchmark datasets."""

    name: str
    description: str
    treatment_col: str
    outcome_col: str
    true_ate: float | None = None  # Known ATE if available
    true_att: float | None = None  # Known ATT if available
    true_cate: np.ndarray | None = None  # Known CATE if available

    @abstractmethod
    def generate(self, n_samples: int | None = None, seed: int = 42) -> pd.DataFrame:
        """Generate or load the benchmark dataset.

        Args:
            n_samples: Number of samples (if synthetic/subsetting)
            seed: Random seed for reproducibility

        Returns:
            DataFrame with treatment, outcome, and covariates
        """
        pass

    def get_ground_truth(self) -> dict[str, Any]:
        """Get ground truth causal effects if known."""
        return {
            "ate": self.true_ate,
            "att": self.true_att,
            "cate": self.true_cate,
        }


@dataclass
class LaLondeDataset(BenchmarkDataset):
    """LaLonde (1986) National Supported Work dataset.

    Classic benchmark for evaluating treatment effect estimators.
    Treatment: Participation in NSW job training program
    Outcome: Earnings in 1978
    Known experimental ATT: ~$1,794 (from RCT)

    This implements:
    1. The original experimental (RCT) data
    2. The observational variant with CPS/PSID comparison groups
    """

    name: str = "lalonde"
    description: str = "LaLonde NSW Job Training - Classic causal benchmark"
    treatment_col: str = "treat"
    outcome_col: str = "re78"
    true_att: float | None = 1794.0  # Experimental ATT estimate

    # Dataset variant
    variant: str = "experimental"  # "experimental", "cps", "psid"

    def generate(self, n_samples: int | None = None, seed: int = 42) -> pd.DataFrame:
        """Generate LaLonde dataset.

        Since we may not have access to the original data files, this
        generates synthetic data matching the known LaLonde distribution.
        """
        np.random.seed(seed)

        # LaLonde data characteristics (from original paper)
        n_treat = 185  # Treatment group size
        n_control = 260 if self.variant == "experimental" else 2490

        total_n = n_treat + n_control
        if n_samples:
            scale = n_samples / total_n
            n_treat = max(20, int(n_treat * scale))
            n_control = max(30, int(n_control * scale))

        # Generate treatment group
        treat_data = self._generate_group(n_treat, treated=True, seed=seed)

        # Generate control group
        if self.variant == "experimental":
            control_data = self._generate_group(n_control, treated=False, seed=seed+1)
        else:
            # Observational control (different distribution - selection bias)
            control_data = self._generate_observational_control(
                n_control, seed=seed+1
            )

        # Combine
        df = pd.concat([treat_data, control_data], ignore_index=True)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        return df

    def _generate_group(
        self, n: int, treated: bool, seed: int
    ) -> pd.DataFrame:
        """Generate data for treatment or control group."""
        np.random.seed(seed)

        # Covariates matching LaLonde demographics
        age = np.clip(np.random.normal(25, 7, n), 17, 55).astype(int)
        education = np.clip(np.random.normal(10, 2, n), 0, 16).astype(int)
        black = np.random.binomial(1, 0.83, n)  # ~83% Black in NSW
        hispanic = np.random.binomial(1, 0.10, n) * (1 - black)  # ~10% Hispanic
        married = np.random.binomial(1, 0.17, n)  # ~17% married
        nodegree = (education < 12).astype(int)

        # Pre-treatment earnings (1974, 1975)
        re74 = np.maximum(0, np.random.exponential(3000, n) * (1 - 0.7 * np.random.binomial(1, 0.3, n)))
        re75 = np.maximum(0, np.random.exponential(3500, n) * (1 - 0.6 * np.random.binomial(1, 0.25, n)))

        # Post-treatment earnings (1978)
        # Base earnings influenced by covariates
        base_earnings = (
            2000
            + 200 * age
            - 5 * age ** 2
            + 500 * education
            + 1000 * married
            - 500 * nodegree
            + 0.3 * re75
            + np.random.normal(0, 3000, n)
        )

        # Treatment effect (heterogeneous)
        treatment_effect = 0
        if treated:
            # ATT around $1,794 with heterogeneity
            treatment_effect = (
                1794
                + 500 * nodegree  # Larger effect for those without degree
                - 100 * (age - 25)  # Smaller effect for older workers
                + np.random.normal(0, 500, n)
            )

        re78 = np.maximum(0, base_earnings + treatment_effect)

        return pd.DataFrame({
            "treat": int(treated),
            "age": age,
            "education": education,
            "black": black,
            "hispanic": hispanic,
            "married": married,
            "nodegree": nodegree,
            "re74": re74,
            "re75": re75,
            "re78": re78,
        })

    def _generate_observational_control(
        self, n: int, seed: int
    ) -> pd.DataFrame:
        """Generate observational control with selection bias."""
        np.random.seed(seed)

        # Observational control has different distribution (higher SES)
        age = np.clip(np.random.normal(33, 11, n), 17, 55).astype(int)
        education = np.clip(np.random.normal(12, 3, n), 0, 18).astype(int)
        black = np.random.binomial(1, 0.25, n)  # Less Black in CPS/PSID
        hispanic = np.random.binomial(1, 0.03, n)
        married = np.random.binomial(1, 0.70, n)  # More married
        nodegree = (education < 12).astype(int)

        # Higher pre-treatment earnings
        re74 = np.maximum(0, np.random.exponential(12000, n))
        re75 = np.maximum(0, np.random.exponential(13000, n))

        # Post-treatment earnings (no treatment effect)
        base_earnings = (
            5000
            + 300 * age
            - 6 * age ** 2
            + 800 * education
            + 3000 * married
            - 1000 * nodegree
            + 0.35 * re75
            + np.random.normal(0, 5000, n)
        )

        re78 = np.maximum(0, base_earnings)

        return pd.DataFrame({
            "treat": 0,
            "age": age,
            "education": education,
            "black": black,
            "hispanic": hispanic,
            "married": married,
            "nodegree": nodegree,
            "re74": re74,
            "re75": re75,
            "re78": re78,
        })


@dataclass
class IHDPDataset(BenchmarkDataset):
    """Infant Health and Development Program semi-synthetic dataset.

    Based on Hill (2011). Uses real covariates from the IHDP study but
    generates synthetic outcomes with known treatment effects.

    Treatment: High-quality child care and home visits
    Outcome: Cognitive test scores
    Ground truth: Known CATE from data generating process
    """

    name: str = "ihdp"
    description: str = "IHDP Semi-synthetic - Known heterogeneous effects"
    treatment_col: str = "treatment"
    outcome_col: str = "y"
    true_ate: float | None = 4.0  # Approx from standard simulation

    # Response surface
    response_type: str = "A"  # "A" or "B" from original paper

    def generate(self, n_samples: int | None = None, seed: int = 42) -> pd.DataFrame:
        """Generate IHDP semi-synthetic dataset.

        Simulates data matching the IHDP structure with known treatment effects.
        """
        np.random.seed(seed)

        n = n_samples or 747  # Original IHDP size

        # Generate covariates
        covariates = self._generate_covariates(n, seed)

        # Generate treatment assignment (with selection on X)
        propensity = self._compute_propensity(covariates)
        treatment = np.random.binomial(1, propensity)

        # Generate potential outcomes
        y0 = self._generate_y0(covariates, seed)
        y1 = self._generate_y1(covariates, y0, seed)

        # Observed outcome
        y = treatment * y1 + (1 - treatment) * y0

        # Store true CATE for evaluation
        self.true_cate = y1 - y0
        self.true_ate = float(np.mean(self.true_cate))
        self.true_att = float(np.mean(self.true_cate[treatment == 1]))

        # Combine into dataframe
        df = pd.DataFrame(covariates)
        df["treatment"] = treatment
        df["y"] = y
        df["y0_true"] = y0  # For evaluation
        df["y1_true"] = y1  # For evaluation
        df["cate_true"] = y1 - y0  # For evaluation

        return df

    def _generate_covariates(self, n: int, seed: int) -> dict[str, np.ndarray]:
        """Generate IHDP-like covariates."""
        np.random.seed(seed)

        return {
            # Continuous
            "bw": np.clip(np.random.normal(2750, 650, n), 1000, 5000),  # Birth weight
            "b.head": np.clip(np.random.normal(33, 2, n), 25, 40),  # Head circumference
            "preterm": np.clip(np.random.normal(37, 3, n), 24, 42),  # Gestational age
            "birth.o": np.random.choice([1, 2, 3, 4], n, p=[0.4, 0.35, 0.15, 0.1]),  # Birth order
            "nnhealth": np.clip(np.random.normal(3.5, 0.8, n), 1, 5),  # Neonatal health
            "momage": np.clip(np.random.normal(23, 6, n), 15, 45),  # Mother's age
            # Binary
            "sex": np.random.binomial(1, 0.5, n),  # Male
            "twin": np.random.binomial(1, 0.03, n),  # Twin birth
            "b.marr": np.random.binomial(1, 0.4, n),  # Mother married
            "mom.lths": np.random.binomial(1, 0.3, n),  # Mom < high school
            "mom.hs": np.random.binomial(1, 0.35, n),  # Mom high school
            "mom.scoll": np.random.binomial(1, 0.35, n),  # Mom some college
            "cig": np.random.binomial(1, 0.2, n),  # Mother smoked
            "first": np.random.binomial(1, 0.4, n),  # First born
            "booze": np.random.binomial(1, 0.1, n),  # Mother drank alcohol
            "drugs": np.random.binomial(1, 0.05, n),  # Mother used drugs
            "work.dur": np.random.binomial(1, 0.6, n),  # Mother worked
            "prenatal": np.random.binomial(1, 0.8, n),  # Prenatal care
            "ark": np.random.binomial(1, 0.1, n),  # Arkansas site
            "ein": np.random.binomial(1, 0.1, n),  # Einstein site
            "har": np.random.binomial(1, 0.1, n),  # Harvard site
            "mit": np.random.binomial(1, 0.1, n),  # MIT site
            "pen": np.random.binomial(1, 0.1, n),  # Penn site
            "tex": np.random.binomial(1, 0.1, n),  # Texas site
            "was": np.random.binomial(1, 0.1, n),  # Washington site
            "yal": np.random.binomial(1, 0.3, n),  # Yale site
        }

    def _compute_propensity(self, X: dict[str, np.ndarray]) -> np.ndarray:
        """Compute propensity scores with selection bias."""
        # Selection based on health and SES
        logit = (
            -1.5
            + 0.5 * (X["bw"] - 2750) / 650
            + 0.3 * X["b.marr"]
            + 0.2 * X["mom.scoll"]
            - 0.3 * X["mom.lths"]
            + 0.2 * (X["nnhealth"] - 3.5) / 0.8
            - 0.2 * X["cig"]
        )
        return 1 / (1 + np.exp(-logit))

    def _generate_y0(self, X: dict[str, np.ndarray], seed: int) -> np.ndarray:
        """Generate control potential outcomes."""
        np.random.seed(seed)

        n = len(X["bw"])

        if self.response_type == "A":
            # Linear response surface
            y0 = (
                80
                + 2 * (X["bw"] - 2750) / 650
                + 1 * X["b.head"]
                + 0.5 * X["preterm"]
                + 3 * X["b.marr"]
                + 2 * X["mom.scoll"]
                - 3 * X["mom.lths"]
                + 1.5 * X["nnhealth"]
                - 2 * X["cig"]
                + np.random.normal(0, 1, n)
            )
        else:
            # Nonlinear response surface (Type B)
            y0 = (
                80
                + np.exp(0.5 * (X["bw"] - 2750) / 650)
                + np.sin(X["b.head"] / 10)
                + 3 * X["b.marr"] * X["mom.scoll"]
                - 3 * X["mom.lths"]
                + 1.5 * X["nnhealth"] ** 2
                - 2 * X["cig"]
                + np.random.normal(0, 1, n)
            )

        return y0

    def _generate_y1(
        self, X: dict[str, np.ndarray], y0: np.ndarray, seed: int
    ) -> np.ndarray:
        """Generate treated potential outcomes with heterogeneous effects."""
        np.random.seed(seed + 1)

        n = len(X["bw"])

        # Heterogeneous treatment effect
        tau = (
            4  # Base effect
            + 2 * X["mom.lths"]  # Larger effect for disadvantaged
            - 1 * X["mom.scoll"]  # Smaller effect for higher SES
            + 1 * (X["bw"] < 2500).astype(float)  # Larger for low birth weight
            + np.random.normal(0, 0.5, n)
        )

        return y0 + tau


@dataclass
class TwinsDataset(BenchmarkDataset):
    """Twins dataset for mortality analysis.

    Based on Louizos et al. (2017). Uses twin births data where
    treatment is defined as being the heavier twin, and outcome
    is one-year mortality.

    This creates a natural experiment because twin pairs share
    genetics and prenatal environment.
    """

    name: str = "twins"
    description: str = "Twins Birth Weight - Natural experiment"
    treatment_col: str = "heavier"
    outcome_col: str = "mortality"
    true_ate: float | None = -0.025  # Being heavier reduces mortality ~2.5%

    def generate(self, n_samples: int | None = None, seed: int = 42) -> pd.DataFrame:
        """Generate Twins dataset.

        Simulates twin pair data with realistic birth weight and
        mortality relationships.
        """
        np.random.seed(seed)

        n_pairs = (n_samples or 5000) // 2

        # Generate twin pairs
        data = self._generate_twin_pairs(n_pairs, seed)

        # Store ground truth
        self.true_ate = float(data["cate_true"].mean())
        self.true_cate = data["cate_true"].values

        return data

    def _generate_twin_pairs(self, n_pairs: int, seed: int) -> pd.DataFrame:
        """Generate twin pair data."""
        np.random.seed(seed)

        records = []

        for pair_id in range(n_pairs):
            # Shared characteristics within pair
            gestation = np.clip(np.random.normal(37, 3), 24, 42)
            prenatal_care = np.random.binomial(1, 0.85)
            mom_age = np.clip(np.random.normal(28, 6), 15, 45)
            mom_education = np.random.choice([1, 2, 3, 4], p=[0.15, 0.35, 0.3, 0.2])
            complications = np.random.binomial(1, 0.15)

            # Birth weights (correlated within pair)
            mean_bw = np.clip(np.random.normal(2800, 500), 1000, 4500)
            bw_diff = np.abs(np.random.normal(0, 200))

            bw1 = mean_bw + bw_diff / 2
            bw2 = mean_bw - bw_diff / 2

            # Determine which is heavier (treatment)
            heavier1 = 1
            heavier2 = 0

            # Mortality probabilities
            base_mort1 = self._mortality_prob(bw1, gestation, complications)
            base_mort2 = self._mortality_prob(bw2, gestation, complications)

            # Add treatment effect (being heavier is protective)
            # True CATE is reduction in mortality
            treatment_effect = -0.025 * (1 + 0.5 * (bw1 < 2500))  # Larger effect for VLBW

            mort_prob1 = np.clip(base_mort1 + treatment_effect, 0, 1)
            mort_prob2 = base_mort2

            # Realize outcomes
            mortality1 = np.random.binomial(1, mort_prob1)
            mortality2 = np.random.binomial(1, mort_prob2)

            # True CATE for this pair (if twin 2 had been heavier instead)
            cate1 = mort_prob1 - base_mort1
            cate2 = self._mortality_prob(bw2, gestation, complications) + treatment_effect - base_mort2

            for twin_idx, (bw, heavier, mort, cate) in enumerate([
                (bw1, heavier1, mortality1, cate1),
                (bw2, heavier2, mortality2, cate2),
            ]):
                records.append({
                    "pair_id": pair_id,
                    "twin_id": twin_idx,
                    "birth_weight": bw,
                    "gestation": gestation,
                    "prenatal_care": prenatal_care,
                    "mom_age": mom_age,
                    "mom_education": mom_education,
                    "complications": complications,
                    "heavier": heavier,
                    "mortality": mort,
                    "cate_true": cate,
                })

        return pd.DataFrame(records)

    def _mortality_prob(
        self,
        birth_weight: float,
        gestation: float,
        complications: int,
    ) -> float:
        """Compute base mortality probability."""
        # Mortality inversely related to birth weight and gestation
        base = 0.02  # 2% base mortality

        # Birth weight effect (exponential increase below 2500g)
        if birth_weight < 1500:
            base += 0.15
        elif birth_weight < 2000:
            base += 0.08
        elif birth_weight < 2500:
            base += 0.03

        # Gestation effect
        if gestation < 32:
            base += 0.10
        elif gestation < 37:
            base += 0.03

        # Complications
        if complications:
            base += 0.05

        return np.clip(base, 0, 1)


@dataclass
class SyntheticBenchmark(BenchmarkDataset):
    """Fully synthetic benchmark with complete ground truth.

    Allows testing with:
    - Known ATE, ATT, CATE
    - Controlled confounding
    - Known treatment effect heterogeneity
    - Controlled selection bias
    """

    name: str = "synthetic"
    description: str = "Synthetic Benchmark - Full ground truth control"
    treatment_col: str = "T"
    outcome_col: str = "Y"

    # Configuration
    n_confounders: int = 5
    treatment_effect: float = 2.0
    effect_heterogeneity: float = 1.0
    selection_bias: float = 0.5
    noise_level: float = 1.0
    nonlinear: bool = False

    def generate(self, n_samples: int | None = None, seed: int = 42) -> pd.DataFrame:
        """Generate synthetic benchmark dataset."""
        np.random.seed(seed)

        n = n_samples or 1000

        # Generate confounders
        X = np.random.normal(0, 1, (n, self.n_confounders))
        X_df = pd.DataFrame(X, columns=[f"X{i}" for i in range(self.n_confounders)])

        # Propensity score (with confounding)
        propensity_logit = (
            self.selection_bias * X[:, 0]
            + 0.3 * X[:, 1]
            - 0.2 * X[:, 2]
        )
        propensity = 1 / (1 + np.exp(-propensity_logit))
        T = np.random.binomial(1, propensity)

        # Treatment effect (heterogeneous)
        if self.nonlinear:
            tau = (
                self.treatment_effect
                + self.effect_heterogeneity * np.sin(X[:, 0])
                + 0.5 * self.effect_heterogeneity * X[:, 1] ** 2
            )
        else:
            tau = (
                self.treatment_effect
                + self.effect_heterogeneity * X[:, 0]
                + 0.5 * self.effect_heterogeneity * X[:, 1]
            )

        # Potential outcomes
        if self.nonlinear:
            Y0 = (
                2 * X[:, 0]
                + X[:, 1] ** 2
                - 1.5 * X[:, 2]
                + np.sin(X[:, 3])
                + np.random.normal(0, self.noise_level, n)
            )
        else:
            Y0 = (
                2 * X[:, 0]
                + 1.5 * X[:, 1]
                - 1 * X[:, 2]
                + 0.5 * X[:, 3]
                + np.random.normal(0, self.noise_level, n)
            )

        Y1 = Y0 + tau

        # Observed outcome
        Y = T * Y1 + (1 - T) * Y0

        # Store ground truth
        self.true_cate = tau
        self.true_ate = float(np.mean(tau))
        self.true_att = float(np.mean(tau[T == 1]))

        # Build dataframe
        df = X_df.copy()
        df["T"] = T
        df["Y"] = Y
        df["Y0_true"] = Y0
        df["Y1_true"] = Y1
        df["cate_true"] = tau
        df["propensity_true"] = propensity

        return df


# Registry of all benchmarks
BENCHMARK_REGISTRY: dict[str, type[BenchmarkDataset]] = {
    "lalonde": LaLondeDataset,
    "lalonde_experimental": LaLondeDataset,
    "lalonde_cps": lambda: LaLondeDataset(variant="cps"),
    "lalonde_psid": lambda: LaLondeDataset(variant="psid"),
    "ihdp": IHDPDataset,
    "ihdp_a": IHDPDataset,
    "ihdp_b": lambda: IHDPDataset(response_type="B"),
    "twins": TwinsDataset,
    "synthetic": SyntheticBenchmark,
    "synthetic_nonlinear": lambda: SyntheticBenchmark(nonlinear=True),
}


def get_benchmark(name: str) -> BenchmarkDataset:
    """Get a benchmark dataset by name.

    Args:
        name: Benchmark name (e.g., "lalonde", "ihdp", "twins", "synthetic")

    Returns:
        BenchmarkDataset instance
    """
    if name not in BENCHMARK_REGISTRY:
        available = list(BENCHMARK_REGISTRY.keys())
        raise ValueError(f"Unknown benchmark: {name}. Available: {available}")

    factory = BENCHMARK_REGISTRY[name]
    if callable(factory) and not isinstance(factory, type):
        return factory()
    return factory()


def get_all_benchmarks() -> list[BenchmarkDataset]:
    """Get all standard benchmarks for comprehensive evaluation."""
    return [
        LaLondeDataset(),
        LaLondeDataset(variant="cps"),
        IHDPDataset(response_type="A"),
        IHDPDataset(response_type="B"),
        TwinsDataset(),
        SyntheticBenchmark(),
        SyntheticBenchmark(nonlinear=True),
    ]
