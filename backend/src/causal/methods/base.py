"""Base class for causal inference methods."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MethodResult:
    """Result from a causal inference method."""

    method: str
    estimand: str  # ATE, ATT, CATE, LATE
    estimate: float
    std_error: float
    ci_lower: float
    ci_upper: float
    p_value: float | None = None
    n_treated: int = 0
    n_control: int = 0
    assumptions_tested: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method,
            "estimand": self.estimand,
            "estimate": self.estimate,
            "std_error": self.std_error,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "p_value": self.p_value,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "assumptions_tested": self.assumptions_tested,
            "diagnostics": self.diagnostics,
            "details": self.details,
        }


class BaseCausalMethod(ABC):
    """Abstract base class for all causal inference methods.

    Each method implements:
    1. fit() - Fit the model to data
    2. estimate() - Compute treatment effect estimate
    3. validate_assumptions() - Check method assumptions
    """

    METHOD_NAME: str = "base"
    ESTIMAND: str = "ATE"  # ATE, ATT, CATE, LATE
    REQUIRED_COLUMNS: list[str] = []

    def __init__(self, confidence_level: float = 0.95):
        """Initialize the method.

        Args:
            confidence_level: Confidence level for intervals (default 0.95)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self._fitted = False
        self._result: MethodResult | None = None
        self._binarize_threshold: float | None = None  # MED3: shared threshold across methods

    @abstractmethod
    def fit(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariates: list[str] | None = None,
        **kwargs: Any,
    ) -> "BaseCausalMethod":
        """Fit the causal model.

        Args:
            df: DataFrame with data
            treatment_col: Name of treatment column
            outcome_col: Name of outcome column
            covariates: List of covariate column names
            **kwargs: Method-specific arguments

        Returns:
            Self for chaining
        """
        pass

    @abstractmethod
    def estimate(self) -> MethodResult:
        """Compute the treatment effect estimate.

        Returns:
            MethodResult with estimate and statistics
        """
        pass

    def validate_assumptions(self, df: pd.DataFrame, treatment_col: str, outcome_col: str) -> list[str]:
        """Validate method assumptions.

        Args:
            df: DataFrame with data
            treatment_col: Treatment column name
            outcome_col: Outcome column name

        Returns:
            List of assumption violations (empty if all pass)
        """
        violations = []

        # Check for missing values
        if df[[treatment_col, outcome_col]].isnull().any().any():
            violations.append("Missing values in treatment or outcome")

        # Check sample size
        if len(df) < 30:
            violations.append(f"Small sample size: {len(df)} < 30")

        return violations

    def _compute_ci(
        self,
        estimate: float,
        std_error: float,
        n: int | None = None,
        k: int = 2,
    ) -> tuple[float, float]:
        """Compute confidence interval.

        Uses t-distribution critical values for small samples (n < 100)
        and z-distribution for large samples (n >= 100).

        Args:
            estimate: Point estimate
            std_error: Standard error
            n: Total sample size (if None, uses z-distribution)
            k: Number of estimated parameters (for degrees of freedom)

        Returns:
            Tuple of (lower, upper) bounds
        """
        from scipy import stats

        import numpy as np

        if np.isnan(std_error) or np.isinf(std_error):
            return (float('nan'), float('nan'))

        if n is not None and n < 100:
            df = max(n - k, 1)
            crit = stats.t.ppf(1 - self.alpha / 2, df)
        else:
            crit = stats.norm.ppf(1 - self.alpha / 2)
        return estimate - crit * std_error, estimate + crit * std_error

    def _compute_p_value(
        self,
        estimate: float,
        std_error: float,
        null_value: float = 0,
        n: int | None = None,
        k: int = 2,
    ) -> float:
        """Compute two-sided p-value.

        Uses t-distribution for small samples (n < 100) and
        z-distribution for large samples (n >= 100).

        Args:
            estimate: Point estimate
            std_error: Standard error
            null_value: Null hypothesis value
            n: Total sample size (if None, uses z-distribution)
            k: Number of estimated parameters (for degrees of freedom)

        Returns:
            P-value
        """
        from scipy import stats

        import numpy as np

        if np.isnan(std_error) or np.isinf(std_error) or std_error <= 0:
            return float('nan')
        t_stat = (estimate - null_value) / std_error

        if n is not None and n < 100:
            df = max(n - k, 1)
            return 2 * (1 - stats.t.cdf(abs(t_stat), df))
        else:
            return 2 * (1 - stats.norm.cdf(abs(t_stat)))

    def _binarize_treatment(
        self, treatment: pd.Series, threshold: float | None = None
    ) -> pd.Series:
        """Binarize treatment if continuous.

        Args:
            treatment: Treatment series
            threshold: Optional pre-computed threshold for consistent
                binarization across methods. If None, uses the median.

        Returns:
            Binary treatment series
        """
        # Fallback: handle string-typed treatments that weren't encoded by the profiler
        if treatment.dtype == object:
            unique_vals = treatment.unique()
            if len(unique_vals) == 2:
                mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                logger.warning(
                    "String treatment '%s' auto-encoded: %s",
                    treatment.name,
                    mapping,
                )
                return treatment.map(mapping).astype(int)
            else:
                raise TypeError(
                    f"Treatment '{treatment.name}' is categorical with {len(unique_vals)} levels "
                    f"({list(unique_vals[:5])}). The data profiler should have determined an encoding "
                    f"strategy. Consider re-running with profiler-guided encoding."
                )

        if treatment.nunique() > 2:
            if threshold is None:
                # MED3: Use shared threshold if available, else compute median
                threshold = self._binarize_threshold if self._binarize_threshold is not None else float(treatment.median())
            logger.warning(
                "Continuous treatment variable '%s' with %d unique values "
                "was median-split at threshold %.4f into binary treatment. "
                "This discards dose-response information and may bias estimates.",
                treatment.name,
                treatment.nunique(),
                threshold,
            )
            return (treatment > threshold).astype(int)
        return treatment.astype(int)

    def _validate_covariates(self, df: pd.DataFrame, covariates: list[str]) -> list[str]:
        """Filter covariates to only columns present in the dataframe.

        Args:
            df: DataFrame to check against
            covariates: List of covariate column names

        Returns:
            Filtered list containing only columns that exist in df
        """
        valid_covariates = [c for c in covariates if c in df.columns]
        if len(valid_covariates) < len(covariates):
            missing = set(covariates) - set(valid_covariates)
            logger.debug("covariates_not_in_dataframe", extra={"missing": list(missing)})
        return valid_covariates

    @property
    def result(self) -> MethodResult | None:
        """Get the result if fitted."""
        return self._result

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._fitted
