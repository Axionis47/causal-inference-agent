"""Effect Estimator Engine - Unified interface for all causal methods."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..methods import (
    AIPWMethod,
    CausalForestMethod,
    DifferenceInDifferencesMethod,
    DoubleMLMethod,
    InstrumentalVariablesMethod,
    IPWMethod,
    MethodResult,
    OLSMethod,
    PSMMethod,
    RegressionDiscontinuityMethod,
    SLearner,
    TLearner,
    XLearner,
)

if TYPE_CHECKING:
    from src.agents.base.state import AnalysisState

from src.agents.base import TreatmentEffectResult

logger = logging.getLogger(__name__)


class EffectEstimatorEngine:
    """Unified engine for running causal effect estimation methods.

    Primary entry points:
    - run_method(): Low-level single method execution
    - run_method_safe(): Agent-facing wrapper with data prep, gating, and error handling
    """

    # Available methods and their classes
    METHODS = {
        "ols": OLSMethod,
        "psm": PSMMethod,
        "ipw": IPWMethod,
        "aipw": AIPWMethod,
        "did": DifferenceInDifferencesMethod,
        "iv": InstrumentalVariablesMethod,
        "rdd": RegressionDiscontinuityMethod,
        "s_learner": SLearner,
        "t_learner": TLearner,
        "x_learner": XLearner,
        "causal_forest": CausalForestMethod,
        "double_ml": DoubleMLMethod,
    }

    # Method requirements
    METHOD_REQUIREMENTS = {
        "ols": {"min_samples": 30, "requires_covariates": False},
        "psm": {"min_samples": 100, "requires_covariates": True},
        "ipw": {"min_samples": 100, "requires_covariates": True},
        "aipw": {"min_samples": 100, "requires_covariates": True},
        "did": {"min_samples": 50, "requires_panel": True},
        "iv": {"min_samples": 100, "requires_instruments": True},
        "rdd": {"min_samples": 100, "requires_running_var": True},
        "s_learner": {"min_samples": 200, "requires_covariates": True},
        "t_learner": {"min_samples": 200, "requires_covariates": True},
        "x_learner": {"min_samples": 200, "requires_covariates": True},
        "causal_forest": {"min_samples": 500, "requires_covariates": True},
        "double_ml": {"min_samples": 200, "requires_covariates": True},
    }

    def __init__(self, confidence_level: float = 0.95):
        """Initialize the engine.

        Args:
            confidence_level: Confidence level for all methods
        """
        self.confidence_level = confidence_level
        self._results: list[MethodResult] = []

    def run_method(
        self,
        method_name: str,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariates: list[str] | None = None,
        **kwargs: Any,
    ) -> MethodResult:
        """Run a single causal estimation method.

        Args:
            method_name: Name of method to run
            df: DataFrame with data
            treatment_col: Treatment variable name
            outcome_col: Outcome variable name
            covariates: Covariate column names
            **kwargs: Method-specific arguments

        Returns:
            MethodResult with estimates
        """
        method_name = method_name.lower().replace("-", "_").replace(" ", "_")

        if method_name not in self.METHODS:
            raise ValueError(f"Unknown method: {method_name}. Available: {list(self.METHODS.keys())}")

        # Get method class
        method_class = self.METHODS[method_name]

        # Initialize and fit
        method = method_class(confidence_level=self.confidence_level)
        # MED3: Pass shared binarization threshold if provided
        binarize_threshold = kwargs.pop("binarize_threshold", None)
        if binarize_threshold is not None:
            method._binarize_threshold = binarize_threshold
        method.fit(df, treatment_col, outcome_col, covariates, **kwargs)

        # Get result
        result = method.estimate()
        self._results.append(result)

        return result

    def run_method_safe(
        self,
        method: str,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariates: list[str],
        current_state: AnalysisState | None = None,
    ) -> TreatmentEffectResult | None:
        """Run a method with data prep, sample-size gating, and safe error handling.

        This is the primary entry point for agents. It:
        1. Filters covariates to numeric columns and drops NaN rows
        2. Binarizes continuous treatments via median split
        3. Checks minimum sample size requirements
        4. Extracts state-dependent kwargs for DID/IV/RDD
        5. Delegates to the registered BaseCausalMethod class
        6. Converts MethodResult to TreatmentEffectResult

        Returns None if the method is inapplicable, data is insufficient,
        or an error occurs during estimation.
        """
        method = method.lower().replace("-", "_").replace(" ", "_")

        # Alias: agent uses "matching", engine uses "psm"
        if method == "matching":
            method = "psm"

        if method not in self.METHODS:
            logger.warning("unknown_method", extra={"method": method})
            return None

        try:
            # --- Apply profiler-guided treatment encoding ---
            if current_state and getattr(current_state, "treatment_encoding", None):
                enc = current_state.treatment_encoding
                if (
                    enc.strategy in ("collapse_to_binary", "label_encode")
                    and enc.value_mapping
                    and treatment_col in df.columns
                    and df[treatment_col].dtype == object
                ):
                    df = df.copy()
                    df[treatment_col] = df[treatment_col].map(enc.value_mapping)
                    df = df.dropna(subset=[treatment_col])
                    df[treatment_col] = df[treatment_col].astype(int)
                    logger.info(
                        "applied_treatment_encoding",
                        extra={
                            "strategy": enc.strategy,
                            "mapping": enc.value_mapping,
                            "n_after": len(df),
                        },
                    )

            # --- Data prep (mirrors the logic from the active agent) ---
            numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
            clean_covariates = [
                c for c in covariates if c in df.columns and c in numeric_cols
            ]

            all_cols = [treatment_col, outcome_col] + clean_covariates
            # Deduplicate while preserving order
            seen: set[str] = set()
            unique_cols: list[str] = []
            for col in all_cols:
                if col not in seen:
                    seen.add(col)
                    unique_cols.append(col)
            df_clean = df[unique_cols].dropna()

            # L7: Warn when a significant fraction of data is dropped
            if len(df) > 0:
                drop_pct = 1 - len(df_clean) / len(df)
                if drop_pct > 0.15:
                    logger.warning(
                        "high_data_loss_after_dropna",
                        extra={
                            "method": method,
                            "drop_pct": f"{drop_pct:.1%}",
                            "original_n": len(df),
                            "clean_n": len(df_clean),
                        },
                    )

            if len(df_clean) < 50:
                return None

            # --- Sample-size gating ---
            reqs = self.METHOD_REQUIREMENTS.get(method, {})
            min_samples = reqs.get("min_samples", 0)
            if len(df_clean) < min_samples:
                return None

            if reqs.get("requires_covariates") and not clean_covariates:
                return None

            # --- State-dependent kwargs for DID / IV / RDD ---
            kwargs: dict[str, Any] = {}
            profile = (
                current_state.data_profile
                if current_state and hasattr(current_state, "data_profile")
                else None
            )

            # MED3: Pass shared binarization threshold if available
            if current_state and getattr(current_state, "treatment_binarization_threshold", None) is not None:
                kwargs["binarize_threshold"] = current_state.treatment_binarization_threshold

            if method == "did":
                if not profile or not getattr(profile, "has_time_dimension", False):
                    return None
                time_col = getattr(profile, "time_column", None)
                if not time_col or time_col not in df_clean.columns:
                    return None
                kwargs["time_col"] = time_col

            elif method == "iv":
                instruments = (
                    getattr(profile, "potential_instruments", None) if profile else None
                )
                if not instruments:
                    return None
                available = [i for i in instruments if i in df_clean.columns]
                if not available:
                    return None
                kwargs["instruments"] = available

            elif method == "rdd":
                candidates = (
                    getattr(profile, "discontinuity_candidates", None)
                    if profile
                    else None
                )
                if not candidates:
                    return None
                run_col = candidates[0]
                if run_col not in df_clean.columns:
                    return None
                kwargs["running_var"] = run_col

            # --- Run the registered method ---
            result = self.run_method(
                method_name=method,
                df=df_clean,
                treatment_col=treatment_col,
                outcome_col=outcome_col,
                covariates=clean_covariates if clean_covariates else None,
                **kwargs,
            )

            # Sanity: filter estimates > 100× outcome std dev (numerical blow-ups)
            outcome_std = df_clean[outcome_col].std()
            if outcome_std > 0 and abs(result.estimate) > 100 * outcome_std:
                logger.warning(
                    "extreme_estimate_filtered",
                    extra={
                        "method": method,
                        "estimate": result.estimate,
                        "outcome_std": outcome_std,
                        "ratio": abs(result.estimate) / outcome_std,
                    },
                )
                return None

            return to_treatment_effect_result(result)

        except Exception:
            logger.exception(
                "method_failed", extra={"method": method}
            )
            return None


def to_treatment_effect_result(result: MethodResult) -> TreatmentEffectResult:
    """Convert a MethodResult to the pipeline's TreatmentEffectResult.

    Merges diagnostics, n_treated, and n_control into the details dict
    so no information is lost.
    """
    merged_details = dict(result.details)
    merged_details["n_treated"] = result.n_treated
    merged_details["n_control"] = result.n_control
    if result.diagnostics:
        merged_details["diagnostics"] = result.diagnostics

    return TreatmentEffectResult(
        method=result.method,
        estimand=result.estimand,
        estimate=result.estimate,
        std_error=result.std_error,
        ci_lower=result.ci_lower,
        ci_upper=result.ci_upper,
        p_value=result.p_value,
        assumptions_tested=result.assumptions_tested,
        details=merged_details,
    )
