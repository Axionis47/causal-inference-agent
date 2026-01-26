"""EDA Agent - ReAct-based Exploratory Data Analysis for Causal Inference.

This agent performs EDA through the ReAct paradigm:
- Queries domain knowledge and data profile for context
- Iteratively investigates data quality issues
- Assesses readiness for causal inference

Key features:
- Pull-based context: Queries domain knowledge for treatment/outcome hints
- ReAct loop: Observe → Reason → Act until EDA is complete
- Focus on causal inference: Balance, confounding, data quality
"""

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import jarque_bera, normaltest, shapiro

from src.agents.base import (
    AnalysisState,
    EDAResult,
    JobStatus,
    ToolResult,
    ToolResultStatus,
)
from src.agents.base.context_tools import ContextTools
from src.agents.base.react_agent import ReActAgent
from src.logging_config.structured import get_logger

logger = get_logger(__name__)


class EDAAgent(ReActAgent, ContextTools):
    """ReAct-based EDA agent that investigates data quality for causal inference.

    This agent:
    1. Queries domain knowledge for treatment/outcome hints
    2. Analyzes variable distributions and data quality
    3. Checks covariate balance between treatment groups
    4. Assesses multicollinearity and outliers
    5. Finalizes with a data quality assessment

    Uses pull-based context - queries for information on demand.
    """

    AGENT_NAME = "eda_agent"
    MAX_STEPS = 15

    SYSTEM_PROMPT = """You are an expert data scientist performing exploratory data analysis
for a causal inference study.

GOAL: Assess data quality and readiness for causal effect estimation.

WORKFLOW:
1. Query domain knowledge for hints about treatment, outcome, confounders
2. Get data overview to understand the structure
3. Analyze treatment and outcome variable distributions
4. Check covariate balance between treatment groups
5. Detect outliers and assess data quality
6. Check for multicollinearity if needed
7. Finalize with your data quality assessment

KEY CONSIDERATIONS:
- Treatment variable: Is it binary? Well-separated? Balanced?
- Outcome variable: Distribution, outliers, transformations needed?
- Confounders: Balance between groups, multicollinearity
- Missing data: Patterns, relationship to treatment

Focus on findings that matter for causal inference. Be thorough but efficient."""

    def __init__(self) -> None:
        """Initialize the EDA agent."""
        super().__init__()

        # Register context query tools from mixin
        self.register_context_tools()

        # Internal state
        self._df: pd.DataFrame | None = None
        self._eda_result: EDAResult | None = None
        self._treatment_var: str | None = None
        self._outcome_var: str | None = None
        self._finalized: bool = False
        self._final_result: dict[str, Any] = {}

        # Track what we've analyzed
        self._analyzed_distributions: dict[str, dict] = {}
        self._outlier_results: dict[str, dict] = {}
        self._correlation_results: dict | None = None
        self._vif_results: dict[str, float] = {}
        self._balance_results: dict[str, dict] = {}
        self._missing_analysis: dict | None = None

        # Register EDA-specific tools
        self._register_eda_tools()

    def _register_eda_tools(self) -> None:
        """Register tools for EDA."""
        self.register_tool(
            name="get_data_overview",
            description="Get overview of the dataset including dimensions, variable types, missing data summary. Call this early to understand the data.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=self._tool_get_overview,
        )

        self.register_tool(
            name="analyze_variable",
            description="Analyze a specific variable's distribution - summary stats, skewness, normality tests. Use for treatment, outcome, or suspicious variables.",
            parameters={
                "type": "object",
                "properties": {
                    "variable": {
                        "type": "string",
                        "description": "Name of the variable to analyze",
                    },
                    "include_normality_tests": {
                        "type": "boolean",
                        "description": "Whether to run normality tests",
                    },
                },
                "required": ["variable"],
            },
            handler=self._tool_analyze_variable,
        )

        self.register_tool(
            name="detect_outliers",
            description="Detect outliers using IQR and Z-score methods. Returns outlier counts and percentages.",
            parameters={
                "type": "object",
                "properties": {
                    "variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Variables to check (empty for all numeric)",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["iqr", "zscore", "both"],
                        "description": "Outlier detection method",
                    },
                },
                "required": [],
            },
            handler=self._tool_detect_outliers,
        )

        self.register_tool(
            name="compute_correlations",
            description="Compute correlation matrix and identify high correlations. Use to check for multicollinearity.",
            parameters={
                "type": "object",
                "properties": {
                    "variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Variables to include (empty for all numeric)",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["pearson", "spearman"],
                        "description": "Correlation method",
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Threshold for flagging high correlations (default: 0.7)",
                    },
                },
                "required": [],
            },
            handler=self._tool_compute_correlations,
        )

        self.register_tool(
            name="compute_vif",
            description="Compute Variance Inflation Factor to assess multicollinearity severity. Use after finding high correlations.",
            parameters={
                "type": "object",
                "properties": {
                    "covariates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Covariates to check (empty for all numeric except treatment/outcome)",
                    },
                },
                "required": [],
            },
            handler=self._tool_compute_vif,
        )

        self.register_tool(
            name="check_covariate_balance",
            description="Check balance of covariates between treatment and control groups using Standardized Mean Difference (SMD). Critical for causal inference.",
            parameters={
                "type": "object",
                "properties": {
                    "covariates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Covariates to check (empty for all potential confounders)",
                    },
                },
                "required": [],
            },
            handler=self._tool_check_balance,
        )

        self.register_tool(
            name="check_missing_patterns",
            description="Analyze missing data patterns and whether missingness relates to treatment.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=self._tool_check_missing,
        )

        self.register_tool(
            name="finalize_eda",
            description="Finalize the EDA with your assessment. Call this ONLY after gathering sufficient evidence.",
            parameters={
                "type": "object",
                "properties": {
                    "data_quality_score": {
                        "type": "number",
                        "description": "Overall data quality score 0-100",
                    },
                    "key_findings": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Most important findings from EDA",
                    },
                    "data_quality_issues": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific data quality issues identified",
                    },
                    "recommendations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Recommendations for causal analysis",
                    },
                    "causal_readiness": {
                        "type": "string",
                        "enum": ["ready", "needs_attention", "significant_concerns"],
                        "description": "Overall readiness for causal inference",
                    },
                },
                "required": [
                    "data_quality_score",
                    "key_findings",
                    "data_quality_issues",
                    "recommendations",
                    "causal_readiness",
                ],
            },
            handler=self._tool_finalize,
        )

    def _get_initial_observation(self, state: AnalysisState) -> str:
        """Generate lean initial observation for EDA."""
        dataset_name = state.dataset_info.name or "dataset"
        n_samples = state.dataset_info.n_samples or "unknown"
        n_features = state.dataset_info.n_features or "unknown"

        obs = f"""Perform exploratory data analysis for causal inference.
Dataset: {dataset_name}
Size: {n_samples} samples x {n_features} features

Start by:
1. Querying domain knowledge for treatment/outcome hints
2. Getting data overview
3. Analyzing key variables (treatment, outcome)
4. Checking covariate balance

Finalize when you have assessed data quality for causal analysis."""

        return obs

    async def is_task_complete(self, state: AnalysisState) -> bool:
        """Check if EDA task is complete."""
        return self._finalized and state.eda_result is not None

    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Execute EDA through ReAct loop.

        Args:
            state: Current analysis state

        Returns:
            Updated state with EDA results
        """
        self.logger.info(
            "eda_agent_start",
            job_id=state.job_id,
            dataset=state.dataset_info.name or state.dataset_info.url,
        )

        state.status = JobStatus.EXPLORATORY_ANALYSIS
        start_time = time.time()

        try:
            # Load the dataset
            self._df = self._load_dataframe(state)
            if self._df is None:
                state.mark_failed("Failed to load dataset for EDA", self.AGENT_NAME)
                return state

            self._eda_result = EDAResult()

            # Resolve treatment/outcome
            self._treatment_var, self._outcome_var = state.get_primary_pair()

            # Reset tracking
            self._analyzed_distributions = {}
            self._outlier_results = {}
            self._correlation_results = None
            self._vif_results = {}
            self._balance_results = {}
            self._missing_analysis = None
            self._finalized = False

            # Store reference to state for context tools
            self._current_state = state

            # Run the ReAct loop
            state = await super().execute(state)

            # If not finalized via tool, auto-finalize
            if not self._finalized:
                self.logger.warning("eda_auto_finalize")
                self._final_result = self._auto_finalize()
                self._finalized = True

            # Populate EDA result
            self._populate_eda_result(self._final_result)

            # Update state
            state.eda_result = self._eda_result

            duration_ms = int((time.time() - start_time) * 1000)
            self.logger.info(
                "eda_complete",
                quality_score=self._eda_result.data_quality_score,
                issues_found=len(self._eda_result.data_quality_issues),
                duration_ms=duration_ms,
            )

        except Exception as e:
            self.logger.exception("eda_failed", error=str(e))
            state.mark_failed(f"EDA failed: {str(e)}", self.AGENT_NAME)

        return state

    def _load_dataframe(self, state: AnalysisState) -> pd.DataFrame | None:
        """Load DataFrame from pickle path."""
        if state.dataframe_path and Path(state.dataframe_path).exists():
            with open(state.dataframe_path, "rb") as f:
                return pickle.load(f)
        return None

    # =========================================================================
    # Tool Handlers
    # =========================================================================

    async def _tool_get_overview(self, state: AnalysisState) -> ToolResult:
        """Get overview of the dataset."""
        if self._df is None:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="Dataset not loaded",
            )

        df = self._df
        n_rows, n_cols = df.shape

        # Variable types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        binary_cols = [c for c in numeric_cols if df[c].dropna().nunique() == 2]

        # Missing data summary
        missing_counts = df.isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0]
        total_missing_pct = df.isnull().sum().sum() / (n_rows * n_cols) * 100

        # Treatment info
        treatment_info = None
        if self._treatment_var and self._treatment_var in df.columns:
            t_vals = df[self._treatment_var].value_counts().to_dict()
            treatment_info = {
                "variable": self._treatment_var,
                "values": {str(k): int(v) for k, v in t_vals.items()},
                "missing": int(df[self._treatment_var].isnull().sum()),
            }

        # Outcome info
        outcome_info = None
        if self._outcome_var and self._outcome_var in df.columns:
            outcome_info = {
                "variable": self._outcome_var,
                "mean": round(float(df[self._outcome_var].mean()), 3),
                "std": round(float(df[self._outcome_var].std()), 3),
                "missing": int(df[self._outcome_var].isnull().sum()),
            }

        # Confounders from profile
        confounders = []
        if state.data_profile and state.data_profile.potential_confounders:
            confounders = state.data_profile.potential_confounders[:10]

        overview = {
            "n_samples": n_rows,
            "n_features": n_cols,
            "numeric_columns": numeric_cols[:10],
            "categorical_columns": categorical_cols[:5],
            "binary_columns": binary_cols[:5],
            "total_missing_pct": round(total_missing_pct, 2),
            "columns_with_missing": len(cols_with_missing),
            "treatment": treatment_info,
            "outcome": outcome_info,
            "potential_confounders": confounders,
        }

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output=overview,
        )

    async def _tool_analyze_variable(
        self,
        state: AnalysisState,
        variable: str,
        include_normality_tests: bool = True,
    ) -> ToolResult:
        """Analyze a specific variable's distribution."""
        if self._df is None:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="Dataset not loaded",
            )

        if variable not in self._df.columns:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error=f"Variable '{variable}' not found",
            )

        data = self._df[variable].dropna()
        n = len(data)

        if n == 0:
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={"variable": variable, "error": "No non-missing values"},
            )

        result_dict = {
            "variable": variable,
            "n": n,
            "missing": int(len(self._df) - n),
        }

        if pd.api.types.is_numeric_dtype(self._df[variable]):
            result_dict["type"] = "numeric"
            result_dict["mean"] = round(float(data.mean()), 4)
            result_dict["median"] = round(float(data.median()), 4)
            result_dict["std"] = round(float(data.std()), 4)
            result_dict["min"] = round(float(data.min()), 4)
            result_dict["max"] = round(float(data.max()), 4)
            result_dict["q1"] = round(float(data.quantile(0.25)), 4)
            result_dict["q3"] = round(float(data.quantile(0.75)), 4)
            result_dict["skewness"] = round(float(stats.skew(data)), 3)
            result_dict["kurtosis"] = round(float(stats.kurtosis(data)), 3)
            result_dict["unique_values"] = int(data.nunique())

            # Interpret skewness
            skew = result_dict["skewness"]
            if abs(skew) > 1:
                result_dict["skewness_interpretation"] = "highly_skewed"
            elif abs(skew) > 0.5:
                result_dict["skewness_interpretation"] = "moderately_skewed"
            else:
                result_dict["skewness_interpretation"] = "approximately_symmetric"

            # Normality tests
            if include_normality_tests and n >= 8:
                normality = {}
                try:
                    if n <= 5000:
                        stat, p = shapiro(data.sample(min(n, 5000), random_state=42))
                        normality["shapiro_p"] = round(float(p), 4)
                        normality["shapiro_normal"] = p > 0.05
                    if n >= 20:
                        stat, p = normaltest(data)
                        normality["dagostino_p"] = round(float(p), 4)
                        normality["dagostino_normal"] = p > 0.05
                    stat, p = jarque_bera(data)
                    normality["jarque_bera_p"] = round(float(p), 4)
                    normality["jarque_bera_normal"] = p > 0.05
                except Exception:
                    pass
                if normality:
                    result_dict["normality_tests"] = normality

            # Store for later
            self._analyzed_distributions[variable] = result_dict

        else:
            # Categorical variable
            result_dict["type"] = "categorical"
            value_counts = data.value_counts().head(10).to_dict()
            result_dict["unique_values"] = int(data.nunique())
            result_dict["top_values"] = {str(k): int(v) for k, v in value_counts.items()}

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output=result_dict,
        )

    async def _tool_detect_outliers(
        self,
        state: AnalysisState,
        variables: list[str] | None = None,
        method: str = "both",
    ) -> ToolResult:
        """Detect outliers in specified variables."""
        if self._df is None:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="Dataset not loaded",
            )

        df = self._df

        # Default to all numeric
        if not variables:
            variables = df.select_dtypes(include=[np.number]).columns.tolist()
            if self._treatment_var in variables:
                variables.remove(self._treatment_var)

        results = []
        for var in variables:
            if var not in df.columns or not pd.api.types.is_numeric_dtype(df[var]):
                continue

            data = df[var].dropna()
            if len(data) < 10:
                continue

            outlier_info = {"variable": var, "n": len(data)}

            # IQR method
            if method in ["iqr", "both"]:
                q1 = data.quantile(0.25)
                q3 = data.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                iqr_outliers = ((data < lower) | (data > upper)).sum()
                outlier_info["iqr_outliers"] = int(iqr_outliers)
                outlier_info["iqr_pct"] = round(float(iqr_outliers / len(data) * 100), 1)

            # Z-score method
            if method in ["zscore", "both"]:
                z_scores = np.abs(stats.zscore(data))
                zscore_outliers = (z_scores > 3).sum()
                outlier_info["zscore_outliers"] = int(zscore_outliers)
                outlier_info["zscore_pct"] = round(float(zscore_outliers / len(data) * 100), 1)

            # Store if outliers found
            if outlier_info.get("iqr_outliers", 0) > 0 or outlier_info.get("zscore_outliers", 0) > 0:
                self._outlier_results[var] = outlier_info

            results.append(outlier_info)

        # Filter to those with outliers
        with_outliers = [r for r in results if r.get("iqr_outliers", 0) > 0 or r.get("zscore_outliers", 0) > 0]

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "method": method,
                "variables_checked": len(results),
                "variables_with_outliers": len(with_outliers),
                "outlier_details": with_outliers[:10],
            },
        )

    async def _tool_compute_correlations(
        self,
        state: AnalysisState,
        variables: list[str] | None = None,
        method: str = "pearson",
        threshold: float = 0.7,
    ) -> ToolResult:
        """Compute correlations and identify high correlations."""
        if self._df is None:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="Dataset not loaded",
            )

        df = self._df

        # Default to all numeric
        if not variables:
            variables = df.select_dtypes(include=[np.number]).columns.tolist()

        # Filter to valid columns
        variables = [v for v in variables if v in df.columns and pd.api.types.is_numeric_dtype(df[v])]

        if len(variables) < 2:
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={"error": "Need at least 2 numeric variables"},
            )

        # Compute correlation matrix
        corr_matrix = df[variables].corr(method=method)

        # Find high correlations
        high_corrs = []
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > threshold:
                        high_corrs.append({
                            "var1": col1,
                            "var2": col2,
                            "correlation": round(float(corr_val), 3),
                        })

        high_corrs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        # Store for later
        self._correlation_results = {
            "method": method,
            "n_variables": len(variables),
            "high_correlations": high_corrs,
        }

        # Store in EDA result
        self._eda_result.correlation_matrix = corr_matrix.to_dict()
        self._eda_result.high_correlations = high_corrs

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "method": method,
                "threshold": threshold,
                "n_variables": len(variables),
                "high_correlations_count": len(high_corrs),
                "high_correlations": high_corrs[:10],
            },
        )

    async def _tool_compute_vif(
        self,
        state: AnalysisState,
        covariates: list[str] | None = None,
    ) -> ToolResult:
        """Compute VIF for multicollinearity assessment."""
        if self._df is None:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="Dataset not loaded",
            )

        df = self._df

        # Default covariates
        if not covariates:
            covariates = df.select_dtypes(include=[np.number]).columns.tolist()
            for var in [self._treatment_var, self._outcome_var]:
                if var in covariates:
                    covariates.remove(var)

        # Filter to valid columns
        covariates = [c for c in covariates if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

        if len(covariates) < 2:
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={"error": "Need at least 2 covariates for VIF"},
            )

        # Prepare data
        X = df[covariates].dropna()
        if len(X) < 10:
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={"error": "Not enough complete cases for VIF"},
            )

        # Compute VIF
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor

            X_with_const = np.column_stack([np.ones(len(X)), X.values])

            vif_results = []
            warnings = []

            for i, col in enumerate(covariates):
                try:
                    vif = variance_inflation_factor(X_with_const, i + 1)
                    vif_results.append({"variable": col, "vif": round(float(vif), 2)})
                    self._vif_results[col] = float(vif)

                    if vif > 10:
                        warnings.append(f"SEVERE: {col} (VIF={vif:.1f})")
                    elif vif > 5:
                        warnings.append(f"MODERATE: {col} (VIF={vif:.1f})")
                except Exception:
                    pass

            # Store
            self._eda_result.vif_scores = self._vif_results
            self._eda_result.multicollinearity_warnings = warnings

            # Sort by VIF
            vif_results.sort(key=lambda x: x.get("vif", 0), reverse=True)

            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={
                    "n_covariates": len(covariates),
                    "severe_count": sum(1 for r in vif_results if r.get("vif", 0) > 10),
                    "moderate_count": sum(1 for r in vif_results if 5 < r.get("vif", 0) <= 10),
                    "top_vif": vif_results[:10],
                    "warnings": warnings,
                },
            )
        except ImportError:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="statsmodels not available for VIF calculation",
            )

    async def _tool_check_balance(
        self,
        state: AnalysisState,
        covariates: list[str] | None = None,
    ) -> ToolResult:
        """Check covariate balance between treatment groups."""
        if self._df is None:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="Dataset not loaded",
            )

        df = self._df
        treatment_col = self._treatment_var

        if not treatment_col or treatment_col not in df.columns:
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={"error": "Treatment variable not specified or not found"},
            )

        # Check binary treatment
        treatment_values = df[treatment_col].dropna().unique()
        if len(treatment_values) != 2:
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={"error": f"Treatment variable is not binary (found {len(treatment_values)} values)"},
            )

        # Get groups
        treated = df[df[treatment_col] == treatment_values.max()]
        control = df[df[treatment_col] == treatment_values.min()]

        # Default covariates
        if not covariates:
            if state.data_profile and state.data_profile.potential_confounders:
                covariates = state.data_profile.potential_confounders
            else:
                covariates = df.select_dtypes(include=[np.number]).columns.tolist()
                for var in [treatment_col, self._outcome_var]:
                    if var in covariates:
                        covariates.remove(var)

        # Check balance for each covariate
        results = []
        imbalanced = []

        for cov in covariates:
            if cov not in df.columns or not pd.api.types.is_numeric_dtype(df[cov]):
                continue

            treated_vals = treated[cov].dropna()
            control_vals = control[cov].dropna()

            if len(treated_vals) < 5 or len(control_vals) < 5:
                continue

            # Compute SMD
            mean_diff = treated_vals.mean() - control_vals.mean()
            pooled_std = np.sqrt((treated_vals.std() ** 2 + control_vals.std() ** 2) / 2)

            smd = abs(mean_diff / pooled_std) if pooled_std > 0 else 0.0

            balance_info = {
                "covariate": cov,
                "treated_mean": round(float(treated_vals.mean()), 3),
                "control_mean": round(float(control_vals.mean()), 3),
                "smd": round(float(smd), 3),
                "is_balanced": smd < 0.1,
            }

            results.append(balance_info)
            self._balance_results[cov] = balance_info

            if smd >= 0.1:
                imbalanced.append({"covariate": cov, "smd": round(smd, 3)})

        # Store in EDA result
        self._eda_result.covariate_balance = self._balance_results

        if imbalanced:
            imb_str = ", ".join([f"{i['covariate']} (SMD={i['smd']})" for i in imbalanced[:5]])
            self._eda_result.balance_summary = f"Imbalanced: {imb_str}"
        else:
            self._eda_result.balance_summary = "All covariates well-balanced (SMD < 0.1)"

        # Sort by SMD
        imbalanced.sort(key=lambda x: x["smd"], reverse=True)

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "treatment_variable": treatment_col,
                "n_treated": len(treated),
                "n_control": len(control),
                "n_covariates_checked": len(results),
                "n_imbalanced": len(imbalanced),
                "imbalanced_covariates": imbalanced[:10],
                "balance_summary": self._eda_result.balance_summary,
            },
        )

    async def _tool_check_missing(self, state: AnalysisState) -> ToolResult:
        """Analyze missing data patterns."""
        if self._df is None:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="Dataset not loaded",
            )

        df = self._df

        # Overall missing
        missing_counts = df.isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0]

        if len(cols_with_missing) == 0:
            self._missing_analysis = {"has_missing": False}
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={"has_missing": False, "message": "No missing values"},
            )

        n_rows = len(df)
        total_missing = df.isnull().sum().sum()
        total_cells = n_rows * len(df.columns)

        # Per-column analysis
        col_analysis = []
        for col in cols_with_missing.index:
            col_analysis.append({
                "column": col,
                "missing": int(cols_with_missing[col]),
                "pct": round(float(cols_with_missing[col] / n_rows * 100), 1),
            })

        col_analysis.sort(key=lambda x: x["missing"], reverse=True)

        # Check if missing is related to treatment
        treatment_col = self._treatment_var
        differential_missing = []
        if treatment_col and treatment_col in df.columns:
            for col in cols_with_missing.index[:10]:
                treated_missing = df[df[treatment_col] == 1][col].isnull().mean()
                control_missing = df[df[treatment_col] == 0][col].isnull().mean()
                if abs(treated_missing - control_missing) > 0.05:
                    differential_missing.append({
                        "column": col,
                        "treated_missing_pct": round(float(treated_missing * 100), 1),
                        "control_missing_pct": round(float(control_missing * 100), 1),
                    })

        self._missing_analysis = {
            "has_missing": True,
            "total_missing_pct": round(float(total_missing / total_cells * 100), 2),
            "n_cols_with_missing": len(cols_with_missing),
            "by_column": col_analysis,
            "differential_missing": differential_missing,
        }

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "has_missing": True,
                "total_missing_pct": self._missing_analysis["total_missing_pct"],
                "n_cols_with_missing": len(cols_with_missing),
                "top_missing_columns": col_analysis[:10],
                "differential_missing": differential_missing,
                "warning": "Differential missing may indicate selection bias!" if differential_missing else None,
            },
        )

    async def _tool_finalize(
        self,
        state: AnalysisState,
        data_quality_score: float,
        key_findings: list[str],
        data_quality_issues: list[str],
        recommendations: list[str],
        causal_readiness: str,
    ) -> ToolResult:
        """Finalize the EDA with assessment."""
        self.logger.info(
            "eda_finalizing",
            data_quality_score=data_quality_score,
            num_findings=len(key_findings),
            num_issues=len(data_quality_issues),
            causal_readiness=causal_readiness,
        )

        self._final_result = {
            "data_quality_score": data_quality_score,
            "key_findings": key_findings,
            "data_quality_issues": data_quality_issues,
            "recommendations": recommendations,
            "causal_readiness": causal_readiness,
        }
        self._finalized = True

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "eda_finalized": True,
                "data_quality_score": data_quality_score,
                "causal_readiness": causal_readiness,
                "n_findings": len(key_findings),
                "n_issues": len(data_quality_issues),
            },
        )

    def _auto_finalize(self) -> dict[str, Any]:
        """Auto-generate final result from collected evidence."""
        # Calculate quality score
        score = 100.0
        issues = []

        # Missing data penalty
        if self._missing_analysis and self._missing_analysis.get("has_missing"):
            missing_pct = self._missing_analysis.get("total_missing_pct", 0)
            if missing_pct > 30:
                score -= 30
                issues.append(f"High missing data: {missing_pct:.1f}%")
            elif missing_pct > 10:
                score -= 15
                issues.append(f"Moderate missing data: {missing_pct:.1f}%")
            elif missing_pct > 5:
                score -= 5
                issues.append(f"Some missing data: {missing_pct:.1f}%")

            if self._missing_analysis.get("differential_missing"):
                score -= 10
                issues.append("Missing data differs by treatment status")

        # Outlier penalty
        if self._outlier_results:
            outlier_cols = len(self._outlier_results)
            if outlier_cols > 5:
                score -= 15
                issues.append(f"Many columns with outliers: {outlier_cols}")
            elif outlier_cols > 2:
                score -= 8
                issues.append(f"Some columns with outliers: {outlier_cols}")

        # Multicollinearity penalty
        if self._vif_results:
            severe = sum(1 for v in self._vif_results.values() if v > 10)
            moderate = sum(1 for v in self._vif_results.values() if 5 < v <= 10)
            if severe > 0:
                score -= 15
                issues.append(f"Severe multicollinearity in {severe} variables")
            elif moderate > 0:
                score -= 8
                issues.append(f"Moderate multicollinearity in {moderate} variables")

        # Balance penalty
        if self._balance_results:
            imbalanced = sum(1 for v in self._balance_results.values() if not v.get("is_balanced", True))
            if imbalanced > 5:
                score -= 15
                issues.append(f"Many imbalanced covariates: {imbalanced}")
            elif imbalanced > 2:
                score -= 8
                issues.append(f"Some imbalanced covariates: {imbalanced}")
            elif imbalanced > 0:
                score -= 5
                issues.append(f"Imbalanced covariates: {imbalanced}")

        score = max(0, score)

        # Generate findings
        key_findings = []
        if score >= 80:
            key_findings.append("Data quality is good for causal analysis")
        elif score >= 60:
            key_findings.append("Data quality is moderate - some issues need attention")
        else:
            key_findings.append("Significant data quality concerns identified")

        key_findings.extend(issues[:5])

        # Generate recommendations
        recommendations = []
        if self._outlier_results:
            recommendations.append("Consider robust estimation methods or outlier treatment")
        if self._vif_results and any(v > 5 for v in self._vif_results.values()):
            recommendations.append("Address multicollinearity through variable selection or regularization")
        if self._balance_results and any(not v.get("is_balanced", True) for v in self._balance_results.values()):
            recommendations.append("Use propensity score methods to address covariate imbalance")
        if not recommendations:
            recommendations = ["Proceed with causal analysis", "Consider multiple estimation methods"]

        # Determine readiness
        if score >= 80:
            causal_readiness = "ready"
        elif score >= 50:
            causal_readiness = "needs_attention"
        else:
            causal_readiness = "significant_concerns"

        return {
            "data_quality_score": score,
            "key_findings": key_findings,
            "data_quality_issues": issues,
            "recommendations": recommendations,
            "causal_readiness": causal_readiness,
        }

    def _populate_eda_result(self, final_result: dict[str, Any]) -> None:
        """Populate EDA result from finalize output."""
        self._eda_result.data_quality_score = final_result.get("data_quality_score", 50.0)
        self._eda_result.data_quality_issues = final_result.get("data_quality_issues", [])
        self._eda_result.summary_table = {
            "key_findings": final_result.get("key_findings", []),
            "recommendations": final_result.get("recommendations", []),
            "causal_readiness": final_result.get("causal_readiness", "needs_attention"),
        }

        # Also populate from collected evidence
        self._eda_result.distribution_stats = self._analyzed_distributions
        self._eda_result.outliers = self._outlier_results
        if not self._eda_result.vif_scores:
            self._eda_result.vif_scores = self._vif_results
        if not self._eda_result.covariate_balance:
            self._eda_result.covariate_balance = self._balance_results
