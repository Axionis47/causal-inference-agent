"""Data Repair Agent - Truly Agentic Data Quality Repair for Causal Inference.

This agent iteratively diagnoses and repairs data quality issues:
- LLM decides what issues to investigate
- Tools provide specific diagnostics on demand
- LLM chooses repair strategies based on evidence
- LLM validates repairs and decides if more work needed
- Iterates until data quality is acceptable

No pre-computation - all diagnostics gathered through tool calls.
"""

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler

from src.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.agents.base.context_tools import ContextTools
from src.agents.base.react_agent import ReActAgent
from src.logging_config.structured import get_logger

logger = get_logger(__name__)


class DataRepairAgent(ReActAgent, ContextTools):
    """Truly agentic data repair agent that iteratively fixes data quality issues.

    Unlike traditional repair that runs all diagnostics upfront, this agent:
    1. LLM investigates data quality issues through tool calls
    2. LLM chooses appropriate repair strategies based on findings
    3. LLM applies repairs and validates results
    4. LLM iterates until satisfied with data quality

    Uses ReAct pattern with pull-based context tools.
    """

    AGENT_NAME = "data_repair"
    MAX_STEPS = 20

    SYSTEM_PROMPT = """You are an expert data scientist specializing in data preparation
for causal inference. Your goal is to repair data quality issues while preserving
the causal structure and avoiding introduction of bias.

CRITICAL: You must ITERATIVELY diagnose and repair issues by calling tools. Do NOT try to
fix everything at once. Instead:
1. Start by understanding the data and identifying issues
2. Prioritize issues that affect causal inference most
3. Apply repairs one at a time
4. Validate each repair before moving on
5. Continue until data quality is acceptable

Key principles:
- PRESERVE causal relationships - repairs must not create spurious associations
- PROTECT treatment and outcome variables - never drop or transform them aggressively
- PREFER conservative repairs - when in doubt, do less
- VALIDATE repairs don't introduce bias

For missing data:
- Check if missing is related to treatment (MNAR = danger!)
- MCAR: Simple imputation is safe
- MAR: Use multiple imputation or model-based methods
- MNAR: Be very careful, document assumptions

For outliers:
- Consider if "outliers" are valid extreme values
- Winsorization is safer than removal for causal inference
- Log transforms help right-skewed distributions
- Never remove outliers from treatment variable

For collinearity:
- High VIF (>10) causes unstable coefficient estimates
- Preserve variables with stronger causal justification
- Remove variables that are effect modifiers carefully

WORKFLOW:
1. Call get_data_summary to understand current state
2. Call check_missing_values to assess missing patterns
3. Call check_outliers for numeric variables
4. Call check_collinearity if many covariates
5. For each issue: repair -> validate -> decide if more repairs needed
6. Call finalize_repairs when data quality is acceptable"""

    def __init__(self) -> None:
        """Initialize the data repair agent."""
        super().__init__()
        self._df: pd.DataFrame | None = None
        self._df_original: pd.DataFrame | None = None
        self._current_state: AnalysisState | None = None
        self._repairs_applied: list[dict] = []
        self._finalized: bool = False

        # Register context tools from mixin
        self.register_context_tools()
        # Register repair-specific tools
        self._register_repair_tools()

    def _register_repair_tools(self) -> None:
        """Register data repair tools."""
        self.register_tool(
            name="get_data_summary",
            description="Get summary of current dataset state including dimensions, missing counts, and treatment/outcome info. Call this FIRST.",
            handler=self._tool_get_data_summary,
            parameters={},
        )

        self.register_tool(
            name="check_missing_values",
            description="Analyze missing value patterns for specified columns or all columns.",
            handler=self._tool_check_missing_values,
            parameters={
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columns to check (empty for all)",
                },
                "check_mcar": {
                    "type": "boolean",
                    "description": "Whether to test if missing is related to treatment (default: true)",
                },
            },
        )

        self.register_tool(
            name="check_outliers",
            description="Detect outliers in specified numeric columns using IQR and Z-score methods.",
            handler=self._tool_check_outliers,
            parameters={
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columns to check (empty for all numeric)",
                },
                "method": {
                    "type": "string",
                    "enum": ["iqr", "zscore", "both"],
                    "description": "Detection method (default: both)",
                },
            },
        )

        self.register_tool(
            name="check_collinearity",
            description="Check correlations and VIF scores for covariates to detect multicollinearity.",
            handler=self._tool_check_collinearity,
            parameters={
                "covariates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Covariates to check (empty for all numeric except treatment/outcome)",
                },
                "correlation_threshold": {
                    "type": "number",
                    "description": "Threshold for flagging high correlations (default: 0.8)",
                },
            },
        )

        self.register_tool(
            name="check_skewness",
            description="Check distribution skewness for numeric columns to assess need for transformation.",
            handler=self._tool_check_skewness,
            parameters={
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columns to check (empty for all numeric)",
                },
            },
        )

        self.register_tool(
            name="repair_missing",
            description="Repair missing values in specified columns using chosen strategy.",
            handler=self._tool_repair_missing,
            parameters={
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columns to repair (empty for all with missing)",
                },
                "strategy": {
                    "type": "string",
                    "enum": ["mean", "median", "mode", "iterative", "drop_rows", "drop_columns"],
                    "description": "Imputation strategy",
                },
                "drop_threshold": {
                    "type": "number",
                    "description": "For drop strategies: missing rate threshold (0-1)",
                },
            },
        )

        self.register_tool(
            name="repair_outliers",
            description="Handle outliers in specified columns using chosen strategy.",
            handler=self._tool_repair_outliers,
            parameters={
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columns to repair",
                },
                "strategy": {
                    "type": "string",
                    "enum": ["winsorize", "clip", "log_transform", "robust_scale", "remove"],
                    "description": "Outlier handling strategy",
                },
                "percentiles": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "For winsorize: [lower, upper] percentiles (default: [1, 99])",
                },
            },
        )

        self.register_tool(
            name="repair_collinearity",
            description="Handle collinearity by dropping or combining highly correlated variables.",
            handler=self._tool_repair_collinearity,
            parameters={
                "columns_to_drop": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columns to drop (for manual selection)",
                },
                "strategy": {
                    "type": "string",
                    "enum": ["drop_specified", "drop_high_vif", "auto_drop_correlated"],
                    "description": "Collinearity reduction strategy",
                },
                "vif_threshold": {
                    "type": "number",
                    "description": "VIF threshold for auto-dropping (default: 10)",
                },
            },
        )

        self.register_tool(
            name="validate_current_state",
            description="Validate current data quality and check if more repairs are needed.",
            handler=self._tool_validate_current_state,
            parameters={},
        )

        self.register_tool(
            name="finalize_repairs",
            description="Finalize repairs and save the repaired dataset. Call this when satisfied with data quality.",
            handler=self._tool_finalize_repairs,
            parameters={
                "quality_assessment": {
                    "type": "string",
                    "description": "Your assessment of final data quality",
                },
                "repairs_summary": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Summary of all repairs applied",
                },
                "cautions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Any cautions for downstream analysis",
                },
            },
        )

    def _get_initial_observation(self, state: AnalysisState) -> str:
        """Build initial observation for data repair."""
        dataset_name = state.dataset_info.name if state.dataset_info else "unknown"
        shape = self._df.shape if self._df is not None else (0, 0)

        return f"""DATA REPAIR TASK
Dataset: {dataset_name}
Treatment variable: {state.treatment_variable}
Outcome variable: {state.outcome_variable}
Shape: {shape}

Your goal is to prepare this data for causal effect estimation while preserving
the causal structure and avoiding introduction of bias.

Start by calling get_data_summary to understand the current state, then
systematically:
1. Check for missing values and their patterns
2. Check for outliers in key variables
3. Check for collinearity among covariates
4. Apply repairs as needed, validating each one
5. Call finalize_repairs when data quality is acceptable

Be conservative - only repair issues that would materially affect causal inference."""

    async def is_task_complete(self, state: AnalysisState) -> bool:
        """Check if data repair is complete."""
        return self._finalized

    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Execute data repair through iterative LLM-driven investigation.

        Args:
            state: Current analysis state

        Returns:
            Updated state with repaired data
        """
        self.logger.info("data_repair_start", job_id=state.job_id)
        start_time = time.time()

        try:
            # Load data
            self._df = self._load_dataframe(state)
            if self._df is None:
                state.mark_failed("No dataframe available for repair", self.AGENT_NAME)
                return state

            self._df_original = self._df.copy()
            self._current_state = state
            self._repairs_applied = []
            self._finalized = False

            self.logger.info("data_loaded", shape=self._df.shape)

            # Run ReAct loop
            await super().execute(state)

            # Auto-finalize if LLM didn't call finalize_repairs
            if not self._finalized:
                self._auto_finalize()

            # Save repaired dataframe
            self._save_dataframe(self._df, state)
            state.data_repairs = self._repairs_applied

            # Create trace
            duration_ms = int((time.time() - start_time) * 1000)
            trace = self.create_trace(
                action="data_repair_complete",
                reasoning="Data repair completed",
                outputs={
                    "repairs_applied": len(self._repairs_applied),
                    "original_shape": list(self._df_original.shape),
                    "final_shape": list(self._df.shape),
                },
                duration_ms=duration_ms,
            )
            state.add_trace(trace)

            self.logger.info(
                "data_repair_complete",
                repairs_applied=len(self._repairs_applied),
                original_shape=self._df_original.shape,
                new_shape=self._df.shape,
            )

        except Exception as e:
            self.logger.exception("data_repair_failed", error=str(e))
            # Don't fail the pipeline - repairs are optional
            state.add_trace(self.create_trace(
                action="repair_skipped",
                reasoning=f"Repair failed but continuing: {str(e)}",
                outputs={"error": str(e)},
            ))

        return state

    def _load_dataframe(self, state: AnalysisState) -> pd.DataFrame | None:
        """Load dataframe from state."""
        if state.dataframe_path and Path(state.dataframe_path).exists():
            try:
                with open(state.dataframe_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.error("load_failed", error=str(e))
        return None

    def _save_dataframe(self, df: pd.DataFrame, state: AnalysisState) -> None:
        """Save repaired dataframe."""
        if state.dataframe_path:
            try:
                with open(state.dataframe_path, "wb") as f:
                    pickle.dump(df, f)
            except Exception as e:
                self.logger.error("save_failed", error=str(e))

    # -------------------------------------------------------------------------
    # Tool Handlers (async, return ToolResult)
    # -------------------------------------------------------------------------

    async def _tool_get_data_summary(self, state: AnalysisState) -> ToolResult:
        """Get summary of current dataset state."""
        df = self._df
        n_rows, n_cols = df.shape

        # Missing summary
        missing_counts = df.isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0].to_dict()
        total_missing_pct = float(df.isnull().sum().sum() / (n_rows * n_cols) * 100)

        # Treatment and outcome info
        treatment_col = self._current_state.treatment_variable
        outcome_col = self._current_state.outcome_variable

        treatment_info = {"status": "Not found"}
        if treatment_col and treatment_col in df.columns:
            t_missing = int(df[treatment_col].isnull().sum())
            t_values = df[treatment_col].value_counts().to_dict()
            treatment_info = {"missing": t_missing, "values": {str(k): int(v) for k, v in t_values.items()}}

        outcome_info = {"status": "Not found"}
        if outcome_col and outcome_col in df.columns:
            o_missing = int(df[outcome_col].isnull().sum())
            outcome_info = {
                "missing": o_missing,
                "mean": float(df[outcome_col].mean()),
                "std": float(df[outcome_col].std()),
            }

        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "shape": [n_rows, n_cols],
                "n_numeric_columns": len(numeric_cols),
                "total_missing_pct": total_missing_pct,
                "cols_with_missing": len(cols_with_missing),
                "treatment_variable": treatment_col,
                "treatment_info": treatment_info,
                "outcome_variable": outcome_col,
                "outcome_info": outcome_info,
                "repairs_applied": len(self._repairs_applied),
            },
        )

    async def _tool_check_missing_values(
        self, state: AnalysisState, columns: list[str] | None = None, check_mcar: bool = True
    ) -> ToolResult:
        """Check missing value patterns."""
        df = self._df
        col_list = columns if columns else df.columns[df.isnull().any()].tolist()

        if not col_list:
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={"message": "No missing values found in the dataset.", "columns": []},
            )

        results = []
        n_rows = len(df)

        for col in col_list[:20]:
            if col not in df.columns:
                continue
            missing = int(df[col].isnull().sum())
            if missing == 0:
                continue
            missing_pct = float(missing / n_rows * 100)
            results.append({"column": col, "missing": missing, "pct": missing_pct})

        # Check if missing is related to treatment (MCAR test)
        mcar_warnings = []
        if check_mcar and self._current_state.treatment_variable in df.columns:
            treatment_col = self._current_state.treatment_variable
            for r in results[:10]:
                col = r["column"]
                try:
                    treated_missing = float(df[df[treatment_col] == 1][col].isnull().mean())
                    control_missing = float(df[df[treatment_col] == 0][col].isnull().mean())
                    diff = abs(treated_missing - control_missing)
                    if diff > 0.05:
                        mcar_warnings.append({
                            "column": col,
                            "treated_missing_pct": treated_missing * 100,
                            "control_missing_pct": control_missing * 100,
                            "diff_pct": diff * 100,
                        })
                except Exception:
                    pass

        results.sort(key=lambda x: x["pct"], reverse=True)

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "columns_analyzed": len(col_list),
                "columns_with_missing": results[:15],
                "mcar_warnings": mcar_warnings,
                "has_mnar_risk": len(mcar_warnings) > 0,
            },
        )

    async def _tool_check_outliers(
        self, state: AnalysisState, columns: list[str] | None = None, method: str = "both"
    ) -> ToolResult:
        """Check for outliers in numeric columns."""
        df = self._df

        if not columns:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude treatment
            if self._current_state.treatment_variable in columns:
                columns.remove(self._current_state.treatment_variable)

        results = []
        for col in columns[:20]:
            if col not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            data = df[col].dropna()
            if len(data) < 10:
                continue

            outlier_info = {"column": col, "n": len(data)}

            # IQR method
            if method in ["iqr", "both"]:
                q1 = data.quantile(0.25)
                q3 = data.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                iqr_out = ((data < lower) | (data > upper)).sum()
                outlier_info["iqr_outliers"] = int(iqr_out)
                outlier_info["iqr_pct"] = float(iqr_out / len(data) * 100)

                # Extreme outliers (3*IQR)
                extreme_lower = q1 - 3 * iqr
                extreme_upper = q3 + 3 * iqr
                extreme_out = ((data < extreme_lower) | (data > extreme_upper)).sum()
                outlier_info["extreme_outliers"] = int(extreme_out)

            # Z-score method
            if method in ["zscore", "both"]:
                z_scores = np.abs(stats.zscore(data))
                zscore_out = (z_scores > 3).sum()
                outlier_info["zscore_outliers"] = int(zscore_out)

            if outlier_info.get("iqr_outliers", 0) > 0 or outlier_info.get("zscore_outliers", 0) > 0:
                results.append(outlier_info)

        results.sort(key=lambda x: x.get("iqr_pct", 0), reverse=True)

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "columns_analyzed": len(columns) if columns else 0,
                "outliers_detected": results[:15],
                "has_outliers": len(results) > 0,
            },
        )

    async def _tool_check_collinearity(
        self,
        state: AnalysisState,
        covariates: list[str] | None = None,
        correlation_threshold: float = 0.8,
    ) -> ToolResult:
        """Check correlations and VIF for collinearity."""
        df = self._df

        if not covariates:
            covariates = df.select_dtypes(include=[np.number]).columns.tolist()
            for var in [self._current_state.treatment_variable, self._current_state.outcome_variable]:
                if var in covariates:
                    covariates.remove(var)

        covariates = [c for c in covariates if c in df.columns]

        if len(covariates) < 2:
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={"message": "Not enough covariates for collinearity analysis."},
            )

        # Correlation analysis
        corr_matrix = df[covariates].corr().abs()
        high_corrs = []

        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:
                    corr_val = corr_matrix.iloc[i, j]
                    if corr_val > correlation_threshold:
                        high_corrs.append({
                            "var1": col1,
                            "var2": col2,
                            "correlation": float(corr_val),
                        })

        high_corrs.sort(key=lambda x: x["correlation"], reverse=True)

        # VIF analysis
        vif_results = []
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor

            X = df[covariates].dropna()
            if len(X) >= 10:
                X_with_const = np.column_stack([np.ones(len(X)), X.values])
                for i, col in enumerate(covariates):
                    try:
                        vif = variance_inflation_factor(X_with_const, i + 1)
                        if vif > 5:
                            vif_results.append({"variable": col, "vif": float(vif)})
                    except Exception:
                        pass
        except Exception:
            pass

        vif_results.sort(key=lambda x: x["vif"], reverse=True)

        # Format output
        output = f"Collinearity Analysis ({len(covariates)} covariates):\n"
        output += "=" * 50 + "\n"

        if high_corrs:
            output += f"\nHigh Correlations (|r| > {correlation_threshold}):\n"
            for hc in high_corrs[:10]:
                output += f"  {hc['var1']} <-> {hc['var2']}: r={hc['correlation']:.3f}\n"
            if len(high_corrs) > 10:
                output += f"  ... and {len(high_corrs) - 10} more pairs\n"
        else:
            output += "\nNo high correlations found.\n"

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "covariates_analyzed": len(covariates),
                "high_correlations": high_corrs[:10],
                "high_vif_variables": vif_results[:10],
                "has_collinearity_issues": len(high_corrs) > 0 or len(vif_results) > 0,
            },
        )

    async def _tool_check_skewness(
        self, state: AnalysisState, columns: list[str] | None = None
    ) -> ToolResult:
        """Check distribution skewness."""
        df = self._df

        if not columns:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        results = []
        for col in columns[:20]:
            if col not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            data = df[col].dropna()
            if len(data) < 10:
                continue

            skew = float(stats.skew(data))
            kurt = float(stats.kurtosis(data))

            if abs(skew) > 1:
                results.append({
                    "column": col,
                    "skewness": skew,
                    "kurtosis": kurt,
                    "recommendation": "log_transform" if skew > 1 and data.min() > 0 else "winsorize",
                })

        results.sort(key=lambda x: abs(x["skewness"]), reverse=True)

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "columns_analyzed": len(columns) if columns else 0,
                "skewed_columns": results[:15],
                "has_skewed_variables": len(results) > 0,
            },
        )

    async def _tool_repair_missing(
        self,
        state: AnalysisState,
        strategy: str,
        columns: list[str] | None = None,
        drop_threshold: float = 0.5,
    ) -> ToolResult:
        """Repair missing values."""
        df = self._df

        if not columns:
            columns = df.columns[df.isnull().any()].tolist()

        # Protect treatment and outcome
        protected = [self._current_state.treatment_variable, self._current_state.outcome_variable]
        if strategy in ["drop_columns"]:
            columns = [c for c in columns if c not in protected]

        columns = [c for c in columns if c in df.columns]

        if not columns:
            return ToolResult(status=ToolResultStatus.SUCCESS, output={"message": "No columns to repair."})

        before_missing = int(df[columns].isnull().sum().sum())

        try:
            if strategy == "drop_rows":
                rows_before = len(df)
                df = df.dropna(subset=columns, thresh=int(len(columns) * (1 - drop_threshold)))
                self._df = df
                rows_dropped = rows_before - len(df)
                self._repairs_applied.append({
                    "type": "missing", "strategy": "drop_rows", "columns": columns, "rows_dropped": rows_dropped,
                })
                return ToolResult(status=ToolResultStatus.SUCCESS, output={
                    "action": "drop_rows", "rows_dropped": rows_dropped, "new_shape": list(df.shape),
                })

            elif strategy == "drop_columns":
                cols_to_drop = [col for col in columns if df[col].isnull().mean() > drop_threshold]
                if cols_to_drop:
                    df = df.drop(columns=cols_to_drop)
                    self._df = df
                    self._repairs_applied.append({"type": "missing", "strategy": "drop_columns", "columns": cols_to_drop})
                    return ToolResult(status=ToolResultStatus.SUCCESS, output={
                        "action": "drop_columns", "columns_dropped": cols_to_drop,
                    })
                return ToolResult(status=ToolResultStatus.SUCCESS, output={"message": "No columns exceeded threshold."})

            elif strategy == "iterative":
                numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
                if numeric_cols:
                    imputer = IterativeImputer(max_iter=10, random_state=42)
                    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                    self._df = df

            elif strategy in ["mean", "median"]:
                numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
                if numeric_cols:
                    imputer = SimpleImputer(strategy=strategy)
                    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                    self._df = df

            elif strategy == "mode":
                for col in columns:
                    if df[col].isnull().any():
                        mode_val = df[col].mode()
                        fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "unknown"
                        df[col].fillna(fill_val, inplace=True)
                self._df = df

            after_missing = int(df[columns].isnull().sum().sum())
            self._repairs_applied.append({
                "type": "missing", "strategy": strategy, "columns": columns,
                "before": before_missing, "after": after_missing,
            })
            return ToolResult(status=ToolResultStatus.SUCCESS, output={
                "strategy": strategy, "before_missing": before_missing, "after_missing": after_missing,
            })

        except Exception as e:
            return ToolResult(status=ToolResultStatus.ERROR, error=f"Error repairing missing values: {str(e)}")

    async def _tool_repair_outliers(
        self,
        state: AnalysisState,
        columns: list[str],
        strategy: str,
        percentiles: list[float] | None = None,
    ) -> ToolResult:
        """Repair outliers."""
        if percentiles is None:
            percentiles = [1, 99]

        df = self._df

        # Protect treatment
        if self._current_state.treatment_variable in columns:
            columns = [c for c in columns if c != self._current_state.treatment_variable]
            if not columns:
                return ToolResult(status=ToolResultStatus.ERROR, error="Cannot modify treatment variable.")

        columns = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

        if not columns:
            return ToolResult(status=ToolResultStatus.ERROR, error="No valid numeric columns to repair.")

        changes = []

        try:
            for col in columns:
                data = df[col].dropna()
                if len(data) < 10:
                    continue

                before_outliers = self._count_outliers(data)

                if strategy == "winsorize":
                    lower_p, upper_p = percentiles[0], percentiles[1]
                    lower = df[col].quantile(lower_p / 100)
                    upper = df[col].quantile(upper_p / 100)
                    df[col] = df[col].clip(lower=lower, upper=upper)

                elif strategy == "clip":
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    df[col] = df[col].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)

                elif strategy == "log_transform":
                    min_val = df[col].min()
                    df[col] = np.log1p(df[col] - min_val + 1) if min_val <= 0 else np.log(df[col])

                elif strategy == "robust_scale":
                    scaler = RobustScaler()
                    df[col] = scaler.fit_transform(df[[col]])

                elif strategy == "remove":
                    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                    iqr = q3 - q1
                    mask = (df[col] >= q1 - 3 * iqr) & (df[col] <= q3 + 3 * iqr)
                    rows_before = len(df)
                    df = df[mask | df[col].isnull()]
                    changes.append({"column": col, "removed": rows_before - len(df)})
                    continue

                after_outliers = self._count_outliers(df[col].dropna())
                changes.append({"column": col, "before": before_outliers, "after": after_outliers})

            self._df = df
            self._repairs_applied.append({"type": "outliers", "strategy": strategy, "columns": columns})

            return ToolResult(status=ToolResultStatus.SUCCESS, output={
                "strategy": strategy, "columns_repaired": len(columns), "changes": changes,
            })

        except Exception as e:
            return ToolResult(status=ToolResultStatus.ERROR, error=f"Error repairing outliers: {str(e)}")

    def _count_outliers(self, data: pd.Series) -> int:
        """Count outliers using IQR method."""
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        return int(((data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)).sum())

    async def _tool_repair_collinearity(
        self,
        state: AnalysisState,
        strategy: str,
        columns_to_drop: list[str] | None = None,
        vif_threshold: float = 10,
    ) -> ToolResult:
        """Repair collinearity issues."""
        df = self._df
        columns_to_drop = columns_to_drop or []

        # Protect treatment and outcome
        protected = [self._current_state.treatment_variable, self._current_state.outcome_variable]

        try:
            if strategy == "drop_specified":
                columns_to_drop = [c for c in columns_to_drop if c in df.columns and c not in protected]
                if not columns_to_drop:
                    return ToolResult(
                        status=ToolResultStatus.SUCCESS,
                        output={"message": "No valid columns to drop (treatment/outcome are protected)."},
                    )
                df = df.drop(columns=columns_to_drop)
                self._df = df
                self._repairs_applied.append({
                    "type": "collinearity",
                    "strategy": "drop_specified",
                    "columns": columns_to_drop,
                })
                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    output={
                        "message": f"Dropped columns: {columns_to_drop}",
                        "new_shape": list(df.shape),
                    },
                )

            elif strategy == "drop_high_vif":
                from statsmodels.stats.outliers_influence import variance_inflation_factor

                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cols = [c for c in numeric_cols if c not in protected]

                dropped = []
                max_iters = 10
                for _ in range(max_iters):
                    if len(numeric_cols) < 3:
                        break

                    X = df[numeric_cols].dropna()
                    if len(X) < 10:
                        break

                    X_with_const = np.column_stack([np.ones(len(X)), X.values])

                    vifs = []
                    for i, col in enumerate(numeric_cols):
                        try:
                            vif = variance_inflation_factor(X_with_const, i + 1)
                            vifs.append((col, vif))
                        except Exception:
                            pass

                    if not vifs:
                        break

                    max_vif_col, max_vif = max(vifs, key=lambda x: x[1])
                    if max_vif > vif_threshold:
                        numeric_cols.remove(max_vif_col)
                        df = df.drop(columns=[max_vif_col])
                        dropped.append(f"{max_vif_col} (VIF={max_vif:.1f})")
                    else:
                        break

                self._df = df

                if dropped:
                    self._repairs_applied.append({
                        "type": "collinearity",
                        "strategy": "drop_high_vif",
                        "columns": [d.split(" ")[0] for d in dropped],
                    })
                    return ToolResult(
                        status=ToolResultStatus.SUCCESS,
                        output={
                            "message": f"Dropped high VIF columns: {dropped}",
                            "new_shape": list(df.shape),
                        },
                    )
                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    output={"message": "No columns exceeded the VIF threshold."},
                )

            elif strategy == "auto_drop_correlated":
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cols = [c for c in numeric_cols if c not in protected]

                corr_matrix = df[numeric_cols].corr().abs()
                dropped = []

                for i, col1 in enumerate(corr_matrix.columns):
                    for j, col2 in enumerate(corr_matrix.columns):
                        if i < j and col1 in df.columns and col2 in df.columns:
                            if corr_matrix.iloc[i, j] > 0.9:
                                # Drop the one with more missing
                                if df[col1].isnull().sum() >= df[col2].isnull().sum():
                                    to_drop = col1
                                else:
                                    to_drop = col2
                                if to_drop not in dropped:
                                    dropped.append(to_drop)
                                    df = df.drop(columns=[to_drop])

                self._df = df

                if dropped:
                    self._repairs_applied.append({
                        "type": "collinearity",
                        "strategy": "auto_drop_correlated",
                        "columns": dropped,
                    })
                    return ToolResult(
                        status=ToolResultStatus.SUCCESS,
                        output={
                            "message": f"Dropped highly correlated columns: {dropped}",
                            "new_shape": list(df.shape),
                        },
                    )
                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    output={"message": "No highly correlated column pairs found to drop."},
                )

            return ToolResult(
                status=ToolResultStatus.ERROR,
                error=f"Unknown strategy: {strategy}",
            )

        except Exception as e:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                error=f"Error repairing collinearity: {str(e)}",
            )

    async def _tool_validate_current_state(self, state: AnalysisState) -> ToolResult:
        """Validate current data quality."""
        df = self._df
        df_orig = self._df_original

        # Calculate quality metrics
        missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        row_loss = (len(df_orig) - len(df)) / len(df_orig) * 100
        col_loss = (len(df_orig.columns) - len(df.columns)) / len(df_orig.columns) * 100

        # Check treatment and outcome
        treatment_ok = self._current_state.treatment_variable in df.columns
        outcome_ok = self._current_state.outcome_variable in df.columns

        # Calculate outlier percentage
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        total_outliers = 0
        total_values = 0
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) >= 10:
                total_outliers += self._count_outliers(data)
                total_values += len(data)

        outlier_pct = total_outliers / total_values * 100 if total_values > 0 else 0

        # Quality score
        score = 100
        score -= missing_pct * 2  # Penalize missing
        score -= min(20, row_loss)  # Penalize row loss
        score -= min(10, col_loss)  # Penalize column loss
        score -= min(20, outlier_pct * 2)  # Penalize outliers
        score = max(0, score)

        # Determine status
        if score >= 80:
            status_msg = "GOOD - Data is ready for causal analysis."
        elif score >= 60:
            status_msg = "ACCEPTABLE - Minor issues remain but analysis can proceed."
        else:
            status_msg = "NEEDS_WORK - Consider additional repairs."

        critical_warning = None
        if not treatment_ok or not outcome_ok:
            critical_warning = "Treatment or outcome variable is missing! Repairs may have been too aggressive."

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "shape": list(df.shape),
                "original_shape": list(df_orig.shape),
                "row_loss_pct": round(row_loss, 1),
                "column_loss_pct": round(col_loss, 1),
                "missing_data_pct": round(missing_pct, 2),
                "outlier_rate_pct": round(outlier_pct, 1),
                "treatment_variable_present": treatment_ok,
                "outcome_variable_present": outcome_ok,
                "quality_score": round(score),
                "status": status_msg,
                "critical_warning": critical_warning,
            },
        )

    async def _tool_finalize_repairs(
        self,
        state: AnalysisState,
        quality_assessment: str,
        repairs_summary: list[str],
        cautions: list[str] | None = None,
    ) -> ToolResult:
        """Finalize repairs and mark task as complete."""
        self.logger.info("finalizing_repairs", quality=quality_assessment)

        # Save repaired data to state
        self._current_state.data_cache["repaired_data"] = self._df

        # Store results
        self._current_state.add_agent_result(
            self.AGENT_NAME,
            {
                "quality_assessment": quality_assessment,
                "repairs_summary": repairs_summary,
                "cautions": cautions or [],
                "final_shape": list(self._df.shape),
                "original_shape": list(self._df_original.shape),
                "repairs_applied": self._repairs_applied,
            },
        )

        self._finalized = True

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "message": "Repairs finalized successfully",
                "quality_assessment": quality_assessment,
                "repairs_applied_count": len(self._repairs_applied),
                "final_shape": list(self._df.shape),
            },
        )

    def _auto_finalize(self) -> dict[str, Any]:
        """Auto-generate final result from repairs applied."""
        self.logger.info("repair_auto_finalize")

        repairs_summary = []
        for r in self._repairs_applied:
            summary = f"{r['type']}: {r.get('strategy', '')} on {len(r.get('columns', []))} columns"
            repairs_summary.append(summary)

        # Quality assessment
        df = self._df
        missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100

        if missing_pct < 1:
            quality = "Good - minimal missing data"
        elif missing_pct < 5:
            quality = "Acceptable - some missing data remains"
        else:
            quality = "Fair - notable missing data remains"

        cautions = []
        if len(df) < len(self._df_original) * 0.8:
            cautions.append("Significant row loss - check for selection bias")
        if len(df.columns) < len(self._df_original.columns) * 0.7:
            cautions.append("Many columns dropped - verify important covariates retained")

        return {
            "quality_assessment": quality,
            "repairs_summary": repairs_summary if repairs_summary else ["No repairs applied"],
            "cautions": cautions,
        }
