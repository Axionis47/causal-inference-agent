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

from src.agents.base import AnalysisState, BaseAgent, JobStatus
from src.logging_config.structured import get_logger

logger = get_logger(__name__)


class DataRepairAgent(BaseAgent):
    """Truly agentic data repair agent that iteratively fixes data quality issues.

    Unlike traditional repair that runs all diagnostics upfront, this agent:
    1. LLM investigates data quality issues through tool calls
    2. LLM chooses appropriate repair strategies based on findings
    3. LLM applies repairs and validates results
    4. LLM iterates until satisfied with data quality

    This approach allows intelligent repair decisions based on causal inference needs.
    """

    AGENT_NAME = "data_repair"

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

    TOOLS = [
        {
            "name": "get_data_summary",
            "description": "Get summary of current dataset state including dimensions, missing counts, and treatment/outcome info. Call this FIRST.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "check_missing_values",
            "description": "Analyze missing value patterns for specified columns or all columns.",
            "parameters": {
                "type": "object",
                "properties": {
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
                "required": [],
            },
        },
        {
            "name": "check_outliers",
            "description": "Detect outliers in specified numeric columns using IQR and Z-score methods.",
            "parameters": {
                "type": "object",
                "properties": {
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
                "required": [],
            },
        },
        {
            "name": "check_collinearity",
            "description": "Check correlations and VIF scores for covariates to detect multicollinearity.",
            "parameters": {
                "type": "object",
                "properties": {
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
                "required": [],
            },
        },
        {
            "name": "check_skewness",
            "description": "Check distribution skewness for numeric columns to assess need for transformation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns to check (empty for all numeric)",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "repair_missing",
            "description": "Repair missing values in specified columns using chosen strategy.",
            "parameters": {
                "type": "object",
                "properties": {
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
                "required": ["strategy"],
            },
        },
        {
            "name": "repair_outliers",
            "description": "Handle outliers in specified columns using chosen strategy.",
            "parameters": {
                "type": "object",
                "properties": {
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
                "required": ["columns", "strategy"],
            },
        },
        {
            "name": "repair_collinearity",
            "description": "Handle collinearity by dropping or combining highly correlated variables.",
            "parameters": {
                "type": "object",
                "properties": {
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
                "required": ["strategy"],
            },
        },
        {
            "name": "validate_current_state",
            "description": "Validate current data quality and check if more repairs are needed.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "finalize_repairs",
            "description": "Finalize repairs and save the repaired dataset. Call this when satisfied with data quality.",
            "parameters": {
                "type": "object",
                "properties": {
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
                "required": ["quality_assessment", "repairs_summary"],
            },
        },
    ]

    def __init__(self):
        """Initialize the data repair agent."""
        super().__init__()
        self._df: pd.DataFrame | None = None
        self._df_original: pd.DataFrame | None = None
        self._state: AnalysisState | None = None
        self._repairs_applied: list[dict] = []

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
            self._state = state
            self._repairs_applied = []

            self.logger.info("data_loaded", shape=self._df.shape)

            # Build the initial prompt
            initial_prompt = self._build_initial_prompt()

            # Run the agentic loop
            final_result = await self._run_agentic_loop(initial_prompt, max_iterations=20)

            # Save repaired dataframe
            self._save_dataframe(self._df, state)
            state.data_repairs = self._repairs_applied

            # Create trace
            duration_ms = int((time.time() - start_time) * 1000)
            trace = self.create_trace(
                action="data_repair_complete",
                reasoning=final_result.get("quality_assessment", "Data repair completed"),
                outputs={
                    "repairs_applied": len(self._repairs_applied),
                    "original_shape": self._df_original.shape,
                    "final_shape": self._df.shape,
                    "repairs_summary": final_result.get("repairs_summary", []),
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
            self.logger.error("data_repair_failed", error=str(e))
            import traceback
            traceback.print_exc()
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

    def _build_initial_prompt(self) -> str:
        """Build the initial prompt for the agentic loop."""
        return f"""You are repairing data quality issues for a causal inference study.

Dataset: {self._state.dataset_info.name or self._state.dataset_info.url}
Treatment variable: {self._state.treatment_variable}
Outcome variable: {self._state.outcome_variable}
Shape: {self._df.shape}

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

    async def _run_agentic_loop(
        self, initial_prompt: str, max_iterations: int = 20
    ) -> dict[str, Any]:
        """Run the agentic loop where LLM iteratively diagnoses and repairs.

        Args:
            initial_prompt: The starting prompt
            max_iterations: Maximum tool-calling iterations

        Returns:
            Final result from finalize_repairs or auto-generated result
        """
        messages = [{"role": "user", "content": initial_prompt}]

        for iteration in range(max_iterations):
            self.logger.info(
                "repair_iteration",
                iteration=iteration,
                max_iterations=max_iterations,
                repairs_so_far=len(self._repairs_applied),
            )

            # Get LLM response with tool calls
            result = await self.reason(
                prompt=messages[-1]["content"],
                context={"iteration": iteration, "max_iterations": max_iterations},
            )

            # Log LLM reasoning if provided
            response_text = result.get("response", "")
            if response_text:
                self.logger.info(
                    "repair_llm_reasoning",
                    reasoning=response_text[:500] + ("..." if len(response_text) > 500 else ""),
                )

            pending_calls = result.get("pending_calls", [])

            if not pending_calls:
                self.logger.info("repair_no_tool_calls", response_preview=response_text[:200])
                if "finalize" in response_text.lower() or iteration > 15:
                    return self._auto_finalize()
                messages.append({
                    "role": "user",
                    "content": "Please continue by calling the appropriate tools. "
                    "If data quality is acceptable, call finalize_repairs.",
                })
                continue

            # Execute tool calls
            tool_results = []
            for call in pending_calls:
                tool_name = call.get("name")
                tool_args = call.get("args", {})

                # Check for finalize with detailed logging
                if tool_name == "finalize_repairs":
                    self.logger.info(
                        "repair_finalizing",
                        quality_assessment=tool_args.get("quality_assessment"),
                        repairs_summary=tool_args.get("repairs_summary", [])[:3],
                        total_repairs=len(self._repairs_applied),
                    )
                    self._save_dataframe(self._df, self._state)
                    return tool_args

                # Log the tool call decision
                self.logger.info(
                    "repair_tool_decision",
                    tool=tool_name,
                    args=tool_args,
                    is_repair="repair" in tool_name.lower(),
                )

                # Execute the tool
                tool_result = self._execute_tool(tool_name, tool_args)
                tool_results.append(f"## {tool_name}\n{tool_result}")

                # Log tool result summary
                self.logger.info(
                    "repair_tool_result",
                    tool=tool_name,
                    result_summary=tool_result[:200] + ("..." if len(tool_result) > 200 else ""),
                )

            # Feed results back to LLM
            results_text = "\n\n".join(tool_results)
            messages.append({
                "role": "user",
                "content": f"Tool results:\n\n{results_text}\n\nAnalyze these results and continue. "
                f"Apply repairs as needed, or call finalize_repairs if data quality is acceptable.",
            })

        # Max iterations reached - auto-finalize
        self.logger.warning("repair_max_iterations_reached")
        return self._auto_finalize()

    def _execute_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        """Execute a tool and return results as string."""
        try:
            if tool_name == "get_data_summary":
                return self._tool_get_data_summary()
            elif tool_name == "check_missing_values":
                return self._tool_check_missing_values(
                    args.get("columns", []),
                    args.get("check_mcar", True),
                )
            elif tool_name == "check_outliers":
                return self._tool_check_outliers(
                    args.get("columns", []),
                    args.get("method", "both"),
                )
            elif tool_name == "check_collinearity":
                return self._tool_check_collinearity(
                    args.get("covariates", []),
                    args.get("correlation_threshold", 0.8),
                )
            elif tool_name == "check_skewness":
                return self._tool_check_skewness(args.get("columns", []))
            elif tool_name == "repair_missing":
                return self._tool_repair_missing(
                    args.get("columns", []),
                    args.get("strategy", "median"),
                    args.get("drop_threshold", 0.5),
                )
            elif tool_name == "repair_outliers":
                return self._tool_repair_outliers(
                    args.get("columns", []),
                    args.get("strategy", "winsorize"),
                    args.get("percentiles", [1, 99]),
                )
            elif tool_name == "repair_collinearity":
                return self._tool_repair_collinearity(
                    args.get("columns_to_drop", []),
                    args.get("strategy", "drop_specified"),
                    args.get("vif_threshold", 10),
                )
            elif tool_name == "validate_current_state":
                return self._tool_validate_current_state()
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            self.logger.error(f"tool_execution_error", tool=tool_name, error=str(e))
            return f"Error executing {tool_name}: {str(e)}"

    def _tool_get_data_summary(self) -> str:
        """Get summary of current dataset state."""
        df = self._df
        n_rows, n_cols = df.shape

        # Missing summary
        missing_counts = df.isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0]
        total_missing_pct = df.isnull().sum().sum() / (n_rows * n_cols) * 100

        # Treatment and outcome info
        treatment_col = self._state.treatment_variable
        outcome_col = self._state.outcome_variable

        treatment_info = "Not found"
        if treatment_col and treatment_col in df.columns:
            t_missing = df[treatment_col].isnull().sum()
            t_values = df[treatment_col].value_counts()
            treatment_info = f"Missing: {t_missing}, Values: {dict(t_values)}"

        outcome_info = "Not found"
        if outcome_col and outcome_col in df.columns:
            o_missing = df[outcome_col].isnull().sum()
            outcome_info = f"Missing: {o_missing}, Mean: {df[outcome_col].mean():.3f}, Std: {df[outcome_col].std():.3f}"

        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        result = f"""Current Dataset State:
- Shape: {n_rows} rows x {n_cols} columns
- Numeric columns: {len(numeric_cols)}
- Total missing: {total_missing_pct:.2f}%
- Columns with missing: {len(cols_with_missing)}

Treatment Variable ({treatment_col}):
{treatment_info}

Outcome Variable ({outcome_col}):
{outcome_info}

Repairs Applied So Far: {len(self._repairs_applied)}"""

        if self._repairs_applied:
            result += "\nRecent repairs:"
            for r in self._repairs_applied[-3:]:
                result += f"\n  - {r['type']}: {r.get('strategy', '')} on {r.get('columns', [])}"

        return result

    def _tool_check_missing_values(self, columns: list[str], check_mcar: bool = True) -> str:
        """Check missing value patterns."""
        df = self._df

        if not columns:
            columns = df.columns[df.isnull().any()].tolist()

        if not columns:
            return "No missing values found in the dataset."

        results = []
        n_rows = len(df)

        for col in columns[:20]:
            if col not in df.columns:
                continue

            missing = df[col].isnull().sum()
            if missing == 0:
                continue

            missing_pct = missing / n_rows * 100
            results.append({
                "column": col,
                "missing": missing,
                "pct": missing_pct,
            })

        # Check if missing is related to treatment (MCAR test)
        mcar_warnings = []
        if check_mcar and self._state.treatment_variable in df.columns:
            treatment_col = self._state.treatment_variable
            for r in results[:10]:
                col = r["column"]
                try:
                    treated_missing = df[df[treatment_col] == 1][col].isnull().mean()
                    control_missing = df[df[treatment_col] == 0][col].isnull().mean()
                    diff = abs(treated_missing - control_missing)
                    if diff > 0.05:
                        mcar_warnings.append(
                            f"{col}: Treated missing={treated_missing*100:.1f}%, "
                            f"Control missing={control_missing*100:.1f}% (diff={diff*100:.1f}%)"
                        )
                except Exception:
                    pass

        # Format output
        output = f"Missing Value Analysis ({len(columns)} columns with missing):\n"
        output += "=" * 50 + "\n"

        results.sort(key=lambda x: x["pct"], reverse=True)
        for r in results[:15]:
            output += f"  {r['column']}: {r['missing']} ({r['pct']:.1f}%)\n"

        if len(results) > 15:
            output += f"  ... and {len(results) - 15} more columns\n"

        if mcar_warnings:
            output += "\nWARNING - Missing may be related to treatment (MNAR risk):\n"
            for w in mcar_warnings:
                output += f"  - {w}\n"
            output += "\nBe cautious with imputation - missing may not be random!\n"

        return output

    def _tool_check_outliers(self, columns: list[str], method: str = "both") -> str:
        """Check for outliers in numeric columns."""
        df = self._df

        if not columns:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude treatment
            if self._state.treatment_variable in columns:
                columns.remove(self._state.treatment_variable)

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

        # Format output
        output = "Outlier Detection Results:\n"
        output += "=" * 50 + "\n"

        if not results:
            return output + "No significant outliers detected."

        results.sort(key=lambda x: x.get("iqr_pct", 0), reverse=True)

        for r in results[:15]:
            output += f"\n{r['column']}:\n"
            if "iqr_outliers" in r:
                output += f"  - IQR method: {r['iqr_outliers']} outliers ({r['iqr_pct']:.1f}%)\n"
                if r.get("extreme_outliers", 0) > 0:
                    output += f"  - Extreme (3*IQR): {r['extreme_outliers']} outliers\n"
            if "zscore_outliers" in r:
                output += f"  - Z-score (>3): {r['zscore_outliers']} outliers\n"

        if len(results) > 15:
            output += f"\n... and {len(results) - 15} more columns with outliers\n"

        return output

    def _tool_check_collinearity(
        self, covariates: list[str], correlation_threshold: float = 0.8
    ) -> str:
        """Check correlations and VIF for collinearity."""
        df = self._df

        if not covariates:
            covariates = df.select_dtypes(include=[np.number]).columns.tolist()
            for var in [self._state.treatment_variable, self._state.outcome_variable]:
                if var in covariates:
                    covariates.remove(var)

        covariates = [c for c in covariates if c in df.columns]

        if len(covariates) < 2:
            return "Not enough covariates for collinearity analysis."

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

        if vif_results:
            output += "\nHigh VIF Variables (VIF > 5):\n"
            for v in vif_results[:10]:
                flag = " *** SEVERE ***" if v["vif"] > 10 else ""
                output += f"  {v['variable']}: VIF={v['vif']:.1f}{flag}\n"
        else:
            output += "\nNo severe multicollinearity detected (all VIF < 5).\n"

        return output

    def _tool_check_skewness(self, columns: list[str]) -> str:
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

        # Format output
        output = "Skewness Analysis:\n"
        output += "=" * 50 + "\n"

        if not results:
            return output + "No highly skewed variables found (|skew| < 1)."

        results.sort(key=lambda x: abs(x["skewness"]), reverse=True)

        for r in results[:15]:
            direction = "right-skewed" if r["skewness"] > 0 else "left-skewed"
            output += f"\n{r['column']}:\n"
            output += f"  - Skewness: {r['skewness']:.2f} ({direction})\n"
            output += f"  - Kurtosis: {r['kurtosis']:.2f}\n"
            output += f"  - Suggested: {r['recommendation']}\n"

        return output

    def _tool_repair_missing(
        self, columns: list[str], strategy: str, drop_threshold: float = 0.5
    ) -> str:
        """Repair missing values."""
        df = self._df

        if not columns:
            columns = df.columns[df.isnull().any()].tolist()

        # Protect treatment and outcome
        protected = [self._state.treatment_variable, self._state.outcome_variable]
        if strategy in ["drop_columns"]:
            columns = [c for c in columns if c not in protected]

        columns = [c for c in columns if c in df.columns]

        if not columns:
            return "No columns to repair."

        before_missing = df[columns].isnull().sum().sum()

        try:
            if strategy == "drop_rows":
                # Drop rows with too much missing
                rows_before = len(df)
                df = df.dropna(subset=columns, thresh=int(len(columns) * (1 - drop_threshold)))
                self._df = df
                rows_dropped = rows_before - len(df)
                self._repairs_applied.append({
                    "type": "missing",
                    "strategy": "drop_rows",
                    "columns": columns,
                    "rows_dropped": rows_dropped,
                })
                return f"Dropped {rows_dropped} rows with high missing rates. New shape: {df.shape}"

            elif strategy == "drop_columns":
                # Drop columns with too much missing
                cols_to_drop = []
                for col in columns:
                    if df[col].isnull().mean() > drop_threshold:
                        cols_to_drop.append(col)
                if cols_to_drop:
                    df = df.drop(columns=cols_to_drop)
                    self._df = df
                    self._repairs_applied.append({
                        "type": "missing",
                        "strategy": "drop_columns",
                        "columns": cols_to_drop,
                    })
                    return f"Dropped columns with >{drop_threshold*100}% missing: {cols_to_drop}"
                return "No columns exceeded the missing threshold."

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

            after_missing = df[columns].isnull().sum().sum()

            self._repairs_applied.append({
                "type": "missing",
                "strategy": strategy,
                "columns": columns,
                "before": int(before_missing),
                "after": int(after_missing),
            })

            return f"Imputed missing values using {strategy}.\nBefore: {before_missing} missing\nAfter: {after_missing} missing"

        except Exception as e:
            return f"Error repairing missing values: {str(e)}"

    def _tool_repair_outliers(
        self, columns: list[str], strategy: str, percentiles: list[float] = None
    ) -> str:
        """Repair outliers."""
        if percentiles is None:
            percentiles = [1, 99]

        df = self._df

        # Protect treatment
        if self._state.treatment_variable in columns:
            columns = [c for c in columns if c != self._state.treatment_variable]
            if not columns:
                return "Cannot modify treatment variable. Please select other columns."

        columns = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

        if not columns:
            return "No valid numeric columns to repair."

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
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    df[col] = df[col].clip(lower=lower, upper=upper)

                elif strategy == "log_transform":
                    min_val = df[col].min()
                    if min_val <= 0:
                        df[col] = np.log1p(df[col] - min_val + 1)
                    else:
                        df[col] = np.log(df[col])

                elif strategy == "robust_scale":
                    scaler = RobustScaler()
                    df[col] = scaler.fit_transform(df[[col]])

                elif strategy == "remove":
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    mask = (df[col] >= q1 - 3 * iqr) & (df[col] <= q3 + 3 * iqr)
                    rows_before = len(df)
                    df = df[mask | df[col].isnull()]
                    changes.append(f"{col}: removed {rows_before - len(df)} extreme outliers")
                    continue

                after_outliers = self._count_outliers(df[col].dropna())
                changes.append(f"{col}: {before_outliers} -> {after_outliers} outliers")

            self._df = df

            self._repairs_applied.append({
                "type": "outliers",
                "strategy": strategy,
                "columns": columns,
            })

            output = f"Applied {strategy} to {len(columns)} columns:\n"
            for c in changes:
                output += f"  - {c}\n"
            return output

        except Exception as e:
            return f"Error repairing outliers: {str(e)}"

    def _count_outliers(self, data: pd.Series) -> int:
        """Count outliers using IQR method."""
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        return int(((data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)).sum())

    def _tool_repair_collinearity(
        self, columns_to_drop: list[str], strategy: str, vif_threshold: float = 10
    ) -> str:
        """Repair collinearity issues."""
        df = self._df

        # Protect treatment and outcome
        protected = [self._state.treatment_variable, self._state.outcome_variable]

        try:
            if strategy == "drop_specified":
                columns_to_drop = [c for c in columns_to_drop if c in df.columns and c not in protected]
                if not columns_to_drop:
                    return "No valid columns to drop (treatment/outcome are protected)."
                df = df.drop(columns=columns_to_drop)
                self._df = df
                self._repairs_applied.append({
                    "type": "collinearity",
                    "strategy": "drop_specified",
                    "columns": columns_to_drop,
                })
                return f"Dropped columns: {columns_to_drop}. New shape: {df.shape}"

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
                    return f"Dropped high VIF columns: {dropped}. New shape: {df.shape}"
                return "No columns exceeded the VIF threshold."

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
                    return f"Dropped highly correlated columns: {dropped}. New shape: {df.shape}"
                return "No highly correlated column pairs found to drop."

            return f"Unknown strategy: {strategy}"

        except Exception as e:
            return f"Error repairing collinearity: {str(e)}"

    def _tool_validate_current_state(self) -> str:
        """Validate current data quality."""
        df = self._df
        df_orig = self._df_original

        # Calculate quality metrics
        missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        row_loss = (len(df_orig) - len(df)) / len(df_orig) * 100
        col_loss = (len(df_orig.columns) - len(df.columns)) / len(df_orig.columns) * 100

        # Check treatment and outcome
        treatment_ok = self._state.treatment_variable in df.columns
        outcome_ok = self._state.outcome_variable in df.columns

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

        output = "Data Quality Validation:\n"
        output += "=" * 50 + "\n"
        output += f"Shape: {df.shape} (original: {df_orig.shape})\n"
        output += f"Row loss: {row_loss:.1f}%\n"
        output += f"Column loss: {col_loss:.1f}%\n"
        output += f"Missing data: {missing_pct:.2f}%\n"
        output += f"Outlier rate: {outlier_pct:.1f}%\n"
        output += f"\nTreatment variable present: {'Yes' if treatment_ok else 'NO - CRITICAL!'}\n"
        output += f"Outcome variable present: {'Yes' if outcome_ok else 'NO - CRITICAL!'}\n"
        output += f"\nQuality Score: {score:.0f}/100\n"

        if score >= 80:
            output += "\nStatus: GOOD - Data is ready for causal analysis."
        elif score >= 60:
            output += "\nStatus: ACCEPTABLE - Minor issues remain but analysis can proceed."
        else:
            output += "\nStatus: NEEDS_WORK - Consider additional repairs."

        if not treatment_ok or not outcome_ok:
            output += "\n\nCRITICAL: Treatment or outcome variable is missing! Repairs may have been too aggressive."

        return output

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
