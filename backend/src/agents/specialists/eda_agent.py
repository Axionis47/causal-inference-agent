"""EDA Agent - Truly Agentic Exploratory Data Analysis for Causal Inference.

This agent performs EDA through iterative LLM-driven investigation:
- LLM decides what to analyze based on causal inference requirements
- Tools provide specific analyses on demand
- LLM interprets results and decides next investigation steps
- Iterates until confident about data quality and readiness

No pre-computation - all evidence gathered through tool calls.
"""

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import jarque_bera, normaltest, shapiro

from src.agents.base import AnalysisState, BaseAgent, EDAResult, JobStatus
from src.logging_config.structured import get_logger

logger = get_logger(__name__)


class EDAAgent(BaseAgent):
    """Truly agentic EDA agent that iteratively investigates data quality.

    Unlike traditional EDA that runs all analyses upfront, this agent:
    1. Starts with a data overview
    2. LLM decides what to investigate based on causal inference needs
    3. Tools execute specific analyses on demand
    4. LLM interprets results and chooses next investigation
    5. Continues until confident about data readiness

    This approach allows the LLM to focus on what matters for causal inference
    and adapt its investigation based on findings.
    """

    AGENT_NAME = "eda_agent"

    SYSTEM_PROMPT = """You are an expert data scientist performing exploratory data analysis
for a causal inference study. Your goal is to thoroughly investigate data quality and
identify issues that could affect causal effect estimation.

CRITICAL: You must ITERATIVELY investigate the data by calling tools. Do NOT try to
analyze everything at once. Instead:
1. Start by understanding the data structure
2. Investigate distributions of key variables (treatment, outcome, confounders)
3. Check for outliers that could bias estimates
4. Examine correlations and multicollinearity among covariates
5. Assess covariate balance between treatment groups
6. Draw conclusions only after gathering sufficient evidence

For causal inference, pay special attention to:
- Treatment variable: Is it binary? Well-separated?
- Outcome variable: Distribution, outliers, transformations needed?
- Confounders: Balance between groups, multicollinearity, missing patterns
- Sample size: Enough for reliable estimation?

WORKFLOW:
1. Call get_data_overview to understand the dataset
2. Call analyze_variable for treatment and outcome variables
3. Call check_covariate_balance for key confounders
4. Call detect_outliers for variables with suspicious distributions
5. Call compute_correlations to check multicollinearity
6. Call compute_vif if correlations suggest multicollinearity
7. When you have enough evidence, call finalize_eda with your assessment

Be thorough but efficient. Focus on findings that matter for causal inference.
Each tool call should have a clear purpose related to assessing causal analysis readiness."""

    TOOLS = [
        {
            "name": "get_data_overview",
            "description": "Get overview of the dataset including dimensions, variable types, missing data summary, and basic statistics. Call this FIRST to understand what you're working with.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "analyze_variable",
            "description": "Analyze a specific variable's distribution including summary stats, skewness, kurtosis, and normality tests. Use this to deeply investigate treatment, outcome, or suspicious variables.",
            "parameters": {
                "type": "object",
                "properties": {
                    "variable": {
                        "type": "string",
                        "description": "Name of the variable to analyze",
                    },
                    "include_normality_tests": {
                        "type": "boolean",
                        "description": "Whether to run normality tests (Shapiro-Wilk, D'Agostino, Jarque-Bera)",
                    },
                },
                "required": ["variable"],
            },
        },
        {
            "name": "detect_outliers",
            "description": "Detect outliers in one or more variables using IQR and Z-score methods. Returns outlier counts and percentages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Variables to check for outliers (empty for all numeric)",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["iqr", "zscore", "both"],
                        "description": "Outlier detection method (default: both)",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "compute_correlations",
            "description": "Compute correlation matrix and identify high correlations. Use this to check for multicollinearity among covariates.",
            "parameters": {
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
                        "description": "Correlation method (default: pearson)",
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Threshold for flagging high correlations (default: 0.7)",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "compute_vif",
            "description": "Compute Variance Inflation Factor for covariates to assess multicollinearity severity. Use after finding high correlations.",
            "parameters": {
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
        },
        {
            "name": "check_covariate_balance",
            "description": "Check balance of covariates between treatment and control groups using Standardized Mean Difference (SMD). Critical for causal inference.",
            "parameters": {
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
        },
        {
            "name": "check_missing_patterns",
            "description": "Analyze missing data patterns - which variables have missing values, correlations between missingness, and whether missing is related to treatment.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "finalize_eda",
            "description": "Finalize the EDA with your assessment. Call this ONLY after you have gathered sufficient evidence through other tools.",
            "parameters": {
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
                    "variables_needing_attention": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Variables that need preprocessing or special handling",
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
        },
    ]

    def __init__(self):
        """Initialize the EDA agent."""
        super().__init__()
        self._df: pd.DataFrame | None = None
        self._state: AnalysisState | None = None
        self._eda_result: EDAResult | None = None
        # Track what we've analyzed for the finalize step
        self._analyzed_distributions: dict[str, dict] = {}
        self._outlier_results: dict[str, dict] = {}
        self._correlation_results: dict | None = None
        self._vif_results: dict[str, float] = {}
        self._balance_results: dict[str, dict] = {}
        self._missing_analysis: dict | None = None

    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Execute EDA through iterative LLM-driven investigation.

        Args:
            state: Current analysis state with dataset info

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

            self._state = state
            self._eda_result = EDAResult()

            # Reset tracking
            self._analyzed_distributions = {}
            self._outlier_results = {}
            self._correlation_results = None
            self._vif_results = {}
            self._balance_results = {}
            self._missing_analysis = None

            # Build the initial prompt
            initial_prompt = self._build_initial_prompt()

            # Run the agentic loop
            final_result = await self._run_agentic_loop(initial_prompt, max_iterations=15)

            # Populate EDA result from final output
            self._populate_eda_result(final_result)

            # Update state
            state.eda_result = self._eda_result

            # Create trace
            duration_ms = int((time.time() - start_time) * 1000)
            trace = self.create_trace(
                action="eda_complete",
                reasoning=f"Completed EDA with quality score {self._eda_result.data_quality_score:.1f}",
                outputs={
                    "quality_score": self._eda_result.data_quality_score,
                    "num_findings": len(self._eda_result.summary_table.get("key_findings", [])),
                    "causal_readiness": self._eda_result.summary_table.get("causal_readiness", "unknown"),
                },
                duration_ms=duration_ms,
            )
            state.add_trace(trace)

            self.logger.info(
                "eda_complete",
                quality_score=self._eda_result.data_quality_score,
                issues_found=len(self._eda_result.data_quality_issues),
            )

        except Exception as e:
            self.logger.error("eda_failed", error=str(e))
            import traceback
            traceback.print_exc()
            state.mark_failed(f"EDA failed: {str(e)}", self.AGENT_NAME)

        return state

    def _load_dataframe(self, state: AnalysisState) -> pd.DataFrame | None:
        """Load DataFrame from pickle path."""
        if state.dataframe_path and Path(state.dataframe_path).exists():
            with open(state.dataframe_path, "rb") as f:
                return pickle.load(f)
        return None

    def _build_initial_prompt(self) -> str:
        """Build the initial prompt for the agentic loop."""
        return f"""You are performing exploratory data analysis for a causal inference study.

Dataset: {self._state.dataset_info.name or self._state.dataset_info.url}
Treatment variable: {self._state.treatment_variable}
Outcome variable: {self._state.outcome_variable}

Your goal is to assess data quality and readiness for causal inference.

Start by calling get_data_overview to understand the dataset structure, then
systematically investigate:
1. Treatment and outcome variable distributions
2. Covariate balance between treatment groups
3. Outliers and data quality issues
4. Correlations and multicollinearity

Call tools iteratively to gather evidence. When you have enough information,
call finalize_eda with your comprehensive assessment."""

    async def _run_agentic_loop(
        self, initial_prompt: str, max_iterations: int = 15
    ) -> dict[str, Any]:
        """Run the agentic loop where LLM iteratively calls tools.

        Args:
            initial_prompt: The starting prompt
            max_iterations: Maximum tool-calling iterations

        Returns:
            Final result from finalize_eda or auto-generated result
        """
        messages = [{"role": "user", "content": initial_prompt}]

        for iteration in range(max_iterations):
            self.logger.info(
                "eda_iteration",
                iteration=iteration,
                max_iterations=max_iterations,
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
                    "eda_llm_reasoning",
                    reasoning=response_text[:500] + ("..." if len(response_text) > 500 else ""),
                )

            pending_calls = result.get("pending_calls", [])

            if not pending_calls:
                self.logger.info("eda_no_tool_calls", response_preview=response_text[:200])
                if "finalize" in response_text.lower() or iteration > 10:
                    return self._auto_finalize()
                messages.append({
                    "role": "user",
                    "content": "Please continue your analysis by calling the appropriate tools. "
                    "If you have gathered enough evidence, call finalize_eda.",
                })
                continue

            # Execute tool calls
            tool_results = []
            for call in pending_calls:
                tool_name = call.get("name")
                tool_args = call.get("args", {})

                # Check for finalize with detailed logging
                if tool_name == "finalize_eda":
                    self.logger.info(
                        "eda_finalizing",
                        data_quality_score=tool_args.get("data_quality_score"),
                        num_findings=len(tool_args.get("key_findings", [])),
                        num_issues=len(tool_args.get("data_quality_issues", [])),
                        causal_readiness=tool_args.get("causal_readiness"),
                    )
                    return tool_args

                # Log the tool call decision
                self.logger.info(
                    "eda_tool_decision",
                    tool=tool_name,
                    args=tool_args,
                )

                # Execute the tool
                tool_result = self._execute_tool(tool_name, tool_args)
                tool_results.append(f"## {tool_name}\n{tool_result}")

                # Log tool result summary
                self.logger.info(
                    "eda_tool_result",
                    tool=tool_name,
                    result_summary=tool_result[:200] + ("..." if len(tool_result) > 200 else ""),
                )

            # Feed results back to LLM
            results_text = "\n\n".join(tool_results)
            messages.append({
                "role": "user",
                "content": f"Tool results:\n\n{results_text}\n\nAnalyze these results and continue your investigation. "
                f"Call more tools if needed, or call finalize_eda when you have enough evidence.",
            })

        # Max iterations reached - auto-finalize
        self.logger.warning("eda_max_iterations_reached")
        return self._auto_finalize()

    def _execute_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        """Execute a tool and return results as string.

        Args:
            tool_name: Name of the tool to execute
            args: Tool arguments

        Returns:
            String representation of tool results
        """
        try:
            if tool_name == "get_data_overview":
                return self._tool_get_data_overview()
            elif tool_name == "analyze_variable":
                return self._tool_analyze_variable(
                    args.get("variable", ""),
                    args.get("include_normality_tests", True),
                )
            elif tool_name == "detect_outliers":
                return self._tool_detect_outliers(
                    args.get("variables", []),
                    args.get("method", "both"),
                )
            elif tool_name == "compute_correlations":
                return self._tool_compute_correlations(
                    args.get("variables", []),
                    args.get("method", "pearson"),
                    args.get("threshold", 0.7),
                )
            elif tool_name == "compute_vif":
                return self._tool_compute_vif(args.get("covariates", []))
            elif tool_name == "check_covariate_balance":
                return self._tool_check_covariate_balance(args.get("covariates", []))
            elif tool_name == "check_missing_patterns":
                return self._tool_check_missing_patterns()
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            self.logger.error("tool_execution_error", tool=tool_name, error=str(e))
            return f"Error executing {tool_name}: {str(e)}"

    def _tool_get_data_overview(self) -> str:
        """Get overview of the dataset."""
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
        treatment_col = self._state.treatment_variable
        treatment_info = "Not specified"
        if treatment_col and treatment_col in df.columns:
            t_vals = df[treatment_col].value_counts()
            treatment_info = f"Values: {dict(t_vals)}, Missing: {df[treatment_col].isnull().sum()}"

        # Outcome info
        outcome_col = self._state.outcome_variable
        outcome_info = "Not specified"
        if outcome_col and outcome_col in df.columns:
            outcome_info = (
                f"Mean: {df[outcome_col].mean():.3f}, "
                f"Std: {df[outcome_col].std():.3f}, "
                f"Missing: {df[outcome_col].isnull().sum()}"
            )

        # Potential confounders
        confounders = []
        if self._state.data_profile and self._state.data_profile.potential_confounders:
            confounders = self._state.data_profile.potential_confounders

        result = f"""Dataset Overview:
- Rows: {n_rows}
- Columns: {n_cols}
- Numeric columns: {len(numeric_cols)} ({', '.join(numeric_cols[:10])}{'...' if len(numeric_cols) > 10 else ''})
- Categorical columns: {len(categorical_cols)} ({', '.join(categorical_cols[:5])}{'...' if len(categorical_cols) > 5 else ''})
- Binary columns: {len(binary_cols)} ({', '.join(binary_cols[:5])}{'...' if len(binary_cols) > 5 else ''})

Missing Data:
- Total missing: {total_missing_pct:.2f}%
- Columns with missing: {len(cols_with_missing)}
{self._format_missing_summary(cols_with_missing, n_rows)}

Treatment Variable ({treatment_col}):
{treatment_info}

Outcome Variable ({outcome_col}):
{outcome_info}

Potential Confounders: {len(confounders)}
{', '.join(confounders[:10])}{'...' if len(confounders) > 10 else ''}"""

        return result

    def _format_missing_summary(self, missing_counts: pd.Series, n_rows: int) -> str:
        """Format missing data summary."""
        if len(missing_counts) == 0:
            return "  No missing values"

        lines = []
        for col, count in missing_counts.head(10).items():
            pct = count / n_rows * 100
            lines.append(f"  - {col}: {count} ({pct:.1f}%)")
        if len(missing_counts) > 10:
            lines.append(f"  ... and {len(missing_counts) - 10} more columns")
        return "\n".join(lines)

    def _tool_analyze_variable(self, variable: str, include_normality_tests: bool = True) -> str:
        """Analyze a specific variable's distribution."""
        if variable not in self._df.columns:
            return f"Variable '{variable}' not found in dataset"

        data = self._df[variable].dropna()
        n = len(data)

        if n == 0:
            return f"Variable '{variable}' has no non-missing values"

        result_dict = {"variable": variable, "n": n}

        # Check if numeric or categorical
        if pd.api.types.is_numeric_dtype(self._df[variable]):
            result_dict["type"] = "numeric"
            result_dict["mean"] = float(data.mean())
            result_dict["median"] = float(data.median())
            result_dict["std"] = float(data.std())
            result_dict["min"] = float(data.min())
            result_dict["max"] = float(data.max())
            result_dict["q1"] = float(data.quantile(0.25))
            result_dict["q3"] = float(data.quantile(0.75))
            result_dict["skewness"] = float(stats.skew(data))
            result_dict["kurtosis"] = float(stats.kurtosis(data))
            result_dict["unique_values"] = int(data.nunique())

            # Normality tests
            if include_normality_tests and n >= 8:
                normality = {}
                try:
                    if n <= 5000:
                        stat, p = shapiro(data.sample(min(n, 5000), random_state=42))
                        normality["shapiro_p"] = float(p)
                        normality["shapiro_normal"] = p > 0.05
                    if n >= 20:
                        stat, p = normaltest(data)
                        normality["dagostino_p"] = float(p)
                        normality["dagostino_normal"] = p > 0.05
                    stat, p = jarque_bera(data)
                    normality["jarque_bera_p"] = float(p)
                    normality["jarque_bera_normal"] = p > 0.05
                except Exception:
                    pass
                result_dict["normality_tests"] = normality

            # Store for later
            self._analyzed_distributions[variable] = result_dict

            # Format output
            output = f"""Variable: {variable} (numeric)
- N: {n}, Missing: {len(self._df) - n}
- Mean: {result_dict['mean']:.4f}, Median: {result_dict['median']:.4f}
- Std: {result_dict['std']:.4f}
- Range: [{result_dict['min']:.4f}, {result_dict['max']:.4f}]
- IQR: [{result_dict['q1']:.4f}, {result_dict['q3']:.4f}]
- Skewness: {result_dict['skewness']:.3f} ({'highly skewed' if abs(result_dict['skewness']) > 1 else 'moderately skewed' if abs(result_dict['skewness']) > 0.5 else 'approximately symmetric'})
- Kurtosis: {result_dict['kurtosis']:.3f} ({'heavy tails' if result_dict['kurtosis'] > 1 else 'light tails' if result_dict['kurtosis'] < -1 else 'normal tails'})
- Unique values: {result_dict['unique_values']}"""

            if "normality_tests" in result_dict:
                norm = result_dict["normality_tests"]
                output += "\n\nNormality Tests:"
                if "shapiro_p" in norm:
                    output += f"\n- Shapiro-Wilk p={norm['shapiro_p']:.4f} ({'Normal' if norm['shapiro_normal'] else 'Non-normal'})"
                if "dagostino_p" in norm:
                    output += f"\n- D'Agostino p={norm['dagostino_p']:.4f} ({'Normal' if norm['dagostino_normal'] else 'Non-normal'})"
                if "jarque_bera_p" in norm:
                    output += f"\n- Jarque-Bera p={norm['jarque_bera_p']:.4f} ({'Normal' if norm['jarque_bera_normal'] else 'Non-normal'})"

        else:
            # Categorical variable
            result_dict["type"] = "categorical"
            value_counts = data.value_counts()
            result_dict["unique_values"] = int(data.nunique())
            result_dict["top_values"] = value_counts.head(10).to_dict()

            output = f"""Variable: {variable} (categorical)
- N: {n}, Missing: {len(self._df) - n}
- Unique values: {result_dict['unique_values']}
- Top values:
"""
            for val, count in value_counts.head(10).items():
                output += f"  - {val}: {count} ({count/n*100:.1f}%)\n"

        return output

    def _tool_detect_outliers(self, variables: list[str], method: str = "both") -> str:
        """Detect outliers in specified variables."""
        df = self._df

        # Default to all numeric if not specified
        if not variables:
            variables = df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude treatment/outcome for focused analysis
            if self._state.treatment_variable in variables:
                variables.remove(self._state.treatment_variable)

        results = []
        for var in variables:
            if var not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[var]):
                continue

            data = df[var].dropna()
            if len(data) < 10:
                continue

            outlier_info = {
                "variable": var,
                "n": len(data),
            }

            # IQR method
            if method in ["iqr", "both"]:
                q1 = data.quantile(0.25)
                q3 = data.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                iqr_outliers = ((data < lower) | (data > upper)).sum()
                outlier_info["iqr_outliers"] = int(iqr_outliers)
                outlier_info["iqr_pct"] = float(iqr_outliers / len(data) * 100)
                outlier_info["iqr_bounds"] = (float(lower), float(upper))

            # Z-score method
            if method in ["zscore", "both"]:
                z_scores = np.abs(stats.zscore(data))
                zscore_outliers = (z_scores > 3).sum()
                outlier_info["zscore_outliers"] = int(zscore_outliers)
                outlier_info["zscore_pct"] = float(zscore_outliers / len(data) * 100)

            # Store for later
            if outlier_info.get("iqr_outliers", 0) > 0 or outlier_info.get("zscore_outliers", 0) > 0:
                self._outlier_results[var] = outlier_info

            results.append(outlier_info)

        # Format output
        output = "Outlier Detection Results:\n"
        output += "=" * 50 + "\n"

        has_outliers = False
        for r in results:
            iqr_out = r.get("iqr_outliers", 0)
            zscore_out = r.get("zscore_outliers", 0)
            if iqr_out > 0 or zscore_out > 0:
                has_outliers = True
                output += f"\n{r['variable']}:\n"
                if "iqr_outliers" in r:
                    output += f"  - IQR method: {iqr_out} outliers ({r['iqr_pct']:.1f}%)\n"
                    output += f"    Bounds: [{r['iqr_bounds'][0]:.3f}, {r['iqr_bounds'][1]:.3f}]\n"
                if "zscore_outliers" in r:
                    output += f"  - Z-score (>3): {zscore_out} outliers ({r['zscore_pct']:.1f}%)\n"

        if not has_outliers:
            output += "\nNo significant outliers detected in the analyzed variables."

        return output

    def _tool_compute_correlations(
        self, variables: list[str], method: str = "pearson", threshold: float = 0.7
    ) -> str:
        """Compute correlations and identify high correlations."""
        df = self._df

        # Default to all numeric if not specified
        if not variables:
            variables = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(variables) < 2:
            return "Need at least 2 numeric variables for correlation analysis"

        # Filter to valid columns
        variables = [v for v in variables if v in df.columns and pd.api.types.is_numeric_dtype(df[v])]

        if len(variables) < 2:
            return "Need at least 2 numeric variables for correlation analysis"

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
                            "correlation": float(corr_val),
                        })

        high_corrs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        # Store for later
        self._correlation_results = {
            "method": method,
            "n_variables": len(variables),
            "high_correlations": high_corrs,
            "matrix": corr_matrix.to_dict(),
        }

        # Store in EDA result
        self._eda_result.correlation_matrix = corr_matrix.to_dict()
        self._eda_result.high_correlations = high_corrs

        # Format output
        output = f"Correlation Analysis ({method}, threshold={threshold}):\n"
        output += f"Analyzed {len(variables)} variables\n"
        output += "=" * 50 + "\n"

        if high_corrs:
            output += f"\nHigh Correlations Found ({len(high_corrs)}):\n"
            for hc in high_corrs[:15]:
                output += f"  - {hc['var1']} <-> {hc['var2']}: r={hc['correlation']:.3f}"
                if abs(hc["correlation"]) > 0.9:
                    output += " (VERY HIGH - multicollinearity concern)"
                elif abs(hc["correlation"]) > 0.8:
                    output += " (HIGH - potential multicollinearity)"
                output += "\n"
            if len(high_corrs) > 15:
                output += f"  ... and {len(high_corrs) - 15} more\n"
        else:
            output += "\nNo correlations above threshold found."

        return output

    def _tool_compute_vif(self, covariates: list[str]) -> str:
        """Compute VIF for multicollinearity assessment."""
        df = self._df

        # Default covariates
        if not covariates:
            covariates = df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove treatment and outcome
            for var in [self._state.treatment_variable, self._state.outcome_variable]:
                if var in covariates:
                    covariates.remove(var)

        # Filter to valid columns
        covariates = [
            c for c in covariates
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]

        if len(covariates) < 2:
            return "Need at least 2 covariates for VIF calculation"

        # Prepare data
        X = df[covariates].dropna()
        if len(X) < 10:
            return "Not enough complete cases for VIF calculation"

        # Compute VIF
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        X_with_const = np.column_stack([np.ones(len(X)), X.values])

        vif_results = []
        warnings = []

        for i, col in enumerate(covariates):
            try:
                vif = variance_inflation_factor(X_with_const, i + 1)
                vif_results.append({"variable": col, "vif": float(vif)})
                self._vif_results[col] = float(vif)

                if vif > 10:
                    warnings.append(f"SEVERE: {col} (VIF={vif:.1f})")
                elif vif > 5:
                    warnings.append(f"MODERATE: {col} (VIF={vif:.1f})")
            except Exception as e:
                vif_results.append({"variable": col, "vif": None, "error": str(e)})

        # Store warnings
        self._eda_result.vif_scores = self._vif_results
        self._eda_result.multicollinearity_warnings = warnings

        # Format output
        vif_results.sort(key=lambda x: x.get("vif", 0) or 0, reverse=True)

        output = "Variance Inflation Factor (VIF) Analysis:\n"
        output += "=" * 50 + "\n"
        output += "VIF > 10: Severe multicollinearity\n"
        output += "VIF > 5: Moderate multicollinearity\n"
        output += "VIF < 5: Generally acceptable\n\n"

        for r in vif_results[:20]:
            if r.get("vif") is not None:
                vif = r["vif"]
                flag = ""
                if vif > 10:
                    flag = " *** SEVERE ***"
                elif vif > 5:
                    flag = " ** MODERATE **"
                output += f"  {r['variable']}: VIF={vif:.2f}{flag}\n"

        if warnings:
            output += f"\nMulticollinearity warnings: {len(warnings)}\n"
            for w in warnings:
                output += f"  - {w}\n"

        return output

    def _tool_check_covariate_balance(self, covariates: list[str]) -> str:
        """Check covariate balance between treatment groups."""
        df = self._df
        treatment_col = self._state.treatment_variable

        if not treatment_col or treatment_col not in df.columns:
            return "Treatment variable not specified or not found"

        # Check binary treatment
        treatment_values = df[treatment_col].dropna().unique()
        if len(treatment_values) != 2:
            return f"Treatment variable is not binary (found {len(treatment_values)} values)"

        # Get groups
        treated = df[df[treatment_col] == treatment_values.max()]
        control = df[df[treatment_col] == treatment_values.min()]

        # Default covariates
        if not covariates:
            if self._state.data_profile and self._state.data_profile.potential_confounders:
                covariates = self._state.data_profile.potential_confounders
            else:
                covariates = df.select_dtypes(include=[np.number]).columns.tolist()
                for var in [treatment_col, self._state.outcome_variable]:
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

            if pooled_std > 0:
                smd = abs(mean_diff / pooled_std)
            else:
                smd = 0.0

            # T-test
            try:
                _, p_value = stats.ttest_ind(treated_vals, control_vals)
            except Exception:
                p_value = 1.0

            balance_info = {
                "covariate": cov,
                "treated_mean": float(treated_vals.mean()),
                "control_mean": float(control_vals.mean()),
                "smd": float(smd),
                "p_value": float(p_value),
                "is_balanced": smd < 0.1,
            }

            results.append(balance_info)
            self._balance_results[cov] = balance_info

            if smd >= 0.1:
                imbalanced.append((cov, smd))

        # Store in EDA result
        self._eda_result.covariate_balance = self._balance_results

        # Format output
        output = f"Covariate Balance Check (Treatment: {treatment_col}):\n"
        output += f"Treated: n={len(treated)}, Control: n={len(control)}\n"
        output += "=" * 50 + "\n"
        output += "SMD < 0.1: Well-balanced\n"
        output += "SMD 0.1-0.25: Moderate imbalance\n"
        output += "SMD > 0.25: Severe imbalance\n\n"

        # Sort by SMD
        results.sort(key=lambda x: x["smd"], reverse=True)

        for r in results[:20]:
            smd = r["smd"]
            flag = ""
            if smd > 0.25:
                flag = " *** SEVERE IMBALANCE ***"
            elif smd >= 0.1:
                flag = " ** IMBALANCED **"
            output += f"  {r['covariate']}: SMD={smd:.3f}, p={r['p_value']:.3f}{flag}\n"

        if len(results) > 20:
            output += f"  ... and {len(results) - 20} more covariates\n"

        output += f"\nSummary: {len(imbalanced)} of {len(results)} covariates are imbalanced (SMD >= 0.1)"

        if imbalanced:
            self._eda_result.balance_summary = f"Imbalanced covariates: {', '.join([f'{c} (SMD={s:.3f})' for c, s in imbalanced[:5]])}"
        else:
            self._eda_result.balance_summary = "All covariates are well-balanced (SMD < 0.1)"

        return output

    def _tool_check_missing_patterns(self) -> str:
        """Analyze missing data patterns."""
        df = self._df

        # Overall missing
        missing_counts = df.isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0]

        if len(cols_with_missing) == 0:
            self._missing_analysis = {"has_missing": False}
            return "No missing values in the dataset."

        n_rows = len(df)
        total_missing = df.isnull().sum().sum()
        total_cells = n_rows * len(df.columns)

        # Per-column analysis
        col_analysis = []
        for col in cols_with_missing.index:
            col_analysis.append({
                "column": col,
                "missing": int(cols_with_missing[col]),
                "pct": float(cols_with_missing[col] / n_rows * 100),
            })

        col_analysis.sort(key=lambda x: x["missing"], reverse=True)

        # Check if missing is related to treatment
        treatment_col = self._state.treatment_variable
        missing_by_treatment = {}
        if treatment_col and treatment_col in df.columns:
            for col in cols_with_missing.index[:10]:
                treated_missing = df[df[treatment_col] == 1][col].isnull().mean()
                control_missing = df[df[treatment_col] == 0][col].isnull().mean()
                if abs(treated_missing - control_missing) > 0.05:
                    missing_by_treatment[col] = {
                        "treated_pct": float(treated_missing * 100),
                        "control_pct": float(control_missing * 100),
                        "diff": float(abs(treated_missing - control_missing) * 100),
                    }

        self._missing_analysis = {
            "has_missing": True,
            "total_missing_pct": float(total_missing / total_cells * 100),
            "n_cols_with_missing": len(cols_with_missing),
            "by_column": col_analysis,
            "differential_missing": missing_by_treatment,
        }

        # Format output
        output = "Missing Data Analysis:\n"
        output += "=" * 50 + "\n"
        output += f"Total missing: {total_missing} cells ({total_missing / total_cells * 100:.2f}%)\n"
        output += f"Columns with missing: {len(cols_with_missing)} of {len(df.columns)}\n\n"

        output += "By Column:\n"
        for ca in col_analysis[:15]:
            output += f"  - {ca['column']}: {ca['missing']} ({ca['pct']:.1f}%)\n"
        if len(col_analysis) > 15:
            output += f"  ... and {len(col_analysis) - 15} more\n"

        if missing_by_treatment:
            output += "\nDifferential Missing (by treatment):\n"
            for col, info in missing_by_treatment.items():
                output += f"  - {col}: Treated={info['treated_pct']:.1f}%, Control={info['control_pct']:.1f}% (diff={info['diff']:.1f}%)\n"
            output += "\nWARNING: Differential missing may indicate selection bias!\n"

        return output

    def _auto_finalize(self) -> dict[str, Any]:
        """Auto-generate final result from collected evidence."""
        self.logger.info("eda_auto_finalize")

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
            recommendations = [
                "Proceed with causal analysis using appropriate methods",
                "Consider multiple estimation methods to check robustness",
            ]

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
