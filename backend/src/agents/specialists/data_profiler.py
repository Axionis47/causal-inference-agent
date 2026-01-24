"""Data Profiler Agent - Truly Agentic Dataset Analysis for Causal Inference.

This agent iteratively explores datasets to identify causal structure:
- LLM decides what to investigate based on causal inference requirements
- Tools provide specific analyses on demand
- LLM interprets findings and chooses next investigation
- Iterates until confident about treatment, outcome, and confounders

No pre-computation of causal structure - all identification through tool calls.
"""

import pickle
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.agents.base import AnalysisState, BaseAgent, DataProfile, JobStatus
from src.logging_config.structured import get_logger

logger = get_logger(__name__)


class DataProfilerAgent(BaseAgent):
    """Truly agentic data profiler that iteratively identifies causal structure.

    Unlike traditional profiling that computes everything upfront, this agent:
    1. Starts with a data overview
    2. LLM investigates columns to understand their role
    3. LLM identifies treatment, outcome, and confounders through investigation
    4. Continues until confident about causal structure

    This approach allows intelligent identification based on actual data patterns.
    """

    AGENT_NAME = "data_profiler"

    SYSTEM_PROMPT = """You are an expert data scientist specializing in causal inference.
Your role is to analyze datasets and identify their causal structure for effect estimation.

CRITICAL: You must ITERATIVELY investigate the data by calling tools. Do NOT try to
identify everything at once. Instead:
1. Start by understanding the dataset structure
2. Investigate promising treatment candidates (binary/categorical with good balance)
3. Investigate outcome candidates (numeric variables of interest)
4. Identify confounders (variables that might affect both treatment and outcome)
5. Draw conclusions only after gathering sufficient evidence

When identifying causal structure, look for:
- Treatment variables: Binary or categorical interventions with reasonable balance
- Outcome variables: Numeric or binary results that could be affected by treatment
- Confounders: Pre-treatment variables that affect both treatment assignment and outcome
- Instruments: Variables that affect treatment but not outcome directly (for IV methods)
- Time variables: For DiD or panel methods
- Running variables: For regression discontinuity designs

Key considerations:
- Column NAMES often reveal their role (treatment, outcome, age, income, etc.)
- Column TYPES matter (binary suggests treatment, continuous suggests outcome/confounder)
- BALANCE of treatment variable is critical (10-90% treated is ideal)
- Exclude ID columns, indices, and clearly irrelevant features

WORKFLOW:
1. Call get_dataset_overview to understand the data
2. Call analyze_column for promising treatment candidates
3. Call check_treatment_balance for best treatment candidates
4. Call analyze_column for outcome candidates
5. Identify confounders based on domain knowledge and correlations
6. Call finalize_profile when you have identified the causal structure"""

    TOOLS = [
        {
            "name": "get_dataset_overview",
            "description": "Get overview of the dataset including shape, columns, types, and missing values. Call this FIRST.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "analyze_column",
            "description": "Analyze a specific column in detail including distribution, unique values, and statistics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Name of the column to analyze",
                    },
                },
                "required": ["column"],
            },
        },
        {
            "name": "check_treatment_balance",
            "description": "Check if a column could be a valid treatment variable by examining its balance and distribution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Name of the potential treatment column",
                    },
                },
                "required": ["column"],
            },
        },
        {
            "name": "check_column_relationship",
            "description": "Check the relationship between two columns (correlation for numeric, association for categorical).",
            "parameters": {
                "type": "object",
                "properties": {
                    "column1": {
                        "type": "string",
                        "description": "First column name",
                    },
                    "column2": {
                        "type": "string",
                        "description": "Second column name",
                    },
                },
                "required": ["column1", "column2"],
            },
        },
        {
            "name": "check_time_dimension",
            "description": "Check if the dataset has a time dimension suitable for DiD or panel methods.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "check_discontinuity_candidates",
            "description": "Check for running variables that could enable regression discontinuity design.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "finalize_profile",
            "description": "Finalize the data profile with identified causal structure. Call this ONLY after investigation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "treatment_candidates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns suitable as treatment variables (best first)",
                    },
                    "outcome_candidates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns suitable as outcome variables (best first)",
                    },
                    "potential_confounders": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns that could be confounders",
                    },
                    "potential_instruments": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns that could be instrumental variables",
                    },
                    "has_time_dimension": {
                        "type": "boolean",
                        "description": "Whether dataset has a time dimension",
                    },
                    "time_column": {
                        "type": "string",
                        "description": "Name of time column if exists",
                    },
                    "discontinuity_candidates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns suitable for RDD running variable",
                    },
                    "recommended_methods": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Causal inference methods suitable for this data",
                    },
                },
                "required": ["treatment_candidates", "outcome_candidates", "potential_confounders"],
            },
        },
    ]

    def __init__(self):
        """Initialize the data profiler agent."""
        super().__init__()
        self._df: pd.DataFrame | None = None
        self._state: AnalysisState | None = None
        self._profile: DataProfile | None = None

    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Execute data profiling through iterative LLM-driven investigation.

        Args:
            state: Current analysis state with dataset info

        Returns:
            Updated state with data profile
        """
        self.logger.info(
            "profiling_start",
            job_id=state.job_id,
            dataset=state.dataset_info.name or state.dataset_info.url,
        )

        state.status = JobStatus.PROFILING
        start_time = time.time()

        try:
            # Load the dataset
            self._df = await self._load_dataset(state)
            if self._df is None:
                state.mark_failed("Failed to load dataset", self.AGENT_NAME)
                return state

            self._state = state

            # Compute basic profile (shape, types, missing)
            self._profile = self._compute_basic_profile(self._df)

            # Save DataFrame for other agents
            df_path = self._save_dataframe(self._df, state.job_id)
            state.dataframe_path = df_path

            # Build the initial prompt
            initial_prompt = self._build_initial_prompt()

            # Run the agentic loop to identify causal structure
            final_result = await self._run_agentic_loop(initial_prompt, max_iterations=15)

            # Populate profile from final result
            self._populate_profile(final_result)

            # Update state
            state.data_profile = self._profile
            state.dataset_info.n_samples = self._profile.n_samples
            state.dataset_info.n_features = self._profile.n_features

            # Create trace
            duration_ms = int((time.time() - start_time) * 1000)
            trace = self.create_trace(
                action="profile_complete",
                reasoning=f"Profiled dataset with {self._profile.n_samples} samples, {self._profile.n_features} features",
                outputs={
                    "treatment_candidates": self._profile.treatment_candidates,
                    "outcome_candidates": self._profile.outcome_candidates,
                    "potential_confounders": self._profile.potential_confounders[:5],
                    "has_time": self._profile.has_time_dimension,
                },
                duration_ms=duration_ms,
            )
            state.add_trace(trace)

            self.logger.info(
                "profiling_complete",
                samples=self._profile.n_samples,
                features=self._profile.n_features,
                treatment_candidates=len(self._profile.treatment_candidates),
                outcome_candidates=len(self._profile.outcome_candidates),
            )

        except Exception as e:
            self.logger.error("profiling_failed", error=str(e))
            import traceback
            traceback.print_exc()
            state.mark_failed(f"Data profiling failed: {str(e)}", self.AGENT_NAME)

        return state

    async def _load_dataset(self, state: AnalysisState) -> pd.DataFrame | None:
        """Load dataset from the source."""
        dataset_info = state.dataset_info

        # If dataframe_path already exists, load from there
        if state.dataframe_path:
            try:
                if state.dataframe_path.endswith(".pkl"):
                    with open(state.dataframe_path, "rb") as f:
                        return pickle.load(f)
                elif state.dataframe_path.endswith(".csv"):
                    return pd.read_csv(state.dataframe_path)
                elif state.dataframe_path.endswith(".parquet"):
                    return pd.read_parquet(state.dataframe_path)
                else:
                    with open(state.dataframe_path, "rb") as f:
                        return pickle.load(f)
            except Exception as e:
                self.logger.error("dataframe_path_load_failed", error=str(e), path=state.dataframe_path)

        # If local path exists, load from there
        if dataset_info.local_path:
            try:
                if dataset_info.local_path.endswith(".csv"):
                    return pd.read_csv(dataset_info.local_path)
                elif dataset_info.local_path.endswith(".parquet"):
                    return pd.read_parquet(dataset_info.local_path)
                else:
                    return pd.read_csv(dataset_info.local_path)
            except Exception as e:
                self.logger.error("local_load_failed", error=str(e))

        # If it's a Kaggle URL, use Kaggle API
        if "kaggle.com" in dataset_info.url:
            return await self._load_from_kaggle(dataset_info.url)

        # Try loading directly as URL
        try:
            return pd.read_csv(dataset_info.url)
        except Exception as e:
            self.logger.error("url_load_failed", error=str(e))
            return None

    async def _load_from_kaggle(self, url: str) -> pd.DataFrame | None:
        """Load dataset from Kaggle."""
        import json
        import os

        from src.config import get_settings

        settings = get_settings()

        # Set Kaggle credentials from config if available
        # Otherwise, Kaggle API will use ~/.kaggle/kaggle.json
        if settings.kaggle_key_value:
            kaggle_key = settings.kaggle_key_value
            kaggle_username = settings.kaggle_username

            # Handle JSON format: {"username": "...", "key": "..."}
            if kaggle_key.startswith("{"):
                try:
                    kaggle_creds = json.loads(kaggle_key)
                    kaggle_username = kaggle_creds.get("username", kaggle_username)
                    kaggle_key = kaggle_creds.get("key", kaggle_key)
                except json.JSONDecodeError:
                    pass  # Not JSON, use as-is

            os.environ["KAGGLE_USERNAME"] = kaggle_username
            os.environ["KAGGLE_KEY"] = kaggle_key

        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()

            parts = url.rstrip("/").split("/")
            if "datasets" in parts:
                idx = parts.index("datasets")
                owner = parts[idx + 1]
                dataset_name = parts[idx + 2]
                dataset_id = f"{owner}/{dataset_name}"
            else:
                self.logger.error("invalid_kaggle_url", url=url)
                return None

            with tempfile.TemporaryDirectory() as tmpdir:
                api.dataset_download_files(dataset_id, path=tmpdir, unzip=True)

                csv_files = list(Path(tmpdir).glob("*.csv"))
                if csv_files:
                    largest = max(csv_files, key=lambda p: p.stat().st_size)
                    return pd.read_csv(largest)

                parquet_files = list(Path(tmpdir).glob("*.parquet"))
                if parquet_files:
                    return pd.read_parquet(parquet_files[0])

                self.logger.error("no_data_files_found", path=tmpdir)
                return None

        except Exception as e:
            self.logger.error("kaggle_load_failed", error=str(e))
            return None

    def _compute_basic_profile(self, df: pd.DataFrame) -> DataProfile:
        """Compute basic statistical profile of the dataset."""
        feature_types = {}
        numeric_stats = {}
        categorical_stats = {}
        missing_values = {}

        for col in df.columns:
            # Determine type
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() <= 2:
                    feature_types[col] = "binary"
                elif df[col].nunique() <= 10:
                    feature_types[col] = "ordinal"
                else:
                    feature_types[col] = "numeric"

                # Compute numeric stats
                numeric_stats[col] = {
                    "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else 0,
                    "std": float(df[col].std()) if not pd.isna(df[col].std()) else 0,
                    "min": float(df[col].min()) if not pd.isna(df[col].min()) else 0,
                    "max": float(df[col].max()) if not pd.isna(df[col].max()) else 0,
                    "median": float(df[col].median()) if not pd.isna(df[col].median()) else 0,
                }
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                feature_types[col] = "datetime"
            else:
                if df[col].nunique() <= 20:
                    feature_types[col] = "categorical"
                    value_counts = df[col].value_counts().head(10).to_dict()
                    categorical_stats[col] = {str(k): int(v) for k, v in value_counts.items()}
                else:
                    feature_types[col] = "text"

            missing_values[col] = int(df[col].isna().sum())

        return DataProfile(
            n_samples=len(df),
            n_features=len(df.columns),
            feature_names=list(df.columns),
            feature_types=feature_types,
            missing_values=missing_values,
            numeric_stats=numeric_stats,
            categorical_stats=categorical_stats,
        )

    def _build_initial_prompt(self) -> str:
        """Build the initial prompt for the agentic loop."""
        return f"""You are profiling a dataset for causal inference analysis.

Dataset: {self._state.dataset_info.name or self._state.dataset_info.url}
Shape: {self._profile.n_samples} samples x {self._profile.n_features} features

Your goal is to identify the causal structure:
1. Which columns could serve as TREATMENT variables
2. Which columns could serve as OUTCOME variables
3. Which columns are potential CONFOUNDERS

Start by calling get_dataset_overview to see all columns and their types,
then systematically investigate promising candidates.

Focus on:
- Binary/categorical columns with good balance for treatment
- Numeric columns that represent meaningful outcomes
- Pre-treatment variables as potential confounders

Call tools to gather evidence, then call finalize_profile with your conclusions."""

    async def _run_agentic_loop(
        self, initial_prompt: str, max_iterations: int = 15
    ) -> dict[str, Any]:
        """Run the agentic loop where LLM iteratively investigates."""
        messages = [{"role": "user", "content": initial_prompt}]

        for iteration in range(max_iterations):
            self.logger.info(
                "profiler_iteration",
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
                    "profiler_llm_reasoning",
                    reasoning=response_text[:500] + ("..." if len(response_text) > 500 else ""),
                )

            pending_calls = result.get("pending_calls", [])

            if not pending_calls:
                self.logger.info("profiler_no_tool_calls", response_preview=response_text[:200])
                if "finalize" in response_text.lower() or iteration > 10:
                    return self._auto_finalize()
                messages.append({
                    "role": "user",
                    "content": "Please continue investigating by calling tools. "
                    "If you have identified the causal structure, call finalize_profile.",
                })
                continue

            # Execute tool calls
            tool_results = []
            for call in pending_calls:
                tool_name = call.get("name")
                tool_args = call.get("args", {})

                # Check for finalize with detailed logging
                if tool_name == "finalize_profile":
                    self.logger.info(
                        "profiler_finalizing",
                        treatment_candidates=tool_args.get("treatment_candidates", []),
                        outcome_candidates=tool_args.get("outcome_candidates", []),
                        confounders=tool_args.get("potential_confounders", [])[:5],
                        recommended_methods=tool_args.get("recommended_methods", []),
                    )
                    return tool_args

                # Log the tool call decision
                self.logger.info(
                    "profiler_tool_decision",
                    tool=tool_name,
                    args=tool_args,
                    reason=f"Investigating {tool_args}" if tool_args else "Getting overview",
                )

                # Execute the tool
                tool_result = self._execute_tool(tool_name, tool_args)
                tool_results.append(f"## {tool_name}\n{tool_result}")

                # Log tool result summary (first 200 chars)
                self.logger.info(
                    "profiler_tool_result",
                    tool=tool_name,
                    result_summary=tool_result[:200] + ("..." if len(tool_result) > 200 else ""),
                )

            # Feed results back to LLM
            results_text = "\n\n".join(tool_results)
            messages.append({
                "role": "user",
                "content": f"Tool results:\n\n{results_text}\n\nContinue investigating or call finalize_profile.",
            })

        # Max iterations reached - auto-finalize
        self.logger.warning("profiler_max_iterations_reached")
        return self._auto_finalize()

    def _execute_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        """Execute a tool and return results as string."""
        try:
            if tool_name == "get_dataset_overview":
                return self._tool_get_dataset_overview()
            elif tool_name == "analyze_column":
                return self._tool_analyze_column(args.get("column", ""))
            elif tool_name == "check_treatment_balance":
                return self._tool_check_treatment_balance(args.get("column", ""))
            elif tool_name == "check_column_relationship":
                return self._tool_check_column_relationship(
                    args.get("column1", ""),
                    args.get("column2", ""),
                )
            elif tool_name == "check_time_dimension":
                return self._tool_check_time_dimension()
            elif tool_name == "check_discontinuity_candidates":
                return self._tool_check_discontinuity_candidates()
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            self.logger.error("tool_execution_error", tool=tool_name, error=str(e))
            return f"Error executing {tool_name}: {str(e)}"

    def _tool_get_dataset_overview(self) -> str:
        """Get overview of the dataset."""
        df = self._df
        profile = self._profile

        # Group columns by type
        binary_cols = [c for c, t in profile.feature_types.items() if t == "binary"]
        numeric_cols = [c for c, t in profile.feature_types.items() if t == "numeric"]
        ordinal_cols = [c for c, t in profile.feature_types.items() if t == "ordinal"]
        categorical_cols = [c for c, t in profile.feature_types.items() if t == "categorical"]
        datetime_cols = [c for c, t in profile.feature_types.items() if t == "datetime"]
        text_cols = [c for c, t in profile.feature_types.items() if t == "text"]

        # Missing summary
        cols_with_missing = {c: v for c, v in profile.missing_values.items() if v > 0}
        total_missing_pct = sum(cols_with_missing.values()) / (df.shape[0] * df.shape[1]) * 100

        output = f"""Dataset Overview:
Shape: {profile.n_samples} samples x {profile.n_features} features
Total missing: {total_missing_pct:.2f}%

Column Types:
- Binary ({len(binary_cols)}): {', '.join(binary_cols[:10])}{'...' if len(binary_cols) > 10 else ''}
- Numeric ({len(numeric_cols)}): {', '.join(numeric_cols[:10])}{'...' if len(numeric_cols) > 10 else ''}
- Ordinal ({len(ordinal_cols)}): {', '.join(ordinal_cols[:5])}{'...' if len(ordinal_cols) > 5 else ''}
- Categorical ({len(categorical_cols)}): {', '.join(categorical_cols[:5])}{'...' if len(categorical_cols) > 5 else ''}
- Datetime ({len(datetime_cols)}): {', '.join(datetime_cols)}
- Text ({len(text_cols)}): {', '.join(text_cols[:3])}{'...' if len(text_cols) > 3 else ''}

Columns with Missing Values ({len(cols_with_missing)}):"""

        if cols_with_missing:
            sorted_missing = sorted(cols_with_missing.items(), key=lambda x: x[1], reverse=True)
            for col, count in sorted_missing[:10]:
                pct = count / df.shape[0] * 100
                output += f"\n  - {col}: {count} ({pct:.1f}%)"

        output += """

SUGGESTIONS for investigation:
- Binary columns are good TREATMENT candidates - investigate balance
- Numeric columns are good OUTCOME candidates - check distributions
- Other columns may be CONFOUNDERS - consider domain relevance"""

        return output

    def _tool_analyze_column(self, column: str) -> str:
        """Analyze a specific column in detail."""
        if column not in self._df.columns:
            return f"Column '{column}' not found in dataset."

        df = self._df
        col_type = self._profile.feature_types.get(column, "unknown")
        data = df[column].dropna()
        n = len(data)
        missing = df[column].isnull().sum()

        output = f"Column Analysis: {column}\n"
        output += "=" * 50 + "\n"
        output += f"Type: {col_type}\n"
        output += f"Non-missing: {n}, Missing: {missing} ({missing/len(df)*100:.1f}%)\n"
        output += f"Unique values: {data.nunique()}\n\n"

        if col_type in ["binary", "ordinal", "categorical"]:
            # Show value distribution
            value_counts = data.value_counts()
            output += "Value Distribution:\n"
            for val, count in value_counts.head(15).items():
                pct = count / n * 100
                bar = "#" * int(pct / 5)
                output += f"  {val}: {count} ({pct:.1f}%) {bar}\n"

            if col_type == "binary":
                # Assess treatment suitability
                if data.nunique() == 2:
                    min_class = value_counts.min()
                    balance = min_class / n
                    output += "\nTreatment Suitability:\n"
                    output += f"  - Balance (minority class): {balance*100:.1f}%\n"
                    if 0.1 <= balance <= 0.5:
                        output += "  - GOOD: Balanced enough for causal inference\n"
                    elif balance < 0.05:
                        output += "  - WARNING: Very imbalanced - may lack power\n"
                    else:
                        output += "  - ACCEPTABLE: Some imbalance but usable\n"

        elif col_type == "numeric":
            # Show numeric statistics
            output += "Statistics:\n"
            output += f"  Mean: {data.mean():.4f}\n"
            output += f"  Std: {data.std():.4f}\n"
            output += f"  Min: {data.min():.4f}\n"
            output += f"  25%: {data.quantile(0.25):.4f}\n"
            output += f"  50%: {data.quantile(0.50):.4f}\n"
            output += f"  75%: {data.quantile(0.75):.4f}\n"
            output += f"  Max: {data.max():.4f}\n"

            # Skewness
            skew = stats.skew(data)
            output += f"\n  Skewness: {skew:.3f}"
            if abs(skew) > 1:
                output += " (highly skewed)"
            elif abs(skew) > 0.5:
                output += " (moderately skewed)"
            else:
                output += " (approximately symmetric)"
            output += "\n"

            # Outcome suitability
            output += "\nOutcome Suitability:\n"
            if data.std() > 0:
                output += "  - Has variance - suitable as outcome\n"
            else:
                output += "  - WARNING: No variance - not suitable as outcome\n"

        elif col_type == "datetime":
            output += "Time Range:\n"
            output += f"  Min: {data.min()}\n"
            output += f"  Max: {data.max()}\n"
            output += f"  Range: {data.max() - data.min()}\n"

        return output

    def _tool_check_treatment_balance(self, column: str) -> str:
        """Check if a column is suitable as a treatment variable."""
        if column not in self._df.columns:
            return f"Column '{column}' not found in dataset."

        df = self._df
        data = df[column].dropna()
        n = len(data)

        output = f"Treatment Balance Check: {column}\n"
        output += "=" * 50 + "\n"

        unique_vals = data.unique()
        n_unique = len(unique_vals)

        if n_unique == 2:
            # Binary treatment - ideal
            value_counts = data.value_counts()
            minority_pct = value_counts.min() / n * 100
            majority_pct = value_counts.max() / n * 100

            output += "Binary Treatment Variable\n"
            output += "\nValue Distribution:\n"
            for val, count in value_counts.items():
                output += f"  {val}: {count} ({count/n*100:.1f}%)\n"

            output += "\nBalance Assessment:\n"
            output += f"  Minority class: {minority_pct:.1f}%\n"
            output += f"  Majority class: {majority_pct:.1f}%\n"

            if minority_pct >= 20:
                output += "\n  EXCELLENT: Well-balanced treatment\n"
                output += "  Suitable for: All methods (IPW, matching, AIPW, etc.)\n"
            elif minority_pct >= 10:
                output += "\n  GOOD: Acceptable balance\n"
                output += "  Suitable for: Most methods, consider trimming extremes\n"
            elif minority_pct >= 5:
                output += "\n  MODERATE: Some imbalance\n"
                output += "  Suitable for: Matching, stabilized weights recommended\n"
            else:
                output += "\n  WARNING: Severely imbalanced\n"
                output += "  May lack statistical power, consider alternative designs\n"

        elif 2 < n_unique <= 5:
            # Multi-level treatment
            output += f"Multi-level Treatment Variable ({n_unique} levels)\n"
            output += "\nValue Distribution:\n"
            for val, count in data.value_counts().items():
                output += f"  {val}: {count} ({count/n*100:.1f}%)\n"
            output += "\n  Can be used for multi-valued treatment analysis\n"
            output += "  Or collapsed to binary for specific comparisons\n"

        elif n_unique > 5 and n_unique <= 20:
            output += f"Categorical Variable ({n_unique} levels)\n"
            output += "  Consider collapsing categories for treatment analysis\n"

        else:
            output += f"Continuous Variable ({n_unique} unique values)\n"
            output += "  Can be used for dose-response analysis\n"
            output += "  Or discretized into treatment bins\n"

        return output

    def _tool_check_column_relationship(self, column1: str, column2: str) -> str:
        """Check relationship between two columns."""
        if column1 not in self._df.columns:
            return f"Column '{column1}' not found."
        if column2 not in self._df.columns:
            return f"Column '{column2}' not found."

        df = self._df
        type1 = self._profile.feature_types.get(column1, "unknown")
        type2 = self._profile.feature_types.get(column2, "unknown")

        output = f"Relationship: {column1} <-> {column2}\n"
        output += "=" * 50 + "\n"

        # Drop rows with missing in either column
        valid_data = df[[column1, column2]].dropna()
        n = len(valid_data)

        if n < 10:
            return output + "Not enough complete cases for relationship analysis."

        # Both numeric
        if type1 in ["numeric", "ordinal", "binary"] and type2 in ["numeric", "ordinal", "binary"]:
            # Pearson correlation
            corr, p_value = stats.pearsonr(valid_data[column1], valid_data[column2])
            output += f"Pearson Correlation: r = {corr:.3f} (p = {p_value:.4f})\n"

            if abs(corr) > 0.7:
                output += "  STRONG relationship\n"
            elif abs(corr) > 0.3:
                output += "  MODERATE relationship\n"
            else:
                output += "  WEAK relationship\n"

            # Spearman for robustness
            spearman_corr, _ = stats.spearmanr(valid_data[column1], valid_data[column2])
            output += f"Spearman Correlation: rho = {spearman_corr:.3f}\n"

        # One binary, one numeric - point-biserial
        elif type1 == "binary" and type2 in ["numeric", "ordinal"]:
            group_0 = valid_data[valid_data[column1] == valid_data[column1].min()][column2]
            group_1 = valid_data[valid_data[column1] == valid_data[column1].max()][column2]

            output += f"Mean {column2} by {column1}:\n"
            output += f"  Group 0: {group_0.mean():.3f} (n={len(group_0)})\n"
            output += f"  Group 1: {group_1.mean():.3f} (n={len(group_1)})\n"
            output += f"  Difference: {group_1.mean() - group_0.mean():.3f}\n"

            # T-test
            t_stat, p_value = stats.ttest_ind(group_0, group_1)
            output += f"\nT-test: t = {t_stat:.3f}, p = {p_value:.4f}\n"

        # Categorical association
        elif type1 in ["categorical", "binary"] and type2 in ["categorical", "binary"]:
            contingency = pd.crosstab(valid_data[column1], valid_data[column2])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            output += f"Chi-squared: χ² = {chi2:.3f}, p = {p_value:.4f}\n"

            # Cramér's V
            n_total = contingency.sum().sum()
            min_dim = min(contingency.shape) - 1
            cramers_v = np.sqrt(chi2 / (n_total * min_dim)) if min_dim > 0 else 0
            output += f"Cramér's V: {cramers_v:.3f}\n"

        else:
            output += f"Types: {type1}, {type2}\n"
            output += "Relationship analysis not available for this combination.\n"

        return output

    def _tool_check_time_dimension(self) -> str:
        """Check for time dimension in the dataset."""
        df = self._df
        profile = self._profile

        output = "Time Dimension Check:\n"
        output += "=" * 50 + "\n"

        time_keywords = ["time", "date", "year", "month", "period", "quarter", "week", "day"]
        time_candidates = []

        for col in df.columns:
            col_lower = col.lower()

            # Check datetime type
            if profile.feature_types.get(col) == "datetime":
                time_candidates.append((col, "datetime", "Datetime type"))
                continue

            # Check for time keywords
            for kw in time_keywords:
                if kw in col_lower:
                    # Check if it could be a time variable
                    if profile.feature_types.get(col) in ["numeric", "ordinal"]:
                        time_candidates.append((col, profile.feature_types.get(col), f"Contains '{kw}'"))
                    break

        if time_candidates:
            output += "Time dimension candidates found:\n"
            for col, col_type, reason in time_candidates:
                unique_vals = df[col].nunique()
                output += f"\n  {col}:\n"
                output += f"    Type: {col_type}\n"
                output += f"    Reason: {reason}\n"
                output += f"    Unique values: {unique_vals}\n"

                if profile.feature_types.get(col) in ["numeric", "ordinal"]:
                    data = df[col].dropna()
                    output += f"    Range: [{data.min()}, {data.max()}]\n"

            output += "\n  DiD/Panel methods may be applicable\n"
            return output

        output += "No obvious time dimension found.\n"
        output += "DiD/Panel methods may not be directly applicable.\n"
        return output

    def _tool_check_discontinuity_candidates(self) -> str:
        """Check for RDD running variable candidates."""
        df = self._df
        profile = self._profile

        output = "Regression Discontinuity Candidates:\n"
        output += "=" * 50 + "\n"

        rdd_keywords = ["score", "threshold", "cutoff", "grade", "test", "rank", "percentile", "rating"]
        candidates = []

        for col in df.columns:
            col_lower = col.lower()

            # Check for RDD keywords in numeric columns
            if profile.feature_types.get(col) in ["numeric", "ordinal"]:
                for kw in rdd_keywords:
                    if kw in col_lower:
                        candidates.append((col, kw))
                        break

        if candidates:
            output += "Potential running variables found:\n"
            for col, keyword in candidates:
                data = df[col].dropna()
                output += f"\n  {col}:\n"
                output += f"    Matched keyword: '{keyword}'\n"
                output += f"    Range: [{data.min():.2f}, {data.max():.2f}]\n"
                output += f"    Mean: {data.mean():.2f}, Median: {data.median():.2f}\n"
                output += f"    Unique values: {data.nunique()}\n"

            output += "\n  RDD may be applicable if there's a known cutoff/threshold\n"
            return output

        output += "No obvious running variables found.\n"
        output += "RDD may not be directly applicable without domain knowledge.\n"
        return output

    def _auto_finalize(self) -> dict[str, Any]:
        """Auto-generate causal structure using heuristics."""
        self.logger.info("profiler_auto_finalize")

        df = self._df
        profile = self._profile

        # Heuristic treatment candidates
        treatment_candidates = []
        treatment_keywords = ["treatment", "treated", "treat", "intervention", "exposed", "program"]

        for col in df.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in treatment_keywords):
                treatment_candidates.append(col)
            elif profile.feature_types.get(col) == "binary":
                data = df[col].dropna()
                if len(data) > 0:
                    balance = data.value_counts().min() / len(data)
                    if 0.1 <= balance <= 0.5:
                        treatment_candidates.append(col)

        # Heuristic outcome candidates
        outcome_candidates = []
        outcome_keywords = ["outcome", "result", "response", "target", "y", "income", "wage", "score"]

        for col in df.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in outcome_keywords):
                outcome_candidates.append(col)
            elif profile.feature_types.get(col) == "numeric":
                if not any(x in col_lower for x in ["id", "index", "key"]):
                    outcome_candidates.append(col)

        # Confounders - everything else that's not ID-like
        confounders = []
        for col in df.columns:
            col_lower = col.lower()
            if col in treatment_candidates[:3] or col in outcome_candidates[:3]:
                continue
            if any(x in col_lower for x in ["id", "index", "key", "_id"]):
                continue
            if profile.feature_types.get(col) in ["categorical", "numeric", "ordinal", "binary"]:
                confounders.append(col)

        # Check time dimension
        has_time = False
        time_column = None
        time_keywords = ["time", "date", "year", "month", "period"]
        for col in df.columns:
            if profile.feature_types.get(col) == "datetime":
                has_time = True
                time_column = col
                break
            if any(kw in col.lower() for kw in time_keywords):
                has_time = True
                time_column = col
                break

        return {
            "treatment_candidates": treatment_candidates[:5],
            "outcome_candidates": outcome_candidates[:5],
            "potential_confounders": confounders,
            "potential_instruments": [],
            "has_time_dimension": has_time,
            "time_column": time_column,
            "discontinuity_candidates": [],
            "recommended_methods": ["IPW", "AIPW", "Matching"],
        }

    def _populate_profile(self, final_result: dict[str, Any]) -> None:
        """Populate profile from finalize result."""
        self._profile.treatment_candidates = final_result.get("treatment_candidates", [])
        self._profile.outcome_candidates = final_result.get("outcome_candidates", [])
        self._profile.potential_confounders = final_result.get("potential_confounders", [])
        self._profile.potential_instruments = final_result.get("potential_instruments", [])
        self._profile.has_time_dimension = final_result.get("has_time_dimension", False)
        self._profile.time_column = final_result.get("time_column")
        self._profile.discontinuity_candidates = final_result.get("discontinuity_candidates", [])

    def _save_dataframe(self, df: pd.DataFrame, job_id: str) -> str:
        """Save DataFrame to disk for other agents."""
        temp_dir = Path(tempfile.gettempdir()) / "causal_orchestrator"
        temp_dir.mkdir(exist_ok=True)

        file_path = temp_dir / f"{job_id}_data.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(df, f)

        return str(file_path)
