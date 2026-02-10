"""Data Profiler Agent - ReAct-based dataset profiling for causal inference.

This agent uses the ReAct paradigm to iteratively explore datasets and identify
causal structure. It queries domain knowledge to get hints about variables
before verifying them statistically.

Key features:
- Pull-based context: Queries domain knowledge instead of receiving dumps
- ReAct loop: Observe → Reason → Act until profile is complete
- Statistical verification: Validates domain hints with actual data patterns
"""

import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.agents.base import (
    AnalysisState,
    DataProfile,
    JobStatus,
    ToolResult,
    ToolResultStatus,
)
from src.agents.base.context_tools import ContextTools
from src.agents.base.react_agent import ReActAgent
from src.agents.registry import register_agent
from src.logging_config.structured import get_logger

logger = get_logger(__name__)


@register_agent("data_profiler")
class DataProfilerAgent(ReActAgent, ContextTools):
    """ReAct-based data profiler that identifies causal structure through investigation.

    This agent:
    1. Queries domain knowledge for hints about treatment/outcome variables
    2. Investigates columns statistically to verify hints
    3. Identifies confounders based on domain knowledge and correlations
    4. Finalizes profile when confident about causal structure

    Uses pull-based context - the agent actively queries for information
    rather than receiving context dumps.
    """

    AGENT_NAME = "data_profiler"
    MAX_STEPS = 15

    SYSTEM_PROMPT = """You are an expert data scientist profiling a dataset for causal inference.

Your goal is to identify the causal structure: treatment variables, outcome variables,
confounders, and potential instruments.

WORKFLOW:
1. First, query domain knowledge for hints about what might be treatment/outcome
2. Get dataset overview to see all columns and their types
3. Investigate promising candidates - verify domain hints with actual data patterns
4. Check treatment balance for binary/categorical treatment candidates
5. Identify confounders (pre-treatment variables affecting both treatment and outcome)
6. Finalize profile when you have high-confidence identifications

KEY PRINCIPLES:
- Domain knowledge provides HINTS, but you must VERIFY with statistical checks
- Binary columns with 10-50% minority class are ideal treatments
- Numeric columns with variance are good outcome candidates
- Pre-treatment variables (demographics, baseline measures) are potential confounders
- Column NAMES often reveal their role (treat, outcome, age, income, etc.)

Be systematic. Don't guess - investigate and verify."""

    def __init__(self) -> None:
        """Initialize the data profiler agent."""
        super().__init__()

        # Register context query tools from mixin
        self.register_context_tools()

        # Internal state
        self._df: pd.DataFrame | None = None
        self._profile: DataProfile | None = None
        self._load_error: str | None = None
        self._finalized: bool = False
        self._final_result: dict[str, Any] = {}

        # Register profiling-specific tools
        self._register_profiling_tools()

    def _register_profiling_tools(self) -> None:
        """Register tools for data profiling."""
        self.register_tool(
            name="get_dataset_overview",
            description="Get overview of the dataset including shape, columns, types, and missing values. Call this early to understand the data structure.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=self._tool_get_overview,
        )

        self.register_tool(
            name="analyze_column",
            description="Analyze a specific column in detail - distribution, unique values, statistics. Use to verify if a column matches domain hints.",
            parameters={
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Name of the column to analyze",
                    },
                },
                "required": ["column"],
            },
            handler=self._tool_analyze_column,
        )

        self.register_tool(
            name="check_treatment_balance",
            description="Check if a column is suitable as a treatment variable by examining its balance and distribution. Binary treatments need 10-50% in minority class.",
            parameters={
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Name of the potential treatment column",
                    },
                },
                "required": ["column"],
            },
            handler=self._tool_check_treatment_balance,
        )

        self.register_tool(
            name="check_column_relationship",
            description="Check relationship between two columns (correlation for numeric, association for categorical). Use to identify confounders.",
            parameters={
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
            handler=self._tool_check_relationship,
        )

        self.register_tool(
            name="check_time_dimension",
            description="Check if the dataset has a time dimension suitable for DiD or panel methods.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=self._tool_check_time_dimension,
        )

        self.register_tool(
            name="check_discontinuity_candidates",
            description="Check for running variables that could enable regression discontinuity design.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=self._tool_check_discontinuity,
        )

        self.register_tool(
            name="finalize_profile",
            description="Finalize the data profile with identified causal structure. Call this ONLY after you have investigated and verified the structure.",
            parameters={
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
            handler=self._tool_finalize_profile,
        )

    def _get_initial_observation(self, state: AnalysisState) -> str:
        """Generate lean initial observation - just enough to start investigation."""
        dataset_name = state.dataset_info.name or state.dataset_info.url.split("/")[-1]

        # Lean observation - don't dump everything
        obs = f"""Profile this dataset for causal inference analysis.
Dataset: {dataset_name}

Start by:
1. Querying domain knowledge for hints about treatment and outcome variables
2. Getting the dataset overview to see columns and types
3. Investigating promising candidates

Use the available tools to gather evidence before finalizing the profile."""

        return obs

    async def is_task_complete(self, state: AnalysisState) -> bool:
        """Check if profiling task is complete."""
        return self._finalized and state.data_profile is not None

    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Execute data profiling through ReAct loop.

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
            # Fetch Kaggle metadata for semantic understanding (if not already present)
            if "kaggle.com" in state.dataset_info.url and not state.raw_metadata:
                await self._fetch_kaggle_metadata(state)

            # Load the dataset first
            self._load_error = None
            self._df = await self._load_dataset(state)
            if self._df is None:
                error_msg = self._load_error or "Failed to load dataset"
                state.mark_failed(error_msg, self.AGENT_NAME)
                return state

            # Compute basic profile (shape, types, missing) - needed for tools
            self._profile = self._compute_basic_profile(self._df)

            # Save DataFrame for other agents
            df_path = self._save_dataframe(self._df, state.job_id)
            state.dataframe_path = df_path

            # Store reference to state for context tools
            self._current_state = state

            # Run the ReAct loop
            state = await super().execute(state)

            # If not finalized via tool, auto-finalize
            if not self._finalized:
                self.logger.warning("profiler_auto_finalize")
                self._final_result = self._auto_finalize()
                self._finalized = True

            # Populate profile from final result
            self._populate_profile(self._final_result)

            # Update state
            state.data_profile = self._profile
            state.dataset_info.n_samples = self._profile.n_samples
            state.dataset_info.n_features = self._profile.n_features

            duration_ms = int((time.time() - start_time) * 1000)
            self.logger.info(
                "profiling_complete",
                samples=self._profile.n_samples,
                features=self._profile.n_features,
                treatment_candidates=self._profile.treatment_candidates,
                outcome_candidates=self._profile.outcome_candidates,
                duration_ms=duration_ms,
            )

        except Exception as e:
            self.logger.exception("profiling_failed", error=str(e))
            state.mark_failed(f"Data profiling failed: {str(e)}", self.AGENT_NAME)

        return state

    # =========================================================================
    # Kaggle Metadata Integration
    # =========================================================================

    async def _fetch_kaggle_metadata(self, state: AnalysisState) -> None:
        """Fetch and store Kaggle metadata for semantic variable analysis.

        This enables agents to understand WHAT variables represent (e.g., 'black'
        is a race indicator, not a treatment) beyond just statistical properties.
        """
        try:
            from src.kaggle.metadata_extractor import KaggleMetadataExtractor

            extractor = KaggleMetadataExtractor()
            metadata = await extractor.extract(state.dataset_info.url)

            if metadata.get("extraction_success"):
                state.raw_metadata = metadata

                # Also populate DatasetInfo convenience fields
                state.dataset_info.kaggle_description = metadata.get("description")
                state.dataset_info.kaggle_column_descriptions = metadata.get(
                    "column_descriptions", {}
                )
                state.dataset_info.kaggle_tags = metadata.get("tags", [])
                state.dataset_info.metadata_quality = metadata.get(
                    "metadata_quality", "unknown"
                )

                # Infer domain from tags
                state.dataset_info.kaggle_domain = self._infer_domain(metadata)

                self.logger.info(
                    "kaggle_metadata_fetched",
                    quality=metadata.get("metadata_quality"),
                    has_column_descs=bool(metadata.get("column_descriptions")),
                    domain=state.dataset_info.kaggle_domain,
                )
        except ImportError:
            self.logger.warning("kaggle_metadata_extractor_not_available")
        except Exception as e:
            self.logger.warning("kaggle_metadata_fetch_failed", error=str(e))

    def _infer_domain(self, metadata: dict) -> str | None:
        """Infer domain from metadata tags for context-aware analysis."""
        tags = [t.lower() for t in metadata.get("tags", [])]

        if any(t in tags for t in ["healthcare", "medical", "clinical", "health"]):
            return "healthcare"
        if any(t in tags for t in ["economics", "economic", "finance", "income", "wages"]):
            return "economics"
        if any(t in tags for t in ["education", "academic", "school", "students"]):
            return "education"
        if any(t in tags for t in ["social", "sociology", "demographics", "census"]):
            return "social_science"
        if any(t in tags for t in ["marketing", "business", "sales"]):
            return "business"

        return None

    # =========================================================================
    # Tool Handlers
    # =========================================================================

    async def _tool_get_overview(self, state: AnalysisState) -> ToolResult:
        """Get overview of the dataset."""
        if self._df is None or self._profile is None:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="Dataset not loaded",
            )

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

        overview = {
            "n_samples": profile.n_samples,
            "n_features": profile.n_features,
            "total_missing_pct": round(total_missing_pct, 2),
            "binary_columns": binary_cols[:10],
            "numeric_columns": numeric_cols[:10],
            "ordinal_columns": ordinal_cols[:5],
            "categorical_columns": categorical_cols[:5],
            "datetime_columns": datetime_cols,
            "text_columns": text_cols[:3],
            "columns_with_missing": list(cols_with_missing.keys())[:10],
            "suggestions": [
                "Binary columns are good TREATMENT candidates - check balance",
                "Numeric columns are good OUTCOME candidates - check variance",
                "Query domain knowledge for hints before investigating",
            ],
        }

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output=overview,
        )

    async def _tool_analyze_column(
        self, state: AnalysisState, column: str
    ) -> ToolResult:
        """Analyze a specific column in detail."""
        if self._df is None:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="Dataset not loaded",
            )

        if column not in self._df.columns:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error=f"Column '{column}' not found. Available: {list(self._df.columns)[:10]}...",
            )

        df = self._df
        col_type = self._profile.feature_types.get(column, "unknown")
        data = df[column].dropna()
        n = len(data)
        missing = df[column].isnull().sum()

        analysis = {
            "column": column,
            "type": col_type,
            "non_missing": n,
            "missing": missing,
            "missing_pct": round(missing / len(df) * 100, 1),
            "unique_values": int(data.nunique()),
        }

        if col_type in ["binary", "ordinal", "categorical"]:
            value_counts = data.value_counts().head(10).to_dict()
            analysis["value_distribution"] = {str(k): int(v) for k, v in value_counts.items()}

            if col_type == "binary" and data.nunique() == 2:
                min_class = data.value_counts().min()
                balance = min_class / n
                analysis["treatment_suitability"] = {
                    "minority_class_pct": round(balance * 100, 1),
                    "is_balanced": 0.1 <= balance <= 0.5,
                    "assessment": "GOOD" if 0.1 <= balance <= 0.5 else "IMBALANCED",
                }

        elif col_type == "numeric":
            analysis["statistics"] = {
                "mean": round(float(data.mean()), 4),
                "std": round(float(data.std()), 4),
                "min": round(float(data.min()), 4),
                "max": round(float(data.max()), 4),
                "median": round(float(data.median()), 4),
            }

            skew = stats.skew(data)
            analysis["skewness"] = round(float(skew), 3)
            analysis["has_variance"] = data.std() > 0
            analysis["outcome_suitability"] = "GOOD" if data.std() > 0 else "NO_VARIANCE"

        elif col_type == "datetime":
            analysis["time_range"] = {
                "min": str(data.min()),
                "max": str(data.max()),
            }

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output=analysis,
        )

    async def _tool_check_treatment_balance(
        self, state: AnalysisState, column: str
    ) -> ToolResult:
        """Check if a column is suitable as a treatment variable."""
        if self._df is None:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="Dataset not loaded",
            )

        if column not in self._df.columns:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error=f"Column '{column}' not found",
            )

        df = self._df
        data = df[column].dropna()
        n = len(data)
        unique_vals = data.unique()
        n_unique = len(unique_vals)

        result = {
            "column": column,
            "n_unique": n_unique,
            "value_counts": data.value_counts().head(10).to_dict(),
        }

        if n_unique == 2:
            # Binary treatment - ideal
            value_counts = data.value_counts()
            minority_pct = value_counts.min() / n * 100
            majority_pct = value_counts.max() / n * 100

            result["treatment_type"] = "binary"
            result["minority_pct"] = round(minority_pct, 1)
            result["majority_pct"] = round(majority_pct, 1)

            if minority_pct >= 20:
                result["assessment"] = "EXCELLENT"
                result["suitable_methods"] = ["IPW", "Matching", "AIPW", "DML"]
            elif minority_pct >= 10:
                result["assessment"] = "GOOD"
                result["suitable_methods"] = ["IPW", "Matching", "AIPW"]
            elif minority_pct >= 5:
                result["assessment"] = "MODERATE"
                result["suitable_methods"] = ["Matching", "Stabilized Weights"]
            else:
                result["assessment"] = "POOR"
                result["suitable_methods"] = ["Consider alternative treatment definition"]

        elif 2 < n_unique <= 5:
            result["treatment_type"] = "multi-level"
            result["assessment"] = "USABLE"
            result["note"] = "Can be used for multi-valued treatment or collapsed to binary"

        elif n_unique > 5 and n_unique <= 20:
            result["treatment_type"] = "categorical"
            result["assessment"] = "CONSIDER_COLLAPSING"
            result["note"] = "Consider collapsing categories for treatment analysis"

        else:
            result["treatment_type"] = "continuous"
            result["assessment"] = "DOSE_RESPONSE"
            result["note"] = "Can be used for dose-response analysis or discretized"

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output=result,
        )

    async def _tool_check_relationship(
        self, state: AnalysisState, column1: str, column2: str
    ) -> ToolResult:
        """Check relationship between two columns."""
        if self._df is None:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="Dataset not loaded",
            )

        if column1 not in self._df.columns:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error=f"Column '{column1}' not found",
            )
        if column2 not in self._df.columns:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error=f"Column '{column2}' not found",
            )

        df = self._df
        type1 = self._profile.feature_types.get(column1, "unknown")
        type2 = self._profile.feature_types.get(column2, "unknown")

        valid_data = df[[column1, column2]].dropna()
        n = len(valid_data)

        if n < 10:
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={"error": "Not enough complete cases", "n": n},
            )

        result = {
            "column1": column1,
            "column2": column2,
            "type1": type1,
            "type2": type2,
            "complete_cases": n,
        }

        # One binary, one numeric - check this FIRST before general numeric case
        if type1 == "binary" and type2 in ["numeric", "ordinal"]:
            group_0 = valid_data[valid_data[column1] == valid_data[column1].min()][column2]
            group_1 = valid_data[valid_data[column1] == valid_data[column1].max()][column2]

            result["group_0_mean"] = round(float(group_0.mean()), 3)
            result["group_1_mean"] = round(float(group_1.mean()), 3)
            result["mean_difference"] = round(float(group_1.mean() - group_0.mean()), 3)

            t_stat, p_value = stats.ttest_ind(group_0, group_1)
            result["t_statistic"] = round(float(t_stat), 3)
            result["p_value"] = round(float(p_value), 4)

        # Both numeric (but not binary-numeric which is handled above)
        elif type1 in ["numeric", "ordinal"] and type2 in ["numeric", "ordinal", "binary"]:
            corr, p_value = stats.pearsonr(valid_data[column1], valid_data[column2])
            result["pearson_correlation"] = round(float(corr), 3)
            result["p_value"] = round(float(p_value), 4)

            if abs(corr) > 0.7:
                result["strength"] = "STRONG"
            elif abs(corr) > 0.3:
                result["strength"] = "MODERATE"
            else:
                result["strength"] = "WEAK"

            # Spearman for robustness
            spearman_corr, _ = stats.spearmanr(valid_data[column1], valid_data[column2])
            result["spearman_correlation"] = round(float(spearman_corr), 3)

        # Categorical association
        elif type1 in ["categorical", "binary"] and type2 in ["categorical", "binary"]:
            contingency = pd.crosstab(valid_data[column1], valid_data[column2])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            result["chi_squared"] = round(float(chi2), 3)
            result["p_value"] = round(float(p_value), 4)

            # Cramér's V
            n_total = contingency.sum().sum()
            min_dim = min(contingency.shape) - 1
            cramers_v = np.sqrt(chi2 / (n_total * min_dim)) if min_dim > 0 else 0
            result["cramers_v"] = round(float(cramers_v), 3)

        else:
            result["note"] = "Relationship analysis not available for this combination"

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output=result,
        )

    async def _tool_check_time_dimension(self, state: AnalysisState) -> ToolResult:
        """Check for time dimension in the dataset."""
        if self._df is None or self._profile is None:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="Dataset not loaded",
            )

        df = self._df
        profile = self._profile

        time_keywords = ["time", "date", "year", "month", "period", "quarter", "week", "day"]
        time_candidates = []

        for col in df.columns:
            col_lower = col.lower()

            if profile.feature_types.get(col) == "datetime":
                time_candidates.append({
                    "column": col,
                    "type": "datetime",
                    "reason": "Datetime type",
                    "unique_values": int(df[col].nunique()),
                })
                continue

            for kw in time_keywords:
                if kw in col_lower:
                    if profile.feature_types.get(col) in ["numeric", "ordinal"]:
                        data = df[col].dropna()
                        time_candidates.append({
                            "column": col,
                            "type": profile.feature_types.get(col),
                            "reason": f"Contains '{kw}'",
                            "unique_values": int(data.nunique()),
                            "range": [float(data.min()), float(data.max())],
                        })
                    break

        result = {
            "has_time_dimension": len(time_candidates) > 0,
            "candidates": time_candidates,
        }

        if time_candidates:
            result["methods_enabled"] = ["DiD", "Panel methods", "Event study"]
        else:
            result["note"] = "No obvious time dimension found"

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output=result,
        )

    async def _tool_check_discontinuity(self, state: AnalysisState) -> ToolResult:
        """Check for RDD running variable candidates."""
        if self._df is None or self._profile is None:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="Dataset not loaded",
            )

        df = self._df
        profile = self._profile

        rdd_keywords = ["score", "threshold", "cutoff", "grade", "test", "rank", "percentile", "rating"]
        candidates = []

        for col in df.columns:
            col_lower = col.lower()

            if profile.feature_types.get(col) in ["numeric", "ordinal"]:
                for kw in rdd_keywords:
                    if kw in col_lower:
                        data = df[col].dropna()
                        candidates.append({
                            "column": col,
                            "matched_keyword": kw,
                            "range": [float(data.min()), float(data.max())],
                            "mean": float(data.mean()),
                            "median": float(data.median()),
                            "unique_values": int(data.nunique()),
                        })
                        break

        result = {
            "has_rdd_candidates": len(candidates) > 0,
            "candidates": candidates,
        }

        if candidates:
            result["note"] = "RDD may be applicable if there's a known cutoff/threshold"
        else:
            result["note"] = "No obvious running variables found"

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output=result,
        )

    async def _tool_finalize_profile(
        self,
        state: AnalysisState,
        treatment_candidates: list[str],
        outcome_candidates: list[str],
        potential_confounders: list[str],
        potential_instruments: list[str] | None = None,
        has_time_dimension: bool = False,
        time_column: str | None = None,
        discontinuity_candidates: list[str] | None = None,
        recommended_methods: list[str] | None = None,
    ) -> ToolResult:
        """Finalize the data profile."""
        self.logger.info(
            "profiler_finalizing",
            treatment_candidates=treatment_candidates,
            outcome_candidates=outcome_candidates,
            confounders=potential_confounders[:5],
            recommended_methods=recommended_methods or [],
        )

        self._final_result = {
            "treatment_candidates": treatment_candidates,
            "outcome_candidates": outcome_candidates,
            "potential_confounders": potential_confounders,
            "potential_instruments": potential_instruments or [],
            "has_time_dimension": has_time_dimension,
            "time_column": time_column,
            "discontinuity_candidates": discontinuity_candidates or [],
            "recommended_methods": recommended_methods or ["IPW", "AIPW", "Matching"],
        }
        self._finalized = True

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "profile_finalized": True,
                "treatment_candidates": treatment_candidates,
                "outcome_candidates": outcome_candidates,
                "n_confounders": len(potential_confounders),
                "recommended_methods": recommended_methods or ["IPW", "AIPW", "Matching"],
            },
        )

    # =========================================================================
    # Dataset Loading and Utility Methods
    # =========================================================================

    async def _load_dataset(self, state: AnalysisState) -> pd.DataFrame | None:
        """Load dataset from the source."""
        dataset_info = state.dataset_info

        # If dataframe_path already exists, load from there
        if state.dataframe_path:
            try:
                if state.dataframe_path.endswith(".csv"):
                    return pd.read_csv(state.dataframe_path)
                else:
                    return pd.read_parquet(state.dataframe_path)
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

        # Save original env values to restore after API use (security best practice)
        original_username = os.environ.get("KAGGLE_USERNAME")
        original_key = os.environ.get("KAGGLE_KEY")

        # Set Kaggle credentials from config if available
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
                    pass

            if not kaggle_username:
                self._load_error = "KAGGLE_USERNAME is not configured. Cannot download datasets."
                self.logger.error("kaggle_missing_username")
                return None

            os.environ["KAGGLE_USERNAME"] = kaggle_username
            os.environ["KAGGLE_KEY"] = kaggle_key

        dataset_id = None
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
            error_str = str(e)
            dataset_ref = dataset_id or url

            if "403" in error_str or "Forbidden" in error_str:
                self._load_error = f"Dataset '{dataset_ref}' is not accessible. It may be private, deleted, or the URL is incorrect."
            elif "401" in error_str or "Unauthorized" in error_str:
                self._load_error = f"Dataset '{dataset_ref}' not found or authentication failed."
            elif "404" in error_str or "Not Found" in error_str:
                self._load_error = f"Dataset '{dataset_ref}' not found. Please verify the URL."
            else:
                self._load_error = f"Failed to load dataset from Kaggle: {error_str[:200]}"

            self.logger.error("kaggle_load_failed", error=self._load_error)
            return None

        finally:
            # Restore original env values to prevent credential leakage
            if original_username is not None:
                os.environ["KAGGLE_USERNAME"] = original_username
            elif "KAGGLE_USERNAME" in os.environ:
                del os.environ["KAGGLE_USERNAME"]

            if original_key is not None:
                os.environ["KAGGLE_KEY"] = original_key
            elif "KAGGLE_KEY" in os.environ:
                del os.environ["KAGGLE_KEY"]

    def _compute_basic_profile(self, df: pd.DataFrame) -> DataProfile:
        """Compute basic statistical profile of the dataset."""
        # Handle empty DataFrame case
        if df is None or len(df) == 0:
            return DataProfile(
                n_samples=0,
                n_features=0 if df is None else len(df.columns),
                feature_names=[] if df is None else list(df.columns),
                feature_types={},
                missing_values={},
                numeric_stats={},
                categorical_stats={},
            )

        feature_types = {}
        numeric_stats = {}
        categorical_stats = {}
        missing_values = {}

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() <= 2:
                    feature_types[col] = "binary"
                elif df[col].nunique() <= 10:
                    feature_types[col] = "ordinal"
                else:
                    feature_types[col] = "numeric"

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

    def _auto_finalize(self) -> dict[str, Any]:
        """Auto-generate causal structure using heuristics when agent doesn't finalize."""
        df = self._df
        profile = self._profile

        # Handle empty DataFrame case
        if df is None or len(df) == 0 or len(df.columns) == 0:
            return {
                "treatment_candidates": [],
                "outcome_candidates": [],
                "potential_confounders": [],
                "potential_instruments": [],
                "has_time_dimension": False,
                "time_column": None,
                "discontinuity_candidates": [],
                "recommended_methods": [],
            }

        # Heuristic treatment candidates
        treatment_candidates = []
        treatment_keywords = ["treatment", "treated", "treat", "intervention", "exposed", "program"]

        for col in df.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in treatment_keywords):
                treatment_candidates.append(col)
            elif profile.feature_types.get(col) == "binary":
                data = df[col].dropna()
                value_counts = data.value_counts()
                if len(data) > 0 and len(value_counts) > 0:
                    balance = value_counts.min() / len(data)
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

        file_path = temp_dir / f"{job_id}_data.parquet"
        df.to_parquet(file_path)

        return str(file_path)
