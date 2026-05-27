"""Pure helpers for the data profiler agent.

These are static utilities: oracle-column detection, basic statistical
profiling, the heuristic auto-finalize used when the LLM never calls
finalize_profile, and the parquet snapshot the orchestrator passes to
downstream agents.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.agents.base import AnalysisState

from .encoding import detect_control_value, detect_strategy
from .output import DataProfile


ORACLE_PATTERNS = [
    "y0_true", "y1_true", "y0", "y1",
    "cate_true", "cate", "ite", "ite_true",
    "propensity_true", "propensity_score", "true_propensity",
    "counterfactual", "cf_",
]


def is_oracle_column(col_name: str) -> bool:
    """True if the column name matches a known oracle/ground-truth pattern."""
    lower = col_name.lower()
    return any(pattern in lower for pattern in ORACLE_PATTERNS)


def exclude_oracle_columns(
    df: pd.DataFrame, state: AnalysisState, logger
) -> pd.DataFrame:
    """Drop suspected oracle columns from the dataframe, preserving any column
    the user explicitly picked as treatment or outcome.

    Records an analysis decision so the audit log shows what we dropped.
    """
    protected: set[str] = set()
    if state.treatment_variable:
        protected.add(state.treatment_variable)
    if state.outcome_variable:
        protected.add(state.outcome_variable)

    excluded = [c for c in df.columns if is_oracle_column(c) and c not in protected]

    if excluded:
        logger.info("oracle_columns_excluded", columns=excluded)
        state.push_decision(
            agent="data_profiler",
            decision_type="columns_excluded",
            choice=", ".join(excluded),
            reason="Excluded suspected oracle/ground-truth columns that would leak into causal analysis",
        )
        df = df.drop(columns=excluded)

    return df


def initial_observation_text(state: AnalysisState) -> str:
    """Lean initial observation - just enough to start investigation."""
    dataset_name = state.dataset_info.name or state.dataset_info.url.split("/")[-1]
    return f"""Profile this dataset for causal inference analysis.
Dataset: {dataset_name}

Start by:
1. Querying domain knowledge for hints about treatment and outcome variables
2. Getting the dataset overview to see columns and types
3. Investigating promising candidates

Use the available tools to gather evidence before finalizing the profile."""


def compute_basic_profile(df: pd.DataFrame) -> DataProfile:
    """Compute the shape, types, missing-count, and per-column stats of a dataframe."""
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

    feature_types: dict[str, str] = {}
    numeric_stats: dict[str, dict[str, float]] = {}
    categorical_stats: dict[str, dict[str, int]] = {}
    missing_values: dict[str, int] = {}

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            unique_count = df[col].nunique()
            if unique_count <= 2:
                feature_types[col] = "binary"
            elif unique_count <= 10:
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


def auto_finalize(df: pd.DataFrame, profile: DataProfile) -> dict[str, Any]:
    """Heuristic finalize used when the agent never calls finalize_profile.

    Picks treatment/outcome candidates from column-name keywords and basic
    type/balance signals, treats the rest as potential confounders, detects
    a time dimension, and seeds a treatment encoding for string-typed
    treatments.
    """
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

    treatment_candidates: list[str] = []
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

    outcome_candidates: list[str] = []
    outcome_keywords = ["outcome", "result", "response", "target", "y", "income", "wage", "score"]

    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in outcome_keywords):
            outcome_candidates.append(col)
        elif profile.feature_types.get(col) == "numeric":
            if not any(x in col_lower for x in ["id", "index", "key"]):
                outcome_candidates.append(col)

    confounders: list[str] = []
    for col in df.columns:
        col_lower = col.lower()
        if col in treatment_candidates[:3] or col in outcome_candidates[:3]:
            continue
        if any(x in col_lower for x in ["id", "index", "key", "_id"]):
            continue
        if profile.feature_types.get(col) in ["categorical", "numeric", "ordinal", "binary"]:
            confounders.append(col)

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

    enc_strategy = None
    enc_control = None
    top_treatment = treatment_candidates[0] if treatment_candidates else None
    if top_treatment and top_treatment in df.columns and df[top_treatment].dtype == object:
        unique_vals = df[top_treatment].dropna().unique().tolist()
        enc_strategy = detect_strategy(unique_vals)
        if enc_strategy == "collapse_to_binary":
            enc_control = detect_control_value(unique_vals, df[top_treatment].value_counts())

    return {
        "treatment_candidates": treatment_candidates[:5],
        "outcome_candidates": outcome_candidates[:5],
        "potential_confounders": confounders,
        "potential_instruments": [],
        "has_time_dimension": has_time,
        "time_column": time_column,
        "discontinuity_candidates": [],
        "recommended_methods": ["IPW", "AIPW", "Matching"],
        "treatment_encoding_strategy": enc_strategy,
        "treatment_control_value": enc_control,
    }


def populate_profile(profile: DataProfile, final_result: dict[str, Any]) -> None:
    """Mutate `profile` in place with the causal-structure fields from final_result."""
    profile.treatment_candidates = final_result.get("treatment_candidates", [])
    profile.outcome_candidates = final_result.get("outcome_candidates", [])
    profile.potential_confounders = final_result.get("potential_confounders", [])
    profile.potential_instruments = final_result.get("potential_instruments", [])
    profile.has_time_dimension = final_result.get("has_time_dimension", False)
    profile.time_column = final_result.get("time_column")
    profile.discontinuity_candidates = final_result.get("discontinuity_candidates", [])


def save_dataframe(df: pd.DataFrame, job_id: str) -> str:
    """Snapshot the dataframe to a job-scoped parquet so downstream agents read
    a stable artifact instead of holding the agent instance.
    """
    temp_dir = Path(tempfile.gettempdir()) / "causal_orchestrator"
    temp_dir.mkdir(exist_ok=True)
    file_path = temp_dir / f"{job_id}_data.parquet"
    df.to_parquet(file_path)
    return str(file_path)
