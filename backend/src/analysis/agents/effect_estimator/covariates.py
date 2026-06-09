"""Covariate selection for a treatment-outcome pair.

Priority chain (most informed first):
  0a. dag_expert's pre-computed adjustment_set on state.proposed_dag
  0b. Compute adjustment set from proposed_dag edges
  1.  confounder_discovery's ranked_confounders
  2.  data_profile's potential_confounders
  3.  All numeric non-ID/non-date columns

Categorical confounders are one-hot encoded on a copy of the
DataFrame; the agent's _df is rebound to the working copy so
downstream estimation tools see the dummy columns.
"""

from __future__ import annotations

import re

import pandas as pd

from src.analysis.agents.base import AnalysisState

from .dag_utils import compute_adjustment_set

_ID_PATTERN = re.compile(
    r"(^id$|_id$|^index$|^row_?num|^unnamed|^Unnamed)", re.IGNORECASE
)


def get_covariates_for_pair(
    agent,
    state: AnalysisState,
    treatment: str,
    outcome: str,
) -> list[str]:
    """Pick covariates and (if needed) one-hot-encode categorical ones on agent._df."""
    raw_confounders: list[str] = []

    # Priority 0a: dag_expert's pre-computed adjustment set.
    if (
        state.proposed_dag
        and state.proposed_dag.adjustment_set
        and len(state.proposed_dag.adjustment_set) > 0
    ):
        raw_confounders = [
            c for c in state.proposed_dag.adjustment_set
            if c in agent._df.columns and c != treatment and c != outcome
        ]
        if raw_confounders:
            agent.logger.info(
                "using_dag_expert_adjustment_set",
                n_vars=len(raw_confounders),
                vars=raw_confounders[:5],
            )

    # Priority 0b: Recompute from DAG edges.
    if not raw_confounders and state.proposed_dag and state.proposed_dag.edges:
        adjustment_set = compute_adjustment_set(
            dag=state.proposed_dag,
            treatment=treatment,
            outcome=outcome,
        )
        if adjustment_set:
            raw_confounders = [
                c for c in adjustment_set
                if c in agent._df.columns and c != treatment and c != outcome
            ]
            if raw_confounders:
                agent.logger.info(
                    "using_dag_adjustment_set",
                    n_vars=len(raw_confounders),
                    vars=raw_confounders[:5],
                )

    # Each lower priority is an independent `if not raw_confounders`, never an
    # elif/else chained to Priority 1. An elif here binds to Priority 1's
    # condition, so when Priority 0a already resolved the adjustment set and
    # Priority 1 is merely absent (no confounder_discovery), the elif/else would
    # still fire and clobber the DAG's confounders with the profiler's shorter
    # potential_confounders list. The guard makes a higher priority's result win.

    # Priority 1: Discovered confounders.
    if not raw_confounders and state.confounder_discovery and state.confounder_discovery.get("ranked_confounders"):
        raw_confounders = [
            c for c in state.confounder_discovery["ranked_confounders"]
            if c in agent._df.columns and c != treatment and c != outcome
        ]

    # Priority 2: Profile's potential confounders.
    if not raw_confounders and state.data_profile and state.data_profile.potential_confounders:
        raw_confounders = [
            c for c in state.data_profile.potential_confounders
            if c in agent._df.columns and c != treatment and c != outcome
        ]

    # Priority 3: All non-ID, non-date columns.
    if not raw_confounders:
        raw_confounders = [
            c for c in agent._df.columns
            if c != treatment
            and c != outcome
            and not _ID_PATTERN.search(c)
            and not pd.api.types.is_datetime64_any_dtype(agent._df[c])
        ]

    numeric_covariates = [
        c for c in raw_confounders
        if pd.api.types.is_numeric_dtype(agent._df[c])
    ]

    cat_confounders = [
        c for c in raw_confounders
        if c in agent._df.columns
        and not pd.api.types.is_numeric_dtype(agent._df[c])
        and agent._df[c].nunique() <= 20
    ]

    # One-hot encode categoricals on a copy so we don't mutate the
    # shared DataFrame; downstream agents would otherwise see dummies.
    dummy_cols: list[str] = []
    if cat_confounders:
        working_df = agent._df.copy()
        dummies = pd.get_dummies(
            working_df[cat_confounders], prefix_sep="_", drop_first=True,
        )
        for col in dummies.columns:
            working_df[col] = dummies[col].astype(float)
        dummy_cols = list(dummies.columns)
        agent._df = working_df

    return numeric_covariates + dummy_cols
