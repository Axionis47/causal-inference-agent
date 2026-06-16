"""Static tool metadata: the description and argument schema the model sees for
each EDA check it may call. Data only; the wrapping logic lives in tools.py.
Keyed by the check function name. Checks absent here fall back to a no-arg
schema and a generic description.
"""
from __future__ import annotations

NONE_SCHEMA: dict = {"type": "object", "properties": {}}

_COVARIATES_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "covariates": {
            "type": "array",
            "items": {"type": "string"},
            "description": "columns to balance on; clamped to the DAG adjustment set",
        }
    },
}

TOOL_DESCRIPTIONS: dict[str, str] = {
    "outcome_distribution":
        "Distribution of the outcome (mean, spread, skew) with a histogram.",
    "exposure_distribution":
        "Distribution of the treatment/exposure (group sizes or continuous range).",
    "group_outcome_comparison":
        "Raw mean outcome by treatment group (descriptive, not an effect).",
    "covariate_balance":
        "Standardized mean differences of the adjustment-set covariates across groups.",
    "propensity_overlap":
        "Overlap of estimated propensity scores between treatment groups.",
    "missingness_by_group":
        "Largest gap in column missingness between treatment groups.",
    "subgroup_outcome_table":
        "Outcome means across moderator subgroups (interaction view).",
    "factor_screening":
        "Association of candidate factors with the outcome and inter-factor collinearity.",
    "dose_response_scatter":
        "Outcome vs a continuous exposure with binned means.",
    "did_trends":
        "Outcome trends over time by group (parallel-trends look).",
    "did_pre_trends":
        "Pre-period slope gap between groups.",
    "rdd_around_cutoff":
        "Outcome vs the running variable near the cutoff.",
    "rdd_treatment_by_side":
        "Treatment rate below vs above the cutoff (sharp/fuzzy).",
    "iv_first_stage":
        "Instrument-treatment correlation (first-stage strength).",
    "series_over_time":
        "Outcome as a time series with any intervention marker.",
    "mediation_pathways":
        "Treatment-mediator-outcome correlations (timing assumed).",
    "survival_overview":
        "Event rate, censoring, and time-to-event distribution.",
    "dag_consistency":
        "Test the conditional independencies the causal graph implies against the data.",
}

TOOL_SCHEMAS: dict[str, dict] = {
    "covariate_balance": _COVARIATES_SCHEMA,
    "propensity_overlap": _COVARIATES_SCHEMA,
}
