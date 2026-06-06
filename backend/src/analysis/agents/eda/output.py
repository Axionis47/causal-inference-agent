"""Typed output for the EDA agent.

`EDAResult` is the single slot this agent writes on AnalysisState. Owning
this model from the agent's own package matches the convention used by
data_profiler and domain_knowledge.
"""

from typing import Any

from pydantic import BaseModel, Field


class EDAResult(BaseModel):
    """Result of exploratory data analysis."""

    # Correlation analysis
    correlation_matrix: dict[str, dict[str, float]] = Field(default_factory=dict)
    high_correlations: list[dict[str, Any]] = Field(default_factory=list)  # pairs with |r| > 0.7

    # Distribution analysis
    distribution_stats: dict[str, dict[str, Any]] = Field(default_factory=dict)  # skewness, kurtosis, normality

    # Outlier detection
    outliers: dict[str, dict[str, Any]] = Field(default_factory=dict)  # column -> {count, indices, method}

    # Multicollinearity
    vif_scores: dict[str, float] = Field(default_factory=dict)  # Variance Inflation Factor
    multicollinearity_warnings: list[str] = Field(default_factory=list)

    # Covariate balance (for treatment/control)
    covariate_balance: dict[str, dict[str, float]] = Field(default_factory=dict)  # SMD, p-values
    balance_summary: str = ""

    # Data quality
    data_quality_score: float = 0.0
    data_quality_issues: list[str] = Field(default_factory=list)

    # Summary statistics. Values are heterogeneous: the finalize step writes
    # key_findings/recommendations as list[str] and causal_readiness as str,
    # so the value type is Any rather than a nested dict.
    summary_table: dict[str, Any] = Field(default_factory=dict)

    # Plot captions written by the agent during finalize, keyed by plot id.
    # The notebook EDA section renders each caption as a markdown cell
    # immediately above the matching plot. Supported keys:
    #   "distribution"          treatment + outcome marginals
    #   "outcome_by_group"      outcome split by treatment group
    #   "correlation_heatmap"   numeric variable correlations
    #   "love_plot"             standardised mean differences by covariate
    plot_captions: dict[str, str] = Field(default_factory=dict)
