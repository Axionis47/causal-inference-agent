"""Utilities for synthesizing analysis results into summaries and consensus metrics."""

from statistics import median

from src.api.schemas.job import (
    DataContextResponse,
    ExecutiveSummaryResponse,
    MethodConsensusResponse,
    TreatmentEffectResponse,
)


def calculate_method_consensus(
    effects: list[TreatmentEffectResponse],
) -> MethodConsensusResponse | None:
    """Calculate consensus metrics across multiple estimation methods.

    Args:
        effects: List of treatment effect results from different methods

    Returns:
        MethodConsensusResponse with consensus metrics, or None if insufficient data
    """
    if len(effects) < 2:
        return None

    estimates = [e.estimate for e in effects]
    p_values = [e.p_value for e in effects if e.p_value is not None]

    # Calculate direction agreement
    positive_count = sum(1 for e in estimates if e > 0)
    negative_count = sum(1 for e in estimates if e < 0)
    direction_agreement = max(positive_count, negative_count) / len(estimates)

    # Check if all significant (p < 0.05)
    all_significant = len(p_values) > 0 and all(p < 0.05 for p in p_values)

    # Estimate range
    estimate_range = (min(estimates), max(estimates))

    # Median estimate
    median_estimate = median(estimates)

    # Determine consensus strength
    range_span = estimate_range[1] - estimate_range[0]
    relative_range = range_span / abs(median_estimate) if median_estimate != 0 else float("inf")

    if direction_agreement >= 0.9 and relative_range < 0.5:
        consensus_strength = "strong"
    elif direction_agreement >= 0.7 and relative_range < 1.0:
        consensus_strength = "moderate"
    else:
        consensus_strength = "weak"

    return MethodConsensusResponse(
        n_methods=len(effects),
        direction_agreement=round(direction_agreement, 2),
        all_significant=all_significant,
        estimate_range=estimate_range,
        median_estimate=round(median_estimate, 4),
        consensus_strength=consensus_strength,
    )


def generate_executive_summary(
    treatment_variable: str | None,
    outcome_variable: str | None,
    effects: list[TreatmentEffectResponse],
    consensus: MethodConsensusResponse | None,
    sensitivity_robust: bool = True,
) -> ExecutiveSummaryResponse | None:
    """Generate an executive summary of the causal analysis.

    Args:
        treatment_variable: Name of treatment variable
        outcome_variable: Name of outcome variable
        effects: List of treatment effect results
        consensus: Pre-calculated consensus metrics
        sensitivity_robust: Whether sensitivity analysis shows robustness

    Returns:
        ExecutiveSummaryResponse with summary information
    """
    if not effects:
        return None

    treatment = treatment_variable or "treatment"
    outcome = outcome_variable or "outcome"

    # Determine effect direction
    if consensus:
        if consensus.direction_agreement >= 0.8:
            if consensus.median_estimate > 0:
                effect_direction = "positive"
            elif consensus.median_estimate < 0:
                effect_direction = "negative"
            else:
                effect_direction = "null"
        else:
            effect_direction = "mixed"
    else:
        # Single method
        effect = effects[0]
        if effect.estimate > 0:
            effect_direction = "positive"
        elif effect.estimate < 0:
            effect_direction = "negative"
        else:
            effect_direction = "null"

    # Determine confidence level
    if consensus:
        if consensus.consensus_strength == "strong" and consensus.all_significant and sensitivity_robust:
            confidence_level = "high"
        elif consensus.consensus_strength in ["strong", "moderate"] and consensus.all_significant:
            confidence_level = "medium"
        else:
            confidence_level = "low"
    else:
        effect = effects[0]
        if effect.p_value and effect.p_value < 0.01 and sensitivity_robust:
            confidence_level = "high"
        elif effect.p_value and effect.p_value < 0.05:
            confidence_level = "medium"
        else:
            confidence_level = "low"

    # Generate headline
    effect_desc = {
        "positive": "positive effect",
        "negative": "negative effect",
        "null": "no significant effect",
        "mixed": "inconclusive effect",
    }[effect_direction]

    confidence_desc = {
        "high": "with high confidence",
        "medium": "with moderate confidence",
        "low": "with low confidence",
    }[confidence_level]

    if consensus:
        estimate_str = f"{consensus.median_estimate:+.3f}"
        method_str = f"across {consensus.n_methods} methods"
    else:
        estimate_str = f"{effects[0].estimate:+.3f}"
        method_str = f"using {effects[0].method}"

    headline = f"{treatment} shows a {effect_desc} on {outcome} ({estimate_str}) {method_str}, {confidence_desc}."

    # Generate key findings
    key_findings = []

    # Finding 1: Effect magnitude and direction
    if consensus:
        key_findings.append(
            f"Median treatment effect estimate: {consensus.median_estimate:+.4f} "
            f"(range: {consensus.estimate_range[0]:.4f} to {consensus.estimate_range[1]:.4f})"
        )
    else:
        effect = effects[0]
        key_findings.append(
            f"Treatment effect estimate: {effect.estimate:+.4f} "
            f"(95% CI: [{effect.ci_lower:.4f}, {effect.ci_upper:.4f}])"
        )

    # Finding 2: Statistical significance
    significant_methods = [e for e in effects if e.p_value and e.p_value < 0.05]
    if len(significant_methods) == len(effects) and len(effects) > 0:
        key_findings.append(f"All {len(effects)} estimation methods show statistical significance (p < 0.05)")
    elif significant_methods:
        key_findings.append(
            f"{len(significant_methods)} of {len(effects)} methods show statistical significance"
        )
    else:
        key_findings.append("No methods show statistical significance at p < 0.05")

    # Finding 3: Method agreement
    if consensus and consensus.direction_agreement >= 0.9:
        key_findings.append(
            f"Strong agreement across methods: {int(consensus.direction_agreement * 100)}% "
            "agree on effect direction"
        )
    elif consensus and consensus.direction_agreement < 0.7:
        key_findings.append(
            "Caution: Methods disagree on effect direction - results may be sensitive to modeling assumptions"
        )

    # Finding 4: Robustness
    if sensitivity_robust:
        key_findings.append("Results appear robust to unmeasured confounding based on sensitivity analysis")
    else:
        key_findings.append("Caution: Results may be sensitive to unmeasured confounding")

    return ExecutiveSummaryResponse(
        headline=headline,
        effect_direction=effect_direction,
        confidence_level=confidence_level,
        key_findings=key_findings,
    )


def build_data_context(results: dict) -> DataContextResponse | None:
    """Build data context response from stored results.

    Args:
        results: Raw results dictionary from storage

    Returns:
        DataContextResponse with data context information
    """
    data_profile = results.get("data_profile", {})
    if not data_profile:
        return None

    n_samples = data_profile.get("n_samples", 0)
    n_features = data_profile.get("n_features", 0)

    # Calculate missing data percentage if available
    missing_values = data_profile.get("missing_values", {})
    total_cells = n_samples * n_features if n_samples and n_features else 1
    total_missing = sum(missing_values.values()) if missing_values else 0
    missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0.0

    # Get treatment/control counts if available
    n_treated = data_profile.get("n_treated")
    n_control = data_profile.get("n_control")

    # Get data quality issues if available
    data_quality_issues = results.get("data_quality_issues", [])

    return DataContextResponse(
        n_samples=n_samples,
        n_features=n_features,
        n_treated=n_treated,
        n_control=n_control,
        missing_data_pct=round(missing_pct, 2),
        data_quality_issues=data_quality_issues,
    )
