"""Pure helpers for the critique agent.

DataFrame loading, the initial-prompt builder, the auto-finalize
heuristic that synthesizes a decision when the LLM exits without
calling finalize_critique, the feedback constructor, and the
heuristic-critique fallback that runs when the LLM call itself fails.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.agents.base import AnalysisState, CritiqueDecision, CritiqueFeedback
from src.logging_config.structured import get_logger

logger = get_logger(__name__)


def load_dataframe(dataframe_path: str | None) -> pd.DataFrame | None:
    """Load the parquet at dataframe_path; return None on missing or unreadable."""
    if dataframe_path and Path(dataframe_path).exists():
        try:
            return pd.read_parquet(dataframe_path)
        except Exception:
            logger.debug("dataframe_load_failed", path=dataframe_path, exc_info=True)
    return None


def build_initial_prompt(
    state: AnalysisState,
    treatment_var: str | None,
    outcome_var: str | None,
) -> str:
    """Compose the opening user message for the agentic loop."""
    n_methods = len(state.treatment_effects)
    n_sens = len(state.sensitivity_results) if state.sensitivity_results else 0

    return f"""You are critically reviewing a causal inference analysis.

Treatment: {treatment_var or 'Unknown'}
Outcome: {outcome_var or 'Unknown'}
Methods used: {n_methods}
Sensitivity analyses: {n_sens}

Your task is to INVESTIGATE the analysis quality by using the available tools.
Do not just accept summaries - verify claims yourself!

Start by calling get_analysis_summary to understand what was done,
then use investigation tools to verify:
1. Is covariate balance adequate?
2. Are estimates consistent across methods?
3. Are there influential observations?
4. Do results hold across subgroups?
5. Are sensitivity results robust?

After gathering evidence, call finalize_critique with your assessment."""


def auto_finalize(
    state: AnalysisState,
    investigation_evidence: list[str],
) -> dict[str, Any]:
    """Synthesize a critique result from state + accumulated evidence.

    Mirrors the keys finalize_critique produces: scores, decision,
    issues, improvements, evidence_summary, reasoning. Decision rules:

    - REJECT only when the analysis is genuinely empty (no methods AND
      no sensitivity ran) — a causal analysis with zero estimates is
      not worth shipping a notebook for.
    - APPROVE when avg score >= 4 with no issues, OR avg >= 3 with at
      most one improvement, OR after >= 2 iterations (avoid infinite
      loops; empty-analysis case is caught above).
    - ITERATE otherwise.
    """
    scores = {
        "statistical_validity": 3,
        "assumption_checking": 3,
        "method_selection": 3,
        "completeness": 3,
        "reproducibility": 3,
        "interpretation": 3,
    }
    issues: list[str] = []
    improvements: list[str] = []

    method_names = {e.method.lower().strip() for e in state.treatment_effects}
    n_distinct_methods = len(method_names)

    if n_distinct_methods >= 3:
        scores["method_selection"] = 5
    elif n_distinct_methods == 2:
        scores["method_selection"] = 4
    elif n_distinct_methods == 1:
        scores["method_selection"] = 2
        improvements.append(
            "Run additional estimation methods (IPW, AIPW, matching) for triangulation"
        )
    else:
        scores["method_selection"] = 1
        improvements.append("No treatment effects estimated — re-run effect estimation")

    if state.sensitivity_results and len(state.sensitivity_results) >= 3:
        scores["completeness"] = 5
    elif state.sensitivity_results:
        scores["completeness"] = 4
    else:
        scores["completeness"] = 2
        improvements.append("Add sensitivity analysis (E-value, placebo tests)")

    has_confounders = bool(
        (state.confounder_discovery and state.confounder_discovery.get("ranked_confounders"))
        or (state.data_profile and state.data_profile.potential_confounders)
    )
    if has_confounders and n_distinct_methods >= 2:
        scores["assumption_checking"] = 4
    elif not has_confounders:
        scores["assumption_checking"] = 2
        improvements.append("Identify and control for confounders in estimation")

    if state.proposed_dag and state.proposed_dag.edges:
        scores["statistical_validity"] = 4
    else:
        scores["statistical_validity"] = 2
        improvements.append("Run causal discovery to identify DAG structure")

    for evidence in investigation_evidence:
        if "CONCERN" in evidence or "WARNING" in evidence:
            issues.append(evidence)

    avg_score = sum(scores.values()) / len(scores)
    analysis_is_empty = (
        n_distinct_methods == 0
        and len(state.sensitivity_results or []) == 0
    )

    if analysis_is_empty:
        decision = "REJECT"
    elif avg_score >= 4 and len(issues) == 0:
        decision = "APPROVE"
    elif avg_score >= 3 and len(improvements) <= 1:
        decision = "APPROVE"
    elif state.iteration_count >= 2:
        decision = "APPROVE"
    else:
        decision = "ITERATE"

    return {
        "scores": scores,
        "decision": decision,
        "issues": issues,
        "improvements": improvements,
        "evidence_summary": " | ".join(investigation_evidence[:5]),
        "reasoning": (
            f"Auto-critique: {n_distinct_methods} distinct methods, "
            f"{len(state.sensitivity_results or [])} sensitivity tests, "
            f"confounders={'yes' if has_confounders else 'no'}, "
            f"Issues: {len(issues)}"
        ),
    }


def create_feedback(result: dict[str, Any], iteration: int) -> CritiqueFeedback:
    """Map an auto- or LLM-finalize result dict to a typed CritiqueFeedback."""
    scores = result.get("scores", {
        "statistical_validity": 3,
        "assumption_checking": 3,
        "method_selection": 3,
        "completeness": 3,
        "reproducibility": 3,
        "interpretation": 3,
    })

    decision_str = result.get("decision", "ITERATE")
    try:
        decision = CritiqueDecision(decision_str)
    except ValueError:
        decision = CritiqueDecision.ITERATE

    reasoning = result.get("reasoning", "")
    evidence = result.get("evidence_summary", "")
    if evidence:
        reasoning = f"{reasoning}\n\nEvidence: {evidence}"

    return CritiqueFeedback(
        decision=decision,
        iteration=iteration,
        scores=scores,
        issues=result.get("issues", []),
        improvements=result.get("improvements", []),
        reasoning=reasoning,
    )


def heuristic_critique(state: AnalysisState, error: str) -> CritiqueFeedback:
    """Fallback when the LLM call itself fails.

    Synthesizes a decision purely from observable state, with no
    investigation tools available.
    """
    issues: list[str] = []
    improvements: list[str] = []
    scores = {
        "statistical_validity": 3,
        "assumption_checking": 3,
        "method_selection": 3,
        "completeness": 3,
        "reproducibility": 3,
        "interpretation": 3,
    }

    if state.treatment_effects:
        n_methods = len(state.treatment_effects)
        if n_methods >= 3:
            scores["method_selection"] = 4
            scores["statistical_validity"] = 4
        elif n_methods == 1:
            scores["method_selection"] = 2
            improvements.append("Use multiple estimation methods for robustness")

        estimates = [e.estimate for e in state.treatment_effects]
        if estimates:
            cv = np.std(estimates) / (np.mean(estimates) + 1e-10)
            if cv < 0.2:
                scores["statistical_validity"] += 1
            elif cv > 0.5:
                issues.append("Estimates vary significantly across methods")
                scores["statistical_validity"] -= 1

    if state.sensitivity_results:
        scores["completeness"] = 4
    else:
        improvements.append("Add sensitivity analysis to test robustness")

    avg_score = sum(scores.values()) / len(scores)
    if avg_score >= 4:
        decision = CritiqueDecision.APPROVE
    elif avg_score >= 3:
        decision = CritiqueDecision.ITERATE
    else:
        decision = CritiqueDecision.REJECT

    return CritiqueFeedback(
        decision=decision,
        iteration=state.iteration_count + 1,
        scores=scores,
        issues=issues,
        improvements=improvements,
        reasoning=(
            f"Heuristic evaluation (LLM error: {error[:100]}). "
            f"Avg score: {avg_score:.1f}/5."
        ),
    )
