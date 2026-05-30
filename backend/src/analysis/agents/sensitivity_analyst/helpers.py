"""Pure helpers for the sensitivity analyst agent.

DataFrame loading, the initial-observation string, and the auto-finalize
heuristic that runs when the LLM exits the loop without calling
finalize_sensitivity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.agents.base import SensitivityResult


def load_dataframe(dataframe_path: str | None) -> pd.DataFrame | None:
    """Load the parquet at dataframe_path; return None when missing."""
    if dataframe_path and Path(dataframe_path).exists():
        return pd.read_parquet(dataframe_path)
    return None


def initial_observation_text(
    treatment_var: str | None,
    outcome_var: str | None,
) -> str:
    """Lean initial observation. Sensitivity pulls upstream context via tools."""
    return f"""You are assessing the robustness of causal effect estimates.

Treatment: {treatment_var or 'Unknown'}
Outcome: {outcome_var or 'Unknown'}

WORKFLOW:
1. Call get_previous_finding with agent="effect_estimator" to review estimates
2. Call ask_domain_knowledge with "What are potential unmeasured confounders?"
3. Run E-value analysis (most fundamental sensitivity test)
4. Based on results, run additional analyses as needed
5. Finalize with overall robustness assessment

START NOW: Call get_previous_finding with agent="effect_estimator"."""


def auto_finalize(results: list[SensitivityResult]) -> dict[str, Any]:
    """Heuristic finalize when the LLM exits the loop without calling finalize_sensitivity.

    Mirrors the keys finalize_sensitivity writes: overall_robustness,
    key_findings, concerns, reasoning.
    """
    if not results:
        return {
            "overall_robustness": "uncertain",
            "key_findings": ["No sensitivity analyses completed"],
            "concerns": ["Unable to assess robustness"],
            "reasoning": "Auto-finalized: no sensitivity results obtained",
        }

    e_value_result = next(
        (r for r in results if "e-value" in r.method.lower()),
        None,
    )
    robustness = "moderate"

    if e_value_result and e_value_result.robustness_value is not None:
        if e_value_result.robustness_value >= 2.5:
            robustness = "high"
        elif e_value_result.robustness_value >= 1.5:
            robustness = "moderate"
        else:
            robustness = "low"

    key_findings = [r.interpretation for r in results[:3]]

    return {
        "overall_robustness": robustness,
        "key_findings": key_findings,
        "concerns": [],
        "reasoning": f"Auto-finalized with {len(results)} analyses completed",
    }
