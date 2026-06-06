"""LLM-driven treatment-outcome pair selection.

The effect estimator's execute() flow first decides which causal
pair(s) to analyze. Priority chain: user-specified -> single
candidates -> LLM filtering when multiple candidates -> fallback to
first candidates.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.analysis.agents.base import AnalysisState

from .method_selector import find_closest_column


async def identify_valid_causal_pairs(
    agent,
    state: AnalysisState,
) -> list[tuple[str, str, str]]:
    """Pick (treatment, outcome, rationale) tuples to analyze.

    Reads agent.llm / agent.logger. Returns up to 3 pairs.
    """
    profile = state.data_profile

    # Priority 1: User specified both variables; fuzzy-match to actual columns.
    if state.treatment_variable and state.outcome_variable:
        treatment = state.treatment_variable
        outcome = state.outcome_variable

        if agent._df is not None:
            matched_treatment = find_closest_column(treatment, list(agent._df.columns))
            matched_outcome = find_closest_column(outcome, list(agent._df.columns))
            if matched_treatment:
                treatment = matched_treatment
            if matched_outcome:
                outcome = matched_outcome

        agent.logger.info(
            "using_user_specified_variables",
            treatment=treatment,
            outcome=outcome,
        )
        return [(treatment, outcome, "User specified")]

    # Priority 2: No profile, no inference possible.
    if profile is None:
        return []

    treatment_candidates = profile.treatment_candidates or []
    outcome_candidates = profile.outcome_candidates or []

    if not treatment_candidates or not outcome_candidates:
        return []

    # Priority 4: Single candidate each — no LLM call needed.
    if len(treatment_candidates) == 1 and len(outcome_candidates) == 1:
        agent.logger.info(
            "single_candidates",
            treatment=treatment_candidates[0],
            outcome=outcome_candidates[0],
        )
        return [(
            treatment_candidates[0],
            outcome_candidates[0],
            "Single candidates identified",
        )]

    # Priority 5: Multiple candidates — let the LLM filter.
    agent.logger.info(
        "multiple_candidates_llm_filtering",
        n_treatments=len(treatment_candidates),
        n_outcomes=len(outcome_candidates),
    )

    prompt = build_pair_selection_prompt(profile)
    try:
        result = await agent.llm.generate(
            prompt=prompt,
            system_instruction="You are an expert in causal inference. Return valid JSON only.",
        )
        pairs = parse_pair_selection({"response": result.text}, profile, agent.logger)
        if pairs:
            return pairs
    except Exception as e:
        agent.logger.warning("pair_selection_llm_failed", error=str(e))

    # Fallback: use the first candidates.
    agent.logger.info("fallback_to_first_candidates")
    return [(
        treatment_candidates[0],
        outcome_candidates[0],
        "Fallback to first candidates",
    )]


def build_pair_selection_prompt(profile) -> str:
    """Compose the JSON-structured prompt for LLM pair filtering."""
    return f"""You are evaluating potential causal relationships in a dataset.

Dataset Profile:
- Total features: {profile.n_features}
- Feature names: {profile.feature_names[:30]}{"..." if len(profile.feature_names) > 30 else ""}
- Treatment candidates: {profile.treatment_candidates}
- Outcome candidates: {profile.outcome_candidates}
- Potential confounders: {profile.potential_confounders[:10]}{"..." if len(profile.potential_confounders) > 10 else ""}

Your task: Identify which treatment-outcome pairs represent VALID causal questions.

A valid causal pair requires:
1. Temporal ordering: Treatment could plausibly precede outcome
2. Manipulability: Treatment is something that could be intervened upon
3. Non-identity: Treatment and outcome measure different concepts
4. Plausible mechanism: There's a reasonable pathway for effect

INVALID pairs include:
- Demographic → Demographic (age cannot cause gender)
- Outcome → Treatment (reverse causality)
- Proxies of each other (revenue_usd → revenue_eur)
- Treatment cannot affect outcome by design

Return your analysis as JSON (no markdown, just raw JSON):
{{
    "valid_pairs": [
        {{"treatment": "var_name", "outcome": "var_name", "rationale": "brief explanation", "priority": 1}}
    ],
    "rejected_pairs": [
        {{"treatment": "var_name", "outcome": "var_name", "reason": "why invalid"}}
    ]
}}

IMPORTANT:
- Limit to at most 3 valid pairs (prioritize the most scientifically interesting)
- Priority 1 = most important, 2 = secondary, 3 = exploratory
- Only include pairs where BOTH variables are in the candidates lists above
"""


def parse_pair_selection(
    result: dict[str, Any],
    profile,
    logger,
) -> list[tuple[str, str, str]]:
    """Extract (treatment, outcome, rationale) tuples from the LLM response.

    Tolerant of fenced JSON and surrounding prose. Drops pairs not in
    the candidates lists.
    """
    response_text = result.get("response", "")

    try:
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response_text[start:end]
            else:
                return []

        data = json.loads(json_str)
        valid_pairs = data.get("valid_pairs", [])

        pairs: list[tuple[str, str, str]] = []
        valid_treatments = set(profile.treatment_candidates or [])
        valid_outcomes = set(profile.outcome_candidates or [])

        for p in valid_pairs[:3]:
            treatment = p.get("treatment")
            outcome = p.get("outcome")
            rationale = p.get("rationale", "LLM selected")

            if treatment in valid_treatments and outcome in valid_outcomes:
                pairs.append((treatment, outcome, rationale))
                logger.info(
                    "valid_pair_identified",
                    treatment=treatment,
                    outcome=outcome,
                    rationale=rationale[:100],
                )

        return pairs

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.warning("pair_selection_parse_failed", error=str(e))
        return []
