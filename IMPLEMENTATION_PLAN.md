# Multi-Causal-Pair Analysis Implementation Plan

## Overview
Add LLM-driven causal pair filtering to analyze multiple valid treatment-outcome combinations when candidates exist.

## Current State
- Effect estimator takes first treatment candidate and first outcome candidate
- Runs multiple estimation methods (OLS, IPW, AIPW, etc.) for that ONE pair
- Works correctly for single-pair scenarios

## Target State
- LLM evaluates all treatment × outcome combinations
- Filters to pairs that make causal sense
- Runs full analysis pipeline for each valid pair
- Results stored per pair with clear labeling

---

## Implementation Details

### File: `effect_estimator.py`

#### Change 1: Add `CausalPair` data structure (in state.py)
```python
class CausalPair(BaseModel):
    """A treatment-outcome pair for causal analysis."""
    treatment: str
    outcome: str
    rationale: str  # Why this pair makes causal sense
    priority: int = 1  # 1 = primary, 2 = secondary
```

#### Change 2: Update `TreatmentEffectResult` (in state.py)
Add fields to track which pair was analyzed:
```python
class TreatmentEffectResult(BaseModel):
    # ... existing fields ...
    treatment_variable: str | None = None  # NEW
    outcome_variable: str | None = None    # NEW
```

#### Change 3: Add `_identify_valid_causal_pairs()` method
```python
async def _identify_valid_causal_pairs(
    self,
    profile: DataProfile
) -> list[tuple[str, str, str]]:
    """Use LLM to identify valid treatment-outcome pairs.

    Returns: List of (treatment, outcome, rationale) tuples
    """
    # If user specified variables, use those (backward compat)
    if self._state.treatment_variable and self._state.outcome_variable:
        return [(
            self._state.treatment_variable,
            self._state.outcome_variable,
            "User specified"
        )]

    # If only one candidate each, no LLM needed
    if (len(profile.treatment_candidates) <= 1 and
        len(profile.outcome_candidates) <= 1):
        t = profile.treatment_candidates[0] if profile.treatment_candidates else None
        o = profile.outcome_candidates[0] if profile.outcome_candidates else None
        if t and o:
            return [(t, o, "Single candidates")]
        return []

    # Multiple candidates - ask LLM to filter
    prompt = self._build_pair_selection_prompt(profile)
    result = await self.reason(prompt, context={"task": "pair_selection"})

    # Parse LLM response for valid pairs
    return self._parse_pair_selection(result, profile)
```

#### Change 4: Modify `execute()` method
```python
async def execute(self, state: AnalysisState) -> AnalysisState:
    # ... data loading stays same ...

    # NEW: Identify all valid pairs to analyze
    pairs = await self._identify_valid_causal_pairs(state.data_profile)

    if not pairs:
        state.mark_failed("No valid treatment-outcome pairs identified", self.AGENT_NAME)
        return state

    self.logger.info("valid_pairs_identified", count=len(pairs))

    # Analyze each pair
    all_results = []
    for treatment, outcome, rationale in pairs:
        self._treatment_var = treatment
        self._outcome_var = outcome
        self._results = []  # Reset for this pair

        self.logger.info("analyzing_pair", treatment=treatment, outcome=outcome)

        # Run agentic loop for this pair
        pair_result = await self._run_agentic_loop(
            self._build_initial_prompt(state),
            max_iterations=15  # Slightly fewer since multiple pairs
        )

        # Tag results with pair info
        for result in self._results:
            result.treatment_variable = treatment
            result.outcome_variable = outcome

        all_results.extend(self._results)

    state.treatment_effects = all_results
    # ... rest stays same ...
```

---

## LLM Prompt for Pair Selection

```python
PAIR_SELECTION_PROMPT = """You are evaluating potential causal relationships in a dataset.

Dataset Profile:
- Features: {feature_names}
- Treatment candidates: {treatment_candidates}
- Outcome candidates: {outcome_candidates}

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

Return your analysis as JSON:
{{
    "valid_pairs": [
        {{"treatment": "...", "outcome": "...", "rationale": "...", "priority": 1}},
        ...
    ],
    "rejected_pairs": [
        {{"treatment": "...", "outcome": "...", "reason": "..."}}
    ]
}}

Limit to at most 3 valid pairs (prioritize the most scientifically interesting).
"""
```

---

## Backward Compatibility Guarantees

| Scenario | Behavior |
|----------|----------|
| User provides treatment + outcome | Uses exactly those (no LLM call) |
| 1 treatment candidate, 1 outcome candidate | Uses those directly (no LLM call) |
| Multiple candidates, no user input | LLM selects valid pairs |
| No candidates found | Fails with clear error |

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| LLM returns invalid JSON | Fallback to first candidates |
| LLM returns 0 valid pairs | Fallback to first candidates with warning |
| Too many pairs (slow) | Cap at 3 pairs max |
| Existing tests break | New behavior only triggers with multiple candidates |

---

## Testing Strategy

1. **Unit test**: Single candidate still works (no regression)
2. **Unit test**: User-specified variables override candidates
3. **Integration test**: Multiple candidates triggers pair selection
4. **E2E test**: Full job with multi-candidate dataset

---

## Files to Modify

1. `backend/src/agents/base/state.py`
   - Add `treatment_variable` and `outcome_variable` to `TreatmentEffectResult`

2. `backend/src/agents/specialists/effect_estimator.py`
   - Add `_identify_valid_causal_pairs()` method
   - Add `_build_pair_selection_prompt()` method
   - Add `_parse_pair_selection()` method
   - Modify `execute()` to loop through pairs

---

## Implementation Order

1. ✅ Add fields to `TreatmentEffectResult` (safe, additive)
2. ✅ Add helper methods (safe, not called yet)
3. ✅ Modify `execute()` with backward-compat checks
4. ✅ Test locally with single-candidate dataset (regression test)
5. ✅ Test with multi-candidate dataset
6. ✅ Deploy

---

## Estimated Changes

- `state.py`: ~5 lines added
- `effect_estimator.py`: ~80 lines added, ~20 lines modified
- No API changes
- No frontend changes
- No database schema changes
