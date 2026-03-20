# Skill: Results Display

Load when: building components to display causal analysis results.

## Results structure from API

```typescript
interface AnalysisResults {
  treatment_variable: string;
  outcome_variable: string;
  treatment_effects: TreatmentEffect[];
  sensitivity_results: SensitivityResult[];
  data_profile: DataProfile;
  eda_result: EDAResult;
  proposed_dag: CausalDAG | null;
  critique_history: CritiqueFeedback[];
  notebook_path: string;
}

interface TreatmentEffect {
  method: string;
  estimand: string;      // ATE, ATT, CATE, LATE
  estimate: number;
  std_error: number;
  ci_lower: number;
  ci_upper: number;
  p_value: number;
}
```

## Display components

### Treatment Effects Table
- Show all estimation methods with estimates, CIs, p-values
- Highlight the primary/recommended estimate
- Color significance: green for p < 0.05, grey for non-significant
- Forest plot visualization (horizontal CI bars)

### Sensitivity Summary
- E-value with interpretation text
- Robustness indicator (fragile / moderate / robust)
- List of sensitivity checks that were run

### DAG Visualization (if available)
- Graph rendering of causal structure
- Highlight treatment -> outcome path
- Show adjustment set variables
- Edge labels (if available)

### Critique Summary
- Overall decision: APPROVE / ITERATE / REJECT
- Dimension scores (methodology, rigor, robustness)
- Key issues identified
- Iteration history (if multiple rounds)

### Notebook Download
- Prominent download button
- File size indication
- "Open in Jupyter" instruction text

## Rules

1. Treatment effects are the hero section — most prominent
2. Always show confidence intervals, not just point estimates
3. Never display raw numbers without labels and units
4. Sensitivity results should be interpreted, not just numbers
5. DAG visualization is optional — only show if proposed_dag exists
6. Critique section shows the quality assessment transparently
