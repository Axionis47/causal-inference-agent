# Skill: Agent Evaluation Rubrics

Load when: writing evaluation rubrics or agent quality scoring.

## What to evaluate per agent

### Effect Estimator (most critical)
- Did it select methods appropriate for the data structure?
- Did it run at least 2 methods for triangulation?
- Did it include OLS baseline?
- Are confidence intervals reasonable (not pathologically wide)?
- Did it check assumptions before interpreting?

### Causal Discovery
- Did it run at least 2 algorithms for comparison?
- Is the resulting DAG acyclic?
- Does the DAG include treatment and outcome?
- Are edge directions plausible?

### Sensitivity Analyst
- Did it compute E-value?
- Did it run at least one additional robustness check?
- Did it interpret results (not just report numbers)?

### Critique Agent
- Did it converge to a decision (not auto-finalize at MAX_STEPS)?
- Are dimension scores justified by evidence?
- Are suggested improvements actionable?

## Scoring scale

| Score | Meaning |
|-------|---------|
| 1 | Agent failed to produce any useful output |
| 2 | Agent produced output but with major errors |
| 3 | Agent produced reasonable output with minor issues |
| 4 | Agent produced good output, appropriate for the data |
| 5 | Agent produced excellent output, methodologically rigorous |

## Eval test structure

```python
def test_effect_estimator_quality(analysis_result):
    effects = analysis_result.treatment_effects
    assert len(effects) >= 2, "Must run at least 2 methods"
    methods = [e.method for e in effects]
    assert "ols" in methods, "OLS baseline required"
    for e in effects:
        assert not np.isnan(e.estimate), f"{e.method} produced NaN"
        assert e.ci_lower < e.ci_upper, f"{e.method} invalid CI"
```
