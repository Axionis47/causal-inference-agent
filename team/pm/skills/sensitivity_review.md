# Skill: Sensitivity Analysis Review

Load when: reviewing robustness checks and sensitivity analysis quality.

## What sensitivity analysis must answer

The central question: "How strong would unmeasured confounding need to be
to explain away the estimated treatment effect?"

If the answer is "trivially weak" — the estimate is fragile.
If the answer is "implausibly strong" — the estimate is robust.

## The 6 sensitivity checks

| Check | What it tests | When to use | Red flag |
|-------|--------------|-------------|----------|
| E-value | Minimum confounding strength to nullify | Always (every analysis) | E-value < 2.0 |
| Rosenbaum bounds | Sensitivity to hidden bias in matched studies | After PSM | Gamma < 1.5 |
| Specification curve | Stability across model specifications | When multiple specs possible | Estimates flip sign |
| Placebo test | False positive rate on non-treated outcomes | When placebo outcome exists | Placebo effect significant |
| Subgroup analysis | Effect consistency across subpopulations | When heterogeneity suspected | Effects reverse in subgroups |
| Bootstrap variance | Estimate precision and stability | Always useful | Wide CI relative to effect |

## E-value interpretation guide

| E-value | Interpretation |
|---------|---------------|
| < 1.5 | Very fragile — even weak confounding could explain it |
| 1.5 - 2.0 | Somewhat fragile — moderate confounding could explain it |
| 2.0 - 3.0 | Moderately robust — would need substantial confounding |
| 3.0 - 5.0 | Robust — would need strong confounding |
| > 5.0 | Very robust — implausibly strong confounding needed |

## Review checklist

- [ ] E-value computed for primary estimate
- [ ] E-value interpreted in context (not just reported as a number)
- [ ] At least one additional robustness check beyond E-value
- [ ] Placebo test run if a suitable placebo outcome exists
- [ ] Specification curve shows estimate stability (if applicable)
- [ ] Subgroup effects consistent in direction (magnitude can vary)
- [ ] Sensitivity results integrated into conclusions (not ignored)
- [ ] Limitations acknowledge what sensitivity analysis cannot test

## Common failures

- Reporting E-value without interpretation
- Running sensitivity only on the best-looking estimate
- Ignoring placebo test failures ("it's just noise")
- Specification curve shows sign changes but conclusions ignore this
- Subgroup analysis used for fishing instead of pre-specified checks
