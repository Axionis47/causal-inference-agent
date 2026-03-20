# Skill: Causal Methodology Review

Load when: reviewing method selection logic, evaluating whether the right estimation
methods are being applied, or defining method applicability criteria.

## The 12 estimation methods — when each applies

| Method | Estimand | Key Assumption | When It Works | When It Breaks |
|--------|----------|----------------|---------------|----------------|
| OLS | ATE | No unmeasured confounders, linear relationship | Baseline. Always run for comparison | Confounders not observed, nonlinear effects |
| IPW | ATE/ATT | Correct propensity model, positivity | Observational data with good overlap | Extreme propensity scores, model misspecification |
| AIPW | ATE | Either PS or outcome model correct (doubly robust) | When you distrust one model | Both models wrong |
| PSM | ATT | Common support, no unmeasured confounders | Small treatment groups, need matched pairs | Poor overlap, high dimensionality |
| DiD | ATT | Parallel trends | Panel data with pre/post treatment | Parallel trends violated, anticipation effects |
| IV/2SLS | LATE | Relevance, exclusion, independence | Valid instrument available | Weak instrument (F < 10), exclusion violated |
| RDD | LATE | Continuity at cutoff | Known assignment threshold | Manipulation of running variable |
| S-Learner | CATE | Correct outcome model | Simple heterogeneity estimation | Treatment effect swamped by outcome variation |
| T-Learner | CATE | Separate models accurate | Clear treatment/control split | Small sample in one arm |
| X-Learner | CATE | Cross-estimation valid | Unequal group sizes | Very small samples |
| Causal Forest | CATE | Honest estimation, sufficient depth | Large samples, complex heterogeneity | Small samples (< 500), high dimensionality |
| Double ML | ATE | Nuisance parameters estimable, Neyman orthogonality | High-dimensional confounders | Small samples, poor first-stage fit |

## The 5 discovery algorithms

| Algorithm | Assumption | Strengths | Limitations |
|-----------|-----------|-----------|-------------|
| PC | Faithfulness, no latent confounders | Fast, well-understood | Sensitive to conditional independence test choice |
| FCI | Allows latent confounders | Handles hidden variables | Produces PAGs (partial), harder to interpret |
| GES | Score-based, BIC penalty | No independence tests needed | Can get stuck in local optima |
| NOTEARS | Continuous optimization, acyclicity | Scalable to many variables | Assumes linear relationships by default |
| LiNGAM | Non-Gaussian errors, linearity | Identifies full causal order | Breaks with Gaussian data |

## Method selection decision tree

```
Is there a valid instrument?
  Yes -> IV/2SLS (check F-stat > 10)
  No ->

Is there a known assignment cutoff?
  Yes -> RDD (check manipulation)
  No ->

Is there panel data with pre/post?
  Yes -> DiD (check parallel trends)
  No ->

Is the treatment binary with good overlap?
  Yes -> IPW, AIPW, PSM (check propensity diagnostics)
  No ->

Is heterogeneous treatment effects the goal?
  Yes -> S/T/X-Learner, Causal Forest (need n > 500)
  No ->

Are there many potential confounders?
  Yes -> Double ML (handles high-dimensional nuisance)
  No -> OLS as baseline
```

Always run OLS as a baseline regardless of other methods.
Always run at least 2 methods and compare estimates for robustness.

## Review checklist

- [ ] Method matches data structure (cross-section vs panel vs RDD)
- [ ] Key assumptions stated and testable ones checked
- [ ] Sample size sufficient for chosen method
- [ ] At least 2 methods run for triangulation
- [ ] OLS baseline included
- [ ] Confidence intervals and p-values reported
- [ ] Effect size interpretation is substantively meaningful
