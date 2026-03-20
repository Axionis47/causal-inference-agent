# Tooltip Glossary and Results Narrative Templates

---

## Section 1: Tooltip Glossary

### Estimand Types

**ATE (Average Treatment Effect)**
The average difference in the outcome if everyone received the treatment versus if no one did. It answers: "What is the overall effect of this treatment across the entire population?"

**ATT (Average Treatment Effect on the Treated)**
The average effect of the treatment specifically among those who actually received it. It answers: "How much did the treatment help the people who got it?"

**CATE (Conditional Average Treatment Effect)**
The average treatment effect for a specific subgroup defined by certain characteristics. It answers: "Does this treatment work differently for men vs. women, or for high-income vs. low-income groups?"

**LATE (Local Average Treatment Effect)**
The average treatment effect among the subgroup of people whose treatment status was changed by an instrument or encouragement. It applies to "compliers," not the full population.

### Methods

**OLS (Ordinary Least Squares)**
Fits a straight-line relationship between the treatment and the outcome while adjusting for other variables.

**IPW (Inverse Probability Weighting)**
Reweights observations so that the treated and untreated groups look comparable, based on each unit's probability of receiving treatment.

**AIPW (Augmented Inverse Probability Weighting)**
Combines outcome modeling with probability weighting, giving a more reliable estimate even if one of the two models is slightly wrong.

**PSM (Propensity Score Matching)**
Pairs each treated unit with an untreated unit that had a similar probability of being treated, then compares their outcomes.

**DiD (Difference-in-Differences)**
Compares the change in outcomes over time between a group that received the treatment and a group that did not, removing shared trends.

**IV/2SLS (Instrumental Variables / Two-Stage Least Squares)**
Uses an external factor (the "instrument") that affects treatment but not the outcome directly, allowing estimation even when there is unmeasured confounding.

**RDD (Regression Discontinuity Design)**
Estimates the treatment effect by comparing units just above and just below a cutoff that determines treatment assignment.

**S-Learner**
Trains a single predictive model on both treated and untreated units, then estimates the treatment effect by comparing predictions with and without treatment.

**T-Learner**
Trains separate predictive models for the treated group and the untreated group, then takes the difference in their predictions as the treatment effect.

**X-Learner**
A two-stage approach that first estimates individual treatment effects using cross-group predictions, then combines them with propensity scores. It performs well when one group is much larger than the other.

**Causal Forest**
A machine-learning method that builds many decision trees to estimate how the treatment effect varies across individuals with different characteristics.

**Double ML (Double Machine Learning)**
Uses machine learning to flexibly control for confounders in a first stage, then estimates the causal effect in a second stage, correcting for overfitting bias.

### Diagnostics and Sensitivity

**p-value**
The probability of seeing a result this extreme (or more extreme) if the treatment truly had no effect. A small p-value (typically below 0.05) suggests the result is unlikely to be due to chance alone.

**Confidence Interval**
A range of values that is likely to contain the true treatment effect. A 95% confidence interval means that if we repeated the study many times, about 95% of such intervals would contain the true value.

**Standard Error**
A measure of how much the estimated effect would vary if we repeated the analysis on different samples from the same population. Smaller standard errors mean more precise estimates.

**E-value**
The minimum strength an unmeasured confounder would need to have (in terms of its association with both treatment and outcome) to fully explain away the observed effect. A large E-value means the result is harder to attribute to hidden bias.

**Rosenbaum Bounds**
A sensitivity check that asks: "How strong would hidden bias have to be to change our conclusion?" Higher bounds mean the finding is more resistant to unobserved confounding.

**Specification Curve**
A diagnostic that re-runs the analysis under many reasonable alternative model choices (different controls, samples, functional forms) and shows how the estimated effect changes. Consistent results across specifications increase confidence.

**Placebo Test**
A check that applies the same analysis to an outcome or time period where no treatment effect should exist. If the method finds a "significant" effect on the placebo, the original result may be suspect.

**Propensity Score**
The estimated probability that a given unit receives the treatment, based on its observed characteristics. It is used to make treated and untreated groups more comparable.

**Common Support / Overlap**
The range of propensity scores where both treated and untreated units exist. If overlap is poor, causal estimates become unreliable because we are comparing units with no good match.

**Covariate Balance (SMD)**
The standardized mean difference between treated and untreated groups on each background variable. An SMD below 0.1 generally indicates good balance, meaning the groups are comparable on that variable.

**DAG (Directed Acyclic Graph)**
A diagram that maps out assumed cause-and-effect relationships between variables. It helps identify which variables to control for and which to leave alone.

**Confounder**
A variable that influences both the treatment and the outcome. Failing to account for a confounder can make a treatment appear to have an effect it does not actually have (or hide a real effect).

**Mediator**
A variable that lies on the causal path between the treatment and the outcome. The treatment causes the mediator, and the mediator in turn causes the outcome. Controlling for a mediator can block part of the true causal effect.

**Collider**
A variable that is caused by both the treatment and the outcome (or by two variables on separate paths). Controlling for a collider can introduce a false association and should generally be avoided.

**Backdoor Criterion**
A rule from causal graph theory that identifies which sets of variables, when controlled for, block all spurious paths between treatment and outcome. If the criterion is satisfied, the causal effect is identifiable from the data.

**Adjustment Set**
The specific set of variables selected for inclusion in the analysis to block confounding paths. A valid adjustment set satisfies the backdoor criterion and does not include colliders or mediators (unless mediation analysis is the goal).

---

## Section 2: Narrative Templates

### Template 1: Strong Finding

Use when the estimated effect is statistically significant (p < 0.05), the confidence interval excludes zero by a meaningful margin, and sensitivity analysis supports the result (e.g., high E-value or stable specification curve).

> **[Treatment] [increases/decreases] [outcome] by [estimate] ([CI]).
> This result holds up under sensitivity analysis: an unmeasured confounder would need [E-value interpretation] to explain the finding away.
> Across [N] alternative model specifications, the direction and significance of the effect remain consistent.**

*Example:*
> Enrolling in the coaching program increases quarterly revenue by $12,400 (95% CI: $8,200 to $16,600).
> This result holds up under sensitivity analysis: an unmeasured confounder would need to be associated with both coaching enrollment and revenue by a factor of at least 3.1 to explain the finding away.
> Across 48 alternative model specifications, the direction and significance of the effect remain consistent.

### Template 2: Moderate Finding

Use when the estimated effect is statistically significant (p < 0.05) but sensitivity analysis raises some concerns (e.g., moderate E-value, some specification instability, or limited overlap in propensity scores).

> **There is evidence that [treatment] affects [outcome] ([estimate], [CI]), though the result is sensitive to certain assumptions.
> [Specific caveat: e.g., "The E-value of 1.8 means a moderately strong unmeasured confounder could account for this finding" or "Results vary across some model specifications" or "Overlap between treated and untreated groups is limited in parts of the distribution."]
> We recommend interpreting this finding with caution and, where possible, validating it with additional data or a prospective test.**

*Example:*
> There is evidence that the discount campaign affects customer retention (estimated increase of 4.2 percentage points, 95% CI: 0.8 to 7.6), though the result is sensitive to certain assumptions.
> The E-value of 1.8 means a moderately strong unmeasured confounder could account for this finding. Additionally, 12% of treated units fall outside the region of common support.
> We recommend interpreting this finding with caution and, where possible, validating it with a prospective A/B test.

### Template 3: Weak or Null Finding

Use when the estimated effect is not statistically significant (p >= 0.05), the confidence interval includes zero, or sensitivity analysis reveals fragility.

> **We did not find strong evidence that [treatment] affects [outcome] ([estimate], [CI]).
> [Possible explanation: e.g., "The sample may be too small to detect a meaningful effect (the study was powered to detect effects of [X] or larger)" or "Residual confounding makes it difficult to isolate the treatment effect" or "The effect, if it exists, appears too small to distinguish from noise in this dataset."]
> This does not prove the treatment has no effect. It means the data do not provide sufficient evidence to conclude that it does.**

*Example:*
> We did not find strong evidence that the new onboarding flow affects 30-day retention (estimated change of 1.1 percentage points, 95% CI: -2.3 to 4.5).
> The sample of 340 users may be too small to detect a meaningful effect. The study was powered to detect retention differences of 5 percentage points or larger.
> This does not prove the new onboarding flow has no effect. It means the data do not provide sufficient evidence to conclude that it does.

---

### Confidence Level Mapping

| Condition | Language to Use |
|---|---|
| p < 0.01, tight CI excluding zero, high E-value (>2.5), stable across specifications | "Strong evidence" |
| p < 0.05, CI excluding zero, moderate E-value (1.5 to 2.5), mostly stable across specifications | "Moderate evidence" or "evidence suggesting" |
| p between 0.05 and 0.10, CI barely excluding or including zero, low E-value (<1.5) | "Weak evidence" or "limited evidence" |
| p >= 0.10, CI clearly includes zero | "No clear evidence" or "we did not find evidence" |

Do not say "we proved" or "we confirmed." Observational causal inference supports conclusions at varying levels of confidence. It does not prove them.

### Direction Language

| Condition | Language to Use |
|---|---|
| Point estimate > 0 and CI excludes zero | "[Treatment] increases [outcome]" |
| Point estimate < 0 and CI excludes zero | "[Treatment] decreases [outcome]" |
| Point estimate is near zero or CI includes zero | "[Treatment] has no clear effect on [outcome]" |
| Point estimate is positive but CI includes zero | "[Treatment] may increase [outcome], but the evidence is inconclusive" |
| Point estimate is negative but CI includes zero | "[Treatment] may decrease [outcome], but the evidence is inconclusive" |

Avoid saying "no effect." Say "no clear effect" or "no detectable effect." The absence of evidence is not evidence of absence.
