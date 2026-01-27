# Causal Methods Reference

This document explains the causal inference methods implemented in the system.

---

## Overview

The system implements three categories of methods:

1. **Causal Discovery** (6 algorithms): Learn causal structure from data
2. **Effect Estimation** (12 methods): Estimate treatment effects
3. **Sensitivity Analysis** (6 tests): Check robustness of findings

---

## Part 1: Causal Discovery Algorithms

These algorithms learn the causal DAG (directed acyclic graph) from observational data.

### PC Algorithm (Peter-Clark)

**What it does:** Starts with a fully connected graph and removes edges when variables are conditionally independent.

**Assumptions:**
- All confounders are observed (no hidden variables)
- Causal faithfulness (independence implies no edge)
- Correct significance level (default α = 0.05)

**How it works:**
1. Start with all variables connected
2. Test if A ⊥ B (unconditional independence)
3. Test if A ⊥ B | C (conditional independence given C)
4. Remove edges where independence holds
5. Orient remaining edges using rules

**Parameters:**
```
ci_test: "fisherz" (linear) or "kernel" (nonlinear)
alpha: 0.05 (significance threshold)
```

**When to use:**
- Good starting point for structure learning
- Works well with n > 50 observations
- Best for linear/Gaussian data with FisherZ test
- Use kernel test for nonlinear relationships

**Limitations:**
- Assumes no hidden confounders
- Sensitive to sample size
- Cannot orient all edges (returns equivalence class)

---

### FCI Algorithm (Fast Causal Inference)

**What it does:** Extension of PC that can detect hidden confounders.

**Output:** Instead of a DAG, outputs a PAG (Partial Ancestral Graph) with special edge marks:
- `-->` : A causes B
- `<->` : A and B share a hidden common cause
- `o->` : A might cause B (uncertain direction)

**When to use:**
- When you suspect unmeasured confounding
- Observational studies where you cannot measure everything
- Need to know which relationships might be confounded

**Example output:**
```
income <-> health     (hidden confounder suspected)
education --> income  (definite causal direction)
age o-> health        (uncertain direction)
```

---

### GES Algorithm (Greedy Equivalence Search)

**What it does:** Searches for the DAG that best fits the data using a score (BIC).

**How it works:**
1. Start with empty graph
2. Add edges that improve BIC score (forward phase)
3. Remove edges that improve BIC score (backward phase)
4. Return best-scoring DAG

**When to use:**
- Prefer score-based over test-based approach
- Have clear sample size for BIC penalty
- Good alternative to PC

---

### NOTEARS Algorithm

**What it does:** Treats DAG learning as a continuous optimization problem with an acyclicity constraint.

**Key innovation:** Guarantees the output is acyclic by construction (no cycles possible).

**Assumptions:**
- Linear relationships (Y = WY + noise)
- Continuous data

**Parameters:**
```
lambda1: 0.1 (sparsity penalty)
w_threshold: 0.3 (minimum edge weight)
```

**When to use:**
- Want guaranteed acyclic output
- Linear relationships
- Prefer optimization to hypothesis testing

**Limitations:**
- Assumes linear relationships
- May miss nonlinear dependencies

---

### LiNGAM Algorithm

**What it does:** Exploits non-Gaussianity to identify causal direction.

**Key insight:** If the data is non-Gaussian, we can tell cause from effect. In Y = aX + e, if e is non-Gaussian, the direction is identifiable.

**Assumptions:**
- **Critical:** Data must be non-Gaussian (not bell-curved)
- Linear relationships
- No cycles

**When to use:**
- Data is clearly non-Gaussian (heavy tails, skewed, uniform, etc.)
- Want to identify direction without needing time information
- Linear relationships

**Limitations:**
- Fails if data is Gaussian
- Sensitive to violations of linearity

---

### Ensemble Discovery

**What it does:** Runs multiple algorithms (PC + GES + NOTEARS) and takes consensus.

**Parameters:**
```
algorithms: ["pc", "ges", "notears"]
voting_threshold: 0.5 (fraction agreeing to keep edge)
```

**When to use:**
- Uncertain which single algorithm is best
- Want robust consensus
- Have computational budget for multiple runs

---

### Bootstrap Stability Selection

**What it does:** Assesses confidence in discovered edges by resampling.

**How it works:**
1. Resample 80% of data
2. Run discovery algorithm
3. Repeat 100 times
4. Keep edges appearing in >50% of runs

**Output:**
- Consensus graph (stable edges only)
- Frequency matrix (how often each edge appeared)

**When to use:**
- Want confidence estimates for edges
- Assess stability of discovered structure
- Use with any discovery algorithm

---

### Conditional Independence Tests

The system uses different tests depending on data type:

| Test | Data Type | What It Tests |
|------|-----------|---------------|
| FisherZ | Continuous/Linear | Partial correlation = 0 |
| Kernel (HSIC) | Any/Nonlinear | Statistical dependence |
| Chi-square | Categorical | Independence of categories |
| G-test | Categorical | Alternative to chi-square |

**FisherZ:** Fast, assumes linear relationships. Uses partial correlation via regression residuals.

**Kernel HSIC:** Slower but handles nonlinear relationships. Uses RBF kernel with median heuristic.

---

## Part 2: Treatment Effect Estimation

### Understanding Estimands

Before choosing a method, understand what you want to estimate:

| Estimand | Full Name | What It Means |
|----------|-----------|---------------|
| ATE | Average Treatment Effect | Effect on everyone |
| ATT | Average Effect on Treated | Effect on those who got treatment |
| LATE | Local Average Treatment Effect | Effect on "compliers" (IV/RDD) |
| CATE | Conditional Average Treatment Effect | Effect for specific subgroups |

---

### OLS Regression

**Estimand:** ATE

**Model:** Y = α + βT + γX + ε

**Assumptions:**
- Linear relationship
- No unmeasured confounders
- Homoscedasticity (constant variance)

**Output:**
- β is the ATE
- Uses HC1 robust standard errors by default

**Diagnostics:**
- R-squared and adjusted R-squared
- F-statistic
- Heteroscedasticity test (Breusch-Pagan)
- Normality of residuals

**When to use:**
- Simple baseline estimate
- Likely linear relationship
- Want interpretable coefficients

**Limitations:**
- Strong linearity assumption
- Sensitive to outliers
- May be biased with many covariates

---

### Propensity Score Matching (PSM)

**Estimand:** ATT (effect on treated)

**Idea:** For each treated unit, find a similar control unit based on propensity score.

**Propensity score:** P(T=1|X) = probability of treatment given covariates

**How it works:**
1. Fit logistic regression: T ~ X
2. Compute propensity score for each unit
3. For each treated unit, find nearest control (within caliper)
4. Compute mean difference in outcomes

**Parameters:**
```
n_neighbors: 1 (how many controls to match)
caliper: 0.2 (max distance in PS standard deviations)
```

**Diagnostics:**
- Match rate (fraction of treated units matched)
- Propensity score overlap
- Balance after matching

**When to use:**
- Want interpretable matched pairs
- Good overlap in propensity scores
- Sufficient sample size

**Limitations:**
- Discards unmatched units
- No estimate if matching fails
- Only estimates ATT, not ATE

---

### Inverse Probability Weighting (IPW)

**Estimand:** ATE

**Idea:** Weight each observation inversely by probability of receiving its treatment.

**Formula:**
```
ATE = mean(T × Y / PS) - mean((1-T) × Y / (1-PS))
```

**How it works:**
1. Estimate propensity scores via logistic regression
2. Clip to [0.01, 0.99] to avoid extreme weights
3. Compute weighted means for treated and control
4. Bootstrap for standard errors (200 iterations)

**Diagnostics:**
- Propensity score range (min, max, mean)
- Effective sample size after weighting

**When to use:**
- Want to use all observations
- Good propensity score overlap
- Treatment/control groups differ substantially

**Limitations:**
- Sensitive to extreme propensity scores
- High variance if weights are extreme
- Requires correct propensity model

---

### AIPW / Doubly Robust

**Estimand:** ATE

**Key advantage:** Consistent if EITHER the propensity model OR the outcome model is correct (not both needed).

**How it works:**
1. Estimate propensity scores (logistic regression)
2. Fit outcome models for treated and control (linear regression)
3. Combine using augmented IPW formula

**Formula:**
```
AIPW = T(Y - μ₁)/PS + μ₁ - (1-T)(Y - μ₀)/(1-PS) - μ₀
```

Where μ₁, μ₀ are predicted outcomes from regression.

**Diagnostics:**
- Propensity score overlap
- Outcome model R² for treated and control

**When to use:**
- Default choice for observational studies
- Want protection against model misspecification
- Good outcome model candidates available

**Limitations:**
- More complex than simple IPW
- Still needs at least one model to be correct

---

### Difference-in-Differences (DiD)

**Estimand:** ATT

**Idea:** Compare change in outcome before/after treatment for treatment vs control groups.

**Key assumption:** Parallel trends (groups would have changed equally without treatment)

**Formula:**
```
ATT = (Y_treat,after - Y_treat,before) - (Y_control,after - Y_control,before)
```

**How it works:**
1. Identify pre and post periods
2. Run regression: Y ~ T + post + T×post + covariates
3. ATT is coefficient on T×post (interaction term)
4. Optional clustered standard errors

**Diagnostics:**
- Raw DiD calculation
- Group means by period
- Pre-treatment balance check
- Cell sizes (warns if <10)

**When to use:**
- Panel data (same units observed before and after)
- Treatment introduced at known time
- Can identify control group with parallel trends

**Limitations:**
- Parallel trends assumption often untestable
- Requires clear before/after periods
- Sensitive to composition changes

---

### Instrumental Variables (IV / 2SLS)

**Estimand:** LATE (effect for "compliers")

**Idea:** Use an instrument (Z) that affects treatment but not outcome directly.

**Requirements for valid instrument:**
1. **Relevance:** Z predicts T (strong correlation)
2. **Exclusion:** Z affects Y only through T (no direct effect)
3. **Independence:** Z is exogenous (random or as-if random)

**How it works (2SLS):**
1. First stage: T ~ Z + X (predict treatment from instrument)
2. Second stage: Y ~ T̂ + X (use predicted treatment)

**Diagnostics:**
- First-stage F-statistic (should be >10)
- Weak instrument warning if F < 10

**When to use:**
- Treatment is endogenous (affected by unmeasured confounders)
- Have a valid instrument
- Want to handle unmeasured confounding

**Limitations:**
- Hard to find valid instruments
- Only estimates LATE (effect for compliers)
- Weak instrument bias if F-statistic low

---

### Regression Discontinuity (RDD)

**Estimand:** LATE (at the cutoff)

**Idea:** Treatment is determined by a cutoff on a running variable. Units just above and below cutoff are similar except for treatment.

**Example:** Scholarship awarded if GPA ≥ 3.5. Students at 3.49 vs 3.51 are similar except for scholarship.

**How it works:**
1. Center running variable at cutoff
2. Filter to bandwidth around cutoff
3. Fit local polynomial regression
4. Effect = jump in outcome at cutoff

**Parameters:**
```
running_var: the variable with cutoff
cutoff: 0 (default, or specify value)
bandwidth: auto-calculated
polynomial_order: 1 (local linear)
fuzzy: False (sharp) or True (fuzzy)
```

**Sharp vs Fuzzy:**
- Sharp: Treatment determined exactly by cutoff (T = 1 if R ≥ c)
- Fuzzy: Treatment probability jumps at cutoff but not 100%

**Diagnostics:**
- Sample size above/below cutoff
- Density ratio (McCrary test for manipulation)
- Raw mean jump at cutoff

**When to use:**
- Treatment assigned by clear cutoff rule
- Running variable is continuous
- Sufficient observations near cutoff (>40 per side)

**Limitations:**
- Only estimates effect at cutoff (limited external validity)
- Sensitive to bandwidth choice
- Manipulation of running variable invalidates design

---

### Meta-Learners (S, T, X-Learner)

These estimate heterogeneous treatment effects (CATE): different effects for different subgroups.

#### S-Learner (Single Model)

**Idea:** Fit one model including treatment as a feature.

**Steps:**
1. Fit Y ~ X + T
2. Predict Y(X, T=1) and Y(X, T=0)
3. CATE = Y(X, T=1) - Y(X, T=0)

**Pros:** Simple, stable
**Cons:** May underestimate heterogeneity

#### T-Learner (Two Models)

**Idea:** Fit separate models for treated and control.

**Steps:**
1. Fit μ₁(X) using treated units only
2. Fit μ₀(X) using control units only
3. CATE = μ₁(X) - μ₀(X)

**Pros:** Better captures heterogeneity
**Cons:** Unstable with imbalanced groups

#### X-Learner

**Idea:** Two-stage approach that handles imbalanced treatment groups.

**Steps:**
1. Fit outcome models for treated and control
2. Compute imputed effects:
   - For treated: D₁ = Y₁ - μ₀(X₁)
   - For control: D₀ = μ₁(X₀) - Y₀
3. Fit CATE models on imputed effects
4. Weight by propensity score

**Pros:** Handles imbalanced treatment/control well
**Cons:** More complex

**Base learners available:**
- `gbm`: Gradient Boosting (default for larger samples)
- `rf`: Random Forest
- `linear`: Linear Regression (for small samples)

---

### Causal Forest

**Estimand:** CATE

**What it does:** Builds a forest of trees that target heterogeneous treatment effects.

**Key features:**
- Honest splitting (different data for split vs estimate)
- Valid confidence intervals for CATE
- Uses EconML's CausalForestDML when available

**When to use:**
- Want heterogeneous effects with valid inference
- Prefer ensemble methods
- Have EconML library

**Limitations:**
- Computationally intensive
- Requires larger sample size (n > 500)

---

### Double ML (Debiased Machine Learning)

**Estimand:** ATE

**Key innovation:** Uses cross-fitting to avoid regularization bias.

**How it works:**
1. Split data into K folds
2. For each fold:
   - Fit nuisance models (propensity, outcome) on other folds
   - Compute residuals on this fold
3. Estimate effect from residualized data

**Why it matters:** ML methods (lasso, forests) introduce bias. Cross-fitting removes this bias.

**When to use:**
- High-dimensional covariates
- Want to use ML for nuisance estimation
- Need valid inference despite regularization

---

## Part 3: Sensitivity Analysis

These methods test how robust findings are to assumption violations.

### E-Value

**What it does:** Quantifies minimum strength of unmeasured confounder needed to explain away the effect.

**Interpretation:**
- E = 1.5: A weak confounder could explain the effect
- E = 2.0: Confounder would need to double the odds
- E = 3.0+: Very strong confounder needed

**Thresholds:**
| E-value | E-value for CI | Interpretation |
|---------|----------------|----------------|
| ≥ 3.0 | ≥ 1.5 | Robust |
| ≥ 2.0 | ≥ 1.2 | Moderately robust |
| ≥ 1.5 | any | Somewhat sensitive |
| < 1.5 | any | Sensitive |

**When to use:** Always. This is the baseline sensitivity test for any observational study.

---

### Rosenbaum Bounds

**What it does:** Sensitivity analysis for matched designs.

**Gamma (Γ):** How much hidden bias could exist while maintaining significance.

**Interpretation:**
| Gamma | Interpretation |
|-------|----------------|
| ≥ 3.0 | Very robust |
| ≥ 2.0 | Moderately robust |
| ≥ 1.5 | Somewhat sensitive |
| < 1.5 | Sensitive |

**When to use:**
- After propensity score matching
- Want discrete sensitivity parameter

---

### Specification Curve Analysis

**What it does:** Tests how estimates vary across different reasonable model specifications.

**How it works:**
1. Define multiple reasonable covariate sets
2. Run estimation with each set
3. Examine variation in estimates

**Metrics:**
- Mean and standard deviation across specifications
- Coefficient of variation (CV = std/mean)
- Sign consistency (do all estimates agree on direction?)

**Interpretation:**
| CV | Sign | Interpretation |
|----|------|----------------|
| < 0.2 | Same | Highly stable |
| < 0.4 | Same | Moderately stable |
| < 0.4 | Varies | Variable |
| any | Flips | Unstable |

**When to use:**
- Multiple plausible models exist
- Want to show robustness to modeling choices

---

### Placebo Tests

**Test 1: Placebo Treatment**
- Randomly shuffle treatment assignment
- Re-estimate effect
- Real effect should be much larger than placebo

**Test 2: Placebo Outcome**
- Use fake outcome (random data)
- Real effect should be much larger

**Interpretation:**
- Real effect > 2× placebo 95th percentile: Robust
- Real effect > placebo 95th percentile: Passed
- Real effect < placebo 95th percentile: Concerning

**When to use:**
- Validate analysis approach
- Check for spurious findings

---

### Subgroup Analysis

**What it does:** Checks if effect is consistent across population subgroups.

**How it works:**
1. Split data by subgroup variable
2. Estimate effect in each subgroup
3. Check consistency

**Interpretation:**
| CV | Sign | Interpretation |
|----|------|----------------|
| < 0.3 | Same | Consistent |
| any | Same | Direction consistent |
| any | Varies | Heterogeneous |

**When to use:**
- Detect effect heterogeneity
- Validate assumptions
- Identify subgroups with different effects

---

### Bootstrap Variance Check

**What it does:** Verifies standard errors by resampling.

**How it works:**
1. Bootstrap sample with replacement
2. Re-estimate effect
3. Repeat 200 times
4. Compare bootstrap SE to reported SE

**Interpretation:**
| SE Ratio | Interpretation |
|----------|----------------|
| 0.8 to 1.2 | Stable |
| < 0.8 | Conservative (reported SE too large) |
| > 1.2 | Unstable (bootstrap SE larger) |

**When to use:**
- Verify analytical SE calculations
- Check for numerical instability

---

## Choosing the Right Method

### Decision Tree for Effect Estimation

```
Do you have a randomized experiment?
├─ Yes → OLS
└─ No (observational data)
    │
    Do you have panel data (before/after)?
    ├─ Yes → DiD
    └─ No
        │
        Do you have a valid instrument?
        ├─ Yes → IV
        └─ No
            │
            Do you have a cutoff/threshold?
            ├─ Yes → RDD
            └─ No
                │
                Do you want heterogeneous effects?
                ├─ Yes → X-Learner or Causal Forest
                └─ No
                    │
                    Default → AIPW (doubly robust)
```

### Sample Size Requirements

| Method | Minimum Per Arm | Minimum Total |
|--------|-----------------|---------------|
| OLS/IPW | 30 | 50 |
| AIPW | 30 | 50 |
| Matching | 50 | 100 |
| Meta-learners | 100 | 200 |
| Causal Forest | 200 | 500 |
| RDD | 20 per side | 40 |

### Discovery Algorithm Selection

```
Is data non-Gaussian?
├─ Yes → LiNGAM
└─ No or unsure
    │
    Do you suspect hidden confounders?
    ├─ Yes → FCI
    └─ No
        │
        Is relationship linear?
        ├─ Yes → NOTEARS or PC (FisherZ)
        └─ No → PC (Kernel test)
        │
        Unsure? → Ensemble (PC + GES + NOTEARS)
```

---

## Summary Tables

### Effect Estimators

| Method | Estimand | Key Assumption | Pros | Cons |
|--------|----------|----------------|------|------|
| OLS | ATE | Linearity | Simple | Strong assumptions |
| PSM | ATT | Unconfoundedness | Interpretable | Discards data |
| IPW | ATE | Unconfoundedness | Uses all data | Sensitive to extremes |
| AIPW | ATE | Unconfoundedness | Doubly robust | More complex |
| DiD | ATT | Parallel trends | Natural experiments | Needs time variation |
| IV | LATE | Valid instrument | Handles endogeneity | Hard to find IVs |
| RDD | LATE | No manipulation | Local randomization | Limited generalizability |
| S-Learner | CATE | Unconfoundedness | Simple | Underestimates heterogeneity |
| T-Learner | CATE | Unconfoundedness | Captures heterogeneity | Imbalance issues |
| X-Learner | CATE | Unconfoundedness | Handles imbalance | Complex |
| Causal Forest | CATE | Unconfoundedness | Valid CIs | Computationally heavy |
| Double ML | ATE | Unconfoundedness | High-dimensional | Needs econml |

### Discovery Algorithms

| Algorithm | Assumptions | Pros | Cons |
|-----------|-------------|------|------|
| PC | No hidden confounders | General purpose | Slow, incomplete orientation |
| FCI | Detects hidden | Identifies latent confounders | Complex output |
| GES | None specific | Score-based, robust | Greedy (local optimum) |
| NOTEARS | Linear | Guaranteed acyclic | Linear only |
| LiNGAM | Non-Gaussian, linear | Full identifiability | Needs non-Gaussianity |

### Sensitivity Tests

| Test | What It Checks | Key Output |
|------|----------------|------------|
| E-value | Unmeasured confounding | Minimum confounder strength |
| Rosenbaum | Hidden bias | Gamma parameter |
| Specification curve | Model dependence | CV across specs |
| Placebo | Spurious findings | Real vs placebo ratio |
| Subgroup | Heterogeneity | CV across subgroups |
| Bootstrap | SE accuracy | SE ratio |
