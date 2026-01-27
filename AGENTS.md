# Agents Reference

This document describes each agent in the Causal Orchestrator system, their tools, and how they work.

---

## Overview

The system has 8 specialized agents. Each agent has a specific job and a set of tools it can use.

| Agent | Job | Tools |
|-------|-----|-------|
| Domain Knowledge | Extract causal hypotheses from metadata | 10 |
| Data Profiler | Identify treatment, outcome, confounders | 7 |
| EDA Agent | Check data quality and balance | 8 |
| Causal Discovery | Build causal DAG | 6 |
| Effect Estimator | Estimate treatment effects | 7 |
| Sensitivity Analyst | Test robustness | 8 |
| Critique Agent | Review and validate | 8 |
| Notebook Generator | Create reproducible notebook | N/A |

---

## How Agents Think (ReAct Pattern)

Most agents use the ReAct pattern. This means they think in loops:

```
1. OBSERVE: Look at current state and tool results
2. REASON: Think about what to do next
3. ACT: Call a tool
4. REPEAT: Until task is complete
```

This is different from planning everything upfront. If something unexpected happens (like finding collinearity), the agent adapts.

**Example from Data Profiler:**

```
Step 1: OBSERVE "I have a dataset with 15 columns"
Step 2: REASON "I should check domain knowledge for treatment hints"
Step 3: ACT query_domain_knowledge("treatment hints")

Step 4: OBSERVE "Domain says 'program' is likely treatment"
Step 5: REASON "Let me verify this statistically"
Step 6: ACT check_treatment_balance("program")

Step 7: OBSERVE "program is binary, 30% treated, good balance"
Step 8: REASON "This is a valid treatment. Now find outcome."
Step 9: ACT analyze_column("earnings")

... continues until confident ...

Final: ACT finalize_profile(treatment="program", outcome="earnings")
```

---

## Agent Details

### 1. Domain Knowledge Agent

**File:** `backend/src/agents/specialists/domain_knowledge_agent.py`

**Purpose:** Reads Kaggle metadata (NOT the actual data) to understand what variables mean and form causal hypotheses.

**Why This Matters:** A column named "black" might look like a binary treatment statistically. But domain knowledge reveals it is a demographic variable, not a treatment.

**Tools:**

| Tool | What It Does |
|------|--------------|
| `read_description` | Gets title and description from Kaggle metadata |
| `list_columns` | Returns column names |
| `investigate_column` | Analyzes column name for clues (treatment, outcome, demographic) |
| `search_metadata` | Full text search across all metadata |
| `get_tags` | Gets dataset tags, infers domain (healthcare, economics, etc.) |
| `hypothesize` | Records a hypothesis with confidence (low/medium/high) |
| `revise_hypothesis` | Updates a hypothesis when new evidence found |
| `set_temporal_ordering` | Documents what happens before what |
| `mark_immutable` | Flags variables that cannot be caused (age, race) |
| `flag_uncertainty` | Records things the agent is unsure about |

**Output:** `state.domain_knowledge` containing:
- Hypotheses about treatment and outcome
- Temporal ordering (what comes before what)
- Immutable variables (demographics, baseline measures)
- Uncertainties that affect analysis

**Key Behavior:** Only reads metadata. Never touches actual data. This forces the agent to think about meaning, not just statistics.

---

### 2. Data Profiler Agent

**File:** `backend/src/agents/specialists/data_profiler.py`

**Purpose:** Loads the dataset and identifies the causal structure through statistical investigation guided by domain knowledge.

**Tools:**

| Tool | What It Does |
|------|--------------|
| `get_dataset_overview` | Returns shape, columns, types, missing values |
| `analyze_column` | Detailed stats for one column (distribution, skewness, unique values) |
| `check_treatment_balance` | Checks if column is suitable as treatment (ideal: 10-50% minority) |
| `check_column_relationship` | Computes correlation, t-test, chi-square between two columns |
| `check_time_dimension` | Looks for time columns (for DiD analysis) |
| `check_discontinuity_candidates` | Looks for running variables (for RDD analysis) |
| `finalize_profile` | Locks in treatment candidates, outcome candidates, confounders |

**Output:** `state.data_profile` containing:
- Treatment candidates
- Outcome candidates
- Potential confounders
- Time dimension (if found)
- Discontinuity candidates (if found)
- Recommended methods based on data structure

**Key Behavior:** Uses both domain hints AND statistical patterns. If domain says "program" is treatment, the agent verifies this with `check_treatment_balance`. If balance is bad (99% treated), it questions the hypothesis.

---

### 3. EDA Agent

**File:** `backend/src/agents/specialists/eda_agent.py`

**Purpose:** Explores data quality and checks assumptions before running causal analysis.

**Tools:**

| Tool | What It Does |
|------|--------------|
| `get_data_overview` | Dataset dimensions, variable types, treatment/outcome info |
| `analyze_variable` | Distribution stats, skewness, normality tests (Shapiro-Wilk, D'Agostino) |
| `detect_outliers` | IQR and Z-score outlier detection with counts |
| `compute_correlations` | Correlation matrix, flags high correlations (>0.7) |
| `compute_vif` | Variance Inflation Factor for multicollinearity |
| `check_covariate_balance` | Standardized Mean Difference between treatment groups |
| `check_missing_patterns` | Missing data analysis, differential missingness by treatment |
| `finalize_eda` | Produces quality score (0-100), issues, recommendations |

**VIF Interpretation:**
- VIF < 5: OK
- VIF 5-10: Moderate multicollinearity
- VIF > 10: Severe multicollinearity
- VIF = infinity: Perfect collinearity (one variable is linear combo of others)

**Balance Interpretation:**
- SMD < 0.1: Balanced
- SMD 0.1-0.25: Moderate imbalance
- SMD > 0.25: Severe imbalance

**Output:** `state.eda_result` containing:
- Quality score (0-100)
- VIF scores for each covariate
- Covariate balance (SMD for each)
- Outlier counts
- Multicollinearity warnings
- Recommendations

**Key Behavior:** Penalizes quality score for issues:
- Missing data >5%: -5 points
- Missing data >10%: -15 points
- Missing data >30%: -30 points
- High multicollinearity: penalty
- Severe imbalance: penalty

---

### 4. Causal Discovery Agent

**File:** `backend/src/agents/specialists/causal_discovery.py`

**Purpose:** Discovers causal structure (DAG) from data using multiple algorithms.

**Tools:**

| Tool | What It Does |
|------|--------------|
| `get_data_characteristics` | Sample size, distributions, Gaussianity, recommends algorithms |
| `run_discovery_algorithm` | Runs PC, GES, NOTEARS, or LiNGAM |
| `inspect_graph` | Shows discovered DAG structure, edges, treatment-outcome path |
| `validate_graph` | Checks for valid treatment-outcome path, reverse causation, density |
| `compare_algorithms` | Runs multiple algorithms, finds consensus |
| `finalize_discovery` | Chooses algorithm, identifies confounders vs colliders |

**Algorithms:**

| Algorithm | Type | Best For |
|-----------|------|----------|
| PC | Constraint-based | Large number of variables, sparse graphs |
| GES | Score-based | Robust to faithfulness violations |
| NOTEARS | Continuous optimization | Dense graphs, linear relationships |
| LiNGAM | Assumes non-Gaussian | When errors are not Gaussian |

**Output:** `state.proposed_dag` containing:
- Nodes (all variables)
- Edges with confidence scores
- Discovery method used
- Confounders identified
- Colliders identified (with warnings)
- Adjustment set (what to control for)

**Key Behavior:**
1. Runs multiple algorithms
2. Uses bootstrap stability selection (only keeps edges in >50% of bootstrap samples)
3. Identifies confounders (common causes of treatment AND outcome)
4. Identifies colliders (common effects, should NOT adjust for these)
5. Warns if adjusting would open confounding paths

**Collider Warning Example:**
```
COLLIDER WARNING: 'hospital_admission' has multiple causes.
Adjusting for it could open confounding paths.
Do NOT include in adjustment set.
```

---

### 5. Effect Estimator Agent

**File:** `backend/src/agents/specialists/effect_estimator.py`

**Purpose:** Estimates causal treatment effects using multiple methods.

**Tools:**

| Tool | What It Does |
|------|--------------|
| `get_data_summary` | Sample sizes, treatment split, recommends methods |
| `check_covariate_balance` | SMD-based balance check |
| `estimate_propensity_scores` | Fits propensity model, checks overlap |
| `run_estimation_method` | Runs any of the 12 estimation methods |
| `check_method_diagnostics` | Residual analysis, influence diagnostics |
| `compare_estimates` | Cross-method comparison, outlier detection |
| `finalize_estimation` | Chooses best estimate with reasoning |

**Methods Available:**

| Method | Estimand | When To Use |
|--------|----------|-------------|
| OLS | ATE | Baseline, randomized experiments |
| IPW | ATE | Selection bias, full sample |
| AIPW (Doubly Robust) | ATE | Want robustness to model misspecification |
| Matching | ATT | Selection bias, want matched pairs |
| S-Learner | CATE | Heterogeneous effects, single model |
| T-Learner | CATE | Heterogeneous effects, separate models |
| X-Learner | CATE | Heterogeneous effects, small treatment group |
| Causal Forest | CATE | Nonlinear heterogeneity |
| Double ML | ATE | High-dimensional covariates |
| DiD | ATT | Panel data, policy changes |
| IV/2SLS | LATE | Endogeneity, have valid instrument |
| RDD | LATE | Sharp/fuzzy cutoff in assignment |

**Sample Size Requirements:**

| Method | Min Per Arm | Min Total |
|--------|-------------|-----------|
| OLS/IPW | 30 | 50 |
| AIPW | 30 | 50 |
| Matching | 50 | 100 |
| Meta-learners | 100 | 200 |
| Causal Forest | 200 | 500 |

**Output:** `state.treatment_effects` list containing:
- Method name
- Point estimate
- Standard error
- 95% confidence interval
- P-value
- Assumptions tested

**Key Behavior:**
1. Gets DAG recommendations for adjustment set
2. Checks sample size requirements
3. Runs multiple methods for robustness
4. Uses stratified bootstrap (samples from both treatment groups)
5. Prefers AIPW if all methods available

---

### 6. Sensitivity Analyst Agent

**File:** `backend/src/agents/specialists/sensitivity_analyst.py`

**Purpose:** Tests how robust findings are to assumption violations.

**Tools:**

| Tool | What It Does |
|------|--------------|
| `get_estimates_summary` | All treatment effects with cross-method summary |
| `compute_e_value` | E-value for unmeasured confounding |
| `compute_rosenbaum_bounds` | Rosenbaum Gamma for hidden bias sensitivity |
| `run_specification_curve` | Tests multiple covariate specifications |
| `run_placebo_test` | Tests on fake treatment/outcome |
| `run_subgroup_analysis` | Effect heterogeneity across subgroups |
| `check_variance_stability` | Bootstrap SE verification |
| `finalize_sensitivity` | Overall robustness assessment |

**E-Value Interpretation:**
- E-value >= 3: Very robust (would need strong unmeasured confounder to nullify)
- E-value 2-3: Moderately robust
- E-value < 2: Sensitive to unmeasured confounding

**Rosenbaum Gamma Interpretation:**
- Gamma >= 3: Very robust
- Gamma 2-3: Moderately robust
- Gamma < 2: Sensitive

**Output:** `state.sensitivity_results` list containing:
- Method name (E-value, Rosenbaum, etc.)
- Robustness value
- Interpretation
- Details

**Key Behavior:**
1. Computes multiple robustness metrics
2. Runs placebo tests (if real effect <= 2x placebo effect, flags concern)
3. Checks subgroup heterogeneity (if CV > 50%, flags substantial heterogeneity)
4. Verifies bootstrap SEs (flags if >20% different from reported)

---

### 7. Critique Agent

**File:** `backend/src/agents/critique/critique_agent.py`

**Purpose:** Quality control. Reviews all outputs before finalization.

**Tools:**

| Tool | What It Does |
|------|--------------|
| `get_analysis_summary` | Overview of all results |
| `check_covariate_balance` | Actual SMD computation |
| `check_estimate_consistency` | Cross-method agreement check |
| `check_subgroup_effects` | Effect heterogeneity |
| `check_influential_observations` | Outlier influence on estimates |
| `verify_propensity_scores` | PS diagnostics check |
| `check_sensitivity_robustness` | Reviews sensitivity findings |
| `finalize_critique` | Scores analysis, decides APPROVE/ITERATE/REJECT |

**Scoring Dimensions (1-5 each):**
- Statistical validity
- Assumption checking
- Method selection
- Completeness
- Reproducibility
- Interpretation

**Decisions:**
- **APPROVE**: Analysis is solid, proceed to notebook
- **ITERATE**: Issues found, re-run specific agents with feedback
- **REJECT**: Fundamental problems, cannot produce valid results

**Output:** `state.critique_history` appended with:
- Decision (APPROVE/ITERATE/REJECT)
- Scores for each dimension
- Specific findings
- Recommendations if ITERATE

**Key Behavior:**
1. Investigates using tools (does not just read summaries)
2. Checks actual balance, not just reported balance
3. Verifies estimates are consistent across methods
4. Can request up to 3 iterations before final decision
5. If ITERATE, specifies which agent should re-run with what changes

---

### 8. Notebook Generator Agent

**File:** `backend/src/agents/specialists/notebook_generator.py`

**Purpose:** Creates a reproducible Jupyter notebook with all analysis steps.

**Note:** This agent does NOT use ReAct. It directly generates notebook cells.

**Notebook Sections:**
1. Introduction (title, dataset info, analysis overview)
2. Setup (library imports)
3. Data Loading (load data, preview, dtypes)
4. EDA (distributions, correlations, balance, VIF)
5. Causal Structure (DAG visualization with NetworkX)
6. Treatment Effects (results table, method implementations)
7. Sensitivity Analysis (robustness metrics)
8. Conclusions (findings, limitations, recommendations)

**Output:** `state.notebook_path` pointing to generated .ipynb file

**Key Behavior:**
- Generates fully executable code
- Includes markdown explanations
- Uses actual column names from the analysis
- Handles optional libraries (causallearn, econml)

---

## How Agents Share Information

All agents read and write to a shared `AnalysisState` object. Here is what each agent contributes:

```
┌─────────────────────┐
│   AnalysisState     │
├─────────────────────┤
│                     │
│  domain_knowledge ◄─────── Domain Knowledge Agent
│  (hypotheses,       │
│   temporal order)   │
│                     │
│  data_profile ◄─────────── Data Profiler Agent
│  (candidates,       │
│   confounders)      │
│                     │
│  dataframe_path ◄───────── Data Profiler Agent
│  (pickled data)     │
│                     │
│  eda_result ◄───────────── EDA Agent
│  (quality, VIF,     │
│   balance)          │
│                     │
│  proposed_dag ◄─────────── Causal Discovery Agent
│  (nodes, edges,     │
│   adjustment set)   │
│                     │
│  treatment_effects ◄────── Effect Estimator Agent
│  (estimates, SE,    │
│   p-values)         │
│                     │
│  sensitivity_results ◄──── Sensitivity Analyst Agent
│  (E-value, Gamma,   │
│   robustness)       │
│                     │
│  critique_history ◄─────── Critique Agent
│  (scores, decision) │
│                     │
│  notebook_path ◄────────── Notebook Generator
│  (output file)      │
│                     │
└─────────────────────┘
```

---

## Pull-Based Context

Agents do not receive all context upfront. They pull what they need using context tools:

```python
# Instead of receiving 5000 tokens of context upfront...

# Agent pulls specific information:
result = query_domain_knowledge("Is there a time dimension?")
# Returns: "Dataset has 'year' column spanning 2010-2020"

result = get_dag_recommendations("program", "earnings")
# Returns: "Adjust for: age, education, married.
#           WARNING: Do not adjust for job_satisfaction (collider)"
```

**Why This Matters:**
1. Reduces token usage
2. Forces agents to think about what they need
3. Improves reasoning quality
4. Makes agent behavior more interpretable

---

## Error Handling

Each agent handles errors gracefully:

1. **Max Steps Limit:** If agent runs too many steps, it auto-finalizes with partial results
2. **LLM API Error:** Automatic retry with exponential backoff (3 retries)
3. **Data Issues:** Empty DataFrame, missing columns handled with clear error messages
4. **Tool Errors:** Caught and logged, agent can try alternative tools

---

## Adding a New Agent

To add a new agent:

1. Create file in `backend/src/agents/specialists/`
2. Inherit from `ReActAgent` and `ContextTools`
3. Register tools in `__init__`
4. Implement `execute()` method
5. Add to orchestrator's available agents
6. Add status to `JobStatus` enum if needed

Example skeleton:

```python
class MyNewAgent(ReActAgent, ContextTools):
    def __init__(self):
        super().__init__(name="my_new_agent")

        self.register_tool(
            name="my_tool",
            description="Does something useful",
            parameters={
                "type": "object",
                "properties": {
                    "column": {"type": "string"}
                },
                "required": ["column"]
            },
            handler=self._tool_my_tool
        )

    async def _tool_my_tool(self, column: str) -> str:
        # Implementation
        return "Result"

    async def execute(self, state: AnalysisState) -> AnalysisState:
        # ReAct loop
        ...
        return state
```
