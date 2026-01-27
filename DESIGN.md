# Design Decisions

This document explains why the system is built the way it is. Each section covers a design choice, the alternatives considered, and why we chose this approach.

---

## Overview

Building an autonomous causal inference system requires solving several hard problems:

1. How do agents decide what to do next?
2. How do agents access information without token explosion?
3. How do we ensure causal validity, not just statistical significance?
4. How do we validate results before presenting them?
5. How do we make the system observable and debuggable?

This document explains our solutions to each.

---

## Decision 1: ReAct Pattern Over Chain-of-Thought

### The Problem

LLM agents need to reason about complex tasks. Two main approaches exist:

**Chain-of-Thought (CoT):** Plan everything upfront, then execute.
```
1. Think through all steps
2. Execute the plan
3. Return results
```

**ReAct:** Interleave reasoning and acting in a loop.
```
1. Observe current state
2. Reason about what to do
3. Act (call a tool)
4. Observe result
5. Repeat until done
```

### Why ReAct?

**Adaptability.** Causal analysis often reveals surprises. The data might have:
- Perfect collinearity (VIF = infinity)
- Severe imbalance (99% treated)
- No valid instruments
- Violated parallel trends

With CoT, the agent plans before seeing these issues. With ReAct, the agent discovers issues and adapts.

**Example:**
```
Step 1: OBSERVE "VIF for education is infinite"
Step 2: REASON "Education is perfectly collinear with another variable.
                I should identify which and consider dropping one."
Step 3: ACT check_correlation("education", all_columns)
Step 4: OBSERVE "Education correlates 1.0 with years_schooling"
Step 5: REASON "These are the same variable. I'll drop one."
```

A CoT agent would have planned to use education as a covariate without knowing it would cause problems.

**Observability.** Each ReAct step is logged. We can see exactly why the agent made each decision. This is critical for debugging and for explaining results to users.

### Trade-offs

**Latency.** ReAct requires multiple LLM calls (one per step). CoT uses fewer calls.

**Mitigation:** We set MAX_STEPS limits and use fast models (Gemini Flash) for routine steps.

**Cost.** More LLM calls means higher API costs.

**Mitigation:** Pull-based context (see below) reduces tokens per call.

### Implementation

See `backend/src/agents/base/react_agent.py` for the ReAct loop implementation.

---

## Decision 2: Pull-Based Context Over Push

### The Problem

Agents need context to make decisions. Two approaches:

**Push (traditional):** Send all context to the LLM upfront.
```python
prompt = f"""
Here is the dataset: {dataset_summary}  # 1000 tokens
Here is the domain knowledge: {domain}   # 500 tokens
Here is the EDA result: {eda}            # 800 tokens
Here is the DAG: {dag}                   # 400 tokens
...
Now decide which method to use.
"""
```

**Pull (our approach):** Agents query for specific information.
```python
# Agent asks only for what it needs
result = query_domain_knowledge("Is there a time dimension?")
# Returns: "Yes, 'year' column spans 2010-2020"

result = get_dag_recommendations("treatment", "outcome")
# Returns: "Adjust for: age, income. WARNING: Do not adjust for hospital (collider)"
```

### Why Pull?

**Token efficiency.** A push approach might send 5000 tokens of context. A pull approach sends 200 tokens per query. With 5 queries, that's 1000 tokens total.

**Focused reasoning.** When the LLM sees everything at once, it may get distracted by irrelevant details. When it pulls specific information, reasoning is more focused.

**Composability.** New agents can be added without changing how context is structured. They just use the same query tools.

**Interpretability.** We can see exactly what information each agent requested. This helps debugging.

### Example Comparison

**Push approach:**
```
Prompt: "Here is everything about the data: [5000 tokens]
         What adjustment set should I use?"

Response: "Based on the data, adjust for age, income, and education."
```

We don't know which part of the 5000 tokens influenced this decision.

**Pull approach:**
```
Step 1: ACT get_dag_recommendations("treatment", "outcome")
        Result: "Confounders: age, income. Collider: satisfaction (DO NOT ADJUST)"

Step 2: REASON "The DAG says adjust for age and income, but NOT satisfaction
                because it's a collider."

Step 3: ACT run_ipw(covariates=["age", "income"])
```

We can trace exactly why satisfaction was excluded.

### Trade-offs

**Multiple calls.** Pull requires multiple tool calls. Each call has latency.

**Mitigation:** Agents can batch related queries. The ReAct loop handles multiple steps naturally.

**Query design.** We must design good query tools. Bad queries give bad results.

**Mitigation:** ContextTools mixin provides tested, well-designed query functions.

### Implementation

See `backend/src/agents/base/context_tools.py` for query tools.

---

## Decision 3: DAG-First Causal Analysis

### The Problem

Traditional causal analysis often skips structure learning:
1. Assume treatment and outcome are defined
2. Pick some covariates
3. Run regression
4. Report results

This ignores the fundamental question: what is the causal structure?

### Why DAG-First?

**Confounders vs Colliders.** The same variable can be helpful or harmful depending on causal structure.

```
Scenario 1: Age causes both Treatment and Outcome
            Age is a CONFOUNDER → MUST adjust

Scenario 2: Treatment and Outcome both cause Hospitalization
            Hospitalization is a COLLIDER → DO NOT adjust
```

Adjusting for a collider can CREATE bias where none existed. Without a DAG, you cannot tell the difference.

**Backdoor Criterion.** With a DAG, we can compute the minimal adjustment set that blocks all backdoor paths. No guessing.

**Transparent Assumptions.** The DAG makes assumptions explicit. Users can see and critique the assumed causal structure.

### Our Approach

1. **Multiple Algorithms:** Run PC, GES, NOTEARS, LiNGAM. Different assumptions, different strengths.

2. **Bootstrap Stability:** Only keep edges that appear in >50% of bootstrap samples. Reduces false positives.

3. **Ensemble Consensus:** Take edges that multiple algorithms agree on. More robust than any single algorithm.

4. **Domain Constraints:** Use temporal ordering (age cannot be caused by treatment) and immutable variables (demographics) to constrain discovery.

5. **Collider Warnings:** Explicitly identify colliders and warn against adjusting for them.

### Example Output

```
DAG Discovery Results:
  Algorithm: PC + Bootstrap (100 samples)

  Confounders (adjust for these):
    - age → treatment (confidence: 0.92)
    - age → outcome (confidence: 0.89)
    - income → treatment (confidence: 0.78)
    - income → outcome (confidence: 0.85)

  Colliders (DO NOT adjust):
    - hospital_admission
      WARNING: Both treatment and outcome cause hospitalization.
      Adjusting would open a confounding path.

  Adjustment Set (backdoor criterion):
    ["age", "income"]
```

### Trade-offs

**Computation.** DAG discovery is expensive. PC on 20 variables with 1000 samples takes seconds. NOTEARS takes longer.

**Mitigation:** We limit variables (top 25 by variance) and use fast implementations.

**Uncertainty.** Discovered DAGs have uncertainty. Edges might be wrong.

**Mitigation:** Bootstrap confidence scores. Collider warnings. Multiple methods comparison.

### Implementation

See `backend/src/causal/dag/discovery.py` for algorithms.

---

## Decision 4: Multiple Methods Comparison

### The Problem

Every causal method has assumptions. If assumptions are violated, estimates are biased. But we rarely know if assumptions hold.

### Why Multiple Methods?

**Robustness Check.** If three different methods with different assumptions give similar answers, we have more confidence.

```
IPW:     ATE = 1543 (95% CI: 930 to 2156)
AIPW:    ATE = 1612 (95% CI: 1021 to 2203)
Matching: ATE = 1489 (95% CI: 876 to 2102)

All three agree: significant positive effect around 1500.
```

**Red Flag Detection.** If methods disagree wildly, something is wrong.

```
IPW:     ATE = 1543
AIPW:    ATE = -892
Matching: ATE = 4521

WARNING: Methods disagree. Check assumptions.
```

**Method Selection Guidance.** Some methods are more appropriate than others:
- DiD needs panel data
- IV needs valid instruments
- RDD needs a cutoff
- AIPW is robust to one model being wrong

By running all applicable methods, we let the data guide selection.

### Our Approach

1. **Automatic Method Selection:** Check data structure. Has time dimension? Try DiD. Has cutoff? Try RDD. Has instruments? Try IV.

2. **Always Run Baseline:** OLS, IPW, AIPW, Matching run on every dataset. These are always applicable (with selection on observables).

3. **Cross-Method Comparison:** Report agreement statistics. Flag large disagreements.

4. **Prefer Doubly Robust:** When all methods available, prefer AIPW (survives one model being wrong).

### Trade-offs

**Computation.** Running 5+ methods takes time.

**Mitigation:** Parallel execution where possible. Cache propensity scores (used by IPW, AIPW, Matching).

**Interpretation.** Multiple estimates can confuse users.

**Mitigation:** Clear reporting. Highlight agreement/disagreement. Explain what each method assumes.

### Implementation

See `backend/src/agents/specialists/effect_estimator.py` for method comparison.

---

## Decision 5: Critique Loop for Quality Control

### The Problem

LLM agents can make mistakes:
- Run wrong method for data type
- Miss assumption violations
- Ignore diagnostics
- Produce invalid results

How do we catch these before presenting to users?

### Why a Critique Agent?

**Separation of Concerns.** Analysis agents focus on doing analysis. Critique agent focuses on checking quality.

**Adversarial Review.** The critique agent looks for problems, not confirmations. Different mindset than analysis.

**Iteration.** If critique finds issues, analysis can be re-run with different parameters.

### What Critique Checks

1. **Assumption Validity**
   - Is positivity satisfied? (check propensity score overlap)
   - Is balance adequate? (check SMD after weighting)
   - Is sample size sufficient?

2. **Method Agreement**
   - Do multiple methods give similar estimates?
   - Same sign? Similar magnitude?

3. **Sensitivity**
   - Is E-value reasonable?
   - Would small unmeasured confounder explain results?

4. **Completeness**
   - Were all relevant methods tried?
   - Were diagnostics checked?

5. **Reproducibility**
   - Can results be reproduced from notebook?
   - Are all parameters recorded?

### Critique Decisions

- **APPROVE:** Analysis is solid. Proceed to notebook generation.
- **ITERATE:** Issues found. Re-run specific agents with feedback.
- **REJECT:** Fundamental problems. Cannot produce valid results.

### Example Iteration

```
Critique: "Propensity score overlap is poor. 15% of treated units have
          PS > 0.95. IPW weights may be unstable."

Action: Re-run effect estimator with:
        - Trimmed weights (drop extreme PS)
        - Matching with caliper instead of IPW

Critique (round 2): "Balance improved. Methods now agree. APPROVE."
```

### Trade-offs

**Added Latency.** Critique adds another agent run.

**Mitigation:** Critique is fast (investigates, doesn't compute). Usually 5-10 steps.

**False Negatives.** Critique might miss issues.

**Mitigation:** Conservative thresholds. Multiple checks. Sensitivity analysis always runs.

### Implementation

See `backend/src/agents/critique/critique_agent.py` for critique logic.

---

## Decision 6: Observability Through Traces

### The Problem

When an autonomous system produces results, users want to know:
- How did it reach this conclusion?
- What decisions did it make?
- Can I trust this?

### Why Traces?

**Debugging.** When results are wrong, traces show where things went wrong.

**Trust.** Users can verify reasoning, not just accept outputs.

**Improvement.** Traces reveal patterns. Common mistakes guide improvements.

### What We Trace

Every agent run produces an AgentTrace:

```python
AgentTrace(
    agent_name="effect_estimator",
    started_at="2024-01-15T10:30:00",
    completed_at="2024-01-15T10:31:23",
    steps=[
        {
            "step": 1,
            "observation": "Dataset has 500 samples, 30% treated",
            "reasoning": "Sample size adequate for IPW and matching",
            "action": "estimate_propensity_scores",
            "result": {"auc": 0.72, "overlap": "good"}
        },
        {
            "step": 2,
            "observation": "PS model AUC=0.72, good overlap",
            "reasoning": "Proceed with IPW estimation",
            "action": "run_ipw",
            "result": {"ate": 1543, "se": 312}
        },
        ...
    ],
    status="completed",
    error=None
)
```

### Trace Access

- **API:** `GET /jobs/{id}/traces` returns all agent traces
- **Frontend:** Traces view shows step-by-step reasoning
- **Storage:** Traces persisted with job results

### Trade-offs

**Storage.** Traces can be large (10+ steps per agent, 8 agents).

**Mitigation:** Store traces separately from results. Compress old traces.

**Privacy.** Traces might contain sensitive reasoning.

**Mitigation:** Traces are job-specific. Access controlled by job ownership.

### Implementation

See `backend/src/agents/base/state.py` for AgentTrace model.

---

## Decision 7: LLM-Based Routing Over Fixed Pipelines

### The Problem

Traditional pipelines have fixed sequences:
```
1. Load data
2. Run EDA
3. Run method X
4. Report results
```

But causal analysis doesn't follow fixed sequences. What if EDA reveals issues? What if the data supports DiD but not matching?

### Why LLM Routing?

**Adaptive Flow.** The orchestrator (an LLM) decides what to do based on current state. Not a fixed sequence.

```
Orchestrator: "EDA shows severe imbalance. Before running IPW,
              I should check propensity score overlap."

Tool call: dispatch_to_agent(
    agent="effect_estimator",
    task="Check propensity score overlap before full estimation"
)
```

**Context-Aware Decisions.** The orchestrator sees all previous results and decides next steps.

**Natural Handling of Edge Cases.** When something unexpected happens, the LLM can reason about it.

### Example Adaptive Flow

```
Normal flow:
  Data Profiler → EDA → Causal Discovery → Effect Estimator → Sensitivity

Actual flow for a specific dataset:
  Data Profiler
    → EDA (finds severe collinearity)
    → Data Profiler (re-profile without collinear variable)
    → EDA (now OK)
    → Causal Discovery (finds no valid instruments)
    → Effect Estimator (skip IV, use IPW/AIPW/Matching only)
    → Sensitivity (E-value is low)
    → Effect Estimator (try matching with caliper for robustness)
    → Critique → APPROVE
    → Notebook Generator
```

No fixed pipeline could handle this.

### Trade-offs

**Unpredictability.** LLM might make unexpected choices.

**Mitigation:** System prompt guides decisions. Critique catches bad choices.

**Latency.** Each routing decision requires LLM call.

**Mitigation:** Orchestrator uses fast model. Most datasets follow standard flow.

### Implementation

See `backend/src/agents/orchestrator/orchestrator_agent.py` for routing logic.

---

## Decision 8: Stratified Bootstrap for Standard Errors

### The Problem

Bootstrap is the standard way to compute standard errors for complex estimators. But naive bootstrap can fail with treatment effects.

**Problem:** If bootstrap sample happens to have 0 treated or 0 control units, the estimate is undefined.

### Our Solution

**Stratified Bootstrap:** Sample separately from treated and control groups, then combine.

```python
for _ in range(n_bootstrap):
    # Sample from treated group
    boot_treated = np.random.choice(treated_idx, size=n_treated, replace=True)

    # Sample from control group
    boot_control = np.random.choice(control_idx, size=n_control, replace=True)

    # Combine
    idx = np.concatenate([boot_treated, boot_control])

    # Compute estimate on bootstrap sample
    boot_estimate = compute_ate(data[idx])
    bootstrap_estimates.append(boot_estimate)

se = np.std(bootstrap_estimates)
```

This guarantees both groups are represented in every bootstrap sample.

### Implementation

See `backend/src/agents/specialists/effect_estimator.py` line 1348-1369.

---

## Decision 9: Defensive Data Handling

### The Problem

Real data has issues:
- Empty DataFrames
- Missing columns
- NaN values in unexpected places
- Perfect collinearity
- Zero variance

These cause cryptic errors if not handled.

### Our Approach

**Early Validation.** Check data structure before computation.

```python
if df is None or len(df) == 0:
    return DataProfile(
        n_samples=0,
        n_features=0,
        ...
    )
```

**Graceful Degradation.** If a computation fails, return sensible default.

```python
if np.isinf(vif) or np.isnan(vif):
    vif_results.append({
        "variable": col,
        "vif": float("inf"),
        "note": "perfect_collinearity"
    })
    warnings.append(f"PERFECT COLLINEARITY: {col}")
```

**NaN-Aware Statistics.** Use `np.nanmean`, `np.nanvar` instead of `np.mean`, `np.var`.

```python
treated_mean = np.nanmean(x[T == 1])
control_mean = np.nanmean(x[T == 0])

if np.isnan(treated_mean) or np.isnan(control_mean):
    continue  # Skip this covariate
```

### Implementation

See defensive checks throughout:
- `data_profiler.py` lines 961-970, 1008-1019
- `eda_agent.py` lines 726-742
- `effect_estimator.py` lines 874-885

---

## Summary: Design Principles

1. **Agents that reason, not scripts that execute.** Use ReAct for adaptability.

2. **Pull context, don't push.** Agents query what they need.

3. **Structure before estimation.** Build DAG first, then estimate.

4. **Multiple methods, not one.** Robustness through comparison.

5. **Critique before output.** Quality control catches mistakes.

6. **Observable by default.** Trace everything for debugging and trust.

7. **Adaptive flow.** LLM routing handles edge cases naturally.

8. **Defensive coding.** Handle data issues gracefully.

These principles combine to create a system that is autonomous yet trustworthy, flexible yet principled.
