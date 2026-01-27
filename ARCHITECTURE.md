# Architecture

This document describes the complete architecture of the Causal Orchestrator system.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                   FRONTEND                                       │
│                              React + TypeScript                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │  HomePage    │  │  JobPage     │  │  JobsList    │  │  ResultsDisplay      │ │
│  │  URL Input   │  │  Progress    │  │  History     │  │  Effects + DAG       │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────────────┘ │
│                                        │                                         │
│                            ┌───────────┴───────────┐                            │
│                            │  TanStack React Query │                            │
│                            │  Polling + Caching    │                            │
│                            └───────────┬───────────┘                            │
└────────────────────────────────────────┼────────────────────────────────────────┘
                                         │ HTTP/REST
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                   BACKEND                                        │
│                              FastAPI + Python                                    │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                              API Layer                                    │   │
│  │  POST /jobs          GET /jobs/{id}        GET /jobs/{id}/results        │   │
│  │  GET /jobs           GET /jobs/{id}/traces GET /jobs/{id}/notebook       │   │
│  │  DELETE /jobs/{id}   POST /jobs/{id}/cancel                              │   │
│  └────────────────────────────────────┬─────────────────────────────────────┘   │
│                                       │                                          │
│  ┌────────────────────────────────────▼─────────────────────────────────────┐   │
│  │                            Job Manager                                    │   │
│  │  - Creates AnalysisState                                                  │   │
│  │  - Spawns async task                                                      │   │
│  │  - Manages job lifecycle (cancel, timeout)                                │   │
│  │  - Thread-safe with asyncio.Lock                                          │   │
│  └────────────────────────────────────┬─────────────────────────────────────┘   │
│                                       │                                          │
│  ┌────────────────────────────────────▼─────────────────────────────────────┐   │
│  │                          Orchestrator Agent                               │   │
│  │                       (Brain of the System)                               │   │
│  │                                                                           │   │
│  │  Tools:                                                                   │   │
│  │  - dispatch_to_agent(agent_name, task, reasoning)                         │   │
│  │  - request_critique(focus_areas)                                          │   │
│  │  - finalize_analysis(summary)                                             │   │
│  │                                                                           │   │
│  │  Reasoning: LLM decides which agent to call next                          │   │
│  │  No hardcoded if-else logic for routing                                   │   │
│  └────────────────────────────────────┬─────────────────────────────────────┘   │
│                                       │                                          │
│           ┌───────────────────────────┼───────────────────────────┐              │
│           │                           │                           │              │
│           ▼                           ▼                           ▼              │
│  ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐        │
│  │    Specialist   │       │    Specialist   │       │    Specialist   │        │
│  │     Agents      │       │     Agents      │       │     Agents      │        │
│  │  (see below)    │       │  (see below)    │       │  (see below)    │        │
│  └─────────────────┘       └─────────────────┘       └─────────────────┘        │
│                                       │                                          │
│  ┌────────────────────────────────────▼─────────────────────────────────────┐   │
│  │                           Critique Agent                                  │   │
│  │  Reviews all outputs. Returns: APPROVE, ITERATE, or REJECT               │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                              Storage Layer                                │   │
│  │  Firestore (production) or Local JSON (development)                       │   │
│  │  - Jobs, Results, Traces                                                  │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## The AnalysisState Object

All agents share a single state object that flows through the pipeline. This is the source of truth.

```python
class AnalysisState:
    # Identity
    job_id: str
    dataset_info: DatasetInfo
    status: JobStatus

    # User inputs (optional)
    treatment_variable: str | None
    outcome_variable: str | None

    # Populated by Domain Knowledge Agent
    raw_metadata: dict                    # Kaggle metadata
    domain_knowledge: dict                # Hypotheses, temporal ordering, immutable vars

    # Populated by Data Profiler
    data_profile: DataProfile             # Columns, types, candidates
    dataframe_path: str                   # Path to pickled DataFrame

    # Populated by EDA Agent
    eda_result: EDAResult                 # VIF, balance, outliers, correlations

    # Populated by Causal Discovery
    proposed_dag: CausalDAG               # Nodes, edges, discovery method

    # Populated by Effect Estimator
    treatment_effects: list[TreatmentEffectResult]
    analyzed_pairs: list[CausalPair]

    # Populated by Sensitivity Analyst
    sensitivity_results: list[SensitivityResult]

    # Critique tracking
    critique_history: list[CritiqueFeedback]
    iteration_count: int
    max_iterations: int = 3

    # Observability
    agent_traces: list[AgentTrace]
```

---

## Job Status Flow

```
PENDING
    │
    ▼
FETCHING_DATA ──────► Downloading from Kaggle
    │
    ▼
PROFILING ──────────► Data Profiler running
    │
    ▼
EXPLORATORY_ANALYSIS ► EDA Agent running
    │
    ▼
DISCOVERING_CAUSAL ──► Causal Discovery Agent running
    │
    ▼
ESTIMATING_EFFECTS ──► Effect Estimator Agent running
    │
    ▼
SENSITIVITY_ANALYSIS ► Sensitivity Analyst running
    │
    ▼
CRITIQUE_REVIEW ────► Critique Agent reviewing
    │
    ├──► ITERATING ──► Back to relevant agent
    │
    ▼
GENERATING_NOTEBOOK ─► Notebook Generator running
    │
    ▼
COMPLETED

At any point:
    ├──► FAILED (error)
    ├──► CANCELLING (user requested)
    └──► CANCELLED (cancelled)
```

---

## Complete Job Flow (Step by Step)

### Step 1: Job Creation

```
User clicks "Analyze" with Kaggle URL
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Frontend: POST /api/v1/jobs                                    │
│  Body: { kaggle_url, treatment_variable?, outcome_variable? }   │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  API Route: create_job()                                        │
│  - Validates request                                            │
│  - Calls JobManager.create_job()                                │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  JobManager.create_job()                                        │
│  1. Generate job_id (UUID)                                      │
│  2. Create AnalysisState object                                 │
│  3. Save to storage (Firestore/local)                           │
│  4. Spawn async task: _run_job(state)                           │
│  5. Store task in _running_jobs dict (protected by lock)        │
│  6. Return job_id                                               │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  Frontend receives job_id                                       │
│  - Redirects to /jobs/{job_id}                                  │
│  - Starts polling for status updates                            │
└─────────────────────────────────────────────────────────────────┘
```

### Step 2: Orchestrator Takes Over

```
┌─────────────────────────────────────────────────────────────────┐
│  JobManager._run_job()                                          │
│  - Sets status = FETCHING_DATA                                  │
│  - Calls orchestrator.execute_with_tracing(state)               │
│  - Wraps with timeout (default: 30 minutes)                     │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  OrchestratorAgent.execute()                                    │
│                                                                 │
│  while not complete:                                            │
│      1. Build prompt with current state                         │
│      2. Call LLM with tools available:                          │
│         - dispatch_to_agent                                     │
│         - request_critique                                      │
│         - finalize_analysis                                     │
│      3. Parse LLM response                                      │
│      4. Execute tool call                                       │
│      5. Update state                                            │
│      6. Loop                                                    │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
                    LLM Decides What To Do
```

### Step 3: Domain Knowledge (if metadata available)

```
┌─────────────────────────────────────────────────────────────────┐
│  Orchestrator LLM Reasoning:                                    │
│  "Metadata is available. I should first extract domain          │
│   knowledge to help downstream agents."                         │
│                                                                 │
│  Tool Call: dispatch_to_agent(                                  │
│      agent_name="domain_knowledge",                             │
│      task="Extract causal hypotheses from Kaggle metadata",     │
│      reasoning="..."                                            │
│  )                                                              │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  DomainKnowledgeAgent.execute()                                 │
│                                                                 │
│  Reads: state.raw_metadata (from Kaggle)                        │
│                                                                 │
│  Extracts:                                                      │
│  - Treatment/outcome hypotheses from column descriptions        │
│  - Temporal ordering (what happens before what)                 │
│  - Immutable variables (demographics, etc.)                     │
│  - Domain category (healthcare, economics, etc.)                │
│                                                                 │
│  Writes: state.domain_knowledge = {                             │
│      "hypotheses": [...],                                       │
│      "temporal_ordering": {...},                                │
│      "immutable_variables": [...],                              │
│      "domain": "healthcare"                                     │
│  }                                                              │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
                    Returns to Orchestrator
```

### Step 4: Data Profiling

```
┌─────────────────────────────────────────────────────────────────┐
│  Orchestrator LLM Reasoning:                                    │
│  "Domain knowledge extracted. Now I need to understand the      │
│   dataset structure and identify causal variables."             │
│                                                                 │
│  Tool Call: dispatch_to_agent(                                  │
│      agent_name="data_profiler",                                │
│      task="Profile dataset and identify causal structure",      │
│      reasoning="..."                                            │
│  )                                                              │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  DataProfilerAgent.execute() [ReAct Loop]                       │
│                                                                 │
│  Step 1: OBSERVE initial state                                  │
│  Step 2: REASON "I should first query domain knowledge"         │
│  Step 3: ACT query_domain_knowledge("treatment hints")          │
│                                                                 │
│  Step 4: OBSERVE domain says "treatment likely 'program'"       │
│  Step 5: REASON "Let me verify this statistically"              │
│  Step 6: ACT get_dataset_overview()                             │
│                                                                 │
│  Step 7: OBSERVE overview shows 'program' is binary, 30% = 1    │
│  Step 8: REASON "Good balance. Let me check treatment balance"  │
│  Step 9: ACT check_treatment_balance("program")                 │
│                                                                 │
│  ... continues until confident ...                              │
│                                                                 │
│  Final: ACT finalize_profile(                                   │
│      treatment_candidates=["program"],                          │
│      outcome_candidates=["earnings"],                           │
│      confounders=["age", "education", "married"]                │
│  )                                                              │
│                                                                 │
│  Writes: state.data_profile = DataProfile(...)                  │
│  Writes: state.dataframe_path = "/tmp/causal_orchestrator/..."  │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
                    Returns to Orchestrator
```

### Step 5: Exploratory Data Analysis

```
┌─────────────────────────────────────────────────────────────────┐
│  Orchestrator LLM Reasoning:                                    │
│  "Dataset profiled. Before estimation, I need to check          │
│   data quality, balance, and potential issues."                 │
│                                                                 │
│  Tool Call: dispatch_to_agent(                                  │
│      agent_name="eda_agent",                                    │
│      task="Check covariate balance, multicollinearity, outliers"│
│      reasoning="..."                                            │
│  )                                                              │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  EDAAgent.execute() [ReAct Loop]                                │
│                                                                 │
│  Tools called:                                                  │
│  - check_balance(covariates) ─► SMD for each covariate          │
│  - check_multicollinearity() ─► VIF scores                      │
│  - detect_outliers() ─────────► IQR/Z-score outliers            │
│  - analyze_distributions() ───► Skewness, normality             │
│  - check_overlap() ───────────► Propensity score overlap        │
│                                                                 │
│  Key outputs:                                                   │
│  - vif_scores: {"age": 1.2, "education": 8.5, ...}             │
│  - covariate_balance: {"age": {"smd": 0.15, "status": "OK"}}   │
│  - multicollinearity_warnings: ["education VIF > 5"]           │
│                                                                 │
│  Writes: state.eda_result = EDAResult(...)                      │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
                    Returns to Orchestrator
```

### Step 6: Causal Discovery

```
┌─────────────────────────────────────────────────────────────────┐
│  Orchestrator LLM Reasoning:                                    │
│  "EDA shows moderate imbalance but no severe issues.            │
│   Now I need to discover the causal structure."                 │
│                                                                 │
│  Tool Call: dispatch_to_agent(                                  │
│      agent_name="causal_discovery",                             │
│      task="Learn causal DAG, identify confounders vs colliders" │
│      reasoning="..."                                            │
│  )                                                              │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  CausalDiscoveryAgent.execute()                                 │
│                                                                 │
│  1. Get domain constraints (temporal ordering, immutable vars)  │
│  2. Run discovery algorithms:                                   │
│     - PC algorithm (constraint-based)                           │
│     - GES (score-based)                                         │
│     - Bootstrap stability selection                             │
│                                                                 │
│  3. Build consensus DAG from multiple algorithms                │
│                                                                 │
│  4. Identify:                                                   │
│     - Confounders (common causes of T and Y)                    │
│     - Colliders (common effects of T and Y) ─► WARNING          │
│     - Mediators (on causal path)                                │
│     - Adjustment set (using backdoor criterion)                 │
│                                                                 │
│  Writes: state.proposed_dag = CausalDAG(                        │
│      nodes=["program", "earnings", "age", "education", ...],    │
│      edges=[                                                    │
│          CausalEdge("age", "program", confidence=0.85),         │
│          CausalEdge("age", "earnings", confidence=0.92),        │
│          CausalEdge("program", "earnings", confidence=0.78),    │
│      ],                                                         │
│      discovery_method="PC + Bootstrap"                          │
│  )                                                              │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
                    Returns to Orchestrator
```

### Step 7: Effect Estimation

```
┌─────────────────────────────────────────────────────────────────┐
│  Orchestrator LLM Reasoning:                                    │
│  "DAG identified. Confounders are age, education, married.      │
│   No colliders. Time to estimate treatment effects."            │
│                                                                 │
│  Tool Call: dispatch_to_agent(                                  │
│      agent_name="effect_estimator",                             │
│      task="Estimate ATE using IPW, AIPW, and Matching",         │
│      reasoning="Multiple methods for robustness"                │
│  )                                                              │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  EffectEstimatorAgent.execute() [ReAct Loop]                    │
│                                                                 │
│  Step 1: ACT get_dag_recommendations()                          │
│          ─► "Adjust for: age, education, married"               │
│                                                                 │
│  Step 2: ACT estimate_propensity_scores(covariates)             │
│          ─► Fits LogisticRegression, returns PS model           │
│                                                                 │
│  Step 3: ACT check_covariate_balance(after_weighting=True)      │
│          ─► Verifies balance improved                           │
│                                                                 │
│  Step 4: ACT run_ipw(treatment, outcome, covariates)            │
│          ─► IPW estimate with stratified bootstrap SE           │
│                                                                 │
│  Step 5: ACT run_aipw(treatment, outcome, covariates)           │
│          ─► Doubly robust estimate                              │
│                                                                 │
│  Step 6: ACT run_matching(treatment, outcome, covariates)       │
│          ─► Propensity score matching estimate                  │
│                                                                 │
│  Writes: state.treatment_effects = [                            │
│      TreatmentEffectResult(                                     │
│          method="IPW",                                          │
│          estimate=1543.21,                                      │
│          std_error=312.45,                                      │
│          ci_lower=930.81,                                       │
│          ci_upper=2155.61,                                      │
│          p_value=0.001                                          │
│      ),                                                         │
│      TreatmentEffectResult(method="AIPW", ...),                 │
│      TreatmentEffectResult(method="Matching", ...),             │
│  ]                                                              │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
                    Returns to Orchestrator
```

### Step 8: Sensitivity Analysis

```
┌─────────────────────────────────────────────────────────────────┐
│  Orchestrator LLM Reasoning:                                    │
│  "Effects estimated. All methods show significant positive      │
│   effect. Now I should check robustness to unobserved           │
│   confounding."                                                 │
│                                                                 │
│  Tool Call: dispatch_to_agent(                                  │
│      agent_name="sensitivity_analyst",                          │
│      task="Run Rosenbaum bounds and E-value analysis",          │
│      reasoning="..."                                            │
│  )                                                              │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  SensitivityAnalystAgent.execute()                              │
│                                                                 │
│  Tools called:                                                  │
│  - run_rosenbaum_bounds() ─► Gamma = 1.8 (robust to moderate)   │
│  - compute_e_value() ──────► E = 2.3 (strong unmeasured needed) │
│  - run_placebo_test() ────► No effect on placebo outcome        │
│                                                                 │
│  Writes: state.sensitivity_results = [                          │
│      SensitivityResult(                                         │
│          method="Rosenbaum Bounds",                             │
│          robustness_value=1.8,                                  │
│          interpretation="Effect robust to moderate confounding" │
│      ),                                                         │
│      SensitivityResult(method="E-value", ...),                  │
│  ]                                                              │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
                    Returns to Orchestrator
```

### Step 9: Critique Review

```
┌─────────────────────────────────────────────────────────────────┐
│  Orchestrator LLM Reasoning:                                    │
│  "Analysis complete. Before finalizing, I should request        │
│   a critique review to ensure quality."                         │
│                                                                 │
│  Tool Call: request_critique(                                   │
│      focus_areas=["assumption validity", "method agreement"]    │
│  )                                                              │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  CritiqueAgent.execute()                                        │
│                                                                 │
│  Reviews:                                                       │
│  - Are positivity assumptions met? (check PS overlap)           │
│  - Do methods agree? (IPW, AIPW, Matching similar?)            │
│  - Is sample size adequate?                                     │
│  - Are there red flags in EDA?                                  │
│  - Is sensitivity analysis reassuring?                          │
│                                                                 │
│  Scoring (1-5):                                                 │
│  - assumption_validity: 4                                       │
│  - method_agreement: 5                                          │
│  - robustness: 4                                                │
│  - interpretability: 4                                          │
│                                                                 │
│  Decision: APPROVE (or ITERATE with feedback)                   │
│                                                                 │
│  Writes: state.critique_history.append(CritiqueFeedback(...))   │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
                    Returns to Orchestrator
```

### Step 9b: Iteration (if ITERATE)

```
┌─────────────────────────────────────────────────────────────────┐
│  If Critique returns ITERATE:                                   │
│                                                                 │
│  Orchestrator reads feedback:                                   │
│  "Positivity assumption may be violated for older age groups.   │
│   Consider trimming extreme propensity scores or using          │
│   matching with caliper."                                       │
│                                                                 │
│  Orchestrator LLM Reasoning:                                    │
│  "Critique raised positivity concerns. I should re-run          │
│   effect estimation with trimmed weights."                      │
│                                                                 │
│  Tool Call: dispatch_to_agent(                                  │
│      agent_name="effect_estimator",                             │
│      task="Re-estimate with trimmed propensity scores",         │
│      reasoning="Address positivity violation"                   │
│  )                                                              │
│                                                                 │
│  state.iteration_count += 1                                     │
│  (Max 3 iterations, then finalize with best effort)             │
└─────────────────────────────────────────────────────────────────┘
```

### Step 10: Notebook Generation

```
┌─────────────────────────────────────────────────────────────────┐
│  Orchestrator LLM Reasoning:                                    │
│  "Critique approved. Time to generate the reproducible          │
│   notebook."                                                    │
│                                                                 │
│  Tool Call: dispatch_to_agent(                                  │
│      agent_name="notebook_generator",                           │
│      task="Generate Jupyter notebook with all analysis steps",  │
│      reasoning="..."                                            │
│  )                                                              │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  NotebookGeneratorAgent.execute()                               │
│                                                                 │
│  Creates notebook with sections:                                │
│  1. Data Loading                                                │
│  2. Exploratory Analysis                                        │
│  3. Causal Graph Visualization                                  │
│  4. Treatment Effect Estimation                                 │
│  5. Sensitivity Analysis                                        │
│  6. Conclusions                                                 │
│                                                                 │
│  Writes: state.notebook_path = "/tmp/.../analysis.ipynb"        │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
                    Returns to Orchestrator
```

### Step 11: Finalization

```
┌─────────────────────────────────────────────────────────────────┐
│  Orchestrator LLM Reasoning:                                    │
│  "All steps complete. Notebook generated. Finalizing."          │
│                                                                 │
│  Tool Call: finalize_analysis(                                  │
│      summary="Analysis complete. Program participation shows    │
│               significant positive effect on earnings..."       │
│  )                                                              │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  OrchestratorAgent sets state.status = COMPLETED                │
│  Returns final state to JobManager                              │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  JobManager._run_job()                                          │
│  - Saves results to storage                                     │
│  - Saves traces to storage                                      │
│  - Updates job status in storage                                │
│  - Removes from _running_jobs dict                              │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  Frontend polling detects status = COMPLETED                    │
│  - Fetches results: GET /jobs/{id}/results                      │
│  - Displays treatment effects, DAG, sensitivity                 │
│  - Shows download button for notebook                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Pull-Based Context Architecture

Instead of dumping all context to agents, agents PULL what they need.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ContextTools Mixin                            │
│                                                                  │
│  Available to all ReAct agents via inheritance                   │
│                                                                  │
│  Query Tools (agents PULL context):                              │
│  ├── query_domain_knowledge(question)                            │
│  │   └── Returns: hypothesis, confidence, reasoning              │
│  ├── get_column_info(column)                                     │
│  │   └── Returns: dtype, stats, missing%, distribution           │
│  ├── get_variable_summary(variables)                             │
│  │   └── Returns: bulk info for multiple variables               │
│  ├── get_dag_recommendations(treatment, outcome)                 │
│  │   └── Returns: confounders, colliders, adjustment set         │
│  └── check_data_quality()                                        │
│      └── Returns: issues, warnings, quality score                │
└─────────────────────────────────────────────────────────────────┘

Example Flow:
─────────────
1. EffectEstimator starts
2. Calls: get_dag_recommendations("program", "earnings")
3. ContextTools reads state.proposed_dag
4. Returns: "Adjust for age, education, married.
            WARNING: Do not adjust for job_satisfaction (collider)"
5. EffectEstimator uses this to configure estimation
```

---

## Error Handling and Recovery

```
┌─────────────────────────────────────────────────────────────────┐
│                     Error Scenarios                              │
│                                                                  │
│  1. Agent Timeout                                                │
│     - Each agent has MAX_STEPS limit                             │
│     - If exceeded, agent auto-finalizes with partial results     │
│     - Orchestrator continues with available data                 │
│                                                                  │
│  2. Job Timeout                                                  │
│     - Total job has 10x agent timeout                            │
│     - If exceeded, job marked FAILED                             │
│     - Partial traces saved for debugging                         │
│                                                                  │
│  3. Agent Failure                                                │
│     - Agent catches exception, logs, marks state.error           │
│     - Orchestrator sees error, decides: retry or fail            │
│     - After 3 agent failures, job fails                          │
│                                                                  │
│  4. LLM API Error                                                │
│     - Automatic retry with exponential backoff                   │
│     - After 3 retries, agent fails                               │
│                                                                  │
│  5. Data Issues                                                  │
│     - Empty DataFrame: Early failure with clear message          │
│     - Missing columns: Agent adapts or requests alternatives     │
│     - Invalid Kaggle URL: Clear error in job creation            │
│                                                                  │
│  6. User Cancellation                                            │
│     - Cancel endpoint sets CANCELLING status                     │
│     - Async task receives CancelledError                         │
│     - Graceful shutdown with partial trace save                  │
│     - Protected by asyncio.Lock to prevent race conditions       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Summary

```
Kaggle URL
    │
    ▼
┌───────────────┐
│ Kaggle API    │ Download dataset + metadata
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ DataFrame     │ Pickled to /tmp/causal_orchestrator/{job_id}_data.pkl
└───────┬───────┘
        │
        ├─────────────────────────────────────────────────────────────┐
        │                                                             │
        ▼                                                             ▼
┌───────────────┐                                           ┌─────────────────┐
│ DataProfile   │ In state.data_profile                     │ raw_metadata    │
│ - n_samples   │                                           │ - description   │
│ - n_features  │                                           │ - column_descs  │
│ - types       │                                           │ - tags          │
│ - candidates  │                                           └────────┬────────┘
└───────┬───────┘                                                    │
        │                                                             │
        │                                                             ▼
        │                                                   ┌─────────────────┐
        │                                                   │ domain_knowledge│
        │                                                   │ - hypotheses    │
        │                                                   │ - temporal_order│
        │                                                   │ - immutable_vars│
        │                                                   └─────────────────┘
        │
        ▼
┌───────────────┐
│ EDAResult     │ In state.eda_result
│ - vif_scores  │
│ - balance     │
│ - outliers    │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ CausalDAG     │ In state.proposed_dag
│ - nodes       │
│ - edges       │
│ - confidence  │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ TreatmentEffects │ In state.treatment_effects
│ - estimates      │
│ - std_errors     │
│ - confidence     │
└───────┬──────────┘
        │
        ▼
┌───────────────┐
│ SensitivityResults │ In state.sensitivity_results
│ - robustness       │
│ - e_values         │
└───────┬────────────┘
        │
        ▼
┌───────────────┐
│ Notebook      │ Generated .ipynb file
└───────────────┘
```

---

## Key Files Reference

| Component | File | Description |
|-----------|------|-------------|
| State definitions | `backend/src/agents/base/state.py` | All Pydantic models |
| ReAct base class | `backend/src/agents/base/react_agent.py` | Observe-Reason-Act loop |
| Context tools | `backend/src/agents/base/context_tools.py` | Pull-based context queries |
| Orchestrator | `backend/src/agents/orchestrator/orchestrator_agent.py` | Central coordinator |
| Job manager | `backend/src/jobs/manager.py` | Job lifecycle management |
| DAG discovery | `backend/src/causal/dag/discovery.py` | PC, FCI, GES, NOTEARS, LiNGAM |
| API routes | `backend/src/api/routes/jobs.py` | REST endpoints |
| Frontend API | `frontend/src/services/api.ts` | Axios client |
| Results display | `frontend/src/components/results/ResultsDisplay.tsx` | UI for results |
