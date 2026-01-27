# Causal Orchestrator

An autonomous multi-agent system for causal inference analysis. Give it a Kaggle dataset URL and it figures out the causal structure, selects appropriate methods, estimates treatment effects, validates results, and generates a reproducible Jupyter notebook.

## What Makes This Different

**Agents that reason, not scripts that execute.** Traditional AutoML tools follow fixed pipelines. This system uses LLM-powered agents that observe data, reason about what to do, and act accordingly. When the data doesn't match expectations, agents adapt.

**Pull-based context, not context dumps.** Most agent systems push all context to the LLM upfront. Our agents actively query for the information they need through specialized tools. This reduces token usage and improves reasoning quality.

**DAG-first causal analysis.** Before estimating effects, the system discovers causal structure using multiple algorithms (PC, FCI, GES, NOTEARS, LiNGAM). It identifies confounders vs colliders and warns against invalid conditioning.

**For the rationale behind these design choices, see [DESIGN.md](DESIGN.md)**

## Key Features

- **ReAct Pattern**: Observe, Reason, Act loops based on the [original paper](https://arxiv.org/abs/2210.03629)
- **12 Causal Methods**: OLS, PSM, IPW, AIPW, DiD, IV/2SLS, RDD, Meta-learners, Causal Forest, Double ML
- **6 DAG Discovery Algorithms**: PC, FCI, GES, NOTEARS, LiNGAM, plus Ensemble consensus
- **10+ Specialized Agents**: Each with domain-specific tools and expertise
- **Critique Loop**: Quality control agent validates all outputs before finalization
- **Multiple LLM Support**: Gemini, Claude, Vertex AI
- **Full Observability**: Agent traces show reasoning at every step

## Tech Stack

**Backend**
- Python 3.11+
- FastAPI with async job management
- DoWhy, EconML, CausalML for causal inference
- NetworkX, pgmpy for graph algorithms
- Firestore for persistence

**Frontend**
- React 18 with TypeScript
- TanStack React Query for data fetching
- Zustand for state management
- Tailwind CSS

## Quick Start

```bash
# Clone and setup
cd causal-orchestrator
cp .env.example .env
# Add your GEMINI_API_KEY and KAGGLE credentials to .env

# Backend
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn src.api.main:app --reload

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

Open http://localhost:5173, paste a Kaggle dataset URL, and watch the agents work.

---

## Architecture

**For the complete architecture with step-by-step flow, see [ARCHITECTURE.md](ARCHITECTURE.md)**

[Architecture diagram placeholder - add image here]

The system has three layers:

1. **Frontend**: React app for submitting jobs and viewing results
2. **API Layer**: FastAPI endpoints for job management
3. **Agent Layer**: Autonomous agents that do the actual analysis

### How a Job Flows Through the System

```
User submits Kaggle URL
        │
        ▼
┌───────────────────┐
│   Job Manager     │  Creates job, starts async task
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│   Orchestrator    │  ReAct agent that coordinates everything
└─────────┬─────────┘
          │
          ├─────────────────────────────────────────┐
          │                                         │
          ▼                                         ▼
┌───────────────────┐                    ┌───────────────────┐
│  Data Profiler    │                    │  Domain Knowledge │
│  - Load dataset   │                    │  - Kaggle metadata│
│  - Find treatment │                    │  - Variable hints │
│  - Find outcome   │                    │                   │
│  - ID confounders │                    │                   │
└─────────┬─────────┘                    └───────────────────┘
          │
          ▼
┌───────────────────┐
│    EDA Agent      │
│  - Balance check  │
│  - VIF analysis   │
│  - Outlier detect │
│  - Distribution   │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Causal Discovery  │
│  - PC algorithm   │
│  - Build DAG      │
│  - Find colliders │
│  - Adjustment set │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Effect Estimator  │
│  - IPW, AIPW      │
│  - Matching       │
│  - DiD, RDD, IV   │
│  - Meta-learners  │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│Sensitivity Analyst│
│  - Rosenbaum      │
│  - E-value        │
│  - Placebo tests  │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Critique Agent   │  Reviews everything, can request re-runs
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│Notebook Generator │  Creates reproducible Jupyter notebook
└───────────────────┘
```

---

## The Agents

**For detailed agent documentation with all tools and behaviors, see [AGENTS.md](AGENTS.md)**

### Orchestrator Agent

The brain of the system. It decides which agent to call next based on the current state. Unlike traditional pipelines, it uses LLM reasoning to make decisions.

**Tools available:**
- `dispatch_to_agent`: Send task to a specialist agent
- `request_critique`: Ask critique agent to review
- `finalize_analysis`: Complete the job

**Key behavior:** The orchestrator does not follow a fixed sequence. If the EDA agent finds severe imbalance, it might ask for propensity score diagnostics before running effect estimation.

### Data Profiler Agent

First agent to touch the data. Identifies the causal structure through investigation.

**Tools available:**
- `get_dataset_overview`: Shape, columns, types, missing values
- `analyze_column`: Detailed stats for a specific column
- `check_treatment_balance`: Is this column suitable as treatment?
- `check_column_relationship`: Correlation, t-tests, chi-square
- `check_time_dimension`: Look for DiD/panel data opportunities
- `check_discontinuity_candidates`: Look for RDD opportunities
- `query_domain_knowledge`: Ask about variable meanings
- `finalize_profile`: Lock in the causal structure

**Key behavior:** Uses both statistical patterns and domain knowledge. A column named "black" with binary values might look like a treatment statistically, but domain knowledge reveals it's a demographic variable.

### EDA Agent

Explores the data quality and checks assumptions before estimation.

**Tools available:**
- `analyze_distributions`: Check normality, skewness
- `detect_outliers`: IQR and Z-score methods
- `check_multicollinearity`: VIF computation with collinearity warnings
- `check_balance`: Standardized mean differences between groups
- `check_overlap`: Propensity score overlap assessment
- `finalize_eda`: Lock in findings

**Key behavior:** If VIF is infinite (perfect collinearity), the agent flags this and the system can decide to drop variables before estimation.

### Causal Discovery Agent

Builds the causal DAG using multiple algorithms.

**Algorithms implemented:**
- **PC**: Constraint-based, uses conditional independence tests
- **FCI**: Handles latent confounders, outputs PAGs
- **GES**: Score-based greedy search
- **NOTEARS**: Continuous optimization with acyclicity constraint
- **LiNGAM**: Assumes non-Gaussian errors
- **Ensemble**: Runs multiple algorithms, takes consensus

**Conditional independence tests:**
- Fisher Z (linear/Gaussian)
- Kernel/HSIC (nonlinear)
- Chi-square (categorical)
- G-test (categorical)
- Missing-values-aware Fisher Z

**Key behavior:** Uses bootstrap stability selection to assign confidence scores to edges. Only edges that appear in >50% of bootstrap samples are considered stable.

### Effect Estimator Agent

Runs the actual causal inference methods.

**Methods available:**
- Inverse Probability Weighting (IPW)
- Augmented IPW (AIPW/Doubly Robust)
- Propensity Score Matching
- OLS Regression
- Difference-in-Differences
- Instrumental Variables (2SLS)
- Regression Discontinuity
- S-Learner, T-Learner, X-Learner
- Causal Forest
- Double ML

**Tools available:**
- `check_covariate_balance`: SMD before/after weighting
- `estimate_propensity_scores`: Fit propensity model
- `run_ipw`, `run_aipw`, `run_matching`: Execute methods
- `run_sensitivity_analysis`: Rosenbaum bounds, E-value
- `compare_methods`: Run multiple methods and compare

**Key behavior:** Uses stratified bootstrap for standard errors to ensure both treatment groups are represented in every bootstrap sample.

### Sensitivity Analyst Agent

Tests how robust the findings are to violations of assumptions.

**Methods:**
- Rosenbaum bounds (unobserved confounding)
- E-value (minimum confounding strength to nullify)
- Placebo tests (falsification)
- Coefficient stability (Oster bounds)

### Critique Agent

Quality control. Reviews all outputs before finalization.

**What it checks:**
- Are assumptions met for chosen methods?
- Is sample size adequate?
- Are confidence intervals reasonable?
- Do multiple methods agree?
- Are there red flags in diagnostics?

**Key behavior:** Can request the orchestrator to re-run agents with different parameters or methods.

### Notebook Generator Agent

Creates a reproducible Jupyter notebook with all analysis steps.

---

## Core Concepts

### The ReAct Pattern

Each agent runs in an observe-reason-act loop:

```
1. OBSERVE: See current state + results of previous action
2. REASON: LLM thinks about what to do next
3. ACT: Execute a tool
4. REPEAT until task complete
```

This is different from chain-of-thought which plans everything upfront. ReAct adapts when things don't go as expected.

**Implementation:** See `backend/src/agents/base/react_agent.py`

### Pull-Based Context

Traditional approach:
```
"Here's everything about the dataset: [5000 tokens of context]
Now decide what method to use."
```

Our approach:
```
Agent: "I need to know if there's a time dimension"
Tool: query_domain_knowledge(question="time dimension")
Result: "Dataset has 'year' column spanning 2010-2020"
Agent: "Good, DiD might be applicable. Let me check treatment timing..."
```

Agents pull context through tools like:
- `query_domain_knowledge`: Ask about variable meanings
- `get_column_info`: Get stats for specific columns
- `get_dag_recommendations`: Ask what the DAG suggests for adjustment

**Implementation:** See `backend/src/agents/base/context_tools.py`

### DAG-Driven Analysis

The system builds a causal DAG before running any estimation:

```
1. Run discovery algorithms (PC, GES, etc.)
2. Identify treatment → outcome path
3. Find confounders (common causes of treatment and outcome)
4. Find colliders (common effects - DO NOT adjust for these)
5. Compute adjustment set using backdoor criterion
6. Warn if adjusting would open confounding paths
```

**Why this matters:** Adjusting for a collider can CREATE bias where none existed. The system explicitly warns:

```
"COLLIDER WARNING: 'hospital_admission' has multiple causes.
Adjusting for it could open confounding paths."
```

**Implementation:** See `backend/src/causal/dag/discovery.py`

### Tool Registration Pattern

Each agent registers tools with:
- Name
- Description (for LLM)
- JSON schema for parameters
- Handler function

```python
self.register_tool(
    name="check_treatment_balance",
    description="Check if column is suitable as treatment",
    parameters={
        "type": "object",
        "properties": {
            "column": {"type": "string"}
        },
        "required": ["column"]
    },
    handler=self._tool_check_treatment_balance
)
```

The LLM sees the tool descriptions and decides which to call. No hardcoded if-else logic.

---

## API Endpoints

**For complete API documentation with request/response examples, see [API.md](API.md)**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /api/v1/jobs` | Create | Submit Kaggle URL for analysis |
| `GET /api/v1/jobs/{id}` | Read | Get job details and status |
| `GET /api/v1/jobs/{id}/results` | Read | Get analysis results |
| `GET /api/v1/jobs/{id}/notebook` | Read | Download Jupyter notebook |
| `GET /api/v1/jobs/{id}/traces` | Read | View agent reasoning traces |
| `DELETE /api/v1/jobs/{id}` | Delete | Delete job and artifacts |
| `POST /api/v1/jobs/{id}/cancel` | Action | Cancel running job |

---

## Causal Methods Reference

**For detailed method documentation with assumptions, formulas, and guidance, see [CAUSAL_METHODS.md](CAUSAL_METHODS.md)**

| Method | Estimand | When to Use |
|--------|----------|-------------|
| OLS Regression | ATE | Baseline, randomized experiments |
| Propensity Score Matching | ATT | Selection bias, want matched pairs |
| Inverse Probability Weighting | ATE | Selection bias, full sample |
| AIPW (Doubly Robust) | ATE | Want robustness to misspecification |
| Difference-in-Differences | ATT | Panel data, policy changes |
| Instrumental Variables | LATE | Endogeneity, have valid instrument |
| Regression Discontinuity | LATE | Sharp/fuzzy cutoff in assignment |
| S/T/X-Learner | CATE | Heterogeneous treatment effects |
| Causal Forest | CATE | Nonlinear heterogeneity |
| Double ML | ATE | High-dimensional covariates |

---

## Configuration

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key | Yes (or other LLM) |
| `ANTHROPIC_API_KEY` | Anthropic Claude API key | Optional |
| `KAGGLE_KEY` | Kaggle API credentials (JSON or key) | Yes |
| `KAGGLE_USERNAME` | Kaggle username | Yes |
| `STORAGE_TYPE` | `firestore` or `local` | No (default: local) |
| `GCP_PROJECT_ID` | GCP project for Firestore | For Firestore |
| `ORCHESTRATOR_MODE` | `standard` or `react` | No (default: standard) |

---

## Testing

```bash
cd backend

# Unit tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Agentic evaluations (tests agent behavior)
pytest -m agentic -v
```

---

## Extending the System

**For development guide with extension patterns, see [DEVELOPMENT.md](DEVELOPMENT.md)**

The system is designed to be extensible. You can add:
- New specialist agents
- New causal inference methods
- New LLM providers (beyond Gemini/Claude/Vertex)
- New storage backends

---

## Project Structure

```
causal-orchestrator/
├── backend/
│   ├── src/
│   │   ├── agents/
│   │   │   ├── base/
│   │   │   │   ├── react_agent.py      # ReAct loop implementation
│   │   │   │   ├── context_tools.py    # Pull-based context tools
│   │   │   │   └── state.py            # Shared state definitions
│   │   │   ├── orchestrator/
│   │   │   │   ├── orchestrator_agent.py
│   │   │   │   └── react_orchestrator.py
│   │   │   ├── specialists/
│   │   │   │   ├── data_profiler.py
│   │   │   │   ├── eda_agent.py
│   │   │   │   ├── effect_estimator.py
│   │   │   │   ├── causal_discovery.py
│   │   │   │   ├── sensitivity_analyst.py
│   │   │   │   └── notebook_generator.py
│   │   │   └── critique/
│   │   │       └── critique_agent.py
│   │   ├── causal/
│   │   │   ├── dag/
│   │   │   │   └── discovery.py        # PC, FCI, GES, NOTEARS, LiNGAM
│   │   │   └── methods/
│   │   │       ├── propensity.py       # IPW, AIPW, Matching
│   │   │       ├── did.py              # Difference-in-Differences
│   │   │       ├── iv.py               # Instrumental Variables
│   │   │       ├── rdd.py              # Regression Discontinuity
│   │   │       └── metalearners.py     # S/T/X-Learner
│   │   ├── api/                        # FastAPI routes
│   │   ├── jobs/                       # Job manager with async tasks
│   │   ├── llm/                        # Gemini, Claude, Vertex clients
│   │   └── storage/                    # Firestore, local storage
│   ├── tests/
│   └── benchmarks/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── results/ResultsDisplay.tsx
│   │   │   └── agents/AgentTraces.tsx
│   │   ├── pages/
│   │   │   ├── HomePage.tsx
│   │   │   ├── JobPage.tsx
│   │   │   └── JobsListPage.tsx
│   │   └── services/api.ts
│   └── package.json
└── README.md
```

---

