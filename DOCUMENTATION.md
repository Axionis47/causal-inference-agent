# Causal Orchestrator - Comprehensive Technical Documentation

**Version:** 1.0.0  
**Last Updated:** 2026-01-23  
**Project Type:** Agentic Causal Inference System  
**Tech Stack:** Python 3.11, FastAPI, React 18, TypeScript, Google Cloud Platform

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Technology Stack](#technology-stack)
4. [Backend Architecture](#backend-architecture)
5. [Frontend Architecture](#frontend-architecture)
6. [Agent System](#agent-system)
7. [Causal Inference Methods](#causal-inference-methods)
8. [Data Flow](#data-flow)
9. [API Reference](#api-reference)
10. [Database Schema](#database-schema)
11. [Infrastructure & Deployment](#infrastructure--deployment)
12. [CI/CD Pipeline](#cicd-pipeline)
13. [Security](#security)
14. [Configuration](#configuration)
15. [Testing Strategy](#testing-strategy)
16. [Monitoring & Observability](#monitoring--observability)
17. [Development Guide](#development-guide)
18. [Troubleshooting](#troubleshooting)

---

## Executive Summary

### What is Causal Orchestrator?

Causal Orchestrator is an **autonomous agentic system** that performs end-to-end causal inference analysis on datasets from Kaggle URLs. It uses Large Language Models (LLMs) to coordinate a team of specialist AI agents that:

1. **Automatically discover** treatment and outcome variables
2. **Select appropriate** causal inference methods based on data characteristics
3. **Estimate treatment effects** using 12 different causal methods
4. **Validate assumptions** and perform sensitivity analysis
5. **Generate reproducible** Jupyter notebooks with complete analysis

### Key Differentiators

- **Truly Agentic**: All method selection decisions made by LLM reasoning, NO hardcoded if-else logic
- **Multi-Method**: Supports 12 causal inference methods (OLS, PSM, IPW, AIPW, DiD, IV/2SLS, RDD, Meta-learners, Causal Forest, Double ML)
- **Quality Control**: Built-in critique agent that reviews and iterates on analysis
- **Production-Ready**: Auto-scaling deployment on GCP Cloud Run with full observability
- **Benchmark-Validated**: Tested on 8 diverse causal inference benchmark datasets

### Use Cases

- **Research**: Automated causal analysis for academic research
- **Business Analytics**: Treatment effect estimation for A/B tests, marketing campaigns
- **Policy Analysis**: Impact evaluation for policy interventions
- **Healthcare**: Treatment effectiveness studies
- **Economics**: Causal inference in observational data

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              React Frontend                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ URL Input   │  │ Job Status  │  │ Results     │  │ Agent Traces       │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FastAPI Backend                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Orchestrator Agent (Claude/Gemini)               │   │
│  │  - Coordinates analysis flow via LLM reasoning                       │   │
│  │  - NO hardcoded if-else logic                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │              │                │               │                     │
│       ▼              ▼                ▼               ▼                     │
│  ┌─────────┐  ┌────────────┐  ┌─────────────┐  ┌───────────────┐           │
│  │ Data    │  │ Effect     │  │ Sensitivity │  │ Notebook      │           │
│  │ Profiler│  │ Estimator  │  │ Analyst     │  │ Generator     │           │
│  └─────────┘  └────────────┘  └─────────────┘  └───────────────┘           │
│                      │                                                      │
│                      ▼                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       Critique Agent (Claude/Gemini)                 │   │
│  │  - Reviews all outputs for quality                                   │   │
│  │  - Can request iterations or reject                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Google Cloud Platform                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  Firestore   │  │ Cloud Storage│  │ Secret Mgr   │  │  Cloud Run   │   │
│  │  (Jobs/Data) │  │  (Notebooks) │  │  (API Keys)  │  │  (Compute)   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

| Component | Technology | Purpose | Scalability |
|-----------|-----------|---------|-------------|
| **Frontend** | React 18 + TypeScript + Vite | User interface for job submission and results | Stateless, CDN-ready |
| **Backend API** | FastAPI + Python 3.11 | REST API, job orchestration | Horizontal scaling on Cloud Run |
| **LLM Provider** | Claude Sonnet 4 / Gemini 1.5 Pro | Agent reasoning and decision-making | API-based, auto-scales |
| **Job Storage** | Google Firestore | Job state, results, traces | Managed, auto-scales |
| **Notebook Storage** | Google Cloud Storage | Generated Jupyter notebooks | Object storage, unlimited |
| **Secret Management** | GCP Secret Manager | API keys (Anthropic, Kaggle) | Encrypted, versioned |
| **Compute** | GCP Cloud Run | Containerized workloads | 0-1000 instances auto-scale |

---

## Technology Stack

### Backend Stack

```yaml
Language: Python 3.11
Framework: FastAPI 0.115.0
ASGI Server: Uvicorn 0.30.6
LLM Clients:
  - Anthropic Claude API (claude-sonnet-4-20250514)
  - Google Gemini API (gemini-1.5-pro)
  - Google Vertex AI (gemini-2.0-flash-exp)
Causal Inference:
  - DoWhy 0.12+
  - EconML 0.15.0+
  - CausalML 0.15.0+
Data Science:
  - NumPy 1.26.4
  - Pandas 2.2.3
  - Scikit-learn 1.5.2
  - Statsmodels 0.14.4
Graph Libraries:
  - NetworkX 3.4.2
  - pgmpy 0.1.26
Cloud SDKs:
  - google-cloud-firestore 2.19.0
  - google-cloud-storage 2.18.2
  - google-cloud-secret-manager 2.20.2
Utilities:
  - Pydantic 2.10.3 (validation)
  - Structlog 24.4.0 (logging)
  - Tenacity 9.0.0 (retries)
  - HTTPX 0.27.2 (async HTTP)
```

### Frontend Stack

```yaml
Language: TypeScript 5.3.3
Framework: React 18.2.0
Build Tool: Vite 5.0.11
Routing: React Router DOM 6.21.0
State Management: Zustand 4.4.7
Data Fetching: TanStack Query 5.17.0
HTTP Client: Axios 1.6.5
Visualization: D3.js 7.8.5
Styling: Tailwind CSS 3.4.1
Icons: Lucide React 0.303.0
Code Display: React Syntax Highlighter 15.5.0
Testing: Vitest 1.2.0
```

### Infrastructure Stack

```yaml
Container Runtime: Docker
Orchestration: Docker Compose (local), GCP Cloud Run (prod)
CI/CD: GitHub Actions
Cloud Provider: Google Cloud Platform
  - Cloud Run (compute)
  - Firestore (database)
  - Cloud Storage (object storage)
  - Secret Manager (secrets)
  - Container Registry (images)
IaC: Terraform (optional, in infrastructure/terraform)
Monitoring: GCP Cloud Logging, Cloud Monitoring
```

---

## Backend Architecture

### Directory Structure

```
backend/
├── src/
│   ├── agents/              # Agentic system
│   │   ├── base/           # Base agent classes
│   │   │   ├── agent.py           # BaseAgent abstract class
│   │   │   ├── react_agent.py     # ReAct pattern implementation
│   │   │   ├── state.py           # AnalysisState, JobStatus enums
│   │   │   └── tool.py            # Tool definitions
│   │   ├── orchestrator/   # Central coordinator agents
│   │   │   ├── orchestrator_agent.py  # Standard orchestrator
│   │   │   └── react_orchestrator.py  # ReAct orchestrator (experimental)
│   │   ├── specialists/    # Domain expert agents
│   │   │   ├── data_profiler.py       # Dataset profiling
│   │   │   ├── eda_agent.py           # Exploratory data analysis
│   │   │   ├── causal_discovery.py    # Causal graph learning
│   │   │   ├── confounder_discovery.py # Confounder identification
│   │   │   ├── effect_estimator.py    # Treatment effect estimation
│   │   │   ├── effect_estimator_react.py # ReAct effect estimator
│   │   │   ├── sensitivity_analyst.py # Robustness checks
│   │   │   ├── ps_diagnostics.py      # Propensity score diagnostics
│   │   │   ├── data_repair.py         # Data quality fixes
│   │   │   └── notebook_generator.py  # Jupyter notebook creation
│   │   ├── critique/       # Quality reviewer
│   │   │   └── critique_agent.py      # Analysis critique
│   │   └── tools/          # Agent tools
│   ├── api/                # FastAPI application
│   │   ├── main.py                # App entry point
│   │   ├── routes/
│   │   │   ├── health.py          # Health check endpoint
│   │   │   └── jobs.py            # Job management endpoints
│   │   ├── schemas/               # Pydantic request/response models
│   │   └── middleware/            # Custom middleware
│   ├── causal/             # Causal inference methods
│   │   ├── methods/
│   │   │   ├── base.py            # BaseCausalMethod
│   │   │   ├── ols.py             # OLS regression
│   │   │   ├── propensity.py      # PSM, IPW, AIPW
│   │   │   ├── did.py             # Difference-in-Differences
│   │   │   ├── iv.py              # Instrumental Variables (2SLS)
│   │   │   ├── rdd.py             # Regression Discontinuity
│   │   │   ├── metalearners.py    # S/T/X-Learners
│   │   │   ├── causal_forest.py   # Causal Forest
│   │   │   └── double_ml.py       # Double Machine Learning
│   │   ├── estimators/            # Effect estimators
│   │   ├── dag/                   # DAG utilities
│   │   └── utils/                 # Causal utilities
│   ├── llm/                # LLM clients
│   │   ├── client.py              # LLM client factory
│   │   ├── claude_client.py       # Anthropic Claude client
│   │   ├── gemini_client.py       # Google Gemini client
│   │   ├── vertex_client.py       # Google Vertex AI client
│   │   └── prompts/               # Prompt templates
│   ├── jobs/               # Job management
│   │   └── manager.py             # JobManager class
│   ├── storage/            # Data persistence
│   │   └── firestore.py           # Firestore client
│   ├── config/             # Configuration
│   │   └── settings.py            # Pydantic settings
│   ├── logging_config/     # Structured logging
│   │   └── structured.py          # Logging setup
│   └── notebook/           # Notebook generation
│       └── templates/             # Notebook templates
├── benchmarks/             # Benchmark datasets & evaluation
│   ├── datasets.py                # 8 benchmark datasets
│   ├── evaluation.py              # Evaluation metrics
│   └── runners/                   # Benchmark runners
├── tests/                  # Test suite
│   ├── unit/                      # Unit tests
│   ├── integration/               # Integration tests
│   └── conftest.py                # Pytest fixtures
├── requirements.txt        # Python dependencies
├── Dockerfile             # Multi-stage Docker build
└── .env.example           # Environment template
```

### Core Classes & Modules

#### 1. AnalysisState (src/agents/base/state.py)

The central state object that flows through the entire analysis pipeline.

```python
@dataclass
class AnalysisState:
    job_id: str
    dataset_info: DatasetInfo
    status: JobStatus
    created_at: datetime
    updated_at: datetime

    # Optional fields populated during analysis
    treatment_variable: str | None = None
    outcome_variable: str | None = None
    data_profile: DataProfile | None = None
    eda_result: EDAResult | None = None
    proposed_dag: CausalDAG | None = None
    treatment_effects: list[TreatmentEffect] = field(default_factory=list)
    sensitivity_results: list[SensitivityResult] = field(default_factory=list)
    critique_history: list[CritiqueResult] = field(default_factory=list)
    agent_traces: list[AgentTrace] = field(default_factory=list)
    notebook_path: str | None = None
    recommendations: list[str] = field(default_factory=list)
    error_message: str | None = None
    error_agent: str | None = None
    iteration_count: int = 0
    max_iterations: int = 3
```

**Key Methods:**
- `add_trace(trace: AgentTrace)` - Record agent execution
- `mark_completed()` - Mark job as complete
- `mark_failed(error: str, agent: str)` - Mark job as failed
- `should_iterate() -> bool` - Check if iteration needed
- `is_approved() -> bool` - Check if critique approved
- `get_latest_critique() -> CritiqueResult | None` - Get latest feedback

#### 2. BaseAgent (src/agents/base/agent.py)

Abstract base class for all agents in the system.

```python
class BaseAgent(ABC):
    AGENT_NAME: str = "base"
    SYSTEM_PROMPT: str = ""
    TOOLS: list[dict[str, Any]] = []

    def __init__(self):
        self.llm_client = get_llm_client()
        self.logger = get_logger(self.AGENT_NAME)

    @abstractmethod
    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Execute the agent's task and return updated state."""
        pass

    async def reason(self, prompt: str, context: dict) -> dict:
        """Use LLM to reason about the task."""
        return await self.llm_client.generate_with_function_calling(
            prompt=prompt,
            system_instruction=self.SYSTEM_PROMPT,
            tools=self.TOOLS
        )

    def create_trace(self, action: str, reasoning: str, **kwargs) -> AgentTrace:
        """Create an execution trace."""
        return AgentTrace(
            agent_name=self.AGENT_NAME,
            timestamp=datetime.utcnow(),
            action=action,
            reasoning=reasoning,
            **kwargs
        )
```

#### 3. OrchestratorAgent (src/agents/orchestrator/orchestrator_agent.py)

The brain of the system - coordinates all specialist agents using LLM reasoning.

**Key Responsibilities:**
1. Analyze current analysis state
2. Decide which specialist agent to dispatch next
3. Handle critique feedback and iterations
4. Finalize analysis and trigger notebook generation

**Tools Available:**
- `dispatch_to_agent` - Send task to specialist (data_profiler, eda_agent, causal_discovery, effect_estimator, sensitivity_analyst, notebook_generator)
- `request_critique` - Request quality review from critique agent
- `finalize_analysis` - Mark analysis complete

**Workflow Logic:**
```
1. ALWAYS start with data_profiler
2. ALWAYS run eda_agent for data quality checks
3. Reason about appropriate causal methods based on profile + EDA
4. Dispatch to effect_estimator with method recommendations
5. Run sensitivity_analyst for robustness checks
6. Request critique review
7. If ITERATE: address feedback and re-dispatch
8. If APPROVED: dispatch to notebook_generator
9. Mark as COMPLETED
```

#### 4. JobManager (src/jobs/manager.py)

Manages the lifecycle of analysis jobs.

**Key Methods:**
- `create_job(kaggle_url, treatment_var, outcome_var)` - Create and start new job
- `get_job(job_id)` - Retrieve job details
- `get_job_status(job_id)` - Get lightweight status
- `list_jobs(status, limit, offset)` - List jobs with pagination
- `cancel_job(job_id)` - Cancel running job
- `get_results(job_id)` - Get analysis results
- `get_traces(job_id)` - Get agent execution traces

**Orchestrator Modes:**
- `standard` - Original orchestrator with guided workflow
- `react` - Fully autonomous ReAct orchestrator (experimental)

#### 5. LLM Clients (src/llm/)

**ClaudeClient** (claude_client.py):
- Uses Anthropic Claude API (claude-sonnet-4-20250514)
- Supports function calling for tool use
- Structured output generation with Pydantic schemas
- Async HTTP client with 120s timeout

**GeminiClient** (gemini_client.py):
- Uses Google Gemini API (gemini-1.5-pro)
- Function calling support
- Structured output via JSON schema

**VertexClient** (vertex_client.py):
- Uses Google Vertex AI (gemini-2.0-flash-exp)
- GCP-managed, uses Application Default Credentials
- Experimental model with rate limits

**Client Selection:**
Configured via `LLM_PROVIDER` environment variable:
```python
# In settings.py
llm_provider: Literal["gemini", "vertex", "claude"] = "claude"
```

#### 6. Firestore Storage (src/storage/firestore.py)

**Collections:**
- `jobs` - Job metadata and status
- `results` - Analysis results (effects, DAG, recommendations)
- `agent_traces/{job_id}/traces` - Agent execution traces

**Key Operations:**
- `create_job(state)` - Create job document
- `update_job(state)` - Update job status
- `save_results(state)` - Save analysis results
- `save_traces(state)` - Save agent traces
- `get_job(job_id)` - Retrieve job
- `list_jobs(status, limit, offset)` - Query jobs

---

## Frontend Architecture

### Directory Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── common/          # Reusable UI components
│   │   │   ├── Header.tsx
│   │   │   ├── Button.tsx
│   │   │   ├── Card.tsx
│   │   │   └── LoadingSpinner.tsx
│   │   ├── agents/          # Agent-related components
│   │   │   ├── AgentTraceViewer.tsx
│   │   │   └── AgentStatusBadge.tsx
│   │   ├── dataset/         # Dataset components
│   │   │   ├── DatasetInput.tsx
│   │   │   └── DataProfileCard.tsx
│   │   ├── job/             # Job components
│   │   │   ├── JobStatusCard.tsx
│   │   │   ├── JobProgressBar.tsx
│   │   │   └── JobList.tsx
│   │   └── results/         # Results components
│   │       ├── TreatmentEffectsTable.tsx
│   │       ├── CausalDAGVisualization.tsx
│   │       ├── SensitivityResults.tsx
│   │       └── NotebookDownload.tsx
│   ├── pages/
│   │   ├── HomePage.tsx         # Job creation page
│   │   ├── JobPage.tsx          # Job details & results
│   │   └── JobsListPage.tsx     # All jobs list
│   ├── services/
│   │   └── api.ts               # API client (Axios)
│   ├── store/
│   │   ├── index.ts             # Zustand store setup
│   │   └── jobStore.ts          # Job state management
│   ├── hooks/
│   │   ├── useCreateJob.ts      # Job creation hook
│   │   ├── useJob.ts            # Job details hook
│   │   ├── useJobsList.ts       # Jobs list hook
│   │   └── useResults.ts        # Results hook
│   ├── types/
│   │   └── index.ts             # TypeScript types
│   ├── utils/
│   │   └── index.ts             # Utility functions
│   ├── App.tsx                  # Root component
│   ├── main.tsx                 # Entry point
│   └── index.css                # Global styles
├── public/                      # Static assets
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js
├── Dockerfile
└── nginx.conf                   # Nginx config for production
```

### Key Components

#### API Client (services/api.ts)

```typescript
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const api = {
  // Jobs
  createJob: (data: CreateJobRequest) =>
    axios.post(`${API_BASE_URL}/api/v1/jobs`, data),

  getJob: (jobId: string) =>
    axios.get(`${API_BASE_URL}/api/v1/jobs/${jobId}`),

  listJobs: (params?: ListJobsParams) =>
    axios.get(`${API_BASE_URL}/api/v1/jobs`, { params }),

  // Results
  getResults: (jobId: string) =>
    axios.get(`${API_BASE_URL}/api/v1/jobs/${jobId}/results`),

  getTraces: (jobId: string) =>
    axios.get(`${API_BASE_URL}/api/v1/jobs/${jobId}/traces`),

  downloadNotebook: (jobId: string) =>
    axios.get(`${API_BASE_URL}/api/v1/jobs/${jobId}/notebook`, {
      responseType: 'blob'
    }),
};
```

#### State Management (store/jobStore.ts)

Uses Zustand for lightweight state management:

```typescript
interface JobStore {
  currentJob: Job | null;
  jobs: Job[];
  setCurrentJob: (job: Job) => void;
  addJob: (job: Job) => void;
  updateJob: (jobId: string, updates: Partial<Job>) => void;
}

export const useJobStore = create<JobStore>((set) => ({
  currentJob: null,
  jobs: [],
  setCurrentJob: (job) => set({ currentJob: job }),
  addJob: (job) => set((state) => ({ jobs: [job, ...state.jobs] })),
  updateJob: (jobId, updates) => set((state) => ({
    jobs: state.jobs.map(j => j.id === jobId ? { ...j, ...updates } : j)
  })),
}));
```

#### Custom Hooks

**useJob** - Real-time job polling:
```typescript
export const useJob = (jobId: string) => {
  return useQuery({
    queryKey: ['job', jobId],
    queryFn: () => api.getJob(jobId),
    refetchInterval: (data) => {
      // Poll every 2s if job is running
      const status = data?.status;
      return ['pending', 'running', 'profiling', 'estimating_effects']
        .includes(status) ? 2000 : false;
    },
  });
};
```

---

## Agent System

### Agent Hierarchy

```
BaseAgent (abstract)
├── OrchestratorAgent - Coordinates workflow
├── ReActOrchestrator - Autonomous ReAct orchestrator
├── Specialist Agents:
│   ├── DataProfilerAgent - Dataset profiling
│   ├── EDAAgent - Exploratory data analysis
│   ├── CausalDiscoveryAgent - Causal graph learning
│   ├── ConfounderDiscoveryAgent - Confounder identification
│   ├── EffectEstimatorAgent - Treatment effect estimation
│   ├── EffectEstimatorReActAgent - ReAct effect estimator
│   ├── SensitivityAnalystAgent - Robustness checks
│   ├── PSDiagnosticsAgent - Propensity score diagnostics
│   ├── DataRepairAgent - Data quality fixes
│   └── NotebookGeneratorAgent - Jupyter notebook creation
└── CritiqueAgent - Quality reviewer
```

### Agent Descriptions

#### 1. DataProfilerAgent

**Purpose:** Analyze dataset characteristics and identify treatment/outcome candidates

**Inputs:**
- Kaggle dataset URL or uploaded CSV
- Optional treatment/outcome hints from user

**Outputs:**
- `DataProfile` object containing:
  - Number of samples, features
  - Feature types (numeric, categorical, binary, datetime)
  - Treatment variable candidates (binary/categorical features)
  - Outcome variable candidates (numeric features)
  - Time dimension detection
  - Potential instrumental variables
  - Discontinuity candidates (for RDD)
  - Missing data summary

**LLM Reasoning:**
- Analyzes feature distributions
- Identifies plausible treatment variables
- Suggests outcome variables
- Detects data structure (panel, cross-sectional, time-series)

#### 2. EDAAgent

**Purpose:** Comprehensive exploratory data analysis

**Outputs:**
- `EDAResult` object containing:
  - Data quality score (0-100)
  - Data quality issues (missing data, outliers, imbalance)
  - High correlations (multicollinearity detection)
  - Covariate balance assessment
  - Distribution summaries
  - Outlier detection results

**Key Checks:**
- Missing data patterns
- Outlier detection (IQR, Z-score)
- Correlation matrix analysis
- Covariate balance (treatment vs control)
- Multicollinearity (VIF scores)

#### 3. CausalDiscoveryAgent

**Purpose:** Learn causal graph structure from data

**Methods:**
- PC Algorithm (constraint-based)
- GES (score-based)
- NOTEARS (continuous optimization)

**Outputs:**
- `CausalDAG` object containing:
  - Nodes (variables)
  - Edges (causal relationships)
  - Discovery method used
  - Confidence scores

**LLM Reasoning:**
- Selects appropriate discovery algorithm
- Interprets discovered graph
- Identifies confounders, mediators, colliders

#### 4. EffectEstimatorAgent

**Purpose:** Estimate treatment effects using appropriate causal methods

**Supported Methods:**
1. **OLS Regression** - Baseline, experimental data
2. **Propensity Score Matching (PSM)** - Selection bias
3. **Inverse Probability Weighting (IPW)** - Covariate adjustment
4. **Doubly Robust (AIPW)** - Robust to misspecification
5. **Difference-in-Differences (DiD)** - Panel data, policy changes
6. **Instrumental Variables (2SLS)** - Endogeneity
7. **Regression Discontinuity (RDD)** - Sharp/fuzzy cutoffs
8. **S-Learner** - Heterogeneous effects
9. **T-Learner** - Heterogeneous effects
10. **X-Learner** - Imbalanced treatment groups
11. **Causal Forest** - Non-linear heterogeneity
12. **Double ML** - High-dimensional covariates

**Outputs:**
- `TreatmentEffect` objects containing:
  - Method name
  - Estimand (ATE, ATT, CATE, LATE)
  - Point estimate
  - Standard error
  - Confidence interval (95%)
  - P-value
  - Assumptions tested

**LLM Reasoning:**
- Analyzes data characteristics
- Selects appropriate methods
- Interprets results
- Checks assumptions

#### 5. SensitivityAnalystAgent

**Purpose:** Perform robustness checks and sensitivity analysis

**Methods:**
- **Rosenbaum Bounds** - Hidden bias sensitivity
- **E-values** - Unmeasured confounding strength
- **Placebo Tests** - Falsification tests
- **Refutation Tests** - Random common cause, data subset

**Outputs:**
- `SensitivityResult` objects containing:
  - Method name
  - Robustness value
  - Interpretation
  - Recommendations

#### 6. NotebookGeneratorAgent

**Purpose:** Generate reproducible Jupyter notebook with complete analysis

**Notebook Sections:**
1. Introduction & Dataset Overview
2. Data Profiling
3. Exploratory Data Analysis
4. Causal Graph Discovery
5. Treatment Effect Estimation
6. Sensitivity Analysis
7. Conclusions & Recommendations
8. References

**Features:**
- Executable Python code
- Visualizations (DAG, distributions, effects)
- Markdown explanations
- Statistical tables
- Reproducible with saved data

#### 7. CritiqueAgent

**Purpose:** Review analysis quality and provide feedback

**Review Criteria:**
- Data quality (missing data, outliers, balance)
- Method appropriateness (assumptions, data structure)
- Statistical significance
- Effect size interpretation
- Sensitivity analysis completeness

**Decisions:**
- `APPROVE` - Analysis is high quality, proceed to notebook
- `ITERATE` - Issues found, re-run with improvements
- `REJECT` - Fatal flaws, cannot proceed

**Outputs:**
- `CritiqueResult` object containing:
  - Decision (APPROVE/ITERATE/REJECT)
  - Issues found
  - Improvements needed
  - Strengths identified

---

## Causal Inference Methods

### Method Selection Decision Tree

```
Data Characteristics → Recommended Methods

Experimental (RCT):
  └─→ OLS Regression (baseline)

Observational + Selection Bias:
  ├─→ Propensity Score Matching (PSM)
  ├─→ Inverse Probability Weighting (IPW)
  └─→ Doubly Robust (AIPW)

Panel Data + Policy Change:
  └─→ Difference-in-Differences (DiD)

Endogeneity + Instrument Available:
  └─→ Instrumental Variables (2SLS)

Sharp/Fuzzy Cutoff:
  └─→ Regression Discontinuity (RDD)

Heterogeneous Effects:
  ├─→ S-Learner (single model)
  ├─→ T-Learner (separate models)
  ├─→ X-Learner (imbalanced groups)
  └─→ Causal Forest (non-linear)

High-Dimensional Covariates:
  └─→ Double Machine Learning (Double ML)
```

### Method Details

#### 1. OLS Regression

**Estimand:** ATE (Average Treatment Effect)

**Formula:**
```
Y = β₀ + β₁·T + β₂·X + ε
ATE = β₁
```

**Assumptions:**
- Linearity
- No omitted confounders
- Homoscedasticity
- Independence

**Use Cases:**
- Randomized experiments
- Baseline comparison
- Linear relationships

**Implementation:** `src/causal/methods/ols.py`

#### 2. Propensity Score Matching (PSM)

**Estimand:** ATT (Average Treatment Effect on Treated)

**Steps:**
1. Estimate propensity scores: P(T=1|X)
2. Match treated units to control units
3. Compute average difference in outcomes

**Assumptions:**
- Unconfoundedness (no hidden confounders)
- Common support (overlap)
- Correct propensity model

**Matching Methods:**
- Nearest neighbor
- Caliper matching
- Kernel matching

**Implementation:** `src/causal/methods/propensity.py`

#### 3. Inverse Probability Weighting (IPW)

**Estimand:** ATE

**Formula:**
```
ATE = E[Y·T/e(X)] - E[Y·(1-T)/(1-e(X))]
where e(X) = P(T=1|X)
```

**Advantages:**
- Uses all data (no matching)
- Consistent under correct propensity model

**Implementation:** `src/causal/methods/propensity.py`

#### 4. Doubly Robust (AIPW)

**Estimand:** ATE

**Formula:**
```
Combines outcome regression + propensity weighting
Robust to misspecification of either model
```

**Advantages:**
- Consistent if EITHER propensity OR outcome model is correct
- More robust than IPW or regression alone

**Implementation:** `src/causal/methods/propensity.py`

#### 5. Difference-in-Differences (DiD)

**Estimand:** ATT

**Formula:**
```
DiD = (Y_treated,post - Y_treated,pre) - (Y_control,post - Y_control,pre)
```

**Assumptions:**
- Parallel trends (pre-treatment)
- No spillovers
- Stable composition

**Use Cases:**
- Policy interventions
- Natural experiments
- Panel data

**Implementation:** `src/causal/methods/did.py`

#### 6. Instrumental Variables (2SLS)

**Estimand:** LATE (Local Average Treatment Effect)

**Steps:**
1. First stage: T = γ₀ + γ₁·Z + γ₂·X + u
2. Second stage: Y = β₀ + β₁·T̂ + β₂·X + ε

**Assumptions:**
- Relevance: Z affects T
- Exclusion: Z affects Y only through T
- Exogeneity: Z uncorrelated with error

**Use Cases:**
- Endogenous treatment
- Compliance issues
- Randomized encouragement

**Implementation:** `src/causal/methods/iv.py`

#### 7. Regression Discontinuity (RDD)

**Estimand:** LATE at cutoff

**Types:**
- Sharp RDD: Treatment deterministic at cutoff
- Fuzzy RDD: Treatment probability jumps at cutoff

**Assumptions:**
- Continuity of potential outcomes
- No manipulation of running variable
- Local randomization near cutoff

**Implementation:** `src/causal/methods/rdd.py`

#### 8-10. Meta-Learners (S/T/X)

**S-Learner:**
```
Single model: μ(X, T) predicts Y
CATE(X) = μ(X, 1) - μ(X, 0)
```

**T-Learner:**
```
Two models: μ₀(X) for control, μ₁(X) for treated
CATE(X) = μ₁(X) - μ₀(X)
```

**X-Learner:**
```
Improved T-Learner for imbalanced groups
Uses propensity scores for weighting
```

**Implementation:** `src/causal/methods/metalearners.py`

#### 11. Causal Forest

**Estimand:** CATE (Conditional Average Treatment Effect)

**Method:**
- Random forest adapted for causal inference
- Honest splitting (separate samples for tree building and estimation)
- Estimates heterogeneous treatment effects

**Advantages:**
- Non-parametric
- Handles non-linear relationships
- Provides confidence intervals

**Implementation:** `src/causal/methods/causal_forest.py`

#### 12. Double Machine Learning (Double ML)

**Estimand:** ATE

**Steps:**
1. Use ML to predict Y from X (nuisance function)
2. Use ML to predict T from X (propensity score)
3. Estimate ATE from residuals (debiased)

**Advantages:**
- Handles high-dimensional X
- Robust to model misspecification
- Valid inference

**Implementation:** `src/causal/methods/double_ml.py`

---

## Data Flow

### Job Lifecycle

```
1. User submits Kaggle URL
   ↓
2. JobManager creates job (status: PENDING)
   ↓
3. JobManager starts async task
   ↓
4. OrchestratorAgent.execute(state)
   ↓
5. Dispatch to DataProfilerAgent (status: PROFILING)
   ├─ Download dataset from Kaggle
   ├─ Analyze features
   ├─ Identify treatment/outcome candidates
   └─ Update state.data_profile
   ↓
6. Dispatch to EDAAgent (status: EXPLORATORY_ANALYSIS)
   ├─ Data quality checks
   ├─ Correlation analysis
   ├─ Covariate balance
   └─ Update state.eda_result
   ↓
7. Dispatch to CausalDiscoveryAgent (status: DISCOVERING_CAUSAL)
   ├─ Learn causal graph
   ├─ Identify confounders
   └─ Update state.proposed_dag
   ↓
8. Dispatch to EffectEstimatorAgent (status: ESTIMATING_EFFECTS)
   ├─ Select appropriate methods (LLM reasoning)
   ├─ Estimate treatment effects
   ├─ Check assumptions
   └─ Update state.treatment_effects
   ↓
9. Dispatch to SensitivityAnalystAgent (status: SENSITIVITY_ANALYSIS)
   ├─ Rosenbaum bounds
   ├─ E-values
   ├─ Placebo tests
   └─ Update state.sensitivity_results
   ↓
10. Request CritiqueAgent review (status: CRITIQUE_REVIEW)
    ├─ Evaluate analysis quality
    ├─ Check assumptions
    └─ Decision: APPROVE / ITERATE / REJECT
    ↓
11a. If ITERATE:
     ├─ Increment iteration_count
     ├─ Address feedback
     └─ Go back to step 8
    ↓
11b. If APPROVE or max_iterations reached:
     ↓
12. Dispatch to NotebookGeneratorAgent (status: GENERATING_NOTEBOOK)
    ├─ Generate Jupyter notebook
    ├─ Upload to Cloud Storage
    └─ Update state.notebook_path
    ↓
13. Mark as COMPLETED
    ├─ Save results to Firestore
    ├─ Save traces to Firestore
    └─ Update job status
```

### State Transitions

```
PENDING → FETCHING_DATA → PROFILING → EXPLORATORY_ANALYSIS →
DISCOVERING_CAUSAL → ESTIMATING_EFFECTS → SENSITIVITY_ANALYSIS →
CRITIQUE_REVIEW → [ITERATING →] GENERATING_NOTEBOOK → COMPLETED

                    ↓ (on error)
                  FAILED
```

---

## API Reference

### Base URL

- **Local:** `http://localhost:8000`
- **Staging:** `https://causal-backend-staging-<hash>.run.app`
- **Production:** `https://causal-backend-<hash>.run.app`

### Authentication

Currently **no authentication** required (allow-unauthenticated on Cloud Run).

**Future:** Add API key authentication or OAuth2.

### Endpoints

#### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-01-23T10:30:00Z"
}
```

#### Create Job

```http
POST /api/v1/jobs
Content-Type: application/json
```

**Request Body:**
```json
{
  "kaggle_url": "https://www.kaggle.com/datasets/username/dataset-name",
  "treatment_variable": "treatment",  // optional
  "outcome_variable": "outcome",      // optional
  "preferences": {                    // optional
    "max_iterations": 3,
    "methods": ["psm", "ipw", "did"]
  }
}
```

**Response:**
```json
{
  "job_id": "a1b2c3d4",
  "status": "pending",
  "created_at": "2026-01-23T10:30:00Z"
}
```

#### Get Job

```http
GET /api/v1/jobs/{job_id}
```

**Response:**
```json
{
  "id": "a1b2c3d4",
  "kaggle_url": "https://www.kaggle.com/datasets/...",
  "status": "estimating_effects",
  "progress_percentage": 50,
  "current_agent": "effect_estimator",
  "treatment_variable": "treatment",
  "outcome_variable": "outcome",
  "iteration_count": 0,
  "created_at": "2026-01-23T10:30:00Z",
  "updated_at": "2026-01-23T10:35:00Z",
  "error_message": null
}
```

#### List Jobs

```http
GET /api/v1/jobs?status={status}&limit={limit}&offset={offset}
```

**Query Parameters:**
- `status` (optional): Filter by status (pending, completed, failed, etc.)
- `limit` (optional): Max results (default: 20)
- `offset` (optional): Skip results (default: 0)

**Response:**
```json
{
  "jobs": [
    {
      "id": "a1b2c3d4",
      "status": "completed",
      "created_at": "2026-01-23T10:30:00Z",
      ...
    }
  ],
  "total": 42,
  "limit": 20,
  "offset": 0
}
```

#### Get Results

```http
GET /api/v1/jobs/{job_id}/results
```

**Response:**
```json
{
  "job_id": "a1b2c3d4",
  "treatment_variable": "treatment",
  "outcome_variable": "outcome",
  "data_profile": {
    "n_samples": 1000,
    "n_features": 15,
    "treatment_candidates": ["treatment", "intervention"],
    "outcome_candidates": ["outcome", "revenue"]
  },
  "causal_graph": {
    "nodes": ["treatment", "outcome", "age", "income"],
    "edges": [
      {"source": "age", "target": "treatment", "type": "directed"},
      {"source": "treatment", "target": "outcome", "type": "directed"}
    ]
  },
  "treatment_effects": [
    {
      "method": "psm",
      "estimand": "ATT",
      "estimate": 12.5,
      "std_error": 2.3,
      "ci_lower": 8.0,
      "ci_upper": 17.0,
      "p_value": 0.001
    }
  ],
  "sensitivity_results": [
    {
      "method": "rosenbaum_bounds",
      "robustness_value": 1.8,
      "interpretation": "Results robust to hidden bias up to Gamma=1.8"
    }
  ],
  "recommendations": [
    "Treatment has significant positive effect",
    "Results robust to moderate hidden bias"
  ]
}
```

#### Get Traces

```http
GET /api/v1/jobs/{job_id}/traces
```

**Response:**
```json
{
  "traces": [
    {
      "agent_name": "data_profiler",
      "timestamp": "2026-01-23T10:31:00Z",
      "action": "profile_dataset",
      "reasoning": "Analyzing dataset characteristics...",
      "inputs": {"url": "https://..."},
      "outputs": {"n_samples": 1000, "n_features": 15},
      "duration_ms": 2500
    }
  ]
}
```

#### Download Notebook

```http
GET /api/v1/jobs/{job_id}/notebook
```

**Response:**
- Content-Type: `application/x-ipynb+json`
- Content-Disposition: `attachment; filename="analysis_{job_id}.ipynb"`

---

## Database Schema

### Firestore Collections

#### jobs

```typescript
{
  id: string;                    // Job ID (8-char UUID)
  kaggle_url: string;            // Dataset URL
  dataset_name: string | null;   // Dataset name
  status: JobStatus;             // Current status
  treatment_variable: string | null;
  outcome_variable: string | null;
  iteration_count: number;       // Current iteration
  max_iterations: number;        // Max allowed iterations
  created_at: Timestamp;
  updated_at: Timestamp;
  completed_at: Timestamp | null;
  error_message: string | null;
  error_agent: string | null;
  notebook_path: string | null;  // GCS path
}
```

**Indexes:**
- `status` (for filtering)
- `created_at` (for sorting)

#### results

```typescript
{
  job_id: string;
  created_at: Timestamp;
  treatment_variable: string;
  outcome_variable: string;
  recommendations: string[];
  notebook_path: string;

  data_profile: {
    n_samples: number;
    n_features: number;
    treatment_candidates: string[];
    outcome_candidates: string[];
    has_time_dimension: boolean;
  };

  causal_graph: {
    nodes: string[];
    edges: Array<{
      source: string;
      target: string;
      type: string;
    }>;
    discovery_method: string;
  };

  treatment_effects: Array<{
    method: string;
    estimand: string;
    estimate: number;
    std_error: number;
    ci_lower: number;
    ci_upper: number;
    p_value: number;
    assumptions_tested: Record<string, boolean>;
  }>;

  sensitivity_results: Array<{
    method: string;
    robustness_value: number;
    interpretation: string;
  }>;
}
```

#### agent_traces/{job_id}/traces

```typescript
{
  agent_name: string;
  timestamp: Timestamp;
  action: string;
  reasoning: string;
  inputs: Record<string, any>;
  outputs: Record<string, any>;
  tools_called: string[];
  duration_ms: number;
}
```

---

## Infrastructure & Deployment

### Google Cloud Platform Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         GCP Project: plotpointe                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Cloud Run (us-central1)                                 │    │
│  │  ├─ causal-backend (0-10 instances)                     │    │
│  │  │   - CPU: 2 vCPU                                      │    │
│  │  │   - Memory: 4 GB                                     │    │
│  │  │   - Timeout: 300s                                    │    │
│  │  │   - Concurrency: 80                                  │    │
│  │  │                                                       │    │
│  │  └─ causal-frontend (0-5 instances)                     │    │
│  │      - CPU: 1 vCPU                                      │    │
│  │      - Memory: 512 MB                                   │    │
│  │      - Timeout: 60s                                     │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Firestore (Native Mode)                                 │    │
│  │  ├─ jobs (collection)                                   │    │
│  │  ├─ results (collection)                                │    │
│  │  └─ agent_traces (collection)                           │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Cloud Storage                                           │    │
│  │  ├─ causal-datasets (bucket)                            │    │
│  │  └─ causal-notebooks (bucket)                           │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Secret Manager                                          │    │
│  │  ├─ gemini-api-key (latest)                             │    │
│  │  ├─ claude-api-key (latest)                             │    │
│  │  └─ kaggle-key (latest)                                 │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Container Registry (gcr.io)                             │    │
│  │  ├─ causal-backend:{sha}                                │    │
│  │  └─ causal-frontend:{sha}                               │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Cloud Logging & Monitoring                              │    │
│  │  ├─ Structured logs (JSON)                              │    │
│  │  ├─ Metrics (requests, latency, errors)                 │    │
│  │  └─ Alerts (error rate, latency)                        │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Deployment Environments

| Environment | URL | Purpose | Auto-scaling |
|-------------|-----|---------|--------------|
| **Local** | localhost:8000 | Development | N/A |
| **Staging** | causal-backend-staging-*.run.app | Pre-production testing | 0-5 instances |
| **Production** | causal-backend-*.run.app | Live system | 1-10 instances |

### Cloud Run Configuration

**Backend Service:**
```yaml
service: causal-backend
region: us-central1
platform: managed
allow-unauthenticated: true

resources:
  cpu: 2
  memory: 4Gi

scaling:
  min-instances: 1  # Production
  max-instances: 10

timeout: 300s
concurrency: 80

env-vars:
  ENVIRONMENT: production
  GCP_PROJECT_ID: plotpointe

secrets:
  GEMINI_API_KEY: gemini-api-key:latest
  CLAUDE_API_KEY: claude-api-key:latest
  KAGGLE_KEY: kaggle-key:latest
```

**Frontend Service:**
```yaml
service: causal-frontend
region: us-central1
platform: managed
allow-unauthenticated: true

resources:
  cpu: 1
  memory: 512Mi

scaling:
  min-instances: 1
  max-instances: 5

timeout: 60s
```

### Manual Deployment

#### Prerequisites

1. **Install gcloud CLI:**
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init
```

2. **Authenticate:**
```bash
gcloud auth login
gcloud config set project plotpointe
```

3. **Enable APIs:**
```bash
gcloud services enable run.googleapis.com
gcloud services enable firestore.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable secretmanager.googleapis.com
```

#### Store Secrets

```bash
# Anthropic Claude API Key
echo -n "your-claude-api-key" | gcloud secrets create claude-api-key \
  --data-file=- \
  --replication-policy="automatic"

# Google Gemini API Key
echo -n "your-gemini-api-key" | gcloud secrets create gemini-api-key \
  --data-file=- \
  --replication-policy="automatic"

# Kaggle API Key
echo -n "your-kaggle-key" | gcloud secrets create kaggle-key \
  --data-file=- \
  --replication-policy="automatic"
```

#### Deploy Backend

```bash
cd causal-orchestrator/backend

gcloud run deploy causal-backend \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --cpu 2 \
  --memory 4Gi \
  --timeout 300 \
  --min-instances 1 \
  --max-instances 10 \
  --set-env-vars ENVIRONMENT=production,GCP_PROJECT_ID=plotpointe \
  --set-secrets CLAUDE_API_KEY=claude-api-key:latest,GEMINI_API_KEY=gemini-api-key:latest,KAGGLE_KEY=kaggle-key:latest
```

#### Deploy Frontend

```bash
cd causal-orchestrator/frontend

# Get backend URL
BACKEND_URL=$(gcloud run services describe causal-backend \
  --region us-central1 \
  --format 'value(status.url)')

gcloud run deploy causal-frontend \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --cpu 1 \
  --memory 512Mi \
  --timeout 60 \
  --min-instances 1 \
  --max-instances 5 \
  --set-env-vars VITE_API_URL=$BACKEND_URL
```

---

## CI/CD Pipeline

### GitHub Actions Workflows

#### Continuous Integration (.github/workflows/ci.yml)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main`

**Jobs:**

1. **backend-lint**
   - Runs Ruff (linter)
   - Runs MyPy (type checker, informational)

2. **backend-test**
   - Runs pytest with coverage
   - Uploads coverage to Codecov

3. **frontend-lint**
   - Runs ESLint
   - Runs TypeScript type checking

4. **frontend-build**
   - Builds production bundle
   - Uploads build artifact

5. **docker-build**
   - Builds Docker images (backend + frontend)
   - Uses GitHub Actions cache for layers

**Status Checks:**
All jobs must pass before merging to `main`.

#### Continuous Deployment (.github/workflows/cd.yml)

**Triggers:**
- Push to `main` branch
- Manual workflow dispatch

**Jobs:**

1. **deploy-staging**
   - Authenticate to GCP
   - Build and push Docker images to GCR
   - Deploy to Cloud Run (staging environment)
   - Set environment: `staging`
   - Load secrets from Secret Manager

2. **benchmark-validation**
   - Run benchmark tests against staging
   - Validate analysis quality
   - Timeout: 10 minutes per benchmark

3. **deploy-production**
   - Requires: `benchmark-validation` success
   - Deploy to Cloud Run (production environment)
   - Set environment: `production`
   - Min instances: 1 (always warm)

**Deployment Flow:**
```
main branch push
  ↓
Deploy to Staging
  ↓
Run Benchmarks
  ↓
Benchmarks Pass?
  ├─ Yes → Deploy to Production
  └─ No → Stop, notify team
```

### Required GitHub Secrets

```bash
# GCP Service Account Key (JSON)
GCP_SA_KEY

# API Keys (for testing)
GEMINI_API_KEY
CLAUDE_API_KEY
KAGGLE_KEY
```

### Setting Up CI/CD

1. **Create GCP Service Account:**
```bash
gcloud iam service-accounts create github-actions \
  --display-name="GitHub Actions"

# Grant permissions
gcloud projects add-iam-policy-binding plotpointe \
  --member="serviceAccount:github-actions@plotpointe.iam.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding plotpointe \
  --member="serviceAccount:github-actions@plotpointe.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

gcloud projects add-iam-policy-binding plotpointe \
  --member="serviceAccount:github-actions@plotpointe.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

# Create key
gcloud iam service-accounts keys create key.json \
  --iam-account=github-actions@plotpointe.iam.gserviceaccount.com
```

2. **Add Secrets to GitHub:**
   - Go to repository Settings → Secrets and variables → Actions
   - Add `GCP_SA_KEY` (paste contents of key.json)
   - Add `GEMINI_API_KEY`, `CLAUDE_API_KEY`, `KAGGLE_KEY`

3. **Enable Workflows:**
   - Workflows are automatically enabled when pushed to repository
   - Check Actions tab to monitor runs

---

## Security

### Security Measures

#### 1. API Key Management

**Storage:**
- ✅ Stored in GCP Secret Manager (encrypted at rest)
- ✅ Versioned (can rotate without downtime)
- ✅ Access controlled via IAM

**Usage:**
- ✅ Loaded as environment variables at runtime
- ✅ Never logged or exposed in responses
- ✅ Used via Pydantic `SecretStr` (prevents accidental logging)

**Example:**
```python
# settings.py
claude_api_key: SecretStr | None = Field(default=None)

# Access
api_key = settings.claude_api_key.get_secret_value()
```

#### 2. Input Validation

**Kaggle URL Validation:**
```python
from pydantic import HttpUrl, validator

class CreateJobRequest(BaseModel):
    kaggle_url: HttpUrl

    @validator('kaggle_url')
    def validate_kaggle_url(cls, v):
        if 'kaggle.com' not in str(v):
            raise ValueError('Must be a Kaggle URL')
        return v
```

**Data Sanitization:**
- All user inputs validated with Pydantic
- SQL injection: N/A (using Firestore, not SQL)
- XSS: Frontend sanitizes all rendered data

#### 3. Container Security

**Dockerfile Best Practices:**
- ✅ Non-root user in production
- ✅ Multi-stage builds (smaller attack surface)
- ✅ Minimal base image (python:3.11-slim)
- ✅ No secrets in image layers

```dockerfile
# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser
USER appuser
```

#### 4. Network Security

**Cloud Run:**
- ✅ HTTPS only (automatic TLS)
- ✅ DDoS protection (GCP infrastructure)
- ✅ Rate limiting (Cloud Armor, optional)

**CORS:**
```python
# Configured in main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,  # Whitelist
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

#### 5. Data Privacy

**Firestore Security Rules:**
```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Jobs: read-only for now (add auth later)
    match /jobs/{jobId} {
      allow read: if true;
      allow write: if false;  // Only backend can write
    }

    // Results: read-only
    match /results/{jobId} {
      allow read: if true;
      allow write: if false;
    }
  }
}
```

**Data Retention:**
- Jobs: Retained indefinitely (add TTL policy later)
- Logs: 30 days retention in Cloud Logging
- Notebooks: Stored in GCS (lifecycle policy: delete after 90 days)

#### 6. Dependency Security

**Automated Scanning:**
- Dependabot enabled (GitHub)
- Scans for known vulnerabilities
- Auto-creates PRs for updates

**Manual Audits:**
```bash
# Python
pip-audit

# Node.js
npm audit
```

### Security Checklist

- [x] API keys stored in Secret Manager
- [x] Secrets never logged
- [x] Input validation with Pydantic
- [x] Non-root container user
- [x] HTTPS only
- [x] CORS whitelist
- [x] Dependency scanning
- [ ] **TODO:** Add authentication (API keys or OAuth2)
- [ ] **TODO:** Add rate limiting per user
- [ ] **TODO:** Add audit logging
- [ ] **TODO:** Add data encryption at rest (Firestore default)

### Known Security Gaps

1. **No Authentication:**
   - Current: Anyone can create jobs
   - Risk: Abuse, cost overruns
   - Mitigation: Add API key auth or OAuth2

2. **No Rate Limiting:**
   - Current: Unlimited requests
   - Risk: DDoS, cost overruns
   - Mitigation: Add Cloud Armor rate limiting

3. **Public Read Access:**
   - Current: Anyone can read any job results
   - Risk: Data leakage
   - Mitigation: Add user-based access control

---

## Configuration

### Environment Variables

#### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `CLAUDE_API_KEY` | Anthropic Claude API key | `sk-ant-...` |
| `GEMINI_API_KEY` | Google Gemini API key | `AIza...` |
| `KAGGLE_KEY` | Kaggle API key | `abc123...` |
| `KAGGLE_USERNAME` | Kaggle username | `siddharth47007` |

#### Optional

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Environment name | `development` |
| `LLM_PROVIDER` | LLM provider (claude/gemini/vertex) | `claude` |
| `GCP_PROJECT_ID` | GCP project ID | `plotpointe` |
| `GCP_REGION` | GCP region | `us-central1` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `MAX_AGENT_ITERATIONS` | Max critique iterations | `3` |
| `AGENT_TIMEOUT_SECONDS` | Agent timeout | `300` |
| `CORS_ORIGINS` | Allowed CORS origins | `http://localhost:5173` |

#### Frontend

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_URL` | Backend API URL | `http://localhost:8000` |

### Configuration Files

#### .env (Local Development)

```bash
# Copy from .env.example
cp .env.example .env

# Edit with your values
CLAUDE_API_KEY=sk-ant-your-key-here
GEMINI_API_KEY=AIza-your-key-here
KAGGLE_KEY=your-kaggle-key
KAGGLE_USERNAME=your-username

ENVIRONMENT=development
LLM_PROVIDER=claude
LOG_LEVEL=DEBUG
```

#### settings.py (Pydantic Settings)

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Loads from environment variables or .env file
    claude_api_key: SecretStr | None = Field(default=None)
    gemini_api_key: SecretStr | None = Field(default=None)
    # ...
```

### LLM Provider Configuration

#### Claude (Anthropic)

```python
# settings.py
llm_provider = "claude"
claude_api_key = "sk-ant-..."
claude_model = "claude-sonnet-4-20250514"
claude_temperature = 0.1
claude_max_tokens = 8192
```

**Advantages:**
- Latest model (Sonnet 4)
- Excellent reasoning
- Function calling support

**Limitations:**
- Requires API key
- Rate limits (tier-based)

#### Gemini (Google AI)

```python
# settings.py
llm_provider = "gemini"
gemini_api_key = "AIza..."
gemini_model = "gemini-1.5-pro"
gemini_temperature = 0.1
gemini_max_tokens = 8192
```

**Advantages:**
- Free tier available
- Good performance
- Function calling support

**Limitations:**
- Rate limits on free tier
- Requires API key

#### Vertex AI (GCP)

```python
# settings.py
llm_provider = "vertex"
vertex_model = "gemini-2.0-flash-exp"
gcp_project_id = "plotpointe"
```

**Advantages:**
- No API key needed (uses ADC)
- Integrated with GCP
- Experimental models

**Limitations:**
- Requires GCP project
- Experimental model (rate limits)
- May have stability issues

---

## Testing Strategy

### Test Pyramid

```
        ┌─────────────┐
        │   E2E (5%)  │  Benchmark tests
        ├─────────────┤
        │ Integration │  Job pipeline tests
        │    (15%)    │
        ├─────────────┤
        │    Unit     │  Agent, method, API tests
        │    (80%)    │
        └─────────────┘
```

### Test Structure

```
tests/
├── unit/                    # Fast, isolated tests
│   ├── test_agents.py      # Agent logic
│   ├── test_causal/        # Causal methods
│   ├── test_api.py         # API endpoints
│   └── test_config.py      # Configuration
├── integration/             # Multi-component tests
│   └── test_job_pipeline.py # Full job flow
├── agentic/                 # Agent behavior tests
└── conftest.py              # Pytest fixtures
```

### Running Tests

```bash
cd backend

# All tests
pytest

# Unit tests only
pytest tests/unit -v

# Integration tests
pytest tests/integration -v

# With coverage
pytest --cov=src --cov-report=html

# Specific test
pytest tests/unit/test_agents.py::test_orchestrator_dispatch -v

# Benchmarks
pytest benchmarks/runners/test_benchmark.py -v --timeout=600
```

### Test Examples

#### Unit Test (Agent)

```python
import pytest
from src.agents import DataProfilerAgent, AnalysisState

@pytest.mark.asyncio
async def test_data_profiler_identifies_treatment():
    agent = DataProfilerAgent()
    state = AnalysisState(
        job_id="test123",
        dataset_info=DatasetInfo(url="https://..."),
        status=JobStatus.PENDING,
    )

    result = await agent.execute(state)

    assert result.data_profile is not None
    assert len(result.data_profile.treatment_candidates) > 0
    assert result.status == JobStatus.PROFILING
```

#### Integration Test (Job Pipeline)

```python
@pytest.mark.asyncio
async def test_full_job_pipeline():
    manager = get_job_manager()

    job_id = await manager.create_job(
        kaggle_url="https://www.kaggle.com/datasets/test/dataset"
    )

    # Wait for completion (with timeout)
    for _ in range(60):
        job = await manager.get_job(job_id)
        if job['status'] in ['completed', 'failed']:
            break
        await asyncio.sleep(5)

    assert job['status'] == 'completed'

    results = await manager.get_results(job_id)
    assert len(results['treatment_effects']) > 0
```

#### Benchmark Test

```python
@pytest.mark.benchmark
@pytest.mark.timeout(600)
async def test_lalonde_benchmark():
    """Test on LaLonde NSW dataset (canonical causal inference benchmark)."""
    manager = get_job_manager()

    job_id = await manager.create_job(
        kaggle_url="https://www.kaggle.com/datasets/seanlahman/lalonde-nsw",
        treatment_variable="treat",
        outcome_variable="re78"
    )

    # Wait for completion
    job = await wait_for_job(job_id, timeout=600)
    assert job['status'] == 'completed'

    results = await manager.get_results(job_id)

    # Known ground truth: ATE ≈ $1,794
    ate_estimates = [e for e in results['treatment_effects'] if e['estimand'] == 'ATE']
    assert len(ate_estimates) > 0

    # Check if any estimate is within reasonable range
    ground_truth = 1794
    tolerance = 1000
    close_estimates = [e for e in ate_estimates
                      if abs(e['estimate'] - ground_truth) < tolerance]
    assert len(close_estimates) > 0, "No estimates close to ground truth"
```

---

## Monitoring & Observability

### Structured Logging

**Log Format (JSON):**
```json
{
  "timestamp": "2026-01-23T10:30:00.123Z",
  "severity": "INFO",
  "message": "Job created",
  "job_id": "a1b2c3d4",
  "agent": "orchestrator",
  "trace_id": "abc123...",
  "labels": {
    "environment": "production",
    "service": "causal-backend"
  }
}
```

**Log Levels:**
- `DEBUG` - Detailed diagnostic info (development only)
- `INFO` - General informational messages
- `WARNING` - Warning messages (non-critical issues)
- `ERROR` - Error messages (handled exceptions)
- `CRITICAL` - Critical errors (system failures)

**Key Log Points:**
```python
# Job lifecycle
logger.info("Job created", extra={"job_id": job_id})
logger.info("Agent dispatched", extra={"job_id": job_id, "agent": agent_name})
logger.info("Job completed", extra={"job_id": job_id, "duration_ms": duration})

# Errors
logger.error("Agent failed", extra={"job_id": job_id, "agent": agent_name, "error": str(e)})

# LLM calls
logger.debug("LLM request", extra={"model": model, "tokens": token_count})
```

### Metrics

**Cloud Monitoring Metrics:**

1. **Request Metrics:**
   - `request_count` - Total requests
   - `request_duration` - Request latency (p50, p95, p99)
   - `request_size` - Request payload size

2. **Job Metrics:**
   - `job_created_count` - Jobs created
   - `job_completed_count` - Jobs completed
   - `job_failed_count` - Jobs failed
   - `job_duration` - Job execution time

3. **Agent Metrics:**
   - `agent_execution_count` - Agent executions
   - `agent_duration` - Agent execution time
   - `agent_error_count` - Agent failures

4. **LLM Metrics:**
   - `llm_request_count` - LLM API calls
   - `llm_token_count` - Tokens consumed
   - `llm_latency` - LLM response time
   - `llm_error_count` - LLM failures

5. **Resource Metrics:**
   - `cpu_utilization` - CPU usage
   - `memory_utilization` - Memory usage
   - `instance_count` - Active instances

### Alerts

**Recommended Alerts:**

1. **High Error Rate:**
   - Condition: Error rate > 5% over 5 minutes
   - Severity: Critical
   - Action: Page on-call engineer

2. **High Latency:**
   - Condition: p95 latency > 30s over 5 minutes
   - Severity: Warning
   - Action: Notify team

3. **Job Failures:**
   - Condition: Job failure rate > 10% over 15 minutes
   - Severity: Warning
   - Action: Notify team

4. **LLM Errors:**
   - Condition: LLM error rate > 1% over 5 minutes
   - Severity: Warning
   - Action: Check API key, rate limits

5. **Resource Exhaustion:**
   - Condition: Memory usage > 90% for 5 minutes
   - Severity: Critical
   - Action: Scale up instances

### Tracing

**Agent Execution Traces:**

Stored in Firestore for debugging and analysis:

```python
{
  "agent_name": "effect_estimator",
  "timestamp": "2026-01-23T10:35:00Z",
  "action": "estimate_effects",
  "reasoning": "Selected PSM and IPW based on selection bias indicators",
  "inputs": {
    "treatment": "treatment",
    "outcome": "outcome",
    "confounders": ["age", "income"]
  },
  "outputs": {
    "methods_used": ["psm", "ipw"],
    "ate_psm": 12.5,
    "ate_ipw": 13.2
  },
  "tools_called": ["estimate_psm", "estimate_ipw"],
  "duration_ms": 8500
}
```

**Viewing Traces:**
```bash
# Via API
curl https://causal-backend-*.run.app/api/v1/jobs/{job_id}/traces

# Via gcloud
gcloud firestore export gs://backup-bucket/traces \
  --collection-ids=agent_traces
```

---

## Development Guide

### Local Setup

#### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- Git

#### Clone Repository

```bash
git clone https://github.com/yourusername/causal-orchestrator.git
cd causal-orchestrator
```

#### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Dev dependencies

# Copy environment template
cp ../.env.example .env

# Edit .env with your API keys
nano .env

# Run tests
pytest

# Start development server
uvicorn src.api.main:app --reload --port 8000
```

#### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Copy environment template
cp .env.example .env.local

# Edit .env.local
echo "VITE_API_URL=http://localhost:8000" > .env.local

# Start development server
npm run dev
```

#### Docker Compose (Recommended)

```bash
# From project root
docker-compose up --build

# Backend: http://localhost:8000
# Frontend: http://localhost:5173
```

### Code Style

#### Python (Backend)

**Linter:** Ruff
```bash
ruff check src/
ruff format src/
```

**Type Checker:** MyPy
```bash
mypy src/
```

**Style Guide:**
- PEP 8 compliant
- Type hints required
- Docstrings for public functions (Google style)
- Max line length: 100 characters

**Example:**
```python
async def estimate_treatment_effect(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    method: str = "psm",
) -> TreatmentEffect:
    """Estimate treatment effect using specified method.

    Args:
        data: Input dataset
        treatment: Treatment variable name
        outcome: Outcome variable name
        method: Causal inference method (default: psm)

    Returns:
        TreatmentEffect object with estimate and confidence interval

    Raises:
        ValueError: If method is not supported
    """
    # Implementation
```

#### TypeScript (Frontend)

**Linter:** ESLint
```bash
npm run lint
npm run lint:fix
```

**Type Checker:** TypeScript
```bash
npm run type-check
```

**Style Guide:**
- Airbnb style guide
- Functional components with hooks
- Type everything (no `any`)
- Props interfaces for components

**Example:**
```typescript
interface JobCardProps {
  job: Job;
  onSelect: (jobId: string) => void;
}

export const JobCard: React.FC<JobCardProps> = ({ job, onSelect }) => {
  const handleClick = () => {
    onSelect(job.id);
  };

  return (
    <div onClick={handleClick}>
      {/* Component JSX */}
    </div>
  );
};
```

### Git Workflow

**Branching Strategy:**
```
main (production)
  \u2514\u2500 develop (staging)
      \u251c\u2500 feature/add-new-method
      \u251c\u2500 fix/bug-in-psm
      \u2514\u2500 docs/update-readme
```

**Commit Messages:**
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation
- `style` - Code style (formatting)
- `refactor` - Code refactoring
- `test` - Tests
- `chore` - Build/tooling

**Example:**
```
feat(agents): add causal forest method

Implement causal forest for heterogeneous treatment effects.
Uses econml's CausalForestDML estimator.

Closes #42
```

### Pull Request Process

1. **Create Feature Branch:**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make Changes:**
   - Write code
   - Add tests
   - Update documentation

3. **Run Tests Locally:**
   ```bash
   pytest
   npm test
   ```

4. **Commit Changes:**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

5. **Push to GitHub:**
   ```bash
   git push origin feature/my-feature
   ```

6. **Create Pull Request:**
   - Go to GitHub
   - Click "New Pull Request"
   - Fill in description
   - Request review

7. **CI Checks:**
   - Wait for CI to pass
   - Fix any failures

8. **Code Review:**
   - Address reviewer comments
   - Push updates

9. **Merge:**
   - Squash and merge to `develop`
   - Delete feature branch

---

## Troubleshooting

### Common Issues

#### 1. Job Stuck in "pending" Status

**Symptoms:**
- Job created but never progresses
- No agent traces

**Causes:**
- Orchestrator not starting
- LLM API key invalid
- Firestore connection issue

**Solutions:**
```bash
# Check logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=causal-backend" --limit 50

# Verify API key
gcloud secrets versions access latest --secret=claude-api-key

# Test Firestore connection
python -c "from google.cloud import firestore; db = firestore.Client(); print(db.collection('jobs').limit(1).get())"
```

#### 2. LLM Rate Limit Errors

**Symptoms:**
- Jobs failing with "rate limit exceeded"
- 429 errors in logs

**Causes:**
- Too many concurrent jobs
- API tier limits

**Solutions:**
```python
# Reduce concurrency in settings.py
MAX_CONCURRENT_JOBS = 2

# Add retry logic with exponential backoff
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def call_llm(...):
    # LLM call
```

#### 3. Memory Errors (OOM)

**Symptoms:**
- Jobs failing with "out of memory"
- Container restarts

**Causes:**
- Large datasets
- Memory leaks
- Insufficient Cloud Run memory

**Solutions:**
```bash
# Increase Cloud Run memory
gcloud run services update causal-backend --memory 8Gi

# Add dataset size limits
MAX_DATASET_SIZE_MB = 100

# Profile memory usage
python -m memory_profiler src/agents/orchestrator/orchestrator_agent.py
```

#### 4. Kaggle Download Failures

**Symptoms:**
- Jobs failing at "fetching_data" stage
- "Dataset not found" errors

**Causes:**
- Invalid Kaggle URL
- Private dataset
- Kaggle API credentials invalid

**Solutions:**
```bash
# Verify Kaggle credentials
kaggle datasets list --user yourusername

# Check dataset accessibility
kaggle datasets download -d username/dataset-name

# Update Kaggle credentials in Secret Manager
echo -n "new-kaggle-key" | gcloud secrets versions add kaggle-key --data-file=-
```

#### 5. Frontend Not Connecting to Backend

**Symptoms:**
- CORS errors in browser console
- API requests failing

**Causes:**
- Incorrect `VITE_API_URL`
- CORS not configured
- Backend not running

**Solutions:**
```bash
# Check backend URL
echo $VITE_API_URL

# Update .env.local
echo "VITE_API_URL=http://localhost:8000" > frontend/.env.local

# Verify CORS settings in backend/src/api/main.py
# Add frontend URL to allow_origins
```

### Debugging Tips

**Enable Debug Logging:**
```bash
# Backend
export LOG_LEVEL=DEBUG
uvicorn src.api.main:app --reload

# View logs
tail -f logs/app.log
```

**Inspect Job State:**
```python
from src.storage.firestore import FirestoreStorage

storage = FirestoreStorage()
job = storage.get_job("job_id")
print(job)
```

**Test Individual Agents:**
```python
from src.agents import DataProfilerAgent, AnalysisState

agent = DataProfilerAgent()
state = AnalysisState(...)
result = await agent.execute(state)
print(result)
```

**Check Cloud Run Logs:**
```bash
gcloud logging read "resource.type=cloud_run_revision" \
  --limit 100 \
  --format json \
  --project plotpointe
```

---

## Benchmark Datasets

The system is validated against 8 canonical causal inference benchmark datasets:

### 1. LaLonde NSW (1986)

**Description:** National Supported Work (NSW) job training program

**Variables:**
- Treatment: `treat` (job training)
- Outcome: `re78` (earnings in 1978)
- Confounders: age, education, race, marital status, prior earnings

**Ground Truth:** ATE ≈ $1,794

**Methods:** PSM, IPW, AIPW

**Source:** https://www.kaggle.com/datasets/seanlahman/lalonde-nsw

### 2. IHDP (Infant Health and Development Program)

**Description:** Early childhood intervention program

**Variables:**
- Treatment: `treat` (intervention)
- Outcome: `iq` (IQ score at age 3)
- Confounders: birth weight, mother's age, mother's education

**Ground Truth:** ATE ≈ 4.0 IQ points

**Methods:** PSM, T-Learner, Causal Forest

**Source:** https://www.fredjo.com/

### 3. Twins Dataset (Almond et al. 2005)

**Description:** Effect of birth weight on mortality

**Variables:**
- Treatment: `low_birth_weight` (<2500g)
- Outcome: `mortality` (1-year mortality)
- Confounders: gestational age, mother's age, prenatal care

**Methods:** IV (twin fixed effects), PSM

**Source:** https://www.nber.org/data/

### 4. Card-Krueger Minimum Wage (1994)

**Description:** Effect of minimum wage increase on employment

**Variables:**
- Treatment: `min_wage_increase` (NJ vs PA)
- Outcome: `employment_change`
- Time: Before/after 1992

**Ground Truth:** No significant negative effect

**Methods:** DiD

**Source:** https://davidcard.berkeley.edu/data_sets.html

### 5. Angrist-Evans Fertility (1998)

**Description:** Effect of children on labor supply

**Variables:**
- Treatment: `more_than_2_children`
- Outcome: `weeks_worked`
- Instrument: `same_sex` (first two children same sex)

**Methods:** IV/2SLS

**Source:** https://economics.mit.edu/people/faculty/josh-angrist/angrist-data-archive

### 6. Lee RDD (2008)

**Description:** Incumbency advantage in elections

**Variables:**
- Treatment: `win_election`
- Outcome: `vote_share_next_election`
- Running variable: `vote_margin`
- Cutoff: 0

**Methods:** RDD (sharp)

**Source:** https://economics.mit.edu/people/faculty/josh-angrist/angrist-data-archive

### 7. ACIC 2016 Competition

**Description:** Synthetic datasets with known ground truth

**Variables:**
- Treatment: Binary
- Outcome: Continuous
- Confounders: 58 variables

**Ground Truth:** Known SATE (Sample ATE)

**Methods:** All methods

**Source:** https://www.synth-inference.com/

### 8. Criteo Uplift Modeling

**Description:** Online advertising uplift

**Variables:**
- Treatment: `ad_exposure`
- Outcome: `conversion`
- Confounders: user features, context

**Methods:** Meta-learners, Causal Forest

**Source:** https://ailab.criteo.com/criteo-uplift-prediction-dataset/

---

## Performance Optimization

### Scaling Considerations

**Current Limits:**
- Max concurrent jobs: 10 (Cloud Run max instances)
- Max dataset size: ~500MB (memory constraints)
- Max job duration: 300s (Cloud Run timeout)

**Scaling Strategies:**

1. **Horizontal Scaling:**
   - Increase Cloud Run max instances
   - Add load balancer for multiple regions
   - Use Cloud Tasks for job queue

2. **Vertical Scaling:**
   - Increase memory (4Gi → 8Gi)
   - Increase CPU (2 → 4 vCPU)

3. **Async Processing:**
   - Move long-running jobs to Cloud Run Jobs
   - Use Pub/Sub for job notifications

### Caching Strategies

**Dataset Caching:**
```python
# Cache downloaded datasets in Cloud Storage
cache_key = hashlib.md5(kaggle_url.encode()).hexdigest()
cache_path = f"gs://causal-datasets/cache/{cache_key}.csv"

if storage.exists(cache_path):
    df = pd.read_csv(cache_path)
else:
    df = download_from_kaggle(kaggle_url)
    df.to_csv(cache_path)
```

**LLM Response Caching:**
```python
# Cache LLM responses for identical inputs
@lru_cache(maxsize=100)
def get_llm_response(prompt: str, model: str) -> str:
    return llm_client.generate(prompt, model)
```

**Propensity Score Caching:**
```python
# Cache propensity scores for reuse across methods
state.propensity_scores = estimate_propensity(data, treatment, confounders)
# Reuse in PSM, IPW, AIPW
```

### Database Optimization

**Firestore Indexes:**
```yaml
indexes:
  - collectionGroup: jobs
    queryScope: COLLECTION
    fields:
      - fieldPath: status
        order: ASCENDING
      - fieldPath: created_at
        order: DESCENDING
```

**Batch Writes:**
```python
# Batch trace writes
batch = db.batch()
for trace in traces:
    ref = db.collection('agent_traces').document(job_id).collection('traces').document()
    batch.set(ref, trace.dict())
batch.commit()
```

---

## Future Enhancements

### Planned Features

1. **Authentication & Authorization**
   - API key authentication
   - User accounts
   - Job ownership
   - Rate limiting per user

2. **Advanced Causal Methods**
   - Synthetic control
   - Bayesian structural time series
   - Mediation analysis
   - Interference/spillover effects

3. **Interactive Notebook**
   - In-browser Jupyter notebook
   - Live code execution
   - Collaborative editing

4. **Model Registry**
   - Save trained models
   - Model versioning
   - A/B testing of methods

5. **Data Connectors**
   - BigQuery integration
   - S3/GCS direct upload
   - Database connectors (PostgreSQL, MySQL)

6. **Visualization Enhancements**
   - Interactive DAG editor
   - Treatment effect heterogeneity plots
   - Sensitivity analysis visualizations

7. **Explainability**
   - SHAP values for treatment effects
   - Feature importance
   - Counterfactual explanations

8. **Multi-treatment Analysis**
   - Multiple treatments simultaneously
   - Treatment combinations
   - Optimal treatment assignment

### Roadmap

**Q1 2026:**
- [ ] Add authentication
- [ ] Implement rate limiting
- [ ] Add synthetic control method

**Q2 2026:**
- [ ] Interactive notebook
- [ ] BigQuery connector
- [ ] Model registry

**Q3 2026:**
- [ ] Advanced visualizations
- [ ] SHAP explainability
- [ ] Multi-treatment support

**Q4 2026:**
- [ ] Mediation analysis
- [ ] Spillover effects
- [ ] Collaborative features

---

## Appendix

### Glossary

**ATE (Average Treatment Effect):** Average effect of treatment across entire population

**ATT (Average Treatment Effect on Treated):** Average effect among those who received treatment

**CATE (Conditional Average Treatment Effect):** Treatment effect conditional on covariates

**Confounder:** Variable that affects both treatment and outcome

**DAG (Directed Acyclic Graph):** Graphical representation of causal relationships

**Estimand:** The quantity we want to estimate (e.g., ATE, ATT)

**Heterogeneous Effects:** Treatment effects that vary across individuals

**LATE (Local Average Treatment Effect):** Effect for compliers in IV analysis

**Propensity Score:** Probability of receiving treatment given covariates

**Unconfoundedness:** Assumption that all confounders are observed

### References

**Causal Inference:**
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*
- Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*

**Methods:**
- Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity score in observational studies
- Athey, S., & Imbens, G. (2016). Recursive partitioning for heterogeneous causal effects
- Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural parameters

**Software:**
- DoWhy: https://github.com/py-why/dowhy
- EconML: https://github.com/py-why/EconML
- CausalML: https://github.com/uber/causalml

### Contact & Support

**Repository:** https://github.com/yourusername/causal-orchestrator

**Issues:** https://github.com/yourusername/causal-orchestrator/issues

**Discussions:** https://github.com/yourusername/causal-orchestrator/discussions

**Email:** support@causalorchestrator.com

---

*Last Updated: 2026-01-23*
*Version: 1.0.0*
*Documentation Generated by: Augment Agent*








```




