# Causal Inference Orchestrator — DESIGN.md

This is a working design document for me, written to be the map I use when I fix the system. It is not a marketing doc, not a tutorial, and not a generic AI-app template. It is specific to this codebase as it stands today.

The system: a multi-agent pipeline that takes a Kaggle dataset URL and emits a reproducible Jupyter notebook with treatment-effect estimates, a discovered causal DAG, sensitivity analysis, and a written audit trail. The interesting design surface is the agent layer; the storage, auth, and HTTP surfaces are conventional.

---

## 1. Goal

Build a system where a non-specialist data scientist can submit a tabular dataset and get back a defensible causal-inference report in one run. The report is not just an effect number; it is a peer-reviewed argument with adjustment-set justification, multiple estimators triangulated, sensitivity bounds, and a critique trail. The pipeline takes the long causal-inference workflow (profile, repair, discover DAG, identify confounders, estimate, sensitivity, write up) and runs the whole thing under LLM coordination so the human does not have to drive each stage by hand.

The architectural commitment that distinguishes this from a single LLM call is twofold: (1) **specialist agents** with bounded responsibilities so each stage is independently evaluable, and (2) a **critique loop** that can send the analysis back for another pass before it ships. Together these mean the system can produce defensible output, not just plausible output.

### Scope caps (deliberately out of scope)

- Not a streaming-data pipeline. Datasets are bounded snapshots (Kaggle CSVs).
- Not a multi-tenant SaaS. Single-user today; auth is optional API key, no user model.
- Not a model trainer. We use pre-built statistical estimators and discovery algorithms.
- Not a general-purpose agent framework. The agent base classes are tuned for causal-inference workflow.
- Not an interactive notebook editor. The notebook is a final artifact, not edited in-app.
- Not a real-time service. Jobs are minutes-long async runs, not request-response.
- Not RAG. There is no vector store and no document corpus. The "context" is the job's state object, pulled by tools.

---

## 2. Resources

The nouns of the system. Each row below is a real entity that appears in the code, plus where it lives.

| Entity | What it is | Where it lives | Lifecycle |
|---|---|---|---|
| Job | One end-to-end analysis run | Firestore doc / local JSON file | Created by `POST /jobs`, terminal at COMPLETED, FAILED, or CANCELLED |
| AnalysisState | The working memory of a job. Every agent reads from and writes to it. | Same store as Job (one doc per job) | Lives for the duration of the job, persisted on each agent boundary |
| Dataset | A Kaggle CSV downloaded for the job | Filesystem at `LOCAL_STORAGE_PATH/{job_id}/data.parquet` | Materialized once during data_profiler stage |
| Agent | Specialist or orchestrator class instance | Backend Python memory, instantiated per job | Created by `create_all_agents()`, garbage-collected when job ends |
| AgentTrace | Per-step record (action, reasoning, tool calls, duration, tokens) | Inside `AnalysisState.agent_traces` list | Append-only during job |
| TreatmentEffect | One method's estimate (method, estimand, estimate, CI, SE, p-value, details) | Inside `AnalysisState.treatment_effects` list | Written by effect_estimator |
| CausalDAG | Nodes, edges, adjustment set, variable roles, forbidden edges | `AnalysisState.proposed_dag` | Written by causal_discovery, refined by dag_expert |
| CritiqueFeedback | Decision, scores, issues, improvements, reasoning | Inside `AnalysisState.critique_history` list | One entry per critique pass, up to 3 |
| Notebook | The final `.ipynb` artifact | Filesystem at `LOCAL_STORAGE_PATH/{job_id}/notebook.ipynb`, path stored in state | Generated once at end of job |
| LLM call | An invocation of Vertex / Claude / Gemini | External provider, response logged | Per agent step |
| Job slot | Concurrency permit | Firestore semaphore + local asyncio.Semaphore | Acquired at job start, released on terminal status |
| User | Whoever submits the job | Implicit. No user table. | N/A |

### Schemas for the load-bearing entities

**Job (as returned by `GET /jobs/{id}`)**

```python
class JobResponse:
    job_id: str
    status: JobStatus  # PENDING, RUNNING, GENERATING_NOTEBOOK, COMPLETED, FAILED, CANCELLED
    dataset_url: str
    created_at: datetime
    updated_at: datetime
    iteration_count: int        # critique loops consumed (0..3)
    error: str | None
    notebook_path: str | None   # set when COMPLETED
```

**AnalysisState (the working memory, abbreviated)**

```python
class AnalysisState:
    job_id: str
    status: JobStatus
    iteration_count: int
    max_iterations: int = 3

    dataset_info: DatasetInfo            # url, name, n_samples hint, kaggle metadata
    raw_metadata: dict | None            # full Kaggle metadata blob
    dataframe_path: str | None           # parquet path

    domain_knowledge: dict | None        # hypotheses, uncertainties, immutable_vars
    data_profile: DataProfile | None     # types, candidates, stats
    eda_result: EDAResult | None         # quality, correlations, balance, outliers
    proposed_dag: CausalDAG | None       # nodes, edges, adjustment_set, variable_roles
    confounder_discovery: dict | None    # ranked confounders, adjustment_strategy
    ps_diagnostics: PSDiagnostics | None
    treatment_effects: list[TreatmentEffect]
    sensitivity_results: list[SensitivityResult]
    critique_history: list[CritiqueFeedback]
    notebook_path: str | None

    agent_traces: list[AgentTrace]       # append-only
    data_repairs: list[Repair]           # append-only

    treatment_variable: str | None
    outcome_variable: str | None
    analyzed_pairs: list[tuple[str, str]]
```

**TreatmentEffect**

```python
class TreatmentEffect:
    method: str          # "OLS", "IPW", "AIPW", ...
    estimand: str        # "ATE", "ATT", "CATE"
    estimate: float
    std_error: float
    ci_lower: float
    ci_upper: float
    p_value: float | None
    details: dict        # method-specific (PS overlap, R^2, ...)
    assumptions_tested: list[str]
```

**CausalDAG**

```python
class CausalDAG:
    nodes: list[str]
    edges: list[Edge]                 # (source, target, edge_type)
    discovery_method: str             # "PC", "FCI", "GES", "NOTEARS", "LiNGAM", or "ensemble"
    treatment_variable: str | None
    outcome_variable: str | None
    adjustment_set: list[str] | None  # set by dag_expert
    variable_roles: dict | None       # confounder, mediator, collider, ...
    forbidden_edges: list[Edge]
    interpretation: str | None
```

**CritiqueFeedback**

```python
class CritiqueFeedback:
    decision: CritiqueDecision  # APPROVE, ITERATE, REJECT
    iteration: int
    scores: dict[str, int]      # six dimensions, 1..5
    issues: list[str]
    improvements: list[str]
    reasoning: str
```

---

## 3. Core flows

### Happy path: Kaggle URL to notebook

1. Client submits `POST /jobs` with `{ "dataset_url": "https://www.kaggle.com/datasets/.../" }`.
2. Route handler validates request, checks API key if configured, applies `@limiter.limit("10/minute")`.
3. `JobManager.create_job` acquires the local `asyncio.Semaphore` (`MAX_CONCURRENT_JOBS`) and increments the Firestore-backed distributed counter. If either is full, returns 503.
4. A new `AnalysisState` is built with `status=PENDING`, persisted to Firestore (or local JSON in dev).
5. `JobManager._run_job_task` is launched as a background asyncio task. The semaphore is released on terminal status, not on return.
6. The runner loads the dataset from Kaggle (via `src.kaggle`), writes parquet to disk, and stores the path in state.
7. The orchestrator (`StandardOrchestrator` by default, or `ReActOrchestrator` if `ORCHESTRATOR_MODE=react`) begins reasoning. It calls `_build_context_prompt(state)` to get a state summary, sends it to the LLM with the two dispatch tools as available functions, and receives a tool call: typically `dispatch_to_agent("domain_knowledge", ...)` if metadata is available.
8. The dispatched agent runs its own ReAct loop: gets a lean initial observation from `_get_initial_observation(state)`, then iteratively reasons, calls pull-based context tools (e.g., `ask_domain_knowledge`, `get_eda_finding`), and writes results into `AnalysisState`. Each step is recorded as an `AgentTrace` and emitted over SSE if `SSE_ENABLED=true`.
9. Control returns to the orchestrator. It rebuilds the context prompt (now with `state.domain_knowledge` populated) and decides the next dispatch. The default suggested order is: domain_knowledge → data_profiler → (eda_agent || causal_discovery in parallel) → dag_expert → confounder_discovery → ps_diagnostics → effect_estimator → sensitivity_analyst → critique.
10. After sensitivity, the orchestrator calls `request_critique`. The critique agent runs as its own agentic loop with investigation tools (`get_analysis_summary`, `check_covariate_balance`, etc.) and emits a `CritiqueFeedback` with decision = APPROVE, ITERATE, or REJECT.
11. On APPROVE, the orchestrator dispatches `notebook_generator`. The generator reads every relevant state field and writes a 15-section `.ipynb` to disk; the path is stored in `state.notebook_path`.
12. Job is marked COMPLETED. Semaphore is released. SSE stream emits a `done` event and closes.
13. Client polls `GET /jobs/{id}/notebook` to download.

### Alternate path A: Critique iterates

After step 10, decision is ITERATE. The orchestrator stores the feedback in `_critique_feedback_for_prompt` and re-dispatches the effect_estimator (or whichever agents the feedback points at). The estimator's initial observation includes the prior critique. After re-running, critique fires again. This loop is bounded by `MAX_AGENT_ITERATIONS` (default 3); on the 3rd ITERATE the system either auto-approves with a warning (heuristic fallback) or finalizes with best effort.

### Alternate path B: Per-agent timeout

Each agent runs under `AGENT_TIMEOUT_SECONDS` (default 300). If exceeded, the agent's task is cancelled, an error trace is appended, and the orchestrator sees the error in state. Depending on which agent timed out, it may skip (eda_agent timeout is non-fatal) or fail the job (data_profiler timeout is fatal because no downstream agent can run without a profile).

### Alternate path C: User cancellation

Client calls `POST /jobs/{id}/cancel`. `JobManager` looks up the running task in `_running_jobs`, calls `.cancel()`, marks the state CANCELLED, releases the semaphore. The current agent step terminates at the next await boundary. Partial state is preserved. The notebook is not generated.

### Alternate path D: Capacity exhausted

`POST /jobs` arrives when both the local semaphore and the Firestore counter are at `MAX_CONCURRENT_JOBS`. The route returns 503 with `Retry-After`. No job document is created; the client is expected to back off and retry.

### Alternate path E: LLM provider error

A specialist's LLM call raises (rate limit, network, 5xx). The agent's `_reason` catches and records the error. ReAct retries up to `MAX_CONSECUTIVE_ERRORS=3` per step. If the third retry fails, the agent exits with a partial result. The orchestrator sees the gap in state and either dispatches a different agent or proceeds with what it has. For the critique agent specifically, persistent LLM failure triggers `_heuristic_critique`, which scores from state alone and returns a bounded decision.

### Alternate path F: Backend instance crash

A backend instance dies mid-job. Its Firestore heartbeat goes stale. On the next instance startup, `JobManager` runs orphan recovery: any job in non-terminal status whose `instance_id` corresponds to a stale heartbeat is marked FAILED with reason "instance_died". The capacity counter is decremented. The job is not auto-resumed.

### Alternate path G: Dataset download failure

Kaggle URL invalid, network failure, or quota exhausted. `data_profiler` (which owns the download since `data_profiler.py:341`) marks state FAILED with the error. The job is terminated. No further agents run.

### Orchestrator modes

There are two orchestrator implementations in the codebase, selected at job-manager construction by `ORCHESTRATOR_MODE` (default `standard`):

- **`StandardOrchestrator`** (`backend/src/agents/orchestrator/standard/agent.py`). Inherits `BaseAgent`. Reasons via a custom decision loop, exposes two dispatch tools to the LLM (`dispatch_to_agent`, `dispatch_parallel_agents`), and rebuilds a state-summary prompt every iteration. Has a fixed default workflow encoded in its system prompt that the LLM can deviate from.
- **`ReActOrchestrator`** (`backend/src/agents/orchestrator/react/agent.py`). Inherits `ReActAgent`. Uses the standard ReAct loop with `check_state` as a pull tool rather than rebuilding state into the prompt. Marketed as "fully autonomous" with no fixed workflow. Currently labeled experimental.

Each package owns its specialist roster (`build_specialists()`). `StandardOrchestrator` uses the default registry instances. `ReActOrchestrator` swaps the standard `EffectEstimatorAgent` for `EffectEstimatorReActAgent` under the same `effect_estimator` key, so the rest of the dispatch surface is unchanged.

Shared infrastructure lives in `backend/src/agents/orchestrator/common/`: the agent-to-`JobStatus` mapping, the required-fields validator, and (through `BaseAgent` / `ReActAgent`) the `execute_with_tracing` wrapper that records `AgentTrace` entries into state and emits structlog events at every agent boundary. Both modes go through that wrapper, so logs and traces look identical regardless of orchestrator.

The `Orchestrator` Protocol in `orchestrator/base.py` defines the contract `JobManager` depends on (`register_specialist`, `execute_with_tracing`). Both orchestrators satisfy it; adding a third mode means writing a new package and pointing `ORCHESTRATOR_MODE` at it, no `JobManager` edit required.

---

## 4. State and data flow map

State is the system's spinal column. Every agent reads and writes the same `AnalysisState`, which is the single source of truth for a job. This section enumerates every place a piece of state lives, who owns it, and what happens to it on common boundaries.

### State map

| State | Bucket | Owner | Source of truth | Survives refresh? | Cross-device? |
|---|---|---|---|---|---|
| Current page URL | Browser address bar | React Router | Browser URL | Yes | No (per-device) |
| Form draft (Kaggle URL input on HomePage) | React component state | HomePage component | Component state | No | No |
| Job list cache | Frontend memory (Zustand store) | Zustand store | Backend `GET /jobs` | No (refetched on mount) | No |
| Current-job snapshot | Frontend memory (Zustand store) | useJob hook | Backend `GET /jobs/{id}` | No (refetched) | No |
| SSE event buffer | Frontend memory | useJob hook | Backend SSE stream | No | No |
| AnalysisState (live, in-flight) | Backend memory (`JobManager._active_states`) | JobManager | The Firestore doc, but updated locally then flushed | No | Yes (after flush) |
| AnalysisState (persisted) | Firestore doc (prod) or local JSON (dev) | Storage client | Itself | Yes | Yes |
| Dataframe (parquet) | Filesystem at `LOCAL_STORAGE_PATH/{job_id}/data.parquet` | data_profiler | Filesystem | Yes | No (per-instance disk) |
| Notebook (.ipynb) | Filesystem at `LOCAL_STORAGE_PATH/{job_id}/notebook.ipynb` | notebook_generator | Filesystem | Yes | No |
| Running job tasks | Backend memory (`JobManager._running_jobs`) | JobManager | Itself | No (lost on restart) | No |
| Job semaphore (local) | Backend memory (`asyncio.Semaphore`) | JobManager | Itself | No | No |
| Job semaphore (distributed) | Firestore counter doc | JobManager | Firestore | Yes | Yes |
| Instance heartbeat | Firestore doc per `instance_id` | JobManager | Firestore | Yes (until stale) | Yes |
| Orchestrator's `_critique_feedback_for_prompt` | Backend memory (per-orchestrator-instance) | OrchestratorAgent | Itself | No | No |
| LLM rate-limit token | Provider-side | LLM provider | Provider | N/A | N/A |
| HTTP rate-limit counter | Backend memory (or Redis if `REDIS_ENABLED`) | slowapi limiter | Either backend memory or Redis | Memory: no, Redis: yes | Memory: no, Redis: yes |
| Logs | stdout (structured JSON via structlog) | All agents | Wherever your log shipper sends them | N/A | N/A |
| Kaggle credentials | `backend/.env` | Operator | Env vars | Yes (file) | No |
| LLM provider keys | `backend/.env` | Operator | Env vars | Yes (file) | No |

### Mental picture (ASCII)

```
                                     HTTP / SSE
   [Browser]  <===================================================>  [FastAPI]
      |                                                                  |
      | (URL, form draft, Zustand cache)                                  |
      |                                                                  v
      |                                                          [JobManager]
      |                                                                  |
      |                                                                  | acquires
      |                                                                  v
      |                                                       [asyncio.Semaphore] + [Firestore counter]
      |                                                                  |
      |                                                                  v
      |                                                       [_run_job_task (async)]
      |                                                                  |
      |                                                                  v
      |                                                       +-------------------+
      |                                                       | AnalysisState     |
      |                                                       | (in memory, ref'd |
      |                                                       |  by _active_states)|
      |                                                       +---------+---------+
      |                                                                 |
      |                                                                 | reads/writes
      |                                                                 v
      |                            +----------------+ +----------------+ +----------------+
      |                            | Orchestrator   |->| Specialist     |->| Critique       |
      |                            | (LLM-driven)   |  | ReAct agents   |  | agentic loop   |
      |                            +-------+--------+ +-------+--------+ +-------+--------+
      |                                    |                  |                  |
      |                                    | LLM tool calls (Vertex/Claude/Gemini)
      |                                    +------------------+------------------+
      |                                                       |
      |                                                       v
      |                                         +-----------------------------+
      |                                         | Pull-based context tools    |
      |                                         | (ask_domain_knowledge,      |
      |                                         |  get_eda_finding,           |
      |                                         |  get_dag_adjustment_set...) |
      |                                         +-----------------------------+
      |                                                       |
      |                                                       v
      |                                          checkpoints to storage
      |                                                       |
      |                                                       v
      |                                +----------------------+----------------------+
      |                                |                                             |
      |                          [Firestore]                                  [Local JSON]
      |                          (prod, USE_FIRESTORE=true)                  (dev, default)
      |                                |                                             |
      |                                v                                             v
      |                       +-----------------+                          +------------------+
      |                       | jobs/{job_id}   |                          | data/{job_id}.json|
      |                       | semaphore_doc   |                          +------------------+
      |                       | instances/{id}  |
      |                       +-----------------+
      |                                                       |
      | SSE stream (status, agent_event, done, ...)           |
      +-------------------------------------------------------+

   Filesystem (per backend instance):
      LOCAL_STORAGE_PATH/{job_id}/
        data.parquet    <- written by data_profiler
        notebook.ipynb  <- written by notebook_generator
```

### Why the single AnalysisState

The single shared state object is the most consequential design choice. Alternatives considered: per-agent local state with explicit message-passing; an event-sourced log; a SQL row per artifact. The shared object wins on three axes for this use case: (1) checkpointing is one serialization, not many, so a crashed instance can resume context without replay; (2) pull-based context tools have one place to read from; (3) the notebook generator can render from a single object instead of joining many. The cost is paid in **multiple-writer hazards** (see Section 9 bug 2) and in the orchestrator's temptation to dump the whole thing into its prompt (see bug 7).

---

## 5. API map

All routes live in `backend/src/api/routes/jobs.py` plus the health router. Routes are grouped under `/jobs`.

| Method | Path | Purpose | Idempotent? | Status codes |
|---|---|---|---|---|
| POST | `/jobs` | Create a job from a Kaggle URL. Rate-limited. | No | 201 created, 400 bad URL, 401 missing key, 429 rate, 503 capacity |
| GET | `/jobs` | List recent jobs. | Yes | 200 |
| GET | `/jobs/agents` | List registered agent names and metadata. Static-ish. | Yes | 200 |
| GET | `/jobs/{id}` | Full job detail (state snapshot). | Yes | 200, 404 |
| GET | `/jobs/{id}/status` | Lightweight status only. | Yes | 200, 404 |
| GET | `/jobs/{id}/stream` | SSE stream of agent events while job runs. | Yes (idempotent connect) | 200 (long-lived) |
| GET | `/jobs/{id}/results` | Structured result summary (effects, DAG, sensitivity). | Yes | 200, 404, 409 if not COMPLETED |
| GET | `/jobs/{id}/notebook` | Download the .ipynb. | Yes | 200, 404 |
| GET | `/jobs/{id}/notebook/bundle` | Download notebook plus data and a runner script as a zip. | Yes | 200, 404 |
| GET | `/jobs/{id}/traces` | Full agent trace log. | Yes | 200, 404 |
| POST | `/jobs/{id}/cancel` | Cancel a running job. | Yes (cancel is idempotent) | 200, 404, 409 if already terminal |
| DELETE | `/jobs/{id}` | Hard-delete the job, dataframe, and notebook. | Yes (idempotent absent) | 204, 404 |
| GET | `/health` | Liveness probe. Does not require API key. | Yes | 200 |
| GET | `/health/ready` | Readiness probe. Checks Firestore reachability if enabled. | Yes | 200, 503 |

Notes:

- **Auth.** A single optional dependency `verify_api_key` is wired onto `job_router` but not `health_router`. If `API_KEY` env var is unset, `verify_api_key` short-circuits and admits the request. This is the silent-prod-auth-hole flagged in Section 9 bug 4.
- **Rate limiting.** Only `POST /jobs` is rate-limited at `10/minute`. Twelve other routes are unprotected. See Section 9 bug 5.
- **SSE.** `GET /jobs/{id}/stream` emits five event types (`status`, `agent_event`, `done`, `timeout`, `heartbeat`). Frontend `useJob.ts` consumes the first three; the other two are intentional but unused. Connection lives until job terminates or client disconnects.
- **Idempotency.** All GETs and DELETE are idempotent. POST `/jobs/{id}/cancel` is naturally idempotent (cancel a cancelled job is a no-op). POST `/jobs` is **not idempotent** and lacks an idempotency key, so a retry on a 503 can create a duplicate if the first request succeeded but the response was lost.

---

## 6. Storage and retrieval map

Two storage backends behind one interface (`backend/src/storage/`). The choice is per-deploy via `USE_FIRESTORE`.

### Choice: Firestore (prod) vs local JSON (dev)

Firestore was chosen for production because it gives us three primitives almost for free that are otherwise expensive to build: (1) atomic counter increments for the distributed semaphore, (2) per-instance heartbeat documents that we can query for staleness, and (3) cross-instance visibility of every job's state without running our own coordinator. We pay for it in (a) no SQL queries, so any analytics over jobs requires either a one-off export or a pre-built index, and (b) per-document write costs that grow with how often we checkpoint state.

Local JSON was chosen for dev because it has zero setup cost. The same `Storage` interface is implemented by writing one JSON file per job to `LOCAL_STORAGE_PATH`. Concurrency primitives are degraded: the "distributed semaphore" becomes a local file lock plus the asyncio semaphore, which is fine because dev is single-instance.

### Data shape in storage

One document per job, keyed by `job_id`. The document is the serialized `AnalysisState`. Whenever an agent writes, the storage client serializes the whole state and writes the doc atomically. We do not do field-level patches: simpler and avoids partial-write bugs, at the cost of larger writes. A typical fully-populated state is 50-300 KB depending on how many treatment effects, traces, and how big the DAG is.

Auxiliary documents:

- `semaphore/global` (single doc): integer counter for distributed concurrency.
- `instances/{instance_id}` (one per backend instance): heartbeat timestamp.

### Filesystem artifacts

The dataframe (parquet) and notebook (.ipynb) are written to the local disk of the backend instance that ran the job, at `LOCAL_STORAGE_PATH/{job_id}/`. **This is per-instance disk**, not shared. If a different instance serves a download request, it has to re-fetch the artifact, which today means it cannot. In practice we pin job-related GETs to the instance that ran the job by virtue of the user-session-affinity that the frontend implicitly does (it reuses the connection it created), but this is fragile. Section 10 tradeoff 11 covers this.

### Dominant queries

| Query | Frequency | Path | Notes |
|---|---|---|---|
| Read one job by id | High (every status poll, every SSE reconnect) | Direct doc fetch by `job_id` | O(1) Firestore |
| List recent jobs | Medium (UI list page) | Query by `created_at desc, limit N` | Needs an index on `created_at` |
| Update one job | High (every agent step, if SSE_ENABLED=true) | Doc replace | Limits write throughput |
| Read semaphore counter | Every job create | Doc fetch | Atomic incr/decr via transaction |
| Read instance heartbeats | Once per backend startup | Collection scan | Small (n = number of instances) |
| Update instance heartbeat | Every 30s per instance | Doc update | Cheap |
| Find orphan jobs | Once per backend startup | Query by `status in [RUNNING, ...] and instance_id == X` | Index on (status, instance_id) |

### Indexes

Firestore composite indexes assumed:

- (`status`, `instance_id`) for orphan recovery
- (`created_at desc`) for list endpoint

If these are not declared in the Firestore index config, those queries will fail at runtime with an "index required" error and a click-through link. Worth verifying.

### Caches

There is **no application-level cache**. Every read goes to Firestore (or local JSON). This is correct for the current load (jobs are unique, low repeat-read volume per job) but if a job's status is polled from the UI every 2 seconds for an hour, that is 1800 reads per job. If polling becomes a problem we either lengthen the poll interval or front Firestore with a memory cache and accept staleness. Today, polling is replaced by SSE for active jobs, which makes the steady-state read pattern fine.

### No vector store

No RAG. The "context" agents pull is the job's own state. There is no document corpus to retrieve from. If we ever add a "look up similar past analyses" feature, we would add a vector store keyed by (dataset description, analysis question) and embed past job summaries. Not on the roadmap.

---

## 7. Auth and access control

This section is intentionally short because the system is single-user-ish today.

### What exists

- **API key auth.** A single shared `API_KEY` env var. If set, every request to `job_router` must carry `X-API-Key: {value}`. Implemented as `Depends(verify_api_key)` on the router.
- **CORS.** `CORS_ORIGINS` env var is a JSON list. Defaults to `localhost:5173` and `localhost:3000`.
- **No user model.** No accounts, no roles, no per-user data isolation. Every job is visible to anyone who has the API key.
- **No tenancy.** No `tenant_id` on jobs.

### What's broken

If `API_KEY` is unset (whether intentionally in dev or by misconfiguration in prod), `verify_api_key` returns without checking. Production deploys that forget to set the env var have an open API. Section 9 bug 4 details the fix: refuse to start if `ENVIRONMENT=production` and `API_KEY` is unset.

### What we'd add for multi-tenant

Not building this now. If we do: a `users` table, JWT auth with refresh tokens, `user_id` on every job document, query rewriting in the storage layer to scope by `user_id`, a tenant_id if cross-org isolation is needed. Section 6's "no SQL" choice would push us toward Firestore subcollections (`users/{user_id}/jobs/{job_id}`) over composite-index-based filtering.

---

## 8. Caching, observability, and deployment

### What's cached

- **HTTP layer:** nothing. No CDN, no `Cache-Control` headers worth talking about. Notebook downloads have `Cache-Control: no-store` since each is unique per job.
- **Application:** nothing. See Section 6.
- **Browser:** Vite default for static assets. Job data is never browser-cached because the frontend uses fetch with no caching directives.
- **LLM:** No prompt caching is configured. Each LLM call rebuilds its prompt from scratch. Anthropic and Vertex both support prompt caching but we do not opt in. Section 9 bug 7 calls this out.

### Per-request observability

Every agent step appends an `AgentTrace` to state with: `agent_name`, `timestamp`, `action`, `reasoning` (LLM thought), `outputs` (tool result summary), `duration_ms`, `token_usage` (if available from provider). These are queryable via `GET /jobs/{id}/traces` and stream over SSE in real time.

Logs use structlog with JSON output. Every log line has `job_id`, `agent`, `instance_id`, and a typed event name. Searchable in any log shipper (Cloud Logging, Datadog, etc.) without parsing.

What is **not** measured today:

- Per-job dollar cost. Token counts are recorded in traces, but no cost roll-up endpoint exists. Compute cost is the dominant operational risk and we should add it.
- Per-agent failure rate over time. Could be derived from logs but no dashboard.
- Tail latency per agent. We have `duration_ms` but no histograms.

### Deployment shape

- **Local dev:** `./start.sh` boots backend (uvicorn on :8000) and frontend (Vite on :5173). Kills stale processes on those ports first.
- **Docker compose:** three profiles. Default is nginx (`:80`) + backend + frontend. `--profile full` adds Redis. `--profile dev` is for hot-reloading.
- **Production target:** Cloud Run. Terraform configs in `infrastructure/`. Per-instance heartbeat and orphan recovery in `JobManager` are the design choices that make multi-instance deploy safe.
- **Concurrency:** `MAX_CONCURRENT_JOBS=3` per instance by default. If running multiple instances, the Firestore semaphore caps total concurrent jobs across the fleet. Local semaphore caps per-instance.
- **Migration story:** None. Schema changes to `AnalysisState` require backfill or in-place handling of old documents. If we add a field, code defaults to `None`. If we rename or remove, we need a script.

---

## 9. Likely failures (the bug bestiary)

Concrete bugs this system will encounter, ranked roughly by likelihood and impact. For each: the symptom you'd see, the root cause, and the fix.

### Bug 1: The Silent REJECT

**Symptom.** Critique agent returns `decision: REJECT` (e.g., the analysis has a clearly broken assumption). The job continues to APPROVE and ships a notebook.

**Root cause.** Neither `OrchestratorAgent` nor `ReActOrchestrator` has a code branch for `CritiqueDecision.REJECT`. They check ITERATE and APPROVE only. REJECT falls through and the orchestrator treats the run as approvable.

**Fix.** Add explicit REJECT handling. On REJECT, mark job FAILED with a `failed_critique` reason, persist the critique reasoning in the failure message, do not generate a notebook. The user gets back what was wrong, not a notebook with a misleading conclusion.

### Bug 2: The DAG Overwrite

**Symptom.** Two agents (`causal_discovery` and `dag_expert`) both write `state.proposed_dag`. If `dag_expert` runs to completion, its refined DAG silently replaces `causal_discovery`'s output. If `dag_expert` fails partway through, what we end up with depends on which writes landed first.

**Root cause.** Both agents declare `WRITES_STATE_FIELDS = ["proposed_dag"]` and the orchestrator's `_DEFAULT_AGENT_WRITES` confirms it. There is no merge or versioning, just last-writer-wins.

**Fix.** Either (a) split into `state.discovered_dag` and `state.refined_dag` so both are preserved, or (b) make `dag_expert` mutate-in-place rather than replace, with explicit fields for what it changed. Option (a) is cleaner because the notebook can show the user how the DAG evolved. Canonical pattern: **expand-then-contract migration** if we want to do this without breaking older state docs.

### Bug 3: The Phantom React Mode (resolved)

**Symptom.** Operator sets some env var hoping to switch to react-mode orchestration. Nothing changes. The default standard orchestrator runs.

**Root cause.** `ORCHESTRATOR_MODE` was not an env var. `JobManager.__init__` took it as a constructor argument; `api/main.py` called `get_job_manager()` with no arguments and got `"standard"`. The "experimental" feature was invisible to operators.

**Resolution.** `Settings.orchestrator_mode` (`Literal["standard", "react"]`) added; `get_job_manager()` falls back to the env var when no explicit mode is passed. `EffectEstimatorReActAgent` now carries `@register_agent("effect_estimator_react")` so the react roster is reachable through the registry rather than a manual hack in `JobManager`. README still needs a section about when to choose react over standard.

### Bug 4: The Production Auth Hole

**Symptom.** Production deploy succeeds and accepts jobs. A pen tester or curious user discovers `/jobs` is open without an `X-API-Key` header.

**Root cause.** `API_KEY` env var is optional. `verify_api_key` short-circuits when the key is unset. If the operator forgets to set it, the API is open. There is no startup-time refusal.

**Fix.** In `Settings.validate()`, raise if `ENVIRONMENT == "production"` and `api_key_value` is None. Make the failure mode "service does not start" rather than "service starts with no auth."

### Bug 5: The Stream Hammer

**Symptom.** A noisy client opens dozens of `GET /jobs/{id}/stream` connections and pins backend workers. CPU climbs, other clients can't connect.

**Root cause.** Only `POST /jobs` has `@limiter.limit`. The 12 other routes including `/stream` are unthrottled.

**Fix.** Apply rate limits per-route or globally. SSE streams should be capped per (api_key, job_id) and globally per (api_key). Also: enforce a max concurrent stream connections per job to stop multi-tab abuse.

### Bug 6: The Forever-Approve

**Symptom.** LLM provider has an outage. Critique agent's `_heuristic_critique` fallback fires. After two iterations, the analysis approves regardless of quality.

**Root cause.** The heuristic fallback (`critique_agent.py:993`) generates only APPROVE or ITERATE. Line 942-944 force-approve after `iteration_count >= 2`. There is no path to REJECT in the fallback.

**Fix.** Add a low-score-and-no-effects REJECT branch in the heuristic. If the LLM is consistently down and quality is genuinely bad (no methods completed, etc.), the right answer is to fail the job, not to ship a heuristically-approved bad analysis.

### Bug 7: The Orchestrator Context Bloat

**Symptom.** On long jobs (multiple critique iterations, many treatment effects), the orchestrator's prompt grows. LLM costs creep up. Latency per orchestrator turn increases.

**Root cause.** `_build_context_prompt` is called every loop iteration and rebuilds a state dump from scratch: full lists of `data_quality_issues`, `multicollinearity_warnings`, every `treatment_effect`, every critique `issues`/`improvements`. The system prompt at line 84-88 calls itself "PULL-based context" but the orchestrator never pulls; it always pushes.

**Fix.** Give the orchestrator the same `ContextTools` mixin every specialist uses. Replace `_build_context_prompt` with a lean initial observation (job_id, dataset, status) and let the orchestrator call `get_state_summary`, `get_latest_results`, `what_did_X_finish` on demand. Also enable Anthropic-style prompt caching on the system prompt portion that does not change.

### Bug 8: The Orphan Job

**Symptom.** A job is stuck in `RUNNING` forever. The user sees "1 of 12 agents complete" and nothing changes.

**Root cause.** The backend instance running it died (OOM, deploy, host failure). Its heartbeat went stale. The job document still says `RUNNING` because the next-state-write never happened.

**Mitigation already present.** On startup, `JobManager` queries Firestore for jobs in non-terminal status, joins against the heartbeats collection, and marks any job whose owner is dead as `FAILED` with reason `instance_died`.

**Remaining bug.** If no instance starts within the heartbeat-staleness window, the job sits in `RUNNING` indefinitely from the user's perspective. Add a periodic sweeper (cron in Cloud Scheduler, or a single-instance heartbeat from any live instance) that recovers orphans every N minutes regardless of restarts.

### Bug 9: The Capacity Race

**Symptom.** Two backend instances both admit a job at the same moment. Total concurrent jobs briefly exceeds `MAX_CONCURRENT_JOBS`.

**Root cause.** The Firestore counter increment is atomic, but the read-then-increment pattern in some implementations is not. If `JobManager` reads "counter == 2", checks against max, and only then increments, two concurrent reads can both see 2 and both increment to 3 (oversubscribed by 1).

**Fix.** Use Firestore's transaction with read-modify-write inside the txn, not outside. Alternatively switch to `firestore.FieldValue.increment(1)` inside a transaction that re-reads after increment and aborts if over cap. Worth verifying the current implementation does this; if not, fix.

### Bug 10: The Notebook Half-Render

**Symptom.** Notebook generated but section 8 ("Causal Structure") is empty or shows a cryptic error. Or generation crashes outright.

**Root cause.** A renderer is called unconditionally even if the prerequisite state field is missing. `notebook/agent.py:111` calls `render_eda_report(state)` whether or not `state.eda_result` exists. If a section renderer doesn't guard, it crashes when it dereferences `None`. If it guards but doesn't communicate, the user gets an empty cell.

**Fix.** Audit each section renderer. Pattern: render a "skipped because X did not run" placeholder cell with the reason. The notebook should always be complete; missing-stage cells are first-class citizens.

### Bug 11: The Lost Trace

**Symptom.** An agent disappears from the trace log. We know it was dispatched (orchestrator logged it) but no trace entry, no error, just nothing.

**Root cause.** An agent throws before `state.add_trace()` is called. `execute_with_tracing` does append an error trace on exception (line 622), but if the throw is during `_load_dataframe` before the try block, or during the trace construction itself, we lose the record.

**Fix.** Move trace construction out of `execute` and into `execute_with_tracing` as the very first line, before any agent-specific work. The trace starts in "in_progress" state and gets finalized at the end. Even if the agent body crashes, the trace is durable.

### Bug 12: The Idempotency Gap

**Symptom.** Client retries `POST /jobs` after a network blip. They end up with two parallel jobs analyzing the same dataset. They get billed for two LLM workflows.

**Root cause.** `POST /jobs` is not idempotent and has no idempotency key. The first request created a job; the response was lost; the retry created another.

**Fix.** Accept an `Idempotency-Key` header. Store last-seen-key per (api_key, key) for 24h. On match, return the original `job_id` instead of creating a new job. Canonical pattern: **idempotent receiver**.

---

## 10. Tradeoffs

Every non-obvious decision in this system, with what we picked and what we gave up. A reader who disagrees with a tradeoff knows exactly where to push back.

### 1. Orchestration: LLM-driven vs hardcoded pipeline

**Q.** Should the order of agents be coded as a pipeline, or decided by an LLM?
**Options.** (a) Hardcoded DAG of agent dependencies. (b) LLM tool-calling decides each next step. (c) Hybrid: hardcoded happy path with LLM-controlled fallbacks.
**Choice.** (b) LLM-driven, with the system prompt suggesting a default order.
**Why.** Causal inference has too many "depends on what we found in the data" branches for a fixed pipeline to handle gracefully. EDA might surface an issue that needs data_repair before profiling can finish. Discovery might fail and need a fallback strategy.
**Gave up.** Determinism. Cost predictability. Easier debugging. We tax-paid this back partly via the suggested-default-order in the system prompt and the bounded critique loop.

### 2. Context propagation: pull vs push

**Q.** When an agent runs, how does it learn what previous agents did?
**Options.** (a) Push the full state into every agent's prompt (context dump). (b) Pull on demand via tools (context_tools mixin). (c) Hybrid.
**Choice.** (b) at the specialist level, accidentally (c) at the orchestrator level.
**Why.** Specialist prompts must stay lean to keep token costs manageable on long jobs. The orchestrator was supposed to be lean too but got built with `_build_context_prompt` instead of pull tools.
**Gave up.** Consistency. The "PULL-based context" claim in the orchestrator's own system prompt is currently false at the orchestrator level. Bug 7 covers the fix.

### 3. State container: shared mutable object vs per-agent isolation

**Q.** How do agents communicate state between stages?
**Options.** (a) Single shared `AnalysisState` they all mutate. (b) Message-passing with explicit envelopes. (c) Event-sourced log replayed by the orchestrator.
**Choice.** (a).
**Why.** Checkpointing is one operation. Notebook generation reads from one place. Pull tools have one source.
**Gave up.** Isolation; multiple-writer collisions (bug 2). The DAG-overwrite hazard is direct fallout.

### 4. Estimation: run-many-and-compare vs auto-pick

**Q.** When estimating, do we pick the best method for the data and run it, or run everything?
**Options.** (a) Auto-select one method based on data characteristics. (b) Run multiple, compare, present all. (c) Let the user pick.
**Choice.** (b). The estimator is told "run OLS, IPW, AIPW and finalize."
**Why.** Triangulation is the credibility argument. One method is one number; three methods agreeing is evidence.
**Gave up.** Speed. Cost. Three methods is roughly 3x the compute of one.

### 5. DAG discovery: ensemble vs single algorithm

**Q.** Discovery has five algorithms. Which do we use?
**Options.** (a) Pick one (say PC). (b) Run all and pick the most stable. (c) Run all and ensemble-vote.
**Choice.** (c) with bootstrap stability for confidence weighting.
**Why.** Each algorithm has known failure modes (PC with unfaithfulness, NOTEARS with non-Gaussian data). Ensembling lets us not bet on which one is right for this dataset.
**Gave up.** Speed. Discovery is the longest-running stage by wall clock. Roughly N times the cost of single-algo where N = 5.

### 6. Storage: Firestore vs Postgres

**Q.** Where do jobs live in production?
**Options.** (a) Postgres with a `jobs` table and JSONB column. (b) Firestore documents. (c) S3 + DynamoDB hybrid.
**Choice.** (b).
**Why.** Firestore gives us atomic counters, per-document leases, and multi-instance visibility for free. Postgres would need pgmq, advisory locks, or a dedicated coordination service to match. Postgres has better analytics ergonomics, but we do not run analytics on jobs today.
**Gave up.** SQL. Joins. Window functions. Migrations are also harder in Firestore: schema changes are by-convention, not enforced.

### 7. Dev storage: local JSON vs SQLite

**Q.** What do we use locally?
**Options.** (a) SQLite. (b) Local JSON files. (c) Use the Firestore emulator.
**Choice.** (b).
**Why.** Zero setup. Inspectable with `cat`. Diffable in git if needed.
**Gave up.** Realism. The local backend can't replicate Firestore concurrency primitives, so concurrency bugs only surface in prod.

### 8. Dataframe cache: parquet vs CSV vs pickle

**Q.** How do we cache the downloaded dataframe?
**Options.** (a) CSV. (b) Parquet. (c) Pickle.
**Choice.** (b).
**Why.** Type fidelity (CSV loses dtypes). Cross-language readability (pickle is Python-only and unsafe to load from untrusted sources). Compression.
**Gave up.** Human-readability of the cached file. We accept this; nobody hand-reads the cached parquet.

### 9. Streaming: SSE vs WebSockets

**Q.** How do we push live progress to the UI?
**Options.** (a) WebSockets. (b) Server-sent events. (c) Long polling.
**Choice.** (b).
**Why.** Unidirectional fits our use case (server emits, client listens; no client-to-server messages mid-stream needed). SSE works through any HTTP proxy without sticky sessions. Browser support is universal. WebSockets give us bidirectional we don't need.
**Gave up.** Bidirectional. If we ever want the UI to send "skip this stage" mid-run, we'd need to add a separate POST endpoint or migrate to WebSockets.

### 10. Background work: in-process tasks vs Celery/RQ

**Q.** How do we run jobs asynchronously?
**Options.** (a) FastAPI background tasks (asyncio task in same process). (b) Celery with Redis/SQS. (c) RQ. (d) Cloud Tasks.
**Choice.** (a).
**Why.** Fewer moving parts. No broker to maintain. Single binary to deploy.
**Gave up.** Survives-restart. If the backend instance dies mid-job, the job dies with it (mitigated by orphan-recovery on the next startup, but the in-flight LLM work is lost). Also gave up cross-instance work distribution.

### 11. Artifact storage: per-instance disk vs shared object store

**Q.** Where do parquet files and notebooks live?
**Options.** (a) Per-instance local disk. (b) GCS / S3.
**Choice.** (a) today.
**Why.** Faster (no network round-trip to read). Simpler (no cloud creds needed in dev). Sufficient for single-instance deploys.
**Gave up.** Multi-instance correctness. A download request hitting a different instance than the one that ran the job will 404 or fall through to a slow re-fetch. We are one instance away from this being a problem; the right fix is GCS.

### 12. Iteration cap: 3 critique loops vs unbounded

**Q.** How many times can the critique send back the analysis?
**Options.** (a) Once. (b) Three. (c) Unbounded with cost-based stop.
**Choice.** (b).
**Why.** Bounds cost. Three is enough for most "iterate on the issue and re-estimate" cases. Beyond that, the issue is usually structural (wrong DAG) and re-estimation won't help.
**Gave up.** The chance to perfect a borderline analysis. Empirically the third iteration rarely improves substantially.

### 13. LLM provider: single vs abstraction

**Q.** Bind to one LLM, or abstract over many?
**Options.** (a) One provider (Vertex/Claude/Gemini). (b) Three behind one interface.
**Choice.** (b).
**Why.** Vendor risk. Claude is best at reasoning, Gemini cheaper, Vertex gives us GCP integration. Switching cost should be hours, not weeks.
**Gave up.** Provider-specific features (Anthropic's prompt caching, Vertex's grounding, Gemini's long context). The abstraction is currently lowest-common-denominator.

### 14. Agent registration: decorator vs explicit dict

**Q.** How are agents wired up?
**Options.** (a) Explicit registry dict in code. (b) `@register_agent("name")` decorator with import-side-effect registration.
**Choice.** (b).
**Why.** Adding an agent is one file plus an import in `__init__.py`. No central edit.
**Gave up.** Explicit dependency tree. It is easy to write `effect_estimator_react` without `@register_agent` and have it silently absent (current bug). A linter rule could catch this.

### 15. Critique agent: ContextTools mixin vs custom tools

**Q.** Should critique use the same context tools as specialists?
**Options.** (a) Inherit ContextTools. (b) Custom tools written for critique only.
**Choice.** (b).
**Why.** Critique needs to investigate (run actual SMD calculations on the dataframe, check estimate consistency), not just retrieve summaries. The generic context tools would be too coarse.
**Gave up.** Code reuse. Critique re-implements its own dispatch loop. Documented in the design comment in `critique_agent.py:51`.

### 16. ReAct vs single-shot for specialists

**Q.** Are specialists multi-step ReAct loops or one-shot completions?
**Options.** (a) Single-shot: one prompt, one structured response. (b) ReAct: plan-act-observe loop with tools.
**Choice.** (b).
**Why.** Specialists need to investigate the data (load it, profile a column, check correlations) before deciding. One-shot completions can't do that. ReAct gives us tool calls and verifiable steps for evals.
**Gave up.** Speed. A ReAct agent with 10 steps takes ~10x the LLM round-trips of a one-shot.

### 17. Orchestrator decision cap: max 15 vs unbounded

**Q.** How many decisions can the standard orchestrator make per job?
**Options.** (a) 15 (current). (b) Unbounded with cost-based stop. (c) Per-stage cap.
**Choice.** (a).
**Why.** Bounds cost. 15 is enough for the default path (~11 dispatches) plus 1-2 critique iterations.
**Gave up.** Edge cases that need more re-dispatching. The auto-finalize at line 351 is the safety net.

### 18. SMD threshold for imbalance: 0.1 vs 0.25

**Q.** At what standardized mean difference do we flag a covariate as imbalanced?
**Options.** (a) 0.1 (Stuart's recommendation). (b) 0.25 (older convention).
**Choice.** (a).
**Why.** Modern causal-inference practice is conservative; 0.1 catches subtler confounding that 0.25 misses.
**Gave up.** Tolerance. Many real datasets have several covariates between 0.1 and 0.25, leading to noisier critique signals.

### 19. Frontend store: Zustand vs Redux vs Context

**Q.** What's the frontend state library?
**Options.** (a) Zustand. (b) Redux Toolkit. (c) React Context + useReducer.
**Choice.** (a).
**Why.** Lightweight (1KB). No provider boilerplate. Hooks-first API matches React 18.
**Gave up.** Devtools maturity. Redux's time-travel debugger is unmatched. Zustand has a basic devtools middleware but it's not in the same league.

### 20. Notebook generator: section-by-section vs template

**Q.** How is the .ipynb built?
**Options.** (a) Jinja-style template with state interpolated. (b) Code that builds JSON cell-by-cell from state.
**Choice.** (b).
**Why.** Conditional rendering is easier in code. Skipping a section because a stage didn't run is one `if` instead of a template flag.
**Gave up.** Visual editability of the notebook layout. To restyle a section we change Python, not a template.

### 21. Effect estimation order: sequential vs parallel

**Q.** Do the 12 estimation methods run in parallel or sequentially?
**Options.** (a) Parallel via asyncio.gather. (b) Sequential.
**Choice.** (b), one method per ReAct step.
**Why.** Each method's success informs the next (e.g., if PSM has bad overlap, the agent might skip ahead to robust methods). Parallel would lose this adaptivity.
**Gave up.** Wall-clock speed. Roughly 12x slower than max-parallel.

---

## 11. Takeaway

### One paragraph

The Causal Inference Orchestrator is a multi-agent system that turns a Kaggle dataset into a peer-reviewed causal-inference notebook. Its load-bearing design choices are: a single shared `AnalysisState` that every agent reads and writes, an LLM-driven orchestrator that decides which specialist to dispatch next, pull-based context tools so each specialist sees only what it needs, and a critique loop bounded at three iterations that can send the analysis back for refinement. The system runs in-process on FastAPI with Firestore for state and per-instance disk for artifacts. The interesting hazards are at the seams where the architectural commitments don't quite hold: the orchestrator pushes context instead of pulling, two agents both write the DAG, the critique can return a decision the system silently ignores, and configuration that should be env-var-driven is hardcoded. Storage and auth are conventional; the agent layer is where the work goes.

### Interview-language one-liners

- "It is a Kaggle-URL-to-notebook system; the interesting part is the agent layer, not the storage."
- "There is one source of truth per job, a shared `AnalysisState` object that gets checkpointed to Firestore on every agent boundary."
- "The orchestrator is LLM-driven via tool-calling; the system prompt suggests an order but the LLM can deviate."
- "Specialists use the ReAct pattern with pull-based context tools so prompts stay lean."
- "The critique loop is bounded at three iterations; that's a deliberate cost cap, not an aspiration."
- "We run twelve estimation methods and present them all, because triangulation across methods is the credibility argument."
- "Discovery is an ensemble of PC, FCI, GES, NOTEARS, and LiNGAM with bootstrap stability, because no single algorithm is reliable for arbitrary data."
- "Storage is Firestore in prod and local JSON in dev, because Firestore gives us the distributed semaphore and heartbeat primitives for free."
- "There is no vector store and no RAG; the only context is the job's own state."
- "The system is single-user today; auth is an optional API key, not a real identity model."
- "What breaks first at scale is the orchestrator prompt growing as state lists accumulate; we'd switch to pull-based context at the orchestrator level and turn on prompt caching."
- "The biggest correctness hazard is the multiple-writer pattern on `proposed_dag`; we'd split it into `discovered_dag` and `refined_dag`."

### Mapping to the 4 north-star questions

**1. What is the data flow?**
Input is a Kaggle URL (Section 3, step 1). Retrieval is the dataset download into parquet (step 6). Context is the pull-based tools each specialist calls into the shared `AnalysisState` (Section 4 mental picture). The model is one of Vertex/Claude/Gemini selected at runtime (Section 2 LLM call). Tools are the dispatch tools (orchestrator) plus the per-agent action tools (specialists). Output is the .ipynb and the structured `treatment_effects` array. Eval is `backend/src/benchmarks/agentic_evals/`, which today covers six of twelve agents (Section 9 surfaces the gap).

**2. Where does each piece of state live, and why?**
See the full state map in Section 4. The headline: AnalysisState is in Firestore; running tasks and live state refs are in backend memory; the dataframe and notebook are on per-instance disk; SSE event buffers are in browser memory; rate-limit counters are either in backend memory or Redis depending on `REDIS_ENABLED`. The "why" for each is in Section 6 (Firestore for distributed primitives) and Section 4 (single AnalysisState for one source of truth).

**3. What tradeoffs did I make, and what did I give up?**
Twenty-one numbered tradeoffs in Section 10. The biggest gives are: determinism (LLM-driven orchestration), per-instance correctness (artifacts on local disk), single-provider feature use (LLM abstraction). The biggest gets are: data-driven adaptivity, defensible output via triangulation and critique, easy multi-instance deploy via Firestore primitives.

**4. What breaks first at scale?**
In order: (1) orchestrator prompt bloat as state lists grow, increasing per-decision cost (bug 7); (2) per-instance disk for artifacts when we add a second instance (tradeoff 11); (3) SSE connection count when many users watch many jobs (bug 5); (4) Firestore write throughput as agents checkpoint frequently; (5) LLM cost in absolute terms because we run twelve estimators on every job. None of these break the system at one user; all of them surface between 10 and 100 concurrent jobs.
