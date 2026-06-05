# ADR-001: Dependency Rationale

**Status**: Accepted  
**Date**: 2025-03-15 (retroactive documentation)

This document explains why key dependencies were chosen over alternatives. It covers storage, logging, serialization, and the causal inference library stack.

---

## 1. Firestore (production storage) + Local JSON (dev)

**Choice**: `google-cloud-firestore` for production, file-based JSON for local development.

**Why Firestore**:
- Serverless — no connection pooling, no instance management, scales to zero when idle.
- Document model maps naturally to `AnalysisState`: each job is one document with nested fields. No ORM, no schema migrations.
- Built-in real-time listeners could power SSE without polling (not yet used, but available).
- Write-once-read-many access pattern (job created, agents append results, client reads final state) matches Firestore's strengths.

**Why not Postgres**: Would require connection pool management (`asyncpg`), schema migrations, and JSON column workarounds for the deeply nested state model. Overkill for a document-per-job workload with no cross-job queries.

**Why local JSON for dev**: Zero setup. `LocalStorage` writes `{job_id}.json` files with `filelock` for concurrency safety. Deterministic for tests — no emulator needed.

---

## 2. structlog (structured logging)

**Choice**: `structlog` with `python-json-logger` for JSON output.

**Why structlog**:
- JSON output in production enables machine-parseable logs for observability pipelines (Cloud Logging, Datadog, etc.).
- Context binding via `structlog.contextvars.bind_contextvars()` attaches `job_id` and `agent_name` to every log line within a request scope — no manual threading of context through function signatures.
- Compatible with stdlib `logging` — library logs from `econml`, `statsmodels`, etc. are captured and formatted consistently.
- `get_logger(__name__)` pattern gives each module a named logger without ceremony.

**Why not loguru**: Fewer production integrations, no stdlib `logging` compatibility (intercepts rather than extends), harder to configure per-deployment.

---

## 3. Parquet for intermediate DataFrames

**Choice**: `pyarrow` + Parquet for `dataframe_path` storage between agents.

**Why Parquet**:
- Type preservation — categorical columns, datetimes, and nullable integers survive round-trips without the silent coercion that CSV causes (e.g., `"NA"` string vs `NaN` float, integer columns becoming float when nulls are present).
- Columnar compression reduces disk I/O. Typical dataset shrinks 3-5x vs CSV.
- Fast partial reads — if an agent only needs specific columns, Parquet supports column projection without reading the full file.
- `pandas.read_parquet()` / `df.to_parquet()` are single-line operations with no configuration.

**Why not Feather/Arrow IPC**: Feather is optimized for same-machine read speed but has no compression. Since DataFrames are written once and read by multiple agents (potentially after process restart in production), Parquet's compression wins.

---

## 4. econml as primary causal ML library

**Choice**: `econml` (Microsoft Research) as the primary library for ML-based causal estimation, complemented by `dowhy` and `causalml`.

**Why econml**:
- `CausalForestDML` provides honest causal forests with valid pointwise confidence intervals out of the box — the only Python implementation with built-in inference (via the Generalized Random Forest framework).
- `LinearDML` implements Chernozhukov et al. 2018 Double/Debiased ML with proper cross-fitting and Neyman-orthogonal inference.
- Active maintenance by Microsoft Research with regular releases.

**Why dowhy is also included**: `dowhy` is the identification layer — it reasons about DAGs, backdoor/frontdoor criteria, and testable implications. It delegates actual estimation to `econml` or `statsmodels`. We use it for DAG-based identification checks, not estimation.

**Why causalml is also included**: Uber's `causalml` provides meta-learner implementations (S/T/X-Learner) with a different API surface. We use it as a fallback when `econml` meta-learners are unavailable or when its uplift modeling interface is more convenient. The overlap is intentional — having multiple implementations enables cross-validation of estimates.

---

## 5. scikit-learn + statsmodels (base ML and statistics)

**Choice**: Both `scikit-learn` and `statsmodels` as foundational libraries.

**Why both**:
- **scikit-learn**: ML pipelines, cross-validation, model selection, preprocessing. Used inside meta-learners, propensity score models, and nuisance model estimation. Its `Pipeline` and `GridSearchCV` abstractions standardize the ML workflow across all methods.
- **statsmodels**: Regression with proper statistical inference — coefficient SEs, p-values, confidence intervals, diagnostic tests (Ramsey RESET, Durbin-Watson, VIF). OLS, WLS, and IV (2SLS) all need inference that scikit-learn intentionally does not provide.

They complement rather than compete: scikit-learn handles prediction, statsmodels handles inference. A typical method uses scikit-learn for nuisance estimation and statsmodels for the final causal parameter.

---

## 6. slowapi for rate limiting

**Choice**: `slowapi` (built on `limits`) with optional Redis backend.

**Why slowapi**: Minimal integration — one decorator per route, automatic 429 responses, works with FastAPI's dependency injection. In-memory storage for single-instance dev, Redis backend for multi-instance production.

**Why not a custom token bucket**: `slowapi` already implements sliding window rate limiting correctly. Custom implementations are bug-prone (race conditions, clock drift) for no benefit at our scale.

---

## 7. pgmpy + networkx (graph and DAG operations)

**Choice**: `pgmpy` for Bayesian network operations, `networkx` for general graph manipulation.

**Why pgmpy**: Provides PC, FCI, and GES constraint-based discovery algorithms with configurable conditional independence tests. Also supports BayesianNetwork scoring (BIC, BDeu) for score-based approaches.

**Why networkx alongside**: `pgmpy` returns DAGs as `networkx.DiGraph` objects. We use `networkx` for graph traversal, d-separation queries, and custom edge operations (forbidden edges, variable role annotation) that `pgmpy` doesn't natively support.
