# API Reference

All endpoints are served by FastAPI. Interactive docs are available at `/docs` (Swagger UI) and `/redoc` when `enable_api_docs` is true in settings.

Source: `backend/src/api/routes/jobs.py`, `backend/src/api/schemas/job.py`

---

## Base URL and Authentication

Routes are mounted at `/jobs` with **no `/api/v1/` prefix**. The full URL for creating a job is `http://host:port/jobs`.

Authentication uses the `X-API-Key` header. When `api_key_value` is not set in config (dev mode), authentication is skipped. Health endpoints (`/health`) do not require authentication.

---

## Endpoints

### POST /jobs

Create a new analysis job. Rate limited to 10 requests per minute.

**Request body:**

```json
{
  "kaggle_url": "https://www.kaggle.com/datasets/owner/dataset-name",
  "treatment_variable": "smoking",
  "outcome_variable": "lung_cancer",
  "analysis_preferences": {},
  "orchestrator_mode": "standard"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `kaggle_url` | string | Yes | Kaggle dataset URL. Must match `https://kaggle.com/datasets/{owner}/{name}`. Max 500 chars. |
| `treatment_variable` | string | No | Hint for the treatment variable. Alphanumeric, underscores, hyphens only. Max 100 chars. |
| `outcome_variable` | string | No | Hint for the outcome variable. Same constraints as above. |
| `analysis_preferences` | object | No | Reserved for future use. |
| `orchestrator_mode` | string | No | `"standard"` (default) or `"react"` (experimental). |

**Response** (201):

```json
{
  "id": "a1b2c3d4",
  "kaggle_url": "https://www.kaggle.com/datasets/owner/dataset-name",
  "status": "pending",
  "created_at": "2025-02-15T12:00:00Z",
  "updated_at": "2025-02-15T12:00:00Z"
}
```

**Errors**: 422 (validation), 429 (rate limit), 500 (internal).

---

### GET /jobs

List jobs with optional filtering and pagination.

**Query parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `status` | string | (none) | Filter by job status. Must be a valid `JobStatus` value. |
| `limit` | int | 20 | Max results (1-100). |
| `offset` | int | 0 | Skip results (0+). |

**Response** (200):

```json
{
  "jobs": [
    {
      "id": "a1b2c3d4",
      "kaggle_url": "...",
      "status": "completed",
      "created_at": "...",
      "updated_at": "..."
    }
  ],
  "total": 42,
  "limit": 20,
  "offset": 0
}
```

**Known bug**: If a `status` query parameter is provided, the variable `status` shadows the `starlette.status` module import on line 101-105. This causes `status.HTTP_400_BAD_REQUEST` to fail with an `AttributeError` when the status value is invalid. The 400 error response is never actually returned.

---

### GET /jobs/{job_id}

Get detailed job information.

**Response** (200):

```json
{
  "id": "a1b2c3d4",
  "kaggle_url": "...",
  "status": "estimating_effects",
  "created_at": "...",
  "updated_at": "...",
  "dataset_name": "Lung Cancer Dataset",
  "current_agent": "effect_estimator",
  "iteration_count": 1,
  "error_message": null,
  "progress_percentage": 56,
  "treatment_variable": "smoking",
  "outcome_variable": "lung_cancer"
}
```

**Errors**: 404 (not found).

---

### GET /jobs/{job_id}/status

Lightweight status check. Returns fewer fields than the detail endpoint.

**Response** (200):

```json
{
  "id": "a1b2c3d4",
  "status": "estimating_effects",
  "progress_percentage": 56,
  "current_agent": "effect_estimator"
}
```

**Errors**: 404 (not found).

---

### GET /jobs/{job_id}/stream

Server-Sent Events stream for real-time status updates. Requires `sse_enabled = true` in settings.

Emits events until the job reaches a terminal state (`completed`, `failed`, `cancelled`).

**Event types:**

| Event | Data | When |
|-------|------|------|
| `status` | JSON `{id, status, progress_percentage, current_agent}` | On every status change |
| `heartbeat` | empty | Every `sse_heartbeat_seconds` (default 15s) |
| `done` | empty | When job reaches terminal state (stream closes) |

**Example:**

```
event: status
data: {"id": "a1b2c3d4", "status": "profiling", "progress_percentage": 20, "current_agent": "data_profiler"}

event: heartbeat
data:

event: status
data: {"id": "a1b2c3d4", "status": "completed", "progress_percentage": 100, "current_agent": null}

event: done
data:
```

**Errors**: 404 (job not found or SSE disabled).

---

### POST /jobs/{job_id}/cancel

Cancel a running job. No-op if already in a terminal state.

**Response** (200):

```json
{
  "job_id": "a1b2c3d4",
  "was_running": true,
  "cancelled": true,
  "status": "cancelled"
}
```

**Errors**: 404 (not found).

---

### DELETE /jobs/{job_id}

Delete a job and all artifacts (storage records, local files).

**Query parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `force` | bool | false | If true, cancel a running job before deleting. |

**Response** (200):

```json
{
  "job_id": "a1b2c3d4",
  "found": true,
  "cancelled": false,
  "firestore_deleted": true,
  "local_artifacts_deleted": {
    "dataframe": true,
    "notebook": true
  }
}
```

**Errors**: 404 (not found), 409 (job running and `force=false`).

---

### GET /jobs/{job_id}/results

Get analysis results. Only available for completed jobs.

**Response** (200):

```json
{
  "job_id": "a1b2c3d4",
  "treatment_variable": "smoking",
  "outcome_variable": "lung_cancer",
  "executive_summary": {
    "headline": "Smoking shows a significant positive effect on lung cancer risk",
    "effect_direction": "positive",
    "confidence_level": "high",
    "key_findings": ["..."]
  },
  "method_consensus": {
    "n_methods": 5,
    "direction_agreement": 1.0,
    "all_significant": true,
    "estimate_range": [0.42, 0.58],
    "median_estimate": 0.50,
    "consensus_strength": "strong"
  },
  "data_context": {
    "n_samples": 10000,
    "n_features": 25,
    "n_treated": 4200,
    "n_control": 5800,
    "missing_data_pct": 2.3,
    "data_quality_issues": []
  },
  "causal_graph": {
    "nodes": ["smoking", "lung_cancer", "age", "exercise"],
    "edges": [{"source": "smoking", "target": "lung_cancer", "edge_type": "directed", "confidence": 0.95}],
    "discovery_method": "PC",
    "interpretation": "..."
  },
  "treatment_effects": [
    {
      "method": "AIPW",
      "estimand": "ATE",
      "estimate": 0.50,
      "std_error": 0.04,
      "ci_lower": 0.42,
      "ci_upper": 0.58,
      "p_value": 0.0001,
      "assumptions_tested": ["Positivity", "Overlap"]
    }
  ],
  "sensitivity_analysis": [
    {
      "method": "E-value",
      "robustness_value": 3.2,
      "interpretation": "Strong robustness: a very strong unmeasured confounder would be needed."
    }
  ],
  "recommendations": ["Consider collecting data on additional confounders."],
  "notebook_url": "/jobs/a1b2c3d4/notebook"
}
```

**Errors**: 400 (job not completed), 404 (not found or no results).

---

### GET /jobs/{job_id}/notebook

Download the generated Jupyter notebook as a file.

**Response**: File download (`application/x-ipynb+json`), filename `causal_analysis_{job_id}.ipynb`.

Security: The endpoint validates that the notebook path:
- Is not a symlink
- Resolves within the expected temp directory (`{tempdir}/causal_orchestrator/`)
- Is a regular file
- Passes TOCTOU inode verification

**Errors**: 403 (path traversal or symlink detected), 404 (notebook not found).

---

### GET /jobs/{job_id}/traces

Get agent reasoning traces for observability and debugging.

**Response** (200):

```json
{
  "job_id": "a1b2c3d4",
  "traces": [
    {
      "agent_name": "data_profiler",
      "timestamp": "2025-02-15T12:01:30Z",
      "action": "step_1_get_dataset_overview",
      "reasoning": "Starting by examining the dataset structure...",
      "duration_ms": 1250
    }
  ]
}
```

**Errors**: 404 (not found).

---

## Health Endpoints

Source: `backend/src/api/routes/health.py`

### GET /health

Returns `{"status": "healthy"}`. No authentication required. Used by Cloud Run probes and load balancers.

### GET /

Returns API metadata: name, version, description, doc URLs, and endpoint map.

---

## Error Format

All error responses follow FastAPI's standard format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

HTTP status codes:
- 400: Bad request (invalid parameters, job not in expected state)
- 401: Missing or invalid API key
- 404: Resource not found
- 409: Conflict (trying to delete a running job without `force=true`)
- 422: Validation error (invalid request body)
- 429: Rate limit exceeded
- 500: Internal server error

---

## Rate Limiting

`POST /jobs` is rate limited to 10 requests per minute per client using `slowapi`. Other endpoints are not rate limited.

---

## CORS

CORS is configured via `settings.cors_origins`. Allowed methods: GET, POST, DELETE, OPTIONS. Allowed headers: Content-Type, X-API-Key, Accept.
