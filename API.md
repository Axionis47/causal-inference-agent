# API Reference

This document describes the REST API for the Causal Orchestrator system.

---

## Base URL

```
http://localhost:8000/api/v1
```

All endpoints are prefixed with `/api/v1`.

---

## Authentication

Currently, the API does not require authentication. In production, add authentication middleware.

---

## Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/jobs` | Create a new analysis job |
| GET | `/jobs` | List all jobs |
| GET | `/jobs/{id}` | Get job details |
| GET | `/jobs/{id}/status` | Get lightweight status |
| GET | `/jobs/{id}/results` | Get analysis results |
| GET | `/jobs/{id}/notebook` | Download Jupyter notebook |
| GET | `/jobs/{id}/traces` | Get agent reasoning traces |
| POST | `/jobs/{id}/cancel` | Cancel a running job |
| DELETE | `/jobs/{id}` | Delete a job |

---

## Create Job

**POST** `/jobs`

Submit a Kaggle dataset URL to start causal inference analysis.

### Request Body

```json
{
  "kaggle_url": "https://www.kaggle.com/datasets/owner/dataset-name",
  "treatment_variable": "treatment",
  "outcome_variable": "outcome",
  "analysis_preferences": {
    "methods": ["ipw", "aipw", "matching"]
  },
  "orchestrator_mode": "standard"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `kaggle_url` | string | Yes | Kaggle dataset URL |
| `treatment_variable` | string | No | Hint for treatment variable |
| `outcome_variable` | string | No | Hint for outcome variable |
| `analysis_preferences` | object | No | Analysis configuration |
| `orchestrator_mode` | string | No | "standard" or "react" (default: "standard") |

### URL Validation

The Kaggle URL must match:
```
https://www.kaggle.com/datasets/{owner}/{dataset-name}
```

### Response

**201 Created**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "kaggle_url": "https://www.kaggle.com/datasets/owner/dataset-name",
  "status": "pending",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

### Errors

| Status | Description |
|--------|-------------|
| 400 | Invalid Kaggle URL format |
| 500 | Failed to create job |

### Example

```bash
curl -X POST http://localhost:8000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "kaggle_url": "https://www.kaggle.com/datasets/lalonde/nsw-job-training",
    "treatment_variable": "treat",
    "outcome_variable": "re78"
  }'
```

---

## List Jobs

**GET** `/jobs`

List all jobs with optional filtering and pagination.

### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `status` | string | null | Filter by job status |
| `limit` | integer | 20 | Max results (1-100) |
| `offset` | integer | 0 | Skip N results |

### Valid Status Values

```
pending, fetching_data, profiling, exploratory_analysis,
discovering_causal, estimating_effects, sensitivity_analysis,
critique_review, iterating, generating_notebook,
completed, failed, cancelling, cancelled
```

### Response

**200 OK**

```json
{
  "jobs": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "kaggle_url": "https://www.kaggle.com/datasets/owner/dataset",
      "status": "completed",
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-01-15T10:35:00Z"
    }
  ],
  "total": 42,
  "limit": 20,
  "offset": 0
}
```

### Example

```bash
# List all jobs
curl http://localhost:8000/api/v1/jobs

# Filter by status
curl "http://localhost:8000/api/v1/jobs?status=completed"

# Pagination
curl "http://localhost:8000/api/v1/jobs?limit=10&offset=20"
```

---

## Get Job Details

**GET** `/jobs/{job_id}`

Get detailed information about a specific job.

### Response

**200 OK**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "kaggle_url": "https://www.kaggle.com/datasets/owner/dataset",
  "status": "estimating_effects",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:33:00Z",
  "dataset_name": "nsw-job-training",
  "current_agent": "effect_estimator",
  "iteration_count": 0,
  "error_message": null,
  "progress_percentage": 65,
  "treatment_variable": "treat",
  "outcome_variable": "re78"
}
```

| Field | Description |
|-------|-------------|
| `dataset_name` | Name of the dataset |
| `current_agent` | Agent currently running |
| `iteration_count` | Number of critique iterations |
| `error_message` | Error details if failed |
| `progress_percentage` | Estimated progress (0-100) |

### Errors

| Status | Description |
|--------|-------------|
| 404 | Job not found |

---

## Get Job Status (Lightweight)

**GET** `/jobs/{job_id}/status`

Get minimal status information. Optimized for polling.

### Response

**200 OK**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "estimating_effects",
  "progress_percentage": 65,
  "current_agent": "effect_estimator"
}
```

Use this endpoint for frequent polling instead of the full details endpoint.

---

## Get Analysis Results

**GET** `/jobs/{job_id}/results`

Get complete analysis results for a completed job.

### Response

**200 OK**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "treatment_variable": "treat",
  "outcome_variable": "re78",

  "executive_summary": {
    "headline": "Job training program increases earnings by $1,794",
    "effect_direction": "positive",
    "confidence_level": "high",
    "key_findings": [
      "Treatment effect is statistically significant (p < 0.01)",
      "All three methods agree on positive effect",
      "Results robust to moderate unmeasured confounding (E-value = 2.3)"
    ]
  },

  "method_consensus": {
    "n_methods": 3,
    "direction_agreement": 1.0,
    "all_significant": true,
    "estimate_range": [1543, 2102],
    "median_estimate": 1794,
    "consensus_strength": "strong"
  },

  "data_context": {
    "n_samples": 445,
    "n_features": 12,
    "n_treated": 185,
    "n_control": 260,
    "missing_data_pct": 0.0,
    "data_quality_issues": []
  },

  "causal_graph": {
    "nodes": ["treat", "re78", "age", "education", "married"],
    "edges": [
      {"from": "age", "to": "treat", "confidence": 0.85},
      {"from": "age", "to": "re78", "confidence": 0.92},
      {"from": "education", "to": "re78", "confidence": 0.78},
      {"from": "treat", "to": "re78", "confidence": 0.88}
    ],
    "discovery_method": "PC + Bootstrap",
    "interpretation": "Age and education are confounders. No colliders detected."
  },

  "treatment_effects": [
    {
      "method": "IPW",
      "estimand": "ATE",
      "estimate": 1794.34,
      "std_error": 312.45,
      "ci_lower": 1181.94,
      "ci_upper": 2406.74,
      "p_value": 0.001,
      "assumptions_tested": ["Positivity", "Unconfoundedness"]
    },
    {
      "method": "AIPW",
      "estimand": "ATE",
      "estimate": 1856.21,
      "std_error": 298.12,
      "ci_lower": 1271.89,
      "ci_upper": 2440.53,
      "p_value": 0.001,
      "assumptions_tested": ["Positivity", "Unconfoundedness", "Outcome model"]
    },
    {
      "method": "Matching",
      "estimand": "ATT",
      "estimate": 1732.45,
      "std_error": 345.67,
      "ci_lower": 1054.94,
      "ci_upper": 2409.96,
      "p_value": 0.003,
      "assumptions_tested": ["Common support", "Unconfoundedness"]
    }
  ],

  "sensitivity_analysis": [
    {
      "method": "E-value",
      "robustness_value": 2.3,
      "interpretation": "Would need unmeasured confounder with RR > 2.3 to explain away effect"
    },
    {
      "method": "Rosenbaum Bounds",
      "robustness_value": 1.8,
      "interpretation": "Effect robust to hidden bias up to Gamma = 1.8"
    }
  ],

  "recommendations": [
    "Consider collecting additional confounders if available",
    "Effect is robust to moderate unmeasured confounding",
    "Multiple methods agree, increasing confidence in findings"
  ],

  "notebook_url": "/jobs/550e8400-e29b-41d4-a716-446655440000/notebook"
}
```

### Key Response Fields

**executive_summary:**
- `headline`: One-line key finding
- `effect_direction`: "positive", "negative", "null", or "mixed"
- `confidence_level`: "high", "medium", or "low"
- `key_findings`: Bullet points

**method_consensus:**
- `direction_agreement`: 1.0 means all methods agree on direction
- `consensus_strength`: "strong" (all agree, all significant), "moderate", or "weak"

**treatment_effects:**
- `estimand`: ATE (Average Treatment Effect), ATT (on Treated), or LATE (Local)
- `p_value`: Two-sided p-value (null if not computed)

### Errors

| Status | Description |
|--------|-------------|
| 400 | Job not completed |
| 404 | Job or results not found |

---

## Download Notebook

**GET** `/jobs/{job_id}/notebook`

Download the generated Jupyter notebook.

### Response

**200 OK**

Returns the notebook file with:
- Content-Type: `application/x-ipynb+json`
- Filename: `causal_analysis_{job_id}.ipynb`

### Security

The endpoint validates:
- Path is not a symlink
- Path is within the allowed temp directory
- File exists and is a regular file

### Errors

| Status | Description |
|--------|-------------|
| 403 | Invalid notebook path (security violation) |
| 404 | Notebook not found |

### Example

```bash
curl -o analysis.ipynb \
  http://localhost:8000/api/v1/jobs/550e8400-e29b-41d4-a716-446655440000/notebook
```

---

## Get Agent Traces

**GET** `/jobs/{job_id}/traces`

Get agent reasoning traces for debugging and observability.

### Response

**200 OK**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "traces": [
    {
      "agent_name": "data_profiler",
      "timestamp": "2024-01-15T10:30:15Z",
      "action": "get_dataset_overview",
      "reasoning": "Starting analysis. Need to understand dataset structure.",
      "duration_ms": 245
    },
    {
      "agent_name": "data_profiler",
      "timestamp": "2024-01-15T10:30:18Z",
      "action": "check_treatment_balance",
      "reasoning": "Dataset has 'treat' column. Checking if suitable as treatment.",
      "duration_ms": 123
    },
    {
      "agent_name": "effect_estimator",
      "timestamp": "2024-01-15T10:32:45Z",
      "action": "run_ipw",
      "reasoning": "DAG suggests age and education as confounders. Running IPW.",
      "duration_ms": 1892
    }
  ]
}
```

Traces show the agent's reasoning at each step, useful for:
- Debugging unexpected results
- Understanding why certain methods were chosen
- Verifying the analysis process

---

## Cancel Job

**POST** `/jobs/{job_id}/cancel`

Cancel a running job gracefully.

### Response

**200 OK**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "was_running": true,
  "cancelled": true,
  "status": "cancelled"
}
```

| Field | Description |
|-------|-------------|
| `was_running` | True if job was actually running |
| `cancelled` | True if cancellation succeeded |
| `status` | Final job status |

### Behavior

- If job is running: Gracefully stops and marks as cancelled
- If job is completed/failed/cancelled: No-op, returns current state

### Errors

| Status | Description |
|--------|-------------|
| 404 | Job not found |

---

## Delete Job

**DELETE** `/jobs/{job_id}`

Permanently delete a job and all its artifacts.

### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `force` | boolean | false | Cancel running job before deletion |

### What Gets Deleted

- Job record from database
- Analysis results from database
- Agent traces from database
- Local temp files (DataFrame, notebook)

### Response

**200 OK**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "found": true,
  "cancelled": false,
  "firestore_deleted": true,
  "local_artifacts_deleted": {
    "dataframe": true,
    "notebook": true
  }
}
```

### Errors

| Status | Description |
|--------|-------------|
| 404 | Job not found |
| 409 | Job is running (use force=true) |

### Example

```bash
# Delete completed job
curl -X DELETE http://localhost:8000/api/v1/jobs/550e8400-...

# Force delete running job
curl -X DELETE "http://localhost:8000/api/v1/jobs/550e8400-...?force=true"
```

---

## Job Status Lifecycle

```
pending
    ↓
fetching_data ────────► (downloading from Kaggle)
    ↓
profiling ────────────► (Data Profiler running)
    ↓
exploratory_analysis ──► (EDA Agent running)
    ↓
discovering_causal ───► (Causal Discovery running)
    ↓
estimating_effects ───► (Effect Estimator running)
    ↓
sensitivity_analysis ─► (Sensitivity Analyst running)
    ↓
critique_review ──────► (Critique Agent reviewing)
    │
    ├──► iterating ──► (back to relevant agent)
    │
    ↓
generating_notebook ──► (Notebook Generator running)
    ↓
completed

At any point:
    ├──► failed (error occurred)
    ├──► cancelling (user requested cancel)
    └──► cancelled (cancel completed)
```

---

## Error Response Format

All errors return JSON with this format:

```json
{
  "detail": "Human-readable error message"
}
```

### Common HTTP Status Codes

| Status | Meaning |
|--------|---------|
| 200 | Success |
| 201 | Created |
| 400 | Bad request (invalid input) |
| 404 | Not found |
| 409 | Conflict (e.g., job running) |
| 500 | Internal server error |

---

## Polling Strategy

For frontend integration, recommended polling:

```javascript
async function pollJobStatus(jobId) {
  const POLL_INTERVAL = 2000; // 2 seconds

  while (true) {
    const response = await fetch(`/api/v1/jobs/${jobId}/status`);
    const status = await response.json();

    updateUI(status);

    if (['completed', 'failed', 'cancelled'].includes(status.status)) {
      break;
    }

    await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL));
  }
}
```

Use `/jobs/{id}/status` for polling (lightweight) and `/jobs/{id}` for full details when needed.

---

## Rate Limits

No rate limits currently implemented. In production, consider:
- 100 requests per minute per IP
- 10 concurrent jobs per user

---

## CORS

CORS is configured to allow requests from:
- `http://localhost:5173` (Vite dev server)
- `http://localhost:3000` (React dev server)

Configure additional origins in `src/api/main.py`.

---

## Health Check

**GET** `/health`

Returns server health status.

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

Use this for load balancer health checks.
