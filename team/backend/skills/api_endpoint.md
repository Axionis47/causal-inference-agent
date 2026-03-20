# Skill: API Endpoint

Load when: creating or modifying FastAPI endpoints in backend/src/api/.

## Current endpoints

```
POST /jobs                    Create analysis job
GET  /jobs                    List all jobs
GET  /jobs/{job_id}/status    Get job status
GET  /jobs/{job_id}/results   Get analysis results
GET  /jobs/{job_id}/notebook  Download notebook file
GET  /jobs/{job_id}/stream    SSE stream for real-time progress
GET  /jobs/{job_id}/traces    Get agent traces
DELETE /jobs/{job_id}         Delete job and artifacts
POST /jobs/{job_id}/cancel    Cancel running job
GET  /health                  Health check
```

## Route pattern

```python
# backend/src/api/routes/jobs.py
from fastapi import APIRouter, Depends
from src.api.schemas.job import CreateJobRequest, JobStatusResponse

router = APIRouter(prefix="/jobs", tags=["jobs"])

@router.post("/", status_code=201)
async def create_job(request: CreateJobRequest):
    ...

@router.get("/{job_id}/status")
async def get_job_status(job_id: str):
    ...
```

## Rules

1. All request/response schemas in src/api/schemas/
2. Use Pydantic models for validation
3. Rate limiting via slowapi (10 req/min on POST /jobs)
4. Auth via X-API-Key header (disabled in dev when API_KEY not set)
5. CORS configured in src/api/main.py
6. SSE streaming for real-time progress updates
7. Return appropriate HTTP status codes (201 for create, 404 for not found, 429 for rate limit)
