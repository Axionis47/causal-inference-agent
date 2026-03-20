# Skill: API Integration

Load when: connecting frontend to backend API or SSE streams.

## API client

All API calls go through frontend/src/services/api.ts.
Never use fetch() directly in components.

## Current API endpoints

```typescript
// POST /jobs — create analysis job
createJob(request: CreateJobRequest): Promise<CreateJobResponse>

// GET /jobs/{id}/status — poll job status
getJobStatus(jobId: string): Promise<JobStatus>

// GET /jobs/{id}/results — get analysis results
getJobResults(jobId: string): Promise<AnalysisResults>

// GET /jobs/{id}/stream — SSE stream for real-time progress
streamJob(jobId: string): EventSource

// GET /jobs/{id}/notebook — download notebook
getNotebook(jobId: string): Promise<Blob>

// GET /jobs — list all jobs
listJobs(): Promise<Job[]>

// POST /jobs/{id}/cancel — cancel running job
cancelJob(jobId: string): Promise<void>
```

## SSE streaming pattern

```typescript
const eventSource = new EventSource(`/jobs/${jobId}/stream`);
eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // Update progress state
};
eventSource.onerror = () => {
  eventSource.close();
  // Fall back to polling
};
```

## Rules

1. All API calls through services/api.ts
2. SSE for real-time progress, polling as fallback
3. Handle network errors with user-facing messages
4. Cancel ongoing requests when component unmounts
5. Type all request/response interfaces in types/index.ts
