# Skill: Loading States

Load when: implementing loading indicators, error handling, progress, or cancel.

## The 4 states — every component must handle all of them

```typescript
type JobState = "idle" | "loading" | "success" | "error";
```

### idle
- Job creation form visible
- No spinners, no error messages

### loading
- Progress bar or stage indicator
- Current pipeline stage name displayed
- Cancel button available
- SSE stream active for real-time updates

### success
- Results displayed (treatment effects, DAG, diagnostics)
- Download notebook button
- New analysis button to return to idle

### error
- Error message displayed with context
- Retry button or new analysis button
- Clear recovery path

## Pipeline progress display

Show the current agent stage with progress percentage:

```
profiling          ████░░░░░░  20%
exploratory        ████████░░  32%
discovering_causal ██████████  44%
estimating_effects ████████████████░░  56%
```

Use the status-to-progress mapping from pipeline.md.

## Cancel pattern

```typescript
const handleCancel = async () => {
  await api.cancelJob(jobId);
  // Close SSE stream
  eventSource?.close();
  // Update state to idle or cancelled
};
```

## Rules

1. Never show a blank screen during loading
2. Always show which pipeline stage is running
3. Cancel must be available during the entire loading state
4. Error messages must suggest a next action (retry or new analysis)
5. SSE disconnect should fall back to polling, not error state
