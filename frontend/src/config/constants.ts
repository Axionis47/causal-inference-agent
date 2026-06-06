// API
export const API_REQUEST_TIMEOUT_MS = 30000;

// Polling intervals
export const DEFAULT_POLL_INTERVAL_MS = 2000;
export const JOB_DETAIL_POLL_INTERVAL_MS = 1000;
export const JOBS_LIST_REFRESH_INTERVAL_MS = 10000;
// Token totals change once per agent, so they don't need the 1s job cadence.
// A gentler interval keeps the /traces poll well under its rate cap.
export const TRACES_POLL_INTERVAL_MS = 5000;
// Dataset blocks resolve over a few seconds during the fetch stage.
export const DATASET_VIEW_POLL_INTERVAL_MS = 2000;

// Toast durations
export const TOAST_DEFAULT_DURATION_MS = 4000;
export const TOAST_SUCCESS_DURATION_MS = 3000;
export const TOAST_ERROR_DURATION_MS = 5000;

// Pagination
export const DEFAULT_PAGE_SIZE = 20;
// Raw dataset rows are fetched a page at a time from the rows endpoint.
export const DATASET_ROWS_PAGE_SIZE = 50;

// Display limits
export const MAX_AGENT_EVENTS = 20;

// Statistical thresholds
export const SIGNIFICANCE_ALPHA = 0.05;

// Animation
export const RESULTS_ANIMATION_DELAY_MS = 50;
