import axios, { AxiosError } from 'axios';
import toast from 'react-hot-toast';
import { API_REQUEST_TIMEOUT_MS } from '../config/constants';

// Runtime config (from docker-entrypoint.sh) > build-time config > fallback
interface RuntimeConfig {
  API_URL?: string;
  API_KEY?: string;
}
const runtimeConfig = (window as unknown as { __CONFIG__?: RuntimeConfig }).__CONFIG__;
const API_BASE_URL = runtimeConfig?.API_URL || import.meta.env.VITE_API_URL || '/api';
const API_KEY = runtimeConfig?.API_KEY || import.meta.env.VITE_API_KEY || '';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: API_REQUEST_TIMEOUT_MS,
  headers: {
    'Content-Type': 'application/json',
    ...(API_KEY ? { 'X-API-Key': API_KEY } : {}),
  },
});

// --- Structured error handling ---

export interface ApiError {
  status: number;
  message: string;
  details?: unknown;
}

function toApiError(error: AxiosError): ApiError {
  if (error.response) {
    const data = error.response.data as Record<string, unknown> | undefined;
    return {
      status: error.response.status,
      message:
        (data?.detail as string) ||
        (data?.message as string) ||
        `Request failed with status ${error.response.status}`,
      details: data,
    };
  }
  if (error.code === 'ECONNABORTED') {
    return { status: 0, message: 'Request timed out. Please try again.' };
  }
  if (error.code === 'ERR_CANCELED') {
    return { status: 0, message: 'Request was cancelled.' };
  }
  return { status: 0, message: error.message || 'Network error. Please check your connection.' };
}

api.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    const apiError = toApiError(error);

    // Don't toast for cancelled requests or network errors during polling
    if (error.code !== 'ERR_CANCELED' && apiError.status !== 0) {
      if (apiError.status === 401) {
        toast.error('Authentication failed. Check your API key.');
      } else if (apiError.status === 429) {
        toast.error('Rate limit exceeded. Please wait a moment.');
      } else if (apiError.status >= 500) {
        toast.error('Server error. Please try again later.');
      }
    }

    return Promise.reject(apiError);
  }
);

export type OrchestratorMode = 'standard' | 'react';

export interface CreateJobRequest {
  kaggle_url: string;
  treatment_variable?: string;
  outcome_variable?: string;
  orchestrator_mode?: OrchestratorMode;
}

export interface Job {
  id: string;
  kaggle_url: string;
  status: string;
  created_at: string;
  updated_at: string;
  // Digest fields populated by /jobs list. Optional because just-
  // created jobs won't have them populated yet.
  dataset_name?: string | null;
  treatment_variable?: string | null;
  outcome_variable?: string | null;
  iteration_count?: number;
}

export interface JobDetail extends Job {
  dataset_name?: string;
  current_agent?: string;
  iteration_count: number;
  error_message?: string;
  progress_percentage: number;
  treatment_variable?: string;
  outcome_variable?: string;
}

export interface JobStatus {
  id: string;
  status: string;
  progress_percentage: number;
  current_agent?: string;
}

export interface TreatmentEffect {
  method: string;
  estimand: string;
  estimate: number;
  std_error: number;
  ci_lower: number;
  ci_upper: number;
  p_value?: number;
  assumptions_tested: string[];
}

export interface CausalGraph {
  nodes: string[];
  edges: { source: string; target: string; type: string }[];
  discovery_method: string;
  interpretation?: string;
}

export interface SensitivityResult {
  method: string;
  robustness_value: number;
  interpretation: string;
}

export interface MethodConsensus {
  n_methods: number;
  direction_agreement: number;
  all_significant: boolean;
  estimate_range: [number, number];
  median_estimate: number;
  consensus_strength: 'strong' | 'moderate' | 'weak';
}

export interface DataContext {
  n_samples: number;
  n_features: number;
  n_treated?: number;
  n_control?: number;
  missing_data_pct: number;
  data_quality_issues: string[];
}

export interface ExecutiveSummary {
  headline: string;
  effect_direction: 'positive' | 'negative' | 'null' | 'mixed';
  confidence_level: 'high' | 'medium' | 'low';
  key_findings: string[];
}

export interface DecisionAlternative {
  option: string;
  reason: string;
}

export interface MethodologyDecision {
  agent: string;
  decision_type: string;
  choice: string;
  reason: string;
  alternatives?: DecisionAlternative[];
  timestamp: string;
}

export interface AnalysisResults {
  job_id: string;
  treatment_variable?: string;
  outcome_variable?: string;
  narrative_summary?: string;
  executive_summary?: ExecutiveSummary;
  method_consensus?: MethodConsensus;
  data_context?: DataContext;
  causal_graph?: CausalGraph;
  treatment_effects: TreatmentEffect[];
  sensitivity_analysis: SensitivityResult[];
  recommendations: string[];
  notebook_url?: string;
  decision_log?: MethodologyDecision[];
}

export interface AgentEvent {
  timestamp: string;
  agent_name: string;
  // A backend-defined discriminator. Beyond agent_started/agent_completed this
  // also carries gate events (approval_required, dag_approval_required, ...) and
  // the live per-agent visibility events (agent_finding, agent_challenge), all
  // routed through the same channel, so it is an open string.
  event_type: string;
  data: Record<string, unknown>;
}

// Payload of an `agent_finding` event: one per agent when it clears the
// readiness gate, built from its sealed brief. Emitted by
// orchestrator/common/visibility.py::publish_brief_events.
export interface AgentFindingPayload {
  agent_name: string;
  headline: string;
  status: string;
  flags: string[];
}

// Payload of an `agent_challenge` event: emitted only when the brief is not
// clean (refused, failed, or carrying a quality flag). Adds the raised issues.
export interface AgentChallengePayload extends AgentFindingPayload {
  issues: string[];
}

// Dataset lifecycle events arrive on the same SSE channel ('agent_event')
// but carry a different event_type discriminator and a different payload.
// The Data panel listens for these to fill in its three blocks live.
export type DatasetEventType =
  | 'dataset_metadata_started'
  | 'dataset_metadata_ready'
  | 'dataset_metadata_failed'
  | 'dataset_download_started'
  | 'dataset_download_complete'
  | 'dataset_load_failed'
  | 'data_profile_ready';

export interface DatasetSseEvent {
  event_type: DatasetEventType;
  data: Record<string, unknown>;
  timestamp: string;
}

export interface FileEntry {
  name: string;
  size_bytes: number;
  format: string;
  used: boolean;
  // From the manifest; absent until the download completes.
  columns?: string[];
  n_rows?: number | null;
  // False for non-tabular files (images, pdfs, freeform text): listed but not
  // previewable. Optional because SSE-derived file lists omit it.
  tabular?: boolean;
}

export interface KaggleMeta {
  description: string | null;
  column_descriptions: Record<string, string>;
  tags: string[];
  domain: string | null;
  metadata_quality: string;
  // Widened at the data-review gate (full Kaggle payload). Optional because
  // the SSE-derived metadata event and older payloads carry only the subset
  // above; the /dataset endpoint always populates these.
  title?: string | null;
  subtitle?: string | null;
  source?: string | null;
  license?: string | null;
  keywords?: string[];
  total_size?: number | null;
  download_count?: number | null;
  vote_count?: number | null;
  usability_rating?: number | null;
}

export interface DataProfileSummary {
  n_samples: number;
  n_features: number;
  // The fields below come from the live profiler. The persisted-results
  // path (completed/evicted jobs) doesn't currently include them, so
  // they must be treated as optional on the frontend.
  feature_types?: Record<string, string>;
  missing_values?: Record<string, number>;
  treatment_candidates: string[];
  outcome_candidates: string[];
  potential_confounders?: string[];
  potential_instruments?: string[];
}

export type BlockStatus =
  | 'pending'
  | 'downloading'
  | 'downloaded'
  | 'loaded'
  | 'unavailable'
  | 'error'
  | 'failed';

export interface DownloadBlock {
  status: BlockStatus;
  url: string | null;
  files: FileEntry[];
  error: string | null;
}

export interface KaggleMetaBlock {
  status: BlockStatus;
  data: KaggleMeta | null;
  error: string | null;
}

export interface ProfileBlock {
  status: BlockStatus;
  data: DataProfileSummary | null;
  error: string | null;
}

export interface DatasetView {
  download: DownloadBlock;
  kaggle_meta: KaggleMetaBlock;
  profile: ProfileBlock;
}

// One page of raw rows for a single downloaded file, fetched on demand.
export interface DatasetRowsPage {
  columns: string[];
  rows: Record<string, unknown>[];
  total_rows: number | null;
  offset: number;
  limit: number;
}

export interface ApprovalRequestBody {
  decision: 'approved' | 'rejected' | 'revise';
  reason?: string;
  appended_context?: string;
}

// Payload carried by the `dag_approval_required` SSE event (and the DAG gate
// snapshot). The frontend renders the graph from `dag` and the review text from
// `justification`. See backend orchestrator/base.py::_build_dag_gate_payload.
export interface DagEdgeView {
  source: string;
  target: string;
  edge_type: string;
}

export interface DagSummary {
  nodes: string[];
  edges: DagEdgeView[];
  adjustment_set: string[];
  variable_roles: Record<string, string>;
  forbidden_edges: Record<string, string>[];
  discovery_method: string;
  interpretation: string;
}

export interface DagJustification {
  identification: string;
  adjustment_set: string[];
  interpretation: string;
  flags: string[];
  concerns: string[];
}

export interface DagGatePayload {
  gate: 'dag';
  treatment_variable: string | null;
  outcome_variable: string | null;
  dag: DagSummary | null;
  justification: DagJustification | null;
}

// How the files in a multi-file bundle relate. Descriptive only; one file still
// feeds the analysis. Carried on the dataset_inspection_complete SSE event
// (`relational_profile`) and the data-gate payload (`relational`). See backend
// src/domain/relational.py.
export interface RelationalFile {
  file: string;
  n_rows: number;
  n_cols: number;
  key_candidates: string[];
}

export interface RelationalProfilePayload {
  shape_hint: 'single' | 'same_schema_shards' | 'multiple_files' | 'unrelated';
  files: RelationalFile[];
  same_schema_groups: string[][];
  note: string;
}

// Payload carried by the `results_approval_required` SSE event. The frontend
// renders a forest plot from `effects` and a balance (Love) plot from
// `balance`. See backend orchestrator/base.py::_build_results_gate_payload.
export interface EffectRow {
  method: string;
  estimand: string;
  estimate: number;
  ci_lower: number;
  ci_upper: number;
  p_value: number | null;
}

export interface SensitivityRow {
  method: string;
  robustness_value: number;
  interpretation: string;
}

export interface BalanceRow {
  covariate: string;
  smd: number | null;
}

export interface ResultsGatePayload {
  gate: 'results';
  treatment_variable: string | null;
  outcome_variable: string | null;
  effects: EffectRow[];
  sensitivity: SensitivityRow[];
  ps_diagnostics: Record<string, unknown> | null;
  balance: BalanceRow[];
}

/** The current human-approval gate for a parked job, rehydrated from the
 *  backend (GET /jobs/:id/approval). `kind` names the gate; `payload` is that
 *  gate's snapshot, the same shape its SSE event carried (cast to
 *  DagGatePayload / ResultsGatePayload by the consumer). */
export interface GateSnapshot {
  kind: 'data' | 'dag' | 'results';
  payload: Record<string, unknown>;
}

export interface ApprovalResult {
  job_id: string;
  resumed: boolean;
  status: string;
}

export interface AgentTrace {
  agent_name: string;
  timestamp: string;
  action: string;
  reasoning: string;
  duration_ms: number;
  inputs: Record<string, unknown>;
  outputs: Record<string, unknown>;
  tools_called: string[];
  token_usage: { input_tokens?: number; output_tokens?: number };
}

export interface CancelJobResponse {
  job_id: string;
  was_running: boolean;
  cancelled: boolean;
  status?: string;
}

export interface DeleteJobResponse {
  job_id: string;
  found: boolean;
  cancelled: boolean;
  firestore_deleted: boolean;
  local_artifacts_deleted: Record<string, boolean>;
}

// API functions
export async function createJob(request: CreateJobRequest): Promise<Job> {
  const response = await api.post('/jobs', request);
  return response.data;
}

export async function getJob(jobId: string): Promise<JobDetail> {
  const response = await api.get(`/jobs/${jobId}`);
  return response.data;
}

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  const response = await api.get(`/jobs/${jobId}/status`);
  return response.data;
}

export async function listJobs(
  status?: string,
  limit: number = 20,
  offset: number = 0
): Promise<{ jobs: Job[]; total: number }> {
  const params = new URLSearchParams();
  if (status) params.set('status', status);
  params.set('limit', limit.toString());
  params.set('offset', offset.toString());

  const response = await api.get(`/jobs?${params.toString()}`);
  return response.data;
}

export async function cancelJob(jobId: string): Promise<CancelJobResponse> {
  const response = await api.post(`/jobs/${jobId}/cancel`);
  return response.data;
}

export async function deleteJob(jobId: string, force: boolean = false): Promise<DeleteJobResponse> {
  const response = await api.delete(`/jobs/${jobId}`, { params: { force } });
  return response.data;
}

export async function getResults(jobId: string): Promise<AnalysisResults> {
  const response = await api.get(`/jobs/${jobId}/results`);
  return response.data;
}

export async function getDatasetView(jobId: string): Promise<DatasetView> {
  const response = await api.get(`/jobs/${jobId}/dataset`);
  return response.data;
}

export async function getDatasetRows(
  jobId: string,
  fileName: string,
  offset: number,
  limit: number
): Promise<DatasetRowsPage> {
  const response = await api.get(
    `/jobs/${jobId}/dataset/files/${encodeURIComponent(fileName)}/rows`,
    { params: { offset, limit } }
  );
  return response.data;
}

export async function submitApproval(
  jobId: string,
  body: ApprovalRequestBody
): Promise<ApprovalResult> {
  const response = await api.post(`/jobs/${jobId}/approval`, body);
  return response.data;
}

export async function getTraces(jobId: string): Promise<AgentTrace[]> {
  const response = await api.get(`/jobs/${jobId}/traces`);
  return response.data.traces;
}

/** The gate a parked job is at, so the panel survives a refresh / SSE drop /
 *  event-buffer eviction. Returns null when the job is not parked (404). */
export async function getGateSnapshot(jobId: string): Promise<GateSnapshot | null> {
  try {
    const response = await api.get(`/jobs/${jobId}/approval`);
    return response.data as GateSnapshot;
  } catch (e) {
    if ((e as ApiError).status === 404) return null;
    throw e;
  }
}

export function getNotebookUrl(jobId: string): string {
  return `${API_BASE_URL}/jobs/${jobId}/notebook`;
}

export function getStreamUrl(jobId: string): string {
  return `${API_BASE_URL}/jobs/${jobId}/stream`;
}

export { API_BASE_URL };
export default api;
