import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

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

export interface AnalysisResults {
  job_id: string;
  treatment_variable?: string;
  outcome_variable?: string;
  executive_summary?: ExecutiveSummary;
  method_consensus?: MethodConsensus;
  data_context?: DataContext;
  causal_graph?: CausalGraph;
  treatment_effects: TreatmentEffect[];
  sensitivity_analysis: SensitivityResult[];
  recommendations: string[];
  notebook_url?: string;
}

export interface AgentTrace {
  agent_name: string;
  timestamp: string;
  action: string;
  reasoning: string;
  duration_ms: number;
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

export async function getTraces(jobId: string): Promise<AgentTrace[]> {
  const response = await api.get(`/jobs/${jobId}/traces`);
  return response.data.traces;
}

export function getNotebookUrl(jobId: string): string {
  return `${API_BASE_URL}/jobs/${jobId}/notebook`;
}

export default api;
