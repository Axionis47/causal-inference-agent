/**
 * TypeScript type definitions for the Causal Orchestrator frontend
 */

// Re-export API types
export type {
  Job,
  JobDetail,
  JobStatus,
  TreatmentEffect,
  CausalGraph,
  SensitivityResult,
  AnalysisResults,
  AgentTrace,
  CreateJobRequest,
} from '../services/api';

/**
 * Job status enum values
 */
export type JobStatusValue =
  | 'pending'
  | 'fetching_data'
  | 'profiling'
  | 'exploratory_analysis'
  | 'discovering_causal'
  | 'estimating_effects'
  | 'sensitivity_analysis'
  | 'critique_review'
  | 'iterating'
  | 'generating_notebook'
  | 'completed'
  | 'failed';

/**
 * Status categories for UI display
 */
export type StatusCategory = 'pending' | 'running' | 'completed' | 'failed';

/**
 * Map job status to category
 */
export function getStatusCategory(status: JobStatusValue): StatusCategory {
  if (status === 'pending') return 'pending';
  if (status === 'completed') return 'completed';
  if (status === 'failed') return 'failed';
  return 'running';
}

/**
 * Human-readable status labels
 */
export const STATUS_LABELS: Record<JobStatusValue, string> = {
  pending: 'Pending',
  fetching_data: 'Fetching Data',
  profiling: 'Profiling Dataset',
  exploratory_analysis: 'Exploratory Analysis',
  discovering_causal: 'Discovering Causal Structure',
  estimating_effects: 'Estimating Effects',
  sensitivity_analysis: 'Sensitivity Analysis',
  critique_review: 'Reviewing Analysis',
  iterating: 'Iterating',
  generating_notebook: 'Generating Notebook',
  completed: 'Completed',
  failed: 'Failed',
};

/**
 * Agent names used in the system
 */
export type AgentName =
  | 'orchestrator'
  | 'data_profiler'
  | 'eda_agent'
  | 'causal_discovery'
  | 'effect_estimator'
  | 'sensitivity_analyst'
  | 'notebook_generator'
  | 'critique';

/**
 * Human-readable agent labels
 */
export const AGENT_LABELS: Record<AgentName, string> = {
  orchestrator: 'Orchestrator',
  data_profiler: 'Data Profiler',
  eda_agent: 'EDA Agent',
  causal_discovery: 'Causal Discovery',
  effect_estimator: 'Effect Estimator',
  sensitivity_analyst: 'Sensitivity Analyst',
  notebook_generator: 'Notebook Generator',
  critique: 'Critique Agent',
};

/**
 * Causal inference method types
 */
export type CausalMethod =
  | 'ols'
  | 'psm'
  | 'ipw'
  | 'aipw'
  | 'did'
  | 'iv'
  | 'rdd'
  | 's_learner'
  | 't_learner'
  | 'x_learner'
  | 'causal_forest'
  | 'double_ml';

/**
 * Human-readable method labels
 */
export const METHOD_LABELS: Record<CausalMethod, string> = {
  ols: 'OLS Regression',
  psm: 'Propensity Score Matching',
  ipw: 'Inverse Probability Weighting',
  aipw: 'Doubly Robust (AIPW)',
  did: 'Difference-in-Differences',
  iv: 'Instrumental Variables',
  rdd: 'Regression Discontinuity',
  s_learner: 'S-Learner',
  t_learner: 'T-Learner',
  x_learner: 'X-Learner',
  causal_forest: 'Causal Forest',
  double_ml: 'Double ML',
};

/**
 * Estimand types
 */
export type Estimand = 'ATE' | 'ATT' | 'ATC' | 'LATE' | 'CATE';

/**
 * Estimand descriptions
 */
export const ESTIMAND_DESCRIPTIONS: Record<Estimand, string> = {
  ATE: 'Average Treatment Effect',
  ATT: 'Average Treatment Effect on the Treated',
  ATC: 'Average Treatment Effect on the Control',
  LATE: 'Local Average Treatment Effect',
  CATE: 'Conditional Average Treatment Effect',
};

/**
 * Graph edge types
 */
export type EdgeType = 'directed' | 'undirected' | 'bidirected';

/**
 * Pagination state
 */
export interface PaginationState {
  currentPage: number;
  pageSize: number;
  totalPages: number;
  totalItems: number;
  hasNextPage: boolean;
  hasPrevPage: boolean;
}

/**
 * Filter options for jobs list
 */
export interface JobFilters {
  status?: JobStatusValue;
  dateFrom?: Date;
  dateTo?: Date;
  searchQuery?: string;
}

/**
 * Sort options
 */
export interface SortOptions {
  field: 'created_at' | 'updated_at' | 'status';
  direction: 'asc' | 'desc';
}

/**
 * Form state for job creation
 */
export interface CreateJobFormState {
  kaggleUrl: string;
  treatmentVariable?: string;
  outcomeVariable?: string;
  orchestratorMode?: 'standard' | 'react';
}

/**
 * Validation errors
 */
export interface ValidationErrors {
  kaggleUrl?: string;
  treatmentVariable?: string;
  outcomeVariable?: string;
}

/**
 * Theme mode
 */
export type ThemeMode = 'light' | 'dark' | 'system';

/**
 * Toast notification types
 */
export type ToastType = 'success' | 'error' | 'warning' | 'info';

/**
 * Toast notification
 */
export interface Toast {
  id: string;
  type: ToastType;
  title: string;
  message?: string;
  duration?: number;
}
