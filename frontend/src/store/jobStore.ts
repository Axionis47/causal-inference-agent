/**
 * Job Store - Zustand state management for causal analysis jobs.
 *
 * NOTE: Polling is handled exclusively by React Query's refetchInterval
 * in the page components. This store is for shared state only — no manual
 * setInterval polling.
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import {
  createJob as apiCreateJob,
  listJobs as apiListJobs,
  cancelJob as apiCancelJob,
  getResults as apiGetResults,
  getTraces as apiGetTraces,
  Job,
  JobDetail,
  AnalysisResults,
  AgentTrace,
} from '../services/api';

interface JobState {
  // Current job being viewed
  currentJob: JobDetail | null;
  currentJobId: string | null;

  // Job results and traces
  results: AnalysisResults | null;
  traces: AgentTrace[];

  // Jobs list
  jobs: Job[];
  totalJobs: number;

  // Loading states
  isLoading: boolean;
  isCreating: boolean;

  // Error state
  error: string | null;
}

interface JobActions {
  // Job CRUD
  createJob: (kaggleUrl: string, treatment?: string, outcome?: string) => Promise<string>;
  fetchJobs: (status?: string, limit?: number, offset?: number) => Promise<void>;
  cancelJob: (jobId: string) => Promise<void>;

  // Results and traces
  fetchResults: (jobId: string) => Promise<void>;
  fetchTraces: (jobId: string) => Promise<void>;

  // State management
  setCurrentJob: (job: JobDetail | null) => void;
  clearError: () => void;
  reset: () => void;
}

const initialState: JobState = {
  currentJob: null,
  currentJobId: null,
  results: null,
  traces: [],
  jobs: [],
  totalJobs: 0,
  isLoading: false,
  isCreating: false,
  error: null,
};

export const useJobStore = create<JobState & JobActions>()(
  devtools(
    persist(
      (set, _get) => ({
        ...initialState,

        createJob: async (kaggleUrl: string, treatment?: string, outcome?: string) => {
          set({ isCreating: true, error: null });
          try {
            const job = await apiCreateJob({
              kaggle_url: kaggleUrl,
              treatment_variable: treatment,
              outcome_variable: outcome,
            });
            set({ isCreating: false, currentJobId: job.id });
            return job.id;
          } catch (error) {
            const message =
              error instanceof Error ? error.message :
              (error as { message?: string })?.message || 'Failed to create job';
            set({ isCreating: false, error: message });
            throw error;
          }
        },

        fetchJobs: async (status?: string, limit = 20, offset = 0) => {
          set({ isLoading: true, error: null });
          try {
            const response = await apiListJobs(status, limit, offset);
            set({
              jobs: response.jobs,
              totalJobs: response.total,
              isLoading: false,
            });
          } catch (error) {
            const message =
              error instanceof Error ? error.message :
              (error as { message?: string })?.message || 'Failed to fetch jobs';
            set({ isLoading: false, error: message });
          }
        },

        cancelJob: async (jobId: string) => {
          try {
            await apiCancelJob(jobId);
            // React Query will refetch automatically via invalidation in the component
          } catch (error) {
            const message =
              error instanceof Error ? error.message :
              (error as { message?: string })?.message || 'Failed to cancel job';
            set({ error: message });
          }
        },

        fetchResults: async (jobId: string) => {
          try {
            const results = await apiGetResults(jobId);
            set({ results });
          } catch (error) {
            console.error('Failed to fetch results:', error);
          }
        },

        fetchTraces: async (jobId: string) => {
          try {
            const traces = await apiGetTraces(jobId);
            set({ traces });
          } catch (error) {
            console.error('Failed to fetch traces:', error);
          }
        },

        setCurrentJob: (job: JobDetail | null) => {
          set({ currentJob: job, currentJobId: job?.id || null });
        },

        clearError: () => {
          set({ error: null });
        },

        reset: () => {
          set(initialState);
        },
      }),
      {
        name: 'causal-orchestrator-jobs',
        partialize: (state) => ({
          // Only persist recent job IDs, not full state
          currentJobId: state.currentJobId,
        }),
      }
    ),
    { name: 'JobStore' }
  )
);

// Selectors for common use cases
export const selectCurrentJob = (state: JobState) => state.currentJob;
export const selectIsJobComplete = (state: JobState) =>
  state.currentJob?.status === 'completed';
export const selectIsJobFailed = (state: JobState) =>
  state.currentJob?.status === 'failed';
export const selectIsJobRunning = (state: JobState) =>
  state.currentJob && !['completed', 'failed'].includes(state.currentJob.status);
export const selectJobProgress = (state: JobState) =>
  state.currentJob?.progress_percentage ?? 0;
