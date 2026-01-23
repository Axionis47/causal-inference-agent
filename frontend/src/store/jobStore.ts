/**
 * Job Store - Zustand state management for causal analysis jobs
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import {
  createJob as apiCreateJob,
  getJob as apiGetJob,
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
  isPolling: boolean;

  // Error state
  error: string | null;

  // Polling
  pollInterval: ReturnType<typeof setInterval> | null;
}

interface JobActions {
  // Job CRUD
  createJob: (kaggleUrl: string, treatment?: string, outcome?: string) => Promise<string>;
  fetchJob: (jobId: string) => Promise<void>;
  fetchJobs: (status?: string, limit?: number, offset?: number) => Promise<void>;
  cancelJob: (jobId: string) => Promise<void>;

  // Results and traces
  fetchResults: (jobId: string) => Promise<void>;
  fetchTraces: (jobId: string) => Promise<void>;

  // Polling
  startPolling: (jobId: string, intervalMs?: number) => void;
  stopPolling: () => void;

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
  isPolling: false,
  error: null,
  pollInterval: null,
};

export const useJobStore = create<JobState & JobActions>()(
  devtools(
    persist(
      (set, get) => ({
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

            // Start polling for the new job
            get().startPolling(job.id);

            return job.id;
          } catch (error) {
            const message = error instanceof Error ? error.message : 'Failed to create job';
            set({ isCreating: false, error: message });
            throw error;
          }
        },

        fetchJob: async (jobId: string) => {
          set({ isLoading: true, error: null });
          try {
            const job = await apiGetJob(jobId);
            set({
              currentJob: job,
              currentJobId: jobId,
              isLoading: false
            });

            // If job is completed, fetch results
            if (job.status === 'completed') {
              get().fetchResults(jobId);
              get().fetchTraces(jobId);
              get().stopPolling();
            }
          } catch (error) {
            const message = error instanceof Error ? error.message : 'Failed to fetch job';
            set({ isLoading: false, error: message });
          }
        },

        fetchJobs: async (status?: string, limit = 20, offset = 0) => {
          set({ isLoading: true, error: null });
          try {
            const response = await apiListJobs(status, limit, offset);
            set({
              jobs: response.jobs,
              totalJobs: response.total,
              isLoading: false
            });
          } catch (error) {
            const message = error instanceof Error ? error.message : 'Failed to fetch jobs';
            set({ isLoading: false, error: message });
          }
        },

        cancelJob: async (jobId: string) => {
          try {
            await apiCancelJob(jobId);
            get().stopPolling();
            get().fetchJob(jobId);
          } catch (error) {
            const message = error instanceof Error ? error.message : 'Failed to cancel job';
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

        startPolling: (jobId: string, intervalMs = 3000) => {
          // Stop any existing polling
          get().stopPolling();

          set({ isPolling: true });

          const pollInterval = setInterval(async () => {
            const { currentJob } = get();

            // Stop polling if job is complete or failed
            if (currentJob?.status === 'completed' || currentJob?.status === 'failed') {
              get().stopPolling();
              return;
            }

            // Fetch latest job status
            await get().fetchJob(jobId);
          }, intervalMs);

          set({ pollInterval });
        },

        stopPolling: () => {
          const { pollInterval } = get();
          if (pollInterval) {
            clearInterval(pollInterval);
          }
          set({ pollInterval: null, isPolling: false });
        },

        setCurrentJob: (job: JobDetail | null) => {
          set({ currentJob: job, currentJobId: job?.id || null });
        },

        clearError: () => {
          set({ error: null });
        },

        reset: () => {
          get().stopPolling();
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
