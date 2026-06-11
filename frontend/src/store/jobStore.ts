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
  listJobs as apiListJobs,
  getResults as apiGetResults,
  getTraces as apiGetTraces,
  Job,
  JobDetail,
  AnalysisResults,
  AgentTrace,
  DatasetView,
  DownloadBlock,
  KaggleMetaBlock,
  ProfileBlock,
} from '../services/api';

interface JobState {
  // Current job being viewed
  currentJob: JobDetail | null;
  currentJobId: string | null;

  // Job results and traces
  results: AnalysisResults | null;
  traces: AgentTrace[];

  // Dataset view (download / kaggle metadata / profile) for the live
  // Data panel. null until first hydrate from /jobs/{id}/dataset; SSE
  // events then patch each block independently.
  datasetView: DatasetView | null;

  // Jobs list
  jobs: Job[];
  totalJobs: number;

  // Loading states
  isLoading: boolean;

  // Error state
  error: string | null;
}

interface JobActions {
  // Job CRUD
  fetchJobs: (status?: string, limit?: number, offset?: number) => Promise<void>;

  // Results and traces
  fetchResults: (jobId: string) => Promise<void>;
  fetchTraces: (jobId: string) => Promise<void>;

  // Dataset view — initial hydrate + per-block patching from SSE.
  setDatasetView: (view: DatasetView | null) => void;
  patchDownload: (partial: Partial<DownloadBlock>) => void;
  patchKaggleMeta: (partial: Partial<KaggleMetaBlock>) => void;
  patchProfile: (partial: Partial<ProfileBlock>) => void;

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
  datasetView: null,
  jobs: [],
  totalJobs: 0,
  isLoading: false,
  error: null,
};

const emptyDatasetView = (): DatasetView => ({
  download: { status: 'pending', url: null, files: [], error: null },
  kaggle_meta: { status: 'pending', data: null, error: null },
  profile: { status: 'pending', data: null, error: null },
});

export const useJobStore = create<JobState & JobActions>()(
  devtools(
    persist(
      (set, _get) => ({
        ...initialState,

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
          // Switching jobs invalidates the previous datasetView so the
          // panel doesn't briefly show stale download/profile data.
          set((state) => ({
            currentJob: job,
            currentJobId: job?.id || null,
            datasetView: job?.id === state.currentJobId ? state.datasetView : null,
          }));
        },

        setDatasetView: (view: DatasetView | null) => {
          set({ datasetView: view });
        },

        patchDownload: (partial: Partial<DownloadBlock>) => {
          set((state) => {
            const current = state.datasetView ?? emptyDatasetView();
            return {
              datasetView: {
                ...current,
                download: { ...current.download, ...partial },
              },
            };
          });
        },

        patchKaggleMeta: (partial: Partial<KaggleMetaBlock>) => {
          set((state) => {
            const current = state.datasetView ?? emptyDatasetView();
            // Deep-merge data: a partial SSE metadata patch (description/tags
            // only) must not clobber the richer fields already loaded over REST
            // (title, subtitle, size, downloads, votes, license).
            const data = partial.data
              ? { ...(current.kaggle_meta.data ?? {}), ...partial.data }
              : current.kaggle_meta.data;
            return {
              datasetView: {
                ...current,
                kaggle_meta: { ...current.kaggle_meta, ...partial, data },
              },
            };
          });
        },

        patchProfile: (partial: Partial<ProfileBlock>) => {
          set((state) => {
            const current = state.datasetView ?? emptyDatasetView();
            return {
              datasetView: {
                ...current,
                profile: { ...current.profile, ...partial },
              },
            };
          });
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
