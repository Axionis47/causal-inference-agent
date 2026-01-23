/**
 * useJob - Hook for fetching and subscribing to a single job
 */

import { useEffect } from 'react';
import { useJobStore } from '../store';

interface UseJobOptions {
  /** Whether to start polling automatically */
  autoPolling?: boolean;
  /** Polling interval in milliseconds */
  pollIntervalMs?: number;
}

/**
 * Hook to fetch and subscribe to a job by ID.
 * Automatically handles polling for running jobs.
 *
 * @param jobId - The job ID to fetch
 * @param options - Configuration options
 * @returns Job state and actions
 */
export function useJob(jobId: string | null, options: UseJobOptions = {}) {
  const { autoPolling = true, pollIntervalMs = 3000 } = options;

  const currentJob = useJobStore((state) => state.currentJob);
  const isLoading = useJobStore((state) => state.isLoading);
  const error = useJobStore((state) => state.error);
  const fetchJob = useJobStore((state) => state.fetchJob);
  const startPolling = useJobStore((state) => state.startPolling);
  const stopPolling = useJobStore((state) => state.stopPolling);
  const cancelJob = useJobStore((state) => state.cancelJob);

  useEffect(() => {
    if (!jobId) return;

    // Fetch the job
    fetchJob(jobId);

    // Start polling if enabled and job is running
    if (autoPolling) {
      startPolling(jobId, pollIntervalMs);
    }

    // Cleanup polling on unmount
    return () => {
      stopPolling();
    };
  }, [jobId, autoPolling, pollIntervalMs, fetchJob, startPolling, stopPolling]);

  const isComplete = currentJob?.status === 'completed';
  const isFailed = currentJob?.status === 'failed';
  const isRunning = currentJob && !['completed', 'failed'].includes(currentJob.status);
  const progress = currentJob?.progress_percentage ?? 0;

  return {
    job: currentJob,
    isLoading,
    isRunning,
    isComplete,
    isFailed,
    progress,
    error,
    refetch: () => jobId && fetchJob(jobId),
    cancel: () => jobId && cancelJob(jobId),
  };
}
