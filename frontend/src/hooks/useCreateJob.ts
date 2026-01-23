/**
 * useCreateJob - Hook for creating new analysis jobs
 */

import { useState, useCallback } from 'react';
import { useJobStore } from '../store';

interface CreateJobParams {
  kaggleUrl: string;
  treatmentVariable?: string;
  outcomeVariable?: string;
}

/**
 * Hook for creating new causal analysis jobs.
 * Provides loading state and error handling.
 *
 * @returns Job creation state and actions
 */
export function useCreateJob() {
  const createJobAction = useJobStore((state) => state.createJob);
  const isCreating = useJobStore((state) => state.isCreating);
  const storeError = useJobStore((state) => state.error);
  const clearError = useJobStore((state) => state.clearError);

  const [localError, setLocalError] = useState<string | null>(null);
  const [createdJobId, setCreatedJobId] = useState<string | null>(null);

  const createJob = useCallback(
    async ({ kaggleUrl, treatmentVariable, outcomeVariable }: CreateJobParams) => {
      setLocalError(null);
      setCreatedJobId(null);
      clearError();

      try {
        // Validate URL format
        if (!kaggleUrl.includes('kaggle.com')) {
          throw new Error('Please provide a valid Kaggle dataset URL');
        }

        const jobId = await createJobAction(kaggleUrl, treatmentVariable, outcomeVariable);
        setCreatedJobId(jobId);
        return jobId;
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to create job';
        setLocalError(message);
        throw err;
      }
    },
    [createJobAction, clearError]
  );

  const reset = useCallback(() => {
    setLocalError(null);
    setCreatedJobId(null);
    clearError();
  }, [clearError]);

  return {
    createJob,
    isCreating,
    error: localError || storeError,
    createdJobId,
    reset,
  };
}
