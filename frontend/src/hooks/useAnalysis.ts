/**
 * useAnalysis — hydrate the analysis view for a job.
 *
 * The query is enabled only once the job is in (or past) the analysis slice
 * (running_analysis | waiting_for_user | completed | failed), or after the
 * view has been seen once (so it survives a status flip to cancelled).
 * A 404 means no analysis run exists yet and resolves to null, not an error;
 * for failed jobs that distinguishes an analysis failure (retryable) from an
 * input-slice failure (not retryable, no run ever existed).
 *
 * SSE analysis_* events invalidate ['analysis', jobId] (see useJob), so the
 * 5s poll while running is only the fallback when the stream drops.
 */

import { useRef } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  AnalysisRunStatus,
  AnalysisViewResponse,
  ApiError,
  getAnalysisView,
} from '../services/api';
import { ANALYSIS_POLL_INTERVAL_MS } from '../config/constants';

const ANALYSIS_JOB_STATUSES = [
  'running_analysis',
  'waiting_for_user',
  'completed',
  'failed',
];

/**
 * Polling cadence for the analysis view: the interval in ms while the run is
 * live and the view has not settled, or false to stop.
 *
 * "Settled" means the view has loaded a state the user reads at rest: a
 * completed / failed / cancelled run, or a parked run (waiting_for_user) whose
 * plan-gate card is now on screen. While the job is in an analysis-active phase
 * (running_analysis or waiting_for_user) and the view has not settled, we poll
 * so the run's progress and the plan-gate card surface even when the SSE stream
 * is silently dead. Polling on waiting_for_user too, not only running_analysis,
 * closes the race where the job status flips to parked before this view has
 * fetched the card.
 */
export function analysisPollInterval(
  jobStatus: string | undefined,
  loadedStatus: AnalysisRunStatus | undefined,
  intervalMs: number,
): number | false {
  if (
    loadedStatus === 'completed' ||
    loadedStatus === 'failed' ||
    loadedStatus === 'cancelled' ||
    loadedStatus === 'waiting_for_user'
  ) {
    return false;
  }
  return jobStatus === 'running_analysis' || jobStatus === 'waiting_for_user'
    ? intervalMs
    : false;
}

export function useAnalysis(
  jobId: string | null,
  jobStatus: string | undefined,
): {
  analysis: AnalysisViewResponse | null;
  isLoading: boolean;
  error: string | null;
} {
  // Once the view has existed, keep the query alive regardless of job status.
  const everExistedRef = useRef(false);
  const statusWantsAnalysis =
    !!jobStatus && ANALYSIS_JOB_STATUSES.includes(jobStatus);

  const query = useQuery<AnalysisViewResponse | null>({
    queryKey: ['analysis', jobId],
    queryFn: async () => {
      try {
        return await getAnalysisView(jobId!);
      } catch (e) {
        if ((e as ApiError).status === 404) return null;
        throw e;
      }
    },
    enabled: !!jobId && (statusWantsAnalysis || everExistedRef.current),
    refetchOnWindowFocus: false,
    refetchInterval: (q: { state: { data?: AnalysisViewResponse | null } }) =>
      analysisPollInterval(
        jobStatus,
        q.state.data?.status,
        ANALYSIS_POLL_INTERVAL_MS,
      ),
  });

  if (query.data) everExistedRef.current = true;

  return {
    analysis: query.data ?? null,
    isLoading: query.isLoading,
    error: query.error
      ? String((query.error as { message?: string })?.message || query.error)
      : null,
  };
}
