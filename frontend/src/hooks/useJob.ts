/**
 * useJob - Hook for fetching and subscribing to a single job.
 *
 * Uses SSE (Server-Sent Events) for real-time updates when available,
 * with automatic fallback to React Query polling.
 */

import { useEffect, useRef } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { getJob, getStreamUrl, JobDetail } from '../services/api';

interface UseJobOptions {
  /** Whether to auto-subscribe for updates (default: true) */
  autoPolling?: boolean;
  /** Polling interval in ms, used as fallback if SSE fails (default: 2000) */
  pollIntervalMs?: number;
}

/**
 * Hook to fetch and subscribe to a job by ID.
 * Tries SSE first for instant updates; falls back to polling on error.
 */
export function useJob(jobId: string | null, options: UseJobOptions = {}) {
  const { autoPolling = true, pollIntervalMs = 2000 } = options;
  const queryClient = useQueryClient();
  const sseRef = useRef<EventSource | null>(null);
  const sseFailedRef = useRef(false);

  // Close SSE on unmount or when jobId changes
  useEffect(() => {
    return () => {
      if (sseRef.current) {
        sseRef.current.close();
        sseRef.current = null;
      }
    };
  }, [jobId]);

  // Set up SSE connection for running jobs
  useEffect(() => {
    if (!jobId || !autoPolling || sseFailedRef.current) return;

    const streamUrl = getStreamUrl(jobId);
    let es: EventSource;

    try {
      es = new EventSource(streamUrl);
      sseRef.current = es;

      es.addEventListener('status', (event: MessageEvent) => {
        try {
          const statusData = JSON.parse(event.data);
          // Update React Query cache with fresh data from SSE
          queryClient.setQueryData(['job', jobId], (old: JobDetail | undefined) => {
            if (!old) return old;
            return {
              ...old,
              status: statusData.status,
              progress_percentage: statusData.progress_percentage,
              current_agent: statusData.current_agent,
            };
          });
        } catch {
          // Ignore parse errors
        }
      });

      es.addEventListener('done', () => {
        es.close();
        sseRef.current = null;
        // Refetch full job data on completion
        queryClient.invalidateQueries({ queryKey: ['job', jobId] });
      });

      es.onerror = () => {
        // SSE failed — fall back to polling
        es.close();
        sseRef.current = null;
        sseFailedRef.current = true;
      };
    } catch {
      // EventSource not supported or URL issue — fall back to polling
      sseFailedRef.current = true;
    }

    return () => {
      if (es) {
        es.close();
      }
    };
  }, [jobId, autoPolling, queryClient]);

  // React Query for initial fetch + fallback polling
  const query = useQuery({
    queryKey: ['job', jobId],
    queryFn: () => getJob(jobId!),
    enabled: !!jobId,
    refetchInterval: autoPolling
      ? (q: { state: { data?: { status?: string } } }) => {
          // If SSE is connected, don't poll
          if (sseRef.current && sseRef.current.readyState === EventSource.OPEN) {
            return false;
          }
          const status = q.state.data?.status;
          const isTerminal =
            status === 'completed' ||
            status === 'failed' ||
            status === 'cancelled';
          return isTerminal ? false : pollIntervalMs;
        }
      : false,
    refetchOnWindowFocus: false,
  });

  const job = query.data ?? null;
  const isComplete = job?.status === 'completed';
  const isFailed = job?.status === 'failed';
  const isRunning = !!job && !['completed', 'failed', 'cancelled'].includes(job.status);
  const progress = job?.progress_percentage ?? 0;

  return {
    job,
    isLoading: query.isLoading,
    isRunning,
    isComplete,
    isFailed,
    progress,
    error: query.error ? String((query.error as { message?: string })?.message || query.error) : null,
    refetch: () => query.refetch(),
  };
}
