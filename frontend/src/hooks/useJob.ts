/**
 * useJob - Hook for fetching and subscribing to a single job.
 *
 * Uses SSE (Server-Sent Events) for real-time updates when available,
 * with automatic fallback to React Query polling.
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  AgentEvent,
  DataProfileSummary,
  DatasetEventType,
  FileEntry,
  JobDetail,
  KaggleMeta,
  getJob,
  getStreamUrl,
} from '../services/api';
import { DEFAULT_POLL_INTERVAL_MS, MAX_AGENT_EVENTS } from '../config/constants';
import { analysisKeysToInvalidate } from './analysisEvents';
import { useJobStore } from '../store/jobStore';

interface UseJobOptions {
  /** Whether to auto-subscribe for updates (default: true) */
  autoPolling?: boolean;
  /** Polling interval in ms, used as fallback if SSE fails */
  pollIntervalMs?: number;
}

/**
 * Hook to fetch and subscribe to a job by ID.
 * Tries SSE first for instant updates; falls back to polling on error.
 */
export function useJob(jobId: string | null, options: UseJobOptions = {}) {
  const { autoPolling = true, pollIntervalMs = DEFAULT_POLL_INTERVAL_MS } = options;
  const queryClient = useQueryClient();
  const sseRef = useRef<EventSource | null>(null);
  const sseFailedRef = useRef(false);
  const [agentEvents, setAgentEvents] = useState<AgentEvent[]>([]);
  const prevStatusRef = useRef<string | null>(null);

  const addAgentEvent = useCallback((event: AgentEvent) => {
    setAgentEvents((prev) => [...prev.slice(-(MAX_AGENT_EVENTS - 1)), event]);
  }, []);

  const patchDownload = useJobStore((s) => s.patchDownload);
  const patchKaggleMeta = useJobStore((s) => s.patchKaggleMeta);
  const patchProfile = useJobStore((s) => s.patchProfile);

  // Dataset events ride the same 'agent_event' SSE channel; route by
  // event_type so each block lights up independently as data arrives.
  const dispatchDatasetEvent = useCallback(
    (eventType: DatasetEventType, data: Record<string, unknown>) => {
      switch (eventType) {
        case 'dataset_metadata_started':
          patchKaggleMeta({ status: 'pending' });
          return;
        case 'dataset_metadata_ready':
          patchKaggleMeta({
            status: 'loaded',
            error: null,
            data: {
              description: (data.description as string | null) ?? null,
              column_descriptions:
                (data.column_descriptions as Record<string, string>) ?? {},
              tags: (data.tags as string[]) ?? [],
              domain: (data.domain as string | null) ?? null,
              metadata_quality: (data.metadata_quality as string) ?? 'unknown',
            } satisfies KaggleMeta,
          });
          return;
        case 'dataset_metadata_failed':
          patchKaggleMeta({
            status: 'error',
            error: (data.error as string) ?? 'metadata fetch failed',
          });
          return;
        case 'dataset_download_started':
          patchDownload({
            status: 'downloading',
            url: (data.url as string) ?? null,
            error: null,
          });
          return;
        case 'dataset_download_complete':
          patchDownload({
            status: 'downloaded',
            files: (data.files as FileEntry[]) ?? [],
            error: null,
          });
          return;
        case 'dataset_load_failed':
          patchDownload({
            status: 'failed',
            error: (data.error as string) ?? 'dataset load failed',
          });
          return;
        case 'data_profile_ready':
          patchProfile({
            status: 'loaded',
            error: null,
            data: data as unknown as DataProfileSummary,
          });
          return;
      }
    },
    [patchDownload, patchKaggleMeta, patchProfile]
  );

  const isDatasetEventType = (t: string): t is DatasetEventType =>
    t === 'dataset_metadata_started' ||
    t === 'dataset_metadata_ready' ||
    t === 'dataset_metadata_failed' ||
    t === 'dataset_download_started' ||
    t === 'dataset_download_complete' ||
    t === 'dataset_load_failed' ||
    t === 'data_profile_ready';

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

      es.addEventListener('agent_event', (event: MessageEvent) => {
        try {
          const parsed = JSON.parse(event.data) as {
            event_type: string;
            data: Record<string, unknown>;
            timestamp: string;
            agent_name?: string;
          };
          if (isDatasetEventType(parsed.event_type)) {
            dispatchDatasetEvent(parsed.event_type, parsed.data);
            return;
          }
          // Analysis lifecycle events: no fine-grained patching, just refetch
          // the analysis view (and the job itself on the terminal events).
          // The event still lands in the tape below; it carries a headline.
          for (const key of analysisKeysToInvalidate(parsed.event_type, jobId)) {
            queryClient.invalidateQueries({ queryKey: key });
          }
          addAgentEvent(parsed as AgentEvent);
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
  }, [jobId, autoPolling, queryClient, addAgentEvent, dispatchDatasetEvent]);

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

  // Fallback: derive activity from status changes when SSE agent_events are unavailable
  useEffect(() => {
    if (!job) return;
    const prev = prevStatusRef.current;
    if (prev && prev !== job.status) {
      addAgentEvent({
        timestamp: new Date().toISOString(),
        agent_name: job.current_agent || 'orchestrator',
        event_type: 'agent_started',
        data: { fallback: true, status: job.status },
      });
    }
    prevStatusRef.current = job.status;
  }, [job?.status, job?.current_agent, addAgentEvent, job]);

  return {
    job,
    isLoading: query.isLoading,
    isRunning,
    isComplete,
    isFailed,
    progress,
    agentEvents,
    error: query.error ? String((query.error as { message?: string })?.message || query.error) : null,
    refetch: () => query.refetch(),
  };
}
