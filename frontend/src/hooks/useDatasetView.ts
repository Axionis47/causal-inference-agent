/**
 * useDatasetView — hydrate the live Data panel for a job.
 *
 * On mount, fetches /jobs/{id}/dataset once to seed the store. After
 * that, SSE events handled in useJob patch the store's datasetView in
 * place, so the UI updates as each block of data arrives without
 * needing to re-fetch the endpoint.
 */

import { useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { DatasetView, getDatasetView } from '../services/api';
import { useJobStore } from '../store/jobStore';

export function useDatasetView(jobId: string | null): {
  view: DatasetView | null;
  isLoading: boolean;
  error: string | null;
} {
  const view = useJobStore((s) => s.datasetView);
  const setDatasetView = useJobStore((s) => s.setDatasetView);

  const query = useQuery({
    queryKey: ['dataset-view', jobId],
    queryFn: () => getDatasetView(jobId!),
    enabled: !!jobId,
    // Single hydrate. SSE patches keep it fresh after this.
    staleTime: Infinity,
    refetchOnWindowFocus: false,
    refetchOnMount: false,
  });

  useEffect(() => {
    if (query.data) {
      setDatasetView(query.data);
    }
  }, [query.data, setDatasetView]);

  return {
    view,
    isLoading: query.isLoading && !view,
    error: query.error
      ? String((query.error as { message?: string })?.message || query.error)
      : null,
  };
}
