/**
 * useDatasetView — hydrate the live Data panel for a job.
 *
 * Polls /jobs/{id}/dataset until every block has settled (download done,
 * profile + kaggle metadata resolved), then stops. The file list (with
 * columns + row counts) arrives over REST, so polling is the live path, not
 * just a backstop. SSE patches in useJob still update status in place between
 * polls. Raw rows are fetched separately and on demand via useDatasetRows.
 */

import { useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { DatasetView, getDatasetView } from '../services/api';
import { useJobStore } from '../store/jobStore';
import { DATASET_VIEW_POLL_INTERVAL_MS } from '../config/constants';

/** True once no block can change further, so polling can stop. */
function isSettled(v: DatasetView): boolean {
  const has = (s: string, ...ok: string[]) => ok.includes(s);
  return (
    has(v.download.status, 'downloaded', 'failed') &&
    has(v.profile.status, 'loaded', 'error') &&
    has(v.kaggle_meta.status, 'loaded', 'unavailable', 'error')
  );
}

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
    refetchOnWindowFocus: false,
    // Poll while any block is still resolving; freeze once all settle.
    refetchInterval: (q) => {
      const data = q.state.data as DatasetView | undefined;
      if (!data) return DATASET_VIEW_POLL_INTERVAL_MS;
      return isSettled(data) ? false : DATASET_VIEW_POLL_INTERVAL_MS;
    },
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
