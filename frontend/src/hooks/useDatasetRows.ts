/**
 * useDatasetRows — fetch one page of raw rows for a downloaded file.
 *
 * Rows are read on demand from the backend (which reads the durable per-job
 * bundle via the manifest), so a dataset of any size is paged through without
 * loading it all. Disabled until a file is selected. The previous page stays
 * visible while the next one loads, so paging does not flicker.
 */

import { keepPreviousData, useQuery } from '@tanstack/react-query';
import { DatasetRowsPage, getDatasetRows } from '../services/api';

export function useDatasetRows(
  jobId: string | null,
  fileName: string | null,
  offset: number,
  limit: number
): { page: DatasetRowsPage | null; isLoading: boolean; error: string | null } {
  const query = useQuery({
    queryKey: ['dataset-rows', jobId, fileName, offset, limit],
    queryFn: () => getDatasetRows(jobId!, fileName!, offset, limit),
    enabled: !!jobId && !!fileName,
    refetchOnWindowFocus: false,
    placeholderData: keepPreviousData,
  });

  return {
    page: query.data ?? null,
    isLoading: query.isLoading,
    error: query.error
      ? String((query.error as { message?: string })?.message || query.error)
      : null,
  };
}
