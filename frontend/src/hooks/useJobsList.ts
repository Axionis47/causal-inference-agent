/**
 * useJobsList - Hook for fetching and managing the jobs list
 */

import { useEffect, useCallback, useState } from 'react';
import { useJobStore } from '../store';

interface UseJobsListOptions {
  /** Filter by job status */
  status?: string;
  /** Number of items per page */
  pageSize?: number;
  /** Whether to fetch on mount */
  fetchOnMount?: boolean;
}

/**
 * Hook for fetching and paginating the jobs list.
 * Supports filtering by status and pagination.
 *
 * @param options - Configuration options
 * @returns Jobs list state and actions
 */
export function useJobsList(options: UseJobsListOptions = {}) {
  const { status, pageSize = 20, fetchOnMount = true } = options;

  const jobs = useJobStore((state) => state.jobs);
  const totalJobs = useJobStore((state) => state.totalJobs);
  const isLoading = useJobStore((state) => state.isLoading);
  const error = useJobStore((state) => state.error);
  const fetchJobs = useJobStore((state) => state.fetchJobs);

  const [currentPage, setCurrentPage] = useState(0);

  const offset = currentPage * pageSize;
  const totalPages = Math.ceil(totalJobs / pageSize);
  const hasNextPage = currentPage < totalPages - 1;
  const hasPrevPage = currentPage > 0;

  useEffect(() => {
    if (fetchOnMount) {
      fetchJobs(status, pageSize, offset);
    }
  }, [status, pageSize, offset, fetchOnMount, fetchJobs]);

  const refetch = useCallback(() => {
    fetchJobs(status, pageSize, offset);
  }, [status, pageSize, offset, fetchJobs]);

  const nextPage = useCallback(() => {
    if (hasNextPage) {
      setCurrentPage((prev) => prev + 1);
    }
  }, [hasNextPage]);

  const prevPage = useCallback(() => {
    if (hasPrevPage) {
      setCurrentPage((prev) => prev - 1);
    }
  }, [hasPrevPage]);

  const goToPage = useCallback(
    (page: number) => {
      if (page >= 0 && page < totalPages) {
        setCurrentPage(page);
      }
    },
    [totalPages]
  );

  // Computed stats
  const stats = {
    total: totalJobs,
    running: jobs.filter((j) => !['completed', 'failed'].includes(j.status)).length,
    completed: jobs.filter((j) => j.status === 'completed').length,
    failed: jobs.filter((j) => j.status === 'failed').length,
  };

  return {
    jobs,
    isLoading,
    error,
    refetch,
    // Pagination
    pagination: {
      currentPage,
      pageSize,
      totalPages,
      totalItems: totalJobs,
      hasNextPage,
      hasPrevPage,
      nextPage,
      prevPage,
      goToPage,
    },
    stats,
  };
}
