/**
 * useResults - Hook for fetching job analysis results
 */

import { useEffect, useMemo } from 'react';
import { useJobStore } from '../store';

interface UseResultsOptions {
  /** Only fetch if job is complete */
  fetchOnComplete?: boolean;
}

/**
 * Hook for fetching and subscribing to job results.
 * Results include treatment effects, causal graph, sensitivity analysis.
 *
 * @param jobId - The job ID to fetch results for
 * @param options - Configuration options
 * @returns Results state and data
 */
export function useResults(jobId: string | null, options: UseResultsOptions = {}) {
  const { fetchOnComplete = true } = options;

  const currentJob = useJobStore((state) => state.currentJob);
  const results = useJobStore((state) => state.results);
  const traces = useJobStore((state) => state.traces);
  const fetchResults = useJobStore((state) => state.fetchResults);
  const fetchTraces = useJobStore((state) => state.fetchTraces);

  const isComplete = currentJob?.status === 'completed';

  useEffect(() => {
    if (!jobId) return;

    // Fetch results if job is complete or if not waiting for completion
    if (!fetchOnComplete || isComplete) {
      fetchResults(jobId);
      fetchTraces(jobId);
    }
  }, [jobId, isComplete, fetchOnComplete, fetchResults, fetchTraces]);

  // Memoized computed values
  const treatmentEffects = useMemo(() => results?.treatment_effects ?? [], [results]);

  const causalGraph = useMemo(() => results?.causal_graph ?? null, [results]);

  const sensitivityAnalysis = useMemo(
    () => results?.sensitivity_analysis ?? [],
    [results]
  );

  const recommendations = useMemo(() => results?.recommendations ?? [], [results]);

  const bestEstimate = useMemo(() => {
    if (treatmentEffects.length === 0) return null;

    // Prefer doubly robust or matching methods over simple OLS
    const preferred = treatmentEffects.find(
      (e) =>
        e.method.toLowerCase().includes('doubly robust') ||
        e.method.toLowerCase().includes('aipw')
    );

    return preferred ?? treatmentEffects[0];
  }, [treatmentEffects]);

  return {
    results,
    traces,
    treatmentEffects,
    causalGraph,
    sensitivityAnalysis,
    recommendations,
    bestEstimate,
    notebookUrl: results?.notebook_url ?? null,
    treatmentVariable: results?.treatment_variable,
    outcomeVariable: results?.outcome_variable,
    hasResults: results !== null,
    refetch: () => {
      if (jobId) {
        fetchResults(jobId);
        fetchTraces(jobId);
      }
    },
  };
}
