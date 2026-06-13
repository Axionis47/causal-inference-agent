// Pure routing for the analysis_* SSE events. No React, no I/O.
//
// Contract: on ANY agent_event whose event_type starts with "analysis"
// (analysis_started, analysis_stage_started, analysis_agent_completed,
// analysis_artifact_emitted, analysis_waiting_for_user, analysis_completed,
// analysis_failed) the frontend invalidates the ['analysis', jobId] query and
// lets the refetch carry the new state; no fine-grained patching. The events
// that flip JobDetail.status (analysis_completed, analysis_failed, and
// analysis_waiting_for_user) also invalidate ['job', jobId], so the job view
// follows the run into its parked or terminal state without a manual refresh.

export function isAnalysisEventType(eventType: string): boolean {
  return eventType.startsWith('analysis');
}

/** Query keys to invalidate for one SSE event. Empty for non-analysis events. */
export function analysisKeysToInvalidate(
  eventType: string,
  jobId: string,
): string[][] {
  if (!isAnalysisEventType(eventType)) return [];
  const keys: string[][] = [['analysis', jobId]];
  if (
    eventType === 'analysis_completed' ||
    eventType === 'analysis_failed' ||
    eventType === 'analysis_waiting_for_user'
  ) {
    keys.push(['job', jobId]);
  }
  return keys;
}
