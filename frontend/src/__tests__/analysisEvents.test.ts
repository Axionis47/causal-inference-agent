import { describe, it, expect } from 'vitest';
import {
  analysisKeysToInvalidate,
  isAnalysisEventType,
} from '../hooks/analysisEvents';

describe('analysisKeysToInvalidate', () => {
  it('returns no keys for non-analysis events', () => {
    expect(analysisKeysToInvalidate('agent_started', 'job-1')).toEqual([]);
    expect(analysisKeysToInvalidate('dataset_download_complete', 'job-1')).toEqual([]);
    expect(analysisKeysToInvalidate('approval_required', 'job-1')).toEqual([]);
  });

  it('invalidates the analysis query for every analysis_* lifecycle event', () => {
    for (const t of [
      'analysis_started',
      'analysis_stage_started',
      'analysis_agent_completed',
      'analysis_artifact_emitted',
      'analysis_waiting_for_user',
    ]) {
      expect(analysisKeysToInvalidate(t, 'job-1')).toEqual([['analysis', 'job-1']]);
    }
  });

  it('also invalidates the job query on the terminal analysis events', () => {
    expect(analysisKeysToInvalidate('analysis_completed', 'job-1')).toEqual([
      ['analysis', 'job-1'],
      ['job', 'job-1'],
    ]);
    expect(analysisKeysToInvalidate('analysis_failed', 'job-1')).toEqual([
      ['analysis', 'job-1'],
      ['job', 'job-1'],
    ]);
  });

  it('treats any future event type starting with "analysis" as an analysis event', () => {
    expect(isAnalysisEventType('analysis_budget_warning')).toBe(true);
    expect(isAnalysisEventType('agent_completed')).toBe(false);
    expect(analysisKeysToInvalidate('analysis_budget_warning', 'j')).toEqual([
      ['analysis', 'j'],
    ]);
  });
});
