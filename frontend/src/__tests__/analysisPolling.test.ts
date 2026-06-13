import { describe, it, expect } from 'vitest';
import { analysisPollInterval } from '../hooks/useAnalysis';

const MS = 5000;

describe('analysisPollInterval', () => {
  it('polls while the run is executing and the view has not settled', () => {
    expect(analysisPollInterval('running_analysis', undefined, MS)).toBe(MS);
    expect(analysisPollInterval('running_analysis', 'running', MS)).toBe(MS);
    expect(analysisPollInterval('running_analysis', 'pending', MS)).toBe(MS);
  });

  it('keeps polling after the job parks until the plan-gate card is fetched', () => {
    // The race: job status flips to waiting_for_user before this view has
    // fetched the card (loaded status still running). It must keep polling so
    // the card surfaces without a manual refresh.
    expect(analysisPollInterval('waiting_for_user', 'running', MS)).toBe(MS);
  });

  it('stops once the parked view has loaded and the card is on screen', () => {
    expect(analysisPollInterval('waiting_for_user', 'waiting_for_user', MS)).toBe(
      false,
    );
  });

  it('stops once the run reaches a terminal state', () => {
    expect(analysisPollInterval('running_analysis', 'completed', MS)).toBe(false);
    expect(analysisPollInterval('running_analysis', 'failed', MS)).toBe(false);
    expect(analysisPollInterval('running_analysis', 'cancelled', MS)).toBe(false);
    expect(analysisPollInterval('completed', 'completed', MS)).toBe(false);
  });

  it('does not poll when the job is outside the analysis phase', () => {
    expect(analysisPollInterval('awaiting_approval', undefined, MS)).toBe(false);
    expect(analysisPollInterval('confirmed', undefined, MS)).toBe(false);
    expect(analysisPollInterval(undefined, undefined, MS)).toBe(false);
  });
});
