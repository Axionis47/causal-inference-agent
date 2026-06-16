import { describe, it, expect } from 'vitest';
import { deriveJobView } from '../components/job/terminal/deriveJobView';
import type { AgentEvent, JobDetail } from '../services/api';

const job = (overrides: Partial<JobDetail> = {}): JobDetail =>
  ({
    status: 'running',
    current_agent: 'eda_agent',
    created_at: '2026-01-01T00:00:00Z',
    ...overrides,
  } as JobDetail);

const ev = (
  agent_name: string,
  event_type: string,
  data: Record<string, unknown> = {},
): AgentEvent => ({ timestamp: '2026-01-01T00:00:01Z', agent_name, event_type, data });

describe('deriveJobView findings and challenges', () => {
  it('maps the focus agent finding and challenge from its events', () => {
    const events = [
      ev('eda_agent', 'agent_finding', { headline: 'h', status: 'done', flags: ['outcome_skewed'] }),
      ev('eda_agent', 'agent_challenge', {
        headline: 'h',
        status: 'done',
        flags: ['outcome_skewed'],
        issues: ['outcome is right-skewed'],
      }),
    ];

    const view = deriveJobView(job(), events, 0);

    expect(view.focusFinding?.data.headline).toBe('h');
    expect(view.focusChallenge?.data.issues).toEqual(['outcome is right-skewed']);
    expect(view.challengedAgents.has('eda_agent')).toBe(true);
  });

  it('pins a selected agent over the live one', () => {
    const events = [ev('dag_expert', 'agent_finding', { headline: 'dag', status: 'done', flags: [] })];

    const view = deriveJobView(job({ current_agent: 'eda_agent' }), events, 0, 'dag_expert');

    expect(view.focusAgent).toBe('dag_expert');
    expect(view.focusFinding?.data.headline).toBe('dag');
  });

  it('keeps a selection after the run finishes (no current agent)', () => {
    const events = [ev('eda_agent', 'agent_finding', { headline: 'h', status: 'done', flags: [] })];

    const view = deriveJobView(
      job({ status: 'completed', current_agent: undefined }),
      events,
      0,
      'eda_agent',
    );

    expect(view.focusAgent).toBe('eda_agent');
    expect(view.focusFinding?.data.headline).toBe('h');
  });

  it('reports no challenge for an agent that only emitted a finding', () => {
    const events = [ev('eda_agent', 'agent_finding', { headline: 'h', status: 'done', flags: [] })];

    const view = deriveJobView(job(), events, 0);

    expect(view.focusChallenge).toBeUndefined();
    expect(view.challengedAgents.size).toBe(0);
  });
});

describe('deriveJobView elapsed clock', () => {
  const startMs = Date.parse('2026-01-01T00:00:00Z');

  it('ticks off nowMs while the job is running', () => {
    const view = deriveJobView(job({ status: 'running' }), [], startMs + 5_000);
    expect(view.elapsed).toBe('00:00:05');
  });

  it('freezes at updated_at once terminal, ignoring a far-later nowMs', () => {
    const view = deriveJobView(
      job({ status: 'completed', updated_at: '2026-01-01T00:02:05Z' }),
      [],
      startMs + 9_999_000, // wall clock has run long past completion
    );
    expect(view.elapsed).toBe('00:02:05');
  });

  it('falls back to nowMs for a terminal job with no updated_at', () => {
    const view = deriveJobView(
      job({ status: 'cancelled', updated_at: undefined as unknown as string }),
      [],
      startMs + 7_000,
    );
    expect(view.elapsed).toBe('00:00:07');
  });
});
