import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import ReviewTimeline from '../components/review/ReviewTimeline';
import type { AgentTrace, MethodologyDecision } from '../services/api';

vi.mock('../services/api', async () => {
  const actual = await vi.importActual<typeof import('../services/api')>(
    '../services/api'
  );
  return {
    ...actual,
    getTraces: vi.fn(),
  };
});

import { getTraces } from '../services/api';

function renderTimeline(
  traces: AgentTrace[],
  decisions: MethodologyDecision[]
) {
  vi.mocked(getTraces).mockResolvedValue(traces);
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <ReviewTimeline jobId="t" decisions={decisions} />
    </QueryClientProvider>
  );
}

const trace1: AgentTrace = {
  agent_name: 'data_profiler',
  timestamp: '2026-05-10T12:00:00Z',
  action: 'profile_dataset',
  reasoning: 'Computed shape and dtypes for the Lalonde dataset.',
  duration_ms: 1240,
  inputs: {},
  outputs: {},
  tools_called: ['get_dataset_overview'],
  token_usage: { input_tokens: 100, output_tokens: 50 },
};

const trace2: AgentTrace = {
  agent_name: 'effect_estimator',
  timestamp: '2026-05-10T12:02:00Z',
  action: 'run_method',
  reasoning: 'Ran OLS against the adjustment set.',
  duration_ms: 320,
  inputs: {},
  outputs: {},
  tools_called: ['run_method', 'run_method'],
  token_usage: { input_tokens: 200, output_tokens: 80 },
};

const decision1: MethodologyDecision = {
  agent: 'orchestrator',
  decision_type: 'agent_dispatched',
  choice: 'data_profiler',
  reason: 'Dataset has not been profiled yet.',
  timestamp: '2026-05-10T11:59:50Z',
};

const decision2: MethodologyDecision = {
  agent: 'effect_estimator',
  decision_type: 'method_selection',
  choice: 'aipw',
  reason: 'OLS and AIPW agreed within 1 SE on the same direction.',
  alternatives: [{ option: 'matching', reason: 'fewer overlapping units' }],
  timestamp: '2026-05-10T12:01:30Z',
};

beforeEach(() => {
  vi.clearAllMocks();
});

describe('ReviewTimeline', () => {
  it('renders empty placeholder when no traces or decisions', async () => {
    renderTimeline([], []);
    // wait one tick for the query to resolve
    expect(
      await screen.findByText(/No traces or decisions recorded/i),
    ).toBeTruthy();
  });

  it('merges decisions and traces in chronological order', async () => {
    renderTimeline([trace1, trace2], [decision1, decision2]);
    // Wait until traces have actually rendered before sampling DOM order.
    await screen.findByText(/Computed shape and dtypes/);
    const body = document.body.textContent || '';
    const idxD1 = body.indexOf('not been profiled');
    const idxT1 = body.indexOf('Computed shape');
    const idxD2 = body.indexOf('AIPW agreed');
    const idxT2 = body.indexOf('OLS against');
    expect(idxD1).toBeGreaterThan(-1);
    expect(idxT1).toBeGreaterThan(-1);
    expect(idxD2).toBeGreaterThan(-1);
    expect(idxT2).toBeGreaterThan(-1);
    expect(idxD1).toBeLessThan(idxT1);
    expect(idxT1).toBeLessThan(idxD2);
    expect(idxD2).toBeLessThan(idxT2);
  });

  it('Decisions filter hides traces', async () => {
    renderTimeline([trace1, trace2], [decision1, decision2]);
    await screen.findByText(/Decisions · 2/);
    fireEvent.click(screen.getByText(/Decisions · 2/));
    expect(screen.queryByText(/Computed shape/)).toBeNull();
    expect(screen.getByText(/not been profiled/)).toBeTruthy();
  });

  it('Traces filter hides decisions', async () => {
    renderTimeline([trace1, trace2], [decision1, decision2]);
    await screen.findByText(/Traces · 2/);
    fireEvent.click(screen.getByText(/Traces · 2/));
    expect(screen.queryByText(/AIPW agreed/)).toBeNull();
    expect(screen.getByText(/OLS against/)).toBeTruthy();
  });

  it('Decision renders agent badge, choice, and reason as a paragraph', async () => {
    renderTimeline([], [decision2]);
    await screen.findByText(/Decisions · 1/);
    expect(screen.getByText('effect_estimator')).toBeTruthy();
    expect(screen.getByText('aipw')).toBeTruthy();
    expect(screen.getByText(/AIPW agreed within 1 SE/)).toBeTruthy();
  });

  it('Decision with alternatives shows toggle that reveals them', async () => {
    renderTimeline([], [decision2]);
    await screen.findByText(/Decisions · 1/);
    expect(screen.queryByText('matching')).toBeNull();
    fireEvent.click(screen.getByText(/1 alternative considered/));
    expect(screen.getByText('matching')).toBeTruthy();
    expect(screen.getByText(/fewer overlapping units/)).toBeTruthy();
  });

  it('Trace renders agent, action, reasoning, and tools chips', async () => {
    renderTimeline([trace1], []);
    await screen.findByText(/Traces · 1/);
    expect(screen.getByText('data_profiler')).toBeTruthy();
    expect(screen.getByText('profile_dataset')).toBeTruthy();
    expect(screen.getByText(/Computed shape and dtypes/)).toBeTruthy();
    expect(screen.getByText('get_dataset_overview')).toBeTruthy();
  });
});
