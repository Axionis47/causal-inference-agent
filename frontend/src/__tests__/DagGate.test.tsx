import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { DagGate } from '../components/job/terminal/DagGate';
import type { DagGatePayload } from '../services/api';

const { submitApproval } = vi.hoisted(() => ({ submitApproval: vi.fn() }));

vi.mock('../services/api', async () => {
  const actual = await vi.importActual<typeof import('../services/api')>('../services/api');
  return { ...actual, submitApproval };
});

function wrap(ui: React.ReactNode) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return <QueryClientProvider client={client}>{ui}</QueryClientProvider>;
}

const payload: DagGatePayload = {
  gate: 'dag',
  treatment_variable: 'treat',
  outcome_variable: 're78',
  dag: {
    nodes: ['treat', 're78', 'age'],
    edges: [
      { source: 'age', target: 'treat', edge_type: 'directed' },
      { source: 'treat', target: 're78', edge_type: 'directed' },
    ],
    adjustment_set: ['age'],
    variable_roles: { age: 'confounder' },
    forbidden_edges: [],
    discovery_method: 'dag_expert',
    interpretation: 'age confounds the treatment',
  },
  justification: {
    identification: 'The effect of treat on re78 is identified by adjusting for age.',
    adjustment_set: ['age'],
    interpretation: 'age confounds the treatment',
    flags: ['collider_suspected'],
    concerns: ['age may be a collider'],
  },
};

beforeEach(() => {
  submitApproval.mockReset();
  submitApproval.mockResolvedValue({ job_id: 't', resumed: true, status: 'discovering_causal' });
});

describe('DagGate', () => {
  it('renders the identification, a concern, and the dag nodes', () => {
    render(wrap(<DagGate jobId="t" payload={payload} />));
    expect(screen.getByText(/identified by adjusting for age/i)).toBeTruthy();
    expect(screen.getByText(/age may be a collider/i)).toBeTruthy();
    // node labels render inside the svg
    expect(screen.getAllByText('age').length).toBeGreaterThan(0);
  });

  it('approves with decision approved', async () => {
    render(wrap(<DagGate jobId="t" payload={payload} />));
    fireEvent.click(screen.getByRole('button', { name: /approve · estimate/i }));
    await waitFor(() =>
      expect(submitApproval).toHaveBeenCalledWith('t', { decision: 'approved' }),
    );
  });

  it('revise is disabled until context is entered, then posts revise', async () => {
    render(wrap(<DagGate jobId="t" payload={payload} />));
    expect(screen.getByRole('button', { name: /revise · redo dag/i })).toBeDisabled();

    fireEvent.change(screen.getByPlaceholderText(/add context to redo the dag/i), {
      target: { value: 'age is a mediator, not a confounder' },
    });
    fireEvent.click(screen.getByRole('button', { name: /revise · redo dag/i }));

    await waitFor(() =>
      expect(submitApproval).toHaveBeenCalledWith('t', {
        decision: 'revise',
        appended_context: 'age is a mediator, not a confounder',
      }),
    );
  });

  it('reject requires a reason then posts rejected', async () => {
    render(wrap(<DagGate jobId="t" payload={payload} />));
    fireEvent.click(screen.getByRole('button', { name: /^reject$/i }));

    const confirm = screen.getByRole('button', { name: /confirm reject/i });
    expect(confirm).toBeDisabled();

    fireEvent.change(screen.getByPlaceholderText(/reason for rejecting the dag/i), {
      target: { value: 'the age to treat edge is backwards' },
    });
    fireEvent.click(confirm);

    await waitFor(() =>
      expect(submitApproval).toHaveBeenCalledWith('t', {
        decision: 'rejected',
        reason: 'the age to treat edge is backwards',
      }),
    );
  });
});
