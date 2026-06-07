import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ResultsGate } from '../components/job/terminal/ResultsGate';
import type { ResultsGatePayload } from '../services/api';

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

const payload: ResultsGatePayload = {
  gate: 'results',
  treatment_variable: 'treat',
  outcome_variable: 're78',
  effects: [
    { method: 'OLS', estimand: 'ATE', estimate: 1.2, ci_lower: 0.5, ci_upper: 1.9, p_value: 0.01 },
    { method: 'IPW', estimand: 'ATE', estimate: 1.0, ci_lower: 0.2, ci_upper: 1.8, p_value: 0.04 },
  ],
  sensitivity: [{ method: 'evalue', robustness_value: 1.8, interpretation: 'moderately robust' }],
  ps_diagnostics: { overlap_pct: 92.0, max_smd_weighted: 0.06 },
  balance: [
    { covariate: 'age', smd: 0.2 },
    { covariate: 'educ', smd: 0.05 },
  ],
};

beforeEach(() => {
  submitApproval.mockReset();
  submitApproval.mockResolvedValue({ job_id: 't', resumed: true, status: 'critique_review' });
});

describe('ResultsGate', () => {
  it('renders the forest methods, robustness, and a balance covariate', () => {
    render(wrap(<ResultsGate jobId="t" payload={payload} />));
    expect(screen.getByText('OLS')).toBeTruthy();
    expect(screen.getByText('IPW')).toBeTruthy();
    expect(screen.getByText(/moderately robust/i)).toBeTruthy();
    expect(screen.getByText('age')).toBeTruthy();
  });

  it('approves with decision approved', async () => {
    render(wrap(<ResultsGate jobId="t" payload={payload} />));
    fireEvent.click(screen.getByRole('button', { name: /approve · finalize/i }));
    await waitFor(() =>
      expect(submitApproval).toHaveBeenCalledWith('t', { decision: 'approved' }),
    );
  });

  it('revise is disabled until context is entered, then posts revise', async () => {
    render(wrap(<ResultsGate jobId="t" payload={payload} />));
    expect(screen.getByRole('button', { name: /revise · re-estimate/i })).toBeDisabled();

    fireEvent.change(screen.getByPlaceholderText(/add context to re-estimate/i), {
      target: { value: 'exclude the 2009 cohort' },
    });
    fireEvent.click(screen.getByRole('button', { name: /revise · re-estimate/i }));

    await waitFor(() =>
      expect(submitApproval).toHaveBeenCalledWith('t', {
        decision: 'revise',
        appended_context: 'exclude the 2009 cohort',
      }),
    );
  });

  it('reject requires a reason then posts rejected', async () => {
    render(wrap(<ResultsGate jobId="t" payload={payload} />));
    fireEvent.click(screen.getByRole('button', { name: /^reject$/i }));

    const confirm = screen.getByRole('button', { name: /confirm reject/i });
    expect(confirm).toBeDisabled();

    fireEvent.change(screen.getByPlaceholderText(/reason for rejecting the results/i), {
      target: { value: 'effect size is implausibly large' },
    });
    fireEvent.click(confirm);

    await waitFor(() =>
      expect(submitApproval).toHaveBeenCalledWith('t', {
        decision: 'rejected',
        reason: 'effect size is implausibly large',
      }),
    );
  });
});
