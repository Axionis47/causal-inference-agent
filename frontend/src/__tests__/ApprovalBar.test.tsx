import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ApprovalBar } from '../components/job/terminal/ApprovalBar';

const { submitApproval } = vi.hoisted(() => ({ submitApproval: vi.fn() }));

vi.mock('../services/api', async () => {
  const actual = await vi.importActual<typeof import('../services/api')>(
    '../services/api'
  );
  return { ...actual, submitApproval };
});

function wrap(ui: React.ReactNode) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return <QueryClientProvider client={client}>{ui}</QueryClientProvider>;
}

beforeEach(() => {
  submitApproval.mockReset();
  submitApproval.mockResolvedValue({ job_id: 't', resumed: true, status: 'discovering_causal' });
});

describe('ApprovalBar', () => {
  it('approves, passing optional notes as appended_context', async () => {
    render(wrap(<ApprovalBar jobId="t" />));
    fireEvent.change(screen.getByPlaceholderText(/optional notes/i), {
      target: { value: 'use re78 as outcome' },
    });
    fireEvent.click(screen.getByRole('button', { name: /approve/i }));
    await waitFor(() =>
      expect(submitApproval).toHaveBeenCalledWith('t', {
        decision: 'approved',
        appended_context: 'use re78 as outcome',
      }),
    );
  });

  it('reopens the dataset overlay via the view-data button', () => {
    // The bar is fixed over the FKeyBar at the gate, so it carries the only
    // reliable way back to the data once the overlay is closed.
    const onOpenData = vi.fn();
    render(wrap(<ApprovalBar jobId="t" onOpenData={onOpenData} />));
    fireEvent.click(screen.getByRole('button', { name: /view data/i }));
    expect(onOpenData).toHaveBeenCalledTimes(1);
  });

  it('requires a reason before it can reject', async () => {
    render(wrap(<ApprovalBar jobId="t" />));
    fireEvent.click(screen.getByRole('button', { name: /^reject$/i }));

    const confirm = screen.getByRole('button', { name: /confirm reject/i });
    expect(confirm).toBeDisabled(); // no reason yet

    fireEvent.change(screen.getByPlaceholderText(/reason for rejecting/i), {
      target: { value: 'wrong dataset' },
    });
    fireEvent.click(confirm);

    await waitFor(() =>
      expect(submitApproval).toHaveBeenCalledWith('t', {
        decision: 'rejected',
        reason: 'wrong dataset',
      }),
    );
  });
});
