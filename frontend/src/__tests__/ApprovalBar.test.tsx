import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ApprovalBar } from '../components/job/terminal/ApprovalBar';

const { submitApproval, confirmDataset } = vi.hoisted(() => ({
  submitApproval: vi.fn(),
  confirmDataset: vi.fn(),
}));

vi.mock('../services/api', async () => {
  const actual = await vi.importActual<typeof import('../services/api')>(
    '../services/api'
  );
  return { ...actual, submitApproval, confirmDataset };
});

function wrap(ui: React.ReactNode) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return <QueryClientProvider client={client}>{ui}</QueryClientProvider>;
}

beforeEach(() => {
  submitApproval.mockReset();
  submitApproval.mockResolvedValue({ job_id: 't', resumed: false, status: 'failed' });
  confirmDataset.mockReset();
  confirmDataset.mockResolvedValue({ job_id: 't', status: 'confirmed' });
});

describe('ApprovalBar', () => {
  it('confirms the dataset', async () => {
    render(wrap(<ApprovalBar jobId="t" />));
    fireEvent.click(screen.getByRole('button', { name: /confirm dataset/i }));
    await waitFor(() => expect(confirmDataset).toHaveBeenCalledWith('t'));
  });

  it('blocks confirm until inputs are valid', () => {
    render(wrap(<ApprovalBar jobId="t" canConfirm={false} />));
    const confirmBtn = screen.getByRole('button', { name: /confirm dataset/i });
    expect(confirmBtn).toBeDisabled();
    fireEvent.click(confirmBtn);
    expect(confirmDataset).not.toHaveBeenCalled();
    expect(screen.getByText(/set a valid treatment/i)).toBeInTheDocument();
  });

  it('reopens the dataset overlay via the view-data button', () => {
    const onOpenData = vi.fn();
    render(wrap(<ApprovalBar jobId="t" onOpenData={onOpenData} />));
    fireEvent.click(screen.getByRole('button', { name: /view data/i }));
    expect(onOpenData).toHaveBeenCalledTimes(1);
  });

  it('requires a reason before it can reject', async () => {
    render(wrap(<ApprovalBar jobId="t" />));
    fireEvent.click(screen.getByRole('button', { name: /^reject$/i }));

    const confirmBtn = screen.getByRole('button', { name: /confirm reject/i });
    expect(confirmBtn).toBeDisabled(); // no reason yet

    fireEvent.change(screen.getByPlaceholderText(/reason for rejecting/i), {
      target: { value: 'wrong dataset' },
    });
    fireEvent.click(confirmBtn);

    await waitFor(() =>
      expect(submitApproval).toHaveBeenCalledWith('t', {
        decision: 'rejected',
        reason: 'wrong dataset',
      }),
    );
  });
});
