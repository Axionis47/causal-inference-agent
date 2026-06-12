import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { RunAnalysisBar } from '../components/job/terminal/RunAnalysisBar';

const { runAnalysis } = vi.hoisted(() => ({ runAnalysis: vi.fn() }));

vi.mock('../services/api', async () => {
  const actual = await vi.importActual<typeof import('../services/api')>(
    '../services/api'
  );
  return { ...actual, runAnalysis };
});

function wrap(ui: React.ReactNode) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return <QueryClientProvider client={client}>{ui}</QueryClientProvider>;
}

beforeEach(() => {
  runAnalysis.mockReset();
  runAnalysis.mockResolvedValue({
    job_id: 't',
    resumed: true,
    status: 'discovering_causal',
  });
});

describe('RunAnalysisBar', () => {
  it('launches the analysis', async () => {
    render(wrap(<RunAnalysisBar jobId="t" />));
    fireEvent.click(screen.getByRole('button', { name: /run analysis/i }));
    await waitFor(() => expect(runAnalysis).toHaveBeenCalledWith('t'));
  });

  it('reopens the dataset overlay via view-data', () => {
    const onOpenData = vi.fn();
    render(wrap(<RunAnalysisBar jobId="t" onOpenData={onOpenData} />));
    fireEvent.click(screen.getByRole('button', { name: /view data/i }));
    expect(onOpenData).toHaveBeenCalledTimes(1);
  });

  it('retry variant relaunches a failed run through the same endpoint', async () => {
    render(wrap(<RunAnalysisBar jobId="t" retry />));
    expect(screen.getByText(/analysis failed/i)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /retry analysis/i }));
    await waitFor(() => expect(runAnalysis).toHaveBeenCalledWith('t'));
  });
});
