import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import JobsListPage from '../pages/JobsListPage';
import type { Job } from '../services/api';

vi.mock('../services/api', async () => {
  const actual = await vi.importActual<typeof import('../services/api')>(
    '../services/api'
  );
  return {
    ...actual,
    listJobs: vi.fn(),
    cancelJob: vi.fn(),
    deleteJob: vi.fn(),
  };
});

import { listJobs } from '../services/api';

function renderPage(jobs: Job[]) {
  vi.mocked(listJobs).mockResolvedValue({ jobs, total: jobs.length });
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <JobsListPage />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

const baseJob: Job = {
  id: 'abcd1234efgh5678',
  kaggle_url: 'https://www.kaggle.com/datasets/samuelzakouri/lalonde',
  status: 'completed',
  created_at: '2026-05-10T12:00:00Z',
  updated_at: '2026-05-10T12:03:10Z',
  dataset_name: 'lalonde',
  causal_question: 'Does training raise income?',
  treatment_variable: 'treat',
  outcome_variable: 're78',
  iteration_count: 2,
};

beforeEach(() => {
  vi.clearAllMocks();
});

describe('JobsListPage digest row', () => {
  it('renders the dataset name and the causal question from the response', async () => {
    renderPage([baseJob]);
    expect(await screen.findByText('lalonde')).toBeTruthy();
    expect(screen.getByText('Does training raise income?')).toBeTruthy();
  });

  it('renders iteration count and a computed duration for terminal jobs', async () => {
    renderPage([baseJob]);
    // 12:03:10 - 12:00:00 = 190s = 3m (rounded)
    expect(await screen.findByText('3m')).toBeTruthy();
    expect(screen.getByText('2')).toBeTruthy(); // iteration count
  });

  it('shows a placeholder when no question is set', async () => {
    renderPage([
      {
        ...baseJob,
        causal_question: null,
      },
    ]);
    expect(await screen.findByText(/no question set/i)).toBeTruthy();
  });

  it('falls back to extracted dataset name when backend did not populate it', async () => {
    renderPage([
      {
        ...baseJob,
        dataset_name: null,
        kaggle_url: 'https://www.kaggle.com/datasets/owner/some-dataset',
      },
    ]);
    expect(await screen.findByText('some-dataset')).toBeTruthy();
  });

  it('renders em-dash for duration on running jobs', async () => {
    renderPage([
      {
        ...baseJob,
        status: 'profiling',
        updated_at: '2026-05-10T12:00:30Z',
      },
    ]);
    // Wait for the row to render first, then assert duration cell is the dash placeholder.
    expect(await screen.findByText('lalonde')).toBeTruthy();
    // The dash text content is "—" in a span with text-ink-300.
    const dashes = screen.getAllByText('—');
    expect(dashes.length).toBeGreaterThan(0);
  });
});
