import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { SampleRowsView } from '../components/job/terminal/SampleRowsView';
import type { DatasetView } from '../services/api';

// Rows come from the paginated endpoint; return a different row per page so
// we can assert that "next" actually advances the offset and refetches.
vi.mock('../services/api', async () => {
  const actual = await vi.importActual<typeof import('../services/api')>(
    '../services/api'
  );
  return {
    ...actual,
    getDatasetRows: vi.fn(async (_jobId: string, _file: string, offset: number) => ({
      columns: ['a', 'b'],
      rows: offset === 0 ? [{ a: 11, b: 22 }] : [{ a: 55, b: 66 }],
      total_rows: 120,
      offset,
      limit: 50,
    })),
  };
});

function makeView(): DatasetView {
  return {
    download: {
      status: 'downloaded',
      url: 'https://www.kaggle.com/datasets/o/n',
      files: [
        {
          name: 'lalonde.csv',
          size_bytes: 9400,
          format: 'csv',
          used: true,
          columns: ['a', 'b'],
          n_rows: 120,
        },
      ],
      error: null,
    },
    kaggle_meta: { status: 'unavailable', data: null, error: null },
    profile: { status: 'pending', data: null, error: null },
  };
}

function wrap(ui: React.ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{ui}</QueryClientProvider>;
}

describe('SampleRowsView', () => {
  it('shows the file, its row count, and the first page of rows', async () => {
    render(wrap(<SampleRowsView view={makeView()} jobId="t" />));
    // Row count from the manifest renders before rows even load.
    expect(screen.getByText(/rows 1-50 of 120/)).toBeTruthy();
    await waitFor(() => expect(screen.getByText('11')).toBeTruthy());
    expect(screen.getByText('22')).toBeTruthy();
  });

  it('pages forward when next is clicked', async () => {
    render(wrap(<SampleRowsView view={makeView()} jobId="t" />));
    await waitFor(() => expect(screen.getByText('11')).toBeTruthy());

    fireEvent.click(screen.getByRole('button', { name: /next/i }));

    await waitFor(() => expect(screen.getByText('55')).toBeTruthy());
    expect(screen.getByText(/rows 51-100 of 120/)).toBeTruthy();
  });

  it('shows a calm message before any file is downloaded', () => {
    const view = makeView();
    view.download = { status: 'downloading', url: null, files: [], error: null };
    render(wrap(<SampleRowsView view={view} jobId="t" />));
    expect(screen.getByText(/Rows appear once the files are downloaded/)).toBeTruthy();
  });
});
