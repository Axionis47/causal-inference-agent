import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { DatasetView } from '../components/job/terminal/DatasetView';
import type { DatasetView as DatasetViewData, KaggleMeta } from '../services/api';

// The metadata panel must never trigger a real rows fetch; stub it so the
// raw-rows section renders its idle state and we can assert on metadata.
vi.mock('../services/api', async () => {
  const actual = await vi.importActual<typeof import('../services/api')>(
    '../services/api'
  );
  return {
    ...actual,
    getDatasetRows: vi.fn().mockRejectedValue(new Error('network disabled')),
  };
});

function wrap(ui: React.ReactNode) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return <QueryClientProvider client={client}>{ui}</QueryClientProvider>;
}

function makeView(meta: Partial<KaggleMeta>): DatasetViewData {
  return {
    download: { status: 'downloaded', url: 'https://kaggle.com/d/o/n', files: [], error: null },
    kaggle_meta: {
      status: 'loaded',
      data: {
        description: null,
        column_descriptions: {},
        tags: [],
        domain: null,
        metadata_quality: 'unknown',
        ...meta,
      },
      error: null,
    },
    profile: { status: 'pending', data: null, error: null },
  };
}

describe('DatasetView metadata panel', () => {
  it('renders the facts Kaggle supplied', () => {
    const view = makeView({
      title: 'Lalonde A/B Testing',
      license: 'CC0-1.0',
      tags: ['business', 'education'],
      download_count: 552,
      usability_rating: 0.647,
    });
    render(wrap(<DatasetView view={view} jobId="t" onClose={() => {}} />));

    expect(screen.getByText('Lalonde A/B Testing')).toBeTruthy();
    expect(screen.getByText('CC0-1.0')).toBeTruthy();
    expect(screen.getByText('business')).toBeTruthy();
    expect(screen.getByText('552')).toBeTruthy();
    expect(screen.getByText('65%')).toBeTruthy(); // usability rounded to a percent
  });

  it('shows a placeholder for fields the source did not provide', () => {
    const view = makeView({ title: 'Only a title' });
    render(wrap(<DatasetView view={view} jobId="t" onClose={() => {}} />));
    // Several fields (license, downloads, votes, ...) are absent -> em-dash.
    expect(screen.getAllByText('—').length).toBeGreaterThan(3);
    expect(screen.getByText('none provided by source')).toBeTruthy(); // columns
  });

  it('strips markdown heading markers from the description', () => {
    const view = makeView({ description: '### Context\nFamous job-training dataset.' });
    render(wrap(<DatasetView view={view} jobId="t" onClose={() => {}} />));
    // The "### " marker must not leak into the rendered text.
    expect(screen.getByText('Context')).toBeTruthy();
    expect(screen.queryByText(/### Context/)).toBeNull();
    expect(screen.getByText('Famous job-training dataset.')).toBeTruthy();
  });
});
