import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import DataPanel from '../components/dataset/DataPanel';
import DownloadSection from '../components/dataset/DownloadSection';
import KaggleMetaSection from '../components/dataset/KaggleMetaSection';
import SchemaSection from '../components/dataset/SchemaSection';
import { useJobStore } from '../store/jobStore';
import type {
  DownloadBlock,
  KaggleMetaBlock,
  ProfileBlock,
} from '../services/api';

// Stub the dataset endpoint so the panel renders against store state
// instead of issuing real network calls. The hook only triggers a fetch
// once on mount; we want it to be a no-op so the test drives state.
vi.mock('../services/api', async () => {
  const actual = await vi.importActual<typeof import('../services/api')>(
    '../services/api'
  );
  return {
    ...actual,
    getDatasetView: vi.fn().mockRejectedValue(new Error('network disabled in test')),
  };
});

function withQueryClient(ui: React.ReactNode) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return <QueryClientProvider client={client}>{ui}</QueryClientProvider>;
}

beforeEach(() => {
  useJobStore.getState().reset();
  window.localStorage.clear();
});

// ─── DataPanel ────────────────────────────────────────────────────────────

describe('DataPanel', () => {
  it('renders the calm Preparing dataset line when all blocks are pending', () => {
    render(withQueryClient(<DataPanel jobId="t" />));
    expect(screen.getByText(/Preparing dataset/i)).toBeTruthy();
  });

  it('renders only the download section when only download is loaded', () => {
    useJobStore.getState().patchDownload({
      status: 'downloaded',
      url: 'https://www.kaggle.com/datasets/o/n',
      files: [{ name: 'lalonde.csv', size_bytes: 9400, format: 'csv', used: true }],
    });
    render(withQueryClient(<DataPanel jobId="t" />));
    expect(screen.getByText(/Downloaded · lalonde.csv/)).toBeTruthy();
    expect(screen.queryByText(/Kaggle metadata/)).toBeNull();
    expect(screen.queryByText(/Schema/)).toBeNull();
    expect(screen.queryByText(/Preparing dataset/)).toBeNull();
  });

  it('shows the metadata-quality chip in the panel header when loaded', () => {
    useJobStore.getState().patchKaggleMeta({
      status: 'loaded',
      data: {
        description: 'X',
        column_descriptions: {},
        tags: [],
        domain: null,
        metadata_quality: 'high',
      },
    });
    render(withQueryClient(<DataPanel jobId="t" />));
    expect(screen.getByText(/quality: high/)).toBeTruthy();
  });

  it('shows the unavailable footnote when kaggle is unavailable', () => {
    useJobStore.getState().patchKaggleMeta({ status: 'unavailable' });
    render(withQueryClient(<DataPanel jobId="t" />));
    expect(screen.getByText(/Kaggle did not provide metadata/)).toBeTruthy();
    expect(screen.queryByText(/Kaggle metadata ·/)).toBeNull();
  });
});

// ─── DownloadSection ──────────────────────────────────────────────────────

describe('DownloadSection', () => {
  const baseFiles = [
    { name: 'a.csv', size_bytes: 1024, format: 'csv', used: true },
    { name: 'README.txt', size_bytes: 64, format: 'txt', used: false },
  ];

  it('renders the failed line when status is failed', () => {
    const block: DownloadBlock = {
      status: 'failed',
      url: null,
      files: [],
      error: 'Dataset not accessible.',
    };
    render(<DownloadSection block={block} />);
    expect(screen.getByText(/Download failed/)).toBeTruthy();
    expect(screen.getByText(/not accessible/)).toBeTruthy();
  });

  it('renders the headline collapsed by default and reveals files on click', () => {
    const block: DownloadBlock = {
      status: 'downloaded',
      url: 'https://www.kaggle.com/datasets/o/n',
      files: baseFiles,
      error: null,
    };
    render(<DownloadSection block={block} />);
    expect(screen.getByText(/Downloaded · a.csv/)).toBeTruthy();
    expect(screen.queryByText('Format')).toBeNull(); // table hidden initially
    fireEvent.click(screen.getByRole('button', { expanded: false }));
    expect(screen.getByText('Format')).toBeTruthy();
    expect(screen.getByText('README.txt')).toBeTruthy();
  });
});

// ─── KaggleMetaSection ────────────────────────────────────────────────────

describe('KaggleMetaSection', () => {
  it('renders error line when status=error', () => {
    const block: KaggleMetaBlock = {
      status: 'error',
      data: null,
      error: 'rate limited',
    };
    render(<KaggleMetaSection block={block} />);
    expect(screen.getByText(/Metadata fetch failed/)).toBeTruthy();
  });

  it('renders headline with description and domain; expand reveals tags', () => {
    const block: KaggleMetaBlock = {
      status: 'loaded',
      data: {
        description: 'A short description.',
        column_descriptions: {},
        tags: ['economics', 'rct'],
        domain: 'econ',
        metadata_quality: 'high',
      },
      error: null,
    };
    render(<KaggleMetaSection block={block} />);
    expect(screen.getByText(/Kaggle metadata · A short description/)).toBeTruthy();
    expect(screen.queryByText('economics')).toBeNull();
    fireEvent.click(screen.getByRole('button', { expanded: false }));
    expect(screen.getByText('economics')).toBeTruthy();
    expect(screen.getByText('rct')).toBeTruthy();
  });

  it('shows the omitted-fields footer when description is the only field', () => {
    const block: KaggleMetaBlock = {
      status: 'loaded',
      data: {
        description: 'Just a description.',
        column_descriptions: {},
        tags: [],
        domain: null,
        metadata_quality: 'medium',
      },
      error: null,
    };
    render(<KaggleMetaSection block={block} />);
    fireEvent.click(screen.getByRole('button', { expanded: false }));
    expect(
      screen.getByText(/Kaggle did not provide:.*tags/i),
    ).toBeTruthy();
  });
});

// ─── SchemaSection ────────────────────────────────────────────────────────

describe('SchemaSection', () => {
  const profileBlock: ProfileBlock = {
    status: 'loaded',
    data: {
      n_samples: 614,
      n_features: 3,
      feature_types: { treat: 'binary', age: 'numeric', re78: 'numeric' },
      missing_values: { treat: 0, age: 154, re78: 5 },
      treatment_candidates: ['treat'],
      outcome_candidates: ['re78'],
      potential_confounders: ['age'],
      potential_instruments: [],
    },
    error: null,
  };

  it('renders headline with rows, columns, T, Y', () => {
    render(<SchemaSection block={profileBlock} />);
    expect(
      screen.getByText(/Schema · 614 rows × 3 columns/),
    ).toBeTruthy();
    expect(screen.getByText(/T: treat/)).toBeTruthy();
    expect(screen.getByText(/Y: re78/)).toBeTruthy();
  });

  it('expand reveals the column table and amber-flags >20% missing', () => {
    render(<SchemaSection block={profileBlock} />);
    fireEvent.click(screen.getByRole('button', { expanded: false }));
    // Column rows render.
    expect(screen.getByText('treat')).toBeTruthy();
    expect(screen.getByText('age')).toBeTruthy();
    // age has 154/614 = 25% missing → amber.
    const ageMissingCell = screen.getByTestId('missing-age');
    expect(ageMissingCell.className).toContain('amber-700');
    // re78 has 5/614 = 0.8% → no amber.
    const re78MissingCell = screen.getByTestId('missing-re78');
    expect(re78MissingCell.className).not.toContain('amber-700');
  });
});
