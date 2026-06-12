import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ArtifactLine } from '../components/job/analysis/artifacts/ArtifactLine';
import type { AnalysisArtifactView } from '../services/api';

function artifact(overrides: Partial<AnalysisArtifactView>): AnalysisArtifactView {
  return {
    artifact_id: 'intake/causal_spec',
    kind: 'json',
    stage: 's1_intake_parsed',
    agent: 'intake',
    title: 'Causal spec',
    summary: null,
    media_type: 'application/json',
    created_at: '2026-06-11T10:00:00Z',
    ...overrides,
  };
}

function mockBody(body: string) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => new Response(body, { status: 200 })),
  );
}

function renderLine(a: AnalysisArtifactView) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <ArtifactLine jobId="job-1" artifact={a} />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  vi.unstubAllGlobals();
});

describe('ArtifactLine', () => {
  it('keeps the raw link pointing at the bytes endpoint', () => {
    renderLine(artifact({}));
    const raw = screen.getByRole('link', { name: 'raw' });
    expect(raw.getAttribute('href')).toContain(
      '/jobs/job-1/analysis/artifacts/intake/causal_spec',
    );
    expect(raw.getAttribute('target')).toBe('_blank');
  });

  it('expands a json artifact into the key/value view, not raw text', async () => {
    mockBody(JSON.stringify({ question_type: 'binary_treatment', covariates: ['x1', 'x2'] }));
    renderLine(artifact({}));
    fireEvent.click(screen.getByRole('button', { name: 'view' }));
    expect(await screen.findByText('question_type')).toBeTruthy();
    expect(screen.getByText('binary_treatment')).toBeTruthy();
    expect(screen.getByText('covariates')).toBeTruthy();
    // the directory description rides along
    expect(screen.getByText(/typed spec/)).toBeTruthy();
    // and collapses again
    fireEvent.click(screen.getByRole('button', { name: 'hide' }));
    expect(screen.queryByText('question_type')).toBeNull();
  });

  it('expands a markdown artifact into rendered prose', async () => {
    mockBody('# Profile\n\nRows look complete.');
    renderLine(
      artifact({
        artifact_id: 'profiling/summary',
        kind: 'markdown',
        title: 'Profile summary',
        media_type: 'text/markdown',
      }),
    );
    fireEvent.click(screen.getByRole('button', { name: 'view' }));
    const heading = await screen.findByRole('heading', { name: 'Profile' });
    expect(heading).toBeTruthy();
    expect(screen.getByText('Rows look complete.')).toBeTruthy();
  });

  it('expands a table artifact into a grid with header cells', async () => {
    mockBody('column,smd\nx9,0.394\nx12,0.21\n');
    renderLine(
      artifact({
        artifact_id: 'eda/tables/covariate_balance',
        kind: 'table',
        title: 'Covariate balance',
        media_type: 'text/csv',
      }),
    );
    fireEvent.click(screen.getByRole('button', { name: 'view' }));
    expect(await screen.findByRole('table')).toBeTruthy();
    expect(screen.getByText('smd')).toBeTruthy();
    expect(screen.getByText('0.394')).toBeTruthy();
  });

  it('shows the fetch failure instead of an empty view', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response('gone', { status: 404 })),
    );
    renderLine(artifact({}));
    fireEvent.click(screen.getByRole('button', { name: 'view' }));
    expect(await screen.findByText(/could not load artifact/)).toBeTruthy();
  });

  it('offers notebooks as an outward link instead of rendering inline', () => {
    renderLine(
      artifact({
        artifact_id: 'notebook/verified',
        kind: 'notebook',
        title: 'Verified notebook',
        media_type: 'application/x-ipynb+json',
      }),
    );
    fireEvent.click(screen.getByRole('button', { name: 'view' }));
    const open = screen.getByRole('link', { name: /open notebook/ });
    expect(open.getAttribute('href')).toContain('notebook/verified');
  });
});
