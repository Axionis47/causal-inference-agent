import { describe, it, expect } from 'vitest';
import { resolveArtifact } from '../components/job/analysis/artifacts/directory';
import { parseCsv } from '../components/job/analysis/artifacts/csv';
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

describe('resolveArtifact', () => {
  it('resolves a known id to its description with the kind renderer', () => {
    const r = resolveArtifact(artifact({}));
    expect(r.description).toMatch(/typed spec/);
    expect(r.renderer).toBe('json');
  });

  it('resolves a dynamic eda check through the prefix family', () => {
    const r = resolveArtifact(
      artifact({ artifact_id: 'eda/propensity_overlap', kind: 'plot' }),
    );
    expect(r.description).toMatch(/EDA check/);
    expect(r.renderer).toBe('plot');
  });

  it('falls back to the kind renderer for an unknown artifact id', () => {
    const r = resolveArtifact(
      artifact({ artifact_id: 'something/new', kind: 'table' }),
    );
    expect(r.description).toBeNull();
    expect(r.renderer).toBe('table');
  });

  it('sends notebooks and html out as links, never inline', () => {
    expect(
      resolveArtifact(artifact({ artifact_id: 'notebook/verified', kind: 'notebook' }))
        .renderer,
    ).toBe('link');
    expect(
      resolveArtifact(artifact({ artifact_id: 'notebook/html_preview', kind: 'html' }))
        .renderer,
    ).toBe('link');
  });
});

describe('parseCsv', () => {
  it('parses a plain header plus rows with a trailing newline', () => {
    expect(parseCsv('a,b\n1,2\n3,4\n')).toEqual([
      ['a', 'b'],
      ['1', '2'],
      ['3', '4'],
    ]);
  });

  it('keeps commas and newlines inside quoted fields', () => {
    expect(parseCsv('name,note\nx,"hello, world"\ny,"line1\nline2"\n')).toEqual([
      ['name', 'note'],
      ['x', 'hello, world'],
      ['y', 'line1\nline2'],
    ]);
  });

  it('unescapes doubled quotes and handles crlf endings', () => {
    expect(parseCsv('q\r\n"say ""hi"""\r\n')).toEqual([['q'], ['say "hi"']]);
  });

  it('returns no rows for empty input', () => {
    expect(parseCsv('')).toEqual([]);
  });
});
