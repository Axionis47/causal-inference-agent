import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import type { DataProfileSummary, ProfileBlock } from '../services/api';
import SchemaSection from '../components/dataset/SchemaSection';

// SchemaSection renders the post-approval data profile (schema, treatment /
// outcome roles, missing-data flags). It is kept for the post-approval schema
// section; it is not part of the pre-approval download-and-display surface.

describe('SchemaSection', () => {
  // useExpandable persists collapse state in localStorage; clear it so each
  // test starts from the collapsed default.
  beforeEach(() => {
    window.localStorage.clear();
  });

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

  it('survives a partial-data shape from the persisted-results path', () => {
    // Persisted storage only saves a summary (n_samples, n_features,
    // candidates) and omits feature_types / missing_values. Component
    // must degrade to an empty schema table, not crash with
    // Object.entries(undefined).
    const partial: ProfileBlock = {
      status: 'loaded',
      data: {
        n_samples: 614,
        n_features: 11,
        treatment_candidates: ['treat'],
        outcome_candidates: ['re78'],
      } as DataProfileSummary,
      error: null,
    };
    render(<SchemaSection block={partial} />);
    // Headline still renders.
    expect(screen.getByText(/Schema · 614 rows × 11 columns/)).toBeTruthy();
    // No crash, and table is empty when expanded.
    fireEvent.click(screen.getByRole('button', { expanded: false }));
    expect(screen.queryByTestId(/^missing-/)).toBeNull();
  });

  it('survives the null → populated data transition without crashing', () => {
    // Catches the Rules-of-Hooks violation that occurred when useMemo
    // was placed after `if (!data) return null` — when data flipped
    // from null to populated, React threw and ErrorBoundary triggered
    // the live page.
    const { rerender } = render(
      <SchemaSection block={{ status: 'pending', data: null, error: null }} />
    );
    rerender(<SchemaSection block={profileBlock} />);
    expect(screen.getByText(/Schema · 614 rows/)).toBeTruthy();
  });
});
