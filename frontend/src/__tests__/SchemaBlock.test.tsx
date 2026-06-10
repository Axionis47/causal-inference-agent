import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { SchemaBlock } from '../components/job/terminal/SchemaBlock';
import type { ProfileBlock } from '../services/api';

const loaded: ProfileBlock = {
  status: 'loaded',
  data: {
    n_samples: 100,
    n_features: 3,
    feature_types: { date: 'datetime', treat: 'binary', sales: 'numeric' },
    missing_values: { date: 0, treat: 0, sales: 25 },
    has_time_dimension: true,
    time_column: 'date',
    treatment_candidates: [],
    outcome_candidates: [],
  },
  error: null,
};

describe('SchemaBlock', () => {
  it('renders each column with its type and missing percent', () => {
    render(<SchemaBlock block={loaded} />);
    expect(screen.getByText('treat')).toBeTruthy();
    expect(screen.getByText('binary')).toBeTruthy();
    // 25 of 100 missing -> 25.0%
    expect(screen.getByTestId('missing-sales').textContent).toBe('25.0%');
  });

  it('shows the time tag when the dataset has a time dimension', () => {
    render(<SchemaBlock block={loaded} />);
    expect(screen.getByText(/time: date/)).toBeTruthy();
  });

  it('shows a status line and no table until the profile loads', () => {
    const pending: ProfileBlock = { status: 'pending', data: null, error: null };
    const { container } = render(<SchemaBlock block={pending} />);
    expect(container.querySelector('table')).toBeNull();
    expect(screen.getByText('schema')).toBeTruthy();
  });
});
