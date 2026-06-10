import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { InputsBlock } from '../components/job/terminal/InputsBlock';
import type { ProfileBlock } from '../services/api';

vi.mock('../services/api', async () => {
  const actual = await vi.importActual<typeof import('../services/api')>(
    '../services/api',
  );
  return { ...actual, updateDatasetInputs: vi.fn() };
});

import { updateDatasetInputs } from '../services/api';

const profile: ProfileBlock = {
  status: 'loaded',
  data: {
    n_samples: 100,
    n_features: 3,
    feature_types: { treat: 'binary', re78: 'numeric', date: 'datetime' },
    missing_values: {},
    has_time_dimension: true,
    time_column: 'date',
    treatment_candidates: [],
    outcome_candidates: [],
  },
  error: null,
};

type Props = Parameters<typeof InputsBlock>[0];

function renderBlock(props: Partial<Props> = {}) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <InputsBlock
        jobId="job-1"
        treatment="treat"
        outcome="re78"
        profile={profile}
        {...props}
      />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe('InputsBlock', () => {
  it('seeds the selectors with the current treatment, outcome, and time column', () => {
    renderBlock();
    expect((screen.getByLabelText('treatment') as HTMLSelectElement).value).toBe('treat');
    expect((screen.getByLabelText('outcome') as HTMLSelectElement).value).toBe('re78');
    expect((screen.getByLabelText('time column') as HTMLSelectElement).value).toBe('date');
  });

  it('flags a treatment that is not a real column', () => {
    renderBlock({ treatment: 'tret' });
    expect(screen.getByText('not a column')).toBeTruthy();
  });

  it('saves a corrected treatment via the endpoint', async () => {
    vi.mocked(updateDatasetInputs).mockResolvedValue({
      treatment_variable: 'treat',
      outcome_variable: 're78',
      time_column: 'date',
      has_time_dimension: true,
    });
    renderBlock({ treatment: 'tret' });
    fireEvent.change(screen.getByLabelText('treatment'), { target: { value: 'treat' } });
    fireEvent.click(screen.getByRole('button', { name: /save inputs/i }));
    await waitFor(() => expect(vi.mocked(updateDatasetInputs)).toHaveBeenCalled());
    const [jobId, body] = vi.mocked(updateDatasetInputs).mock.calls[0];
    expect(jobId).toBe('job-1');
    expect(body.treatment_variable).toBe('treat');
    expect(body.outcome_variable).toBe('re78');
  });

  it('can clear the time column to none', async () => {
    vi.mocked(updateDatasetInputs).mockResolvedValue({
      treatment_variable: 'treat',
      outcome_variable: 're78',
      time_column: null,
      has_time_dimension: false,
    });
    renderBlock();
    fireEvent.change(screen.getByLabelText('time column'), { target: { value: '' } });
    fireEvent.click(screen.getByRole('button', { name: /save inputs/i }));
    await waitFor(() => expect(vi.mocked(updateDatasetInputs)).toHaveBeenCalled());
    expect(vi.mocked(updateDatasetInputs).mock.calls[0][1].time_column).toBeNull();
  });

  it('stays read-only until the schema loads', () => {
    renderBlock({ profile: { status: 'pending', data: null, error: null } });
    expect(screen.queryByRole('button', { name: /save inputs/i })).toBeNull();
  });
});
