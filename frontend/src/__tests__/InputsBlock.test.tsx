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
        causalQuestion="Does training raise income?"
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
  it('seeds the question and the time column from the current values', () => {
    renderBlock();
    expect((screen.getByLabelText('causal question') as HTMLTextAreaElement).value).toBe(
      'Does training raise income?',
    );
    expect((screen.getByLabelText('time column') as HTMLSelectElement).value).toBe('date');
  });

  it('marks an empty question as not ready to confirm', () => {
    renderBlock({ causalQuestion: '' });
    expect(screen.getByText(/a question is required/i)).toBeTruthy();
  });

  it('saves a refined question via the endpoint', async () => {
    vi.mocked(updateDatasetInputs).mockResolvedValue({
      causal_question: 'Does training raise earnings?',
      time_column: 'date',
      has_time_dimension: true,
    });
    renderBlock();
    fireEvent.change(screen.getByLabelText('causal question'), {
      target: { value: 'Does training raise earnings?' },
    });
    fireEvent.click(screen.getByRole('button', { name: /save inputs/i }));
    await waitFor(() => expect(vi.mocked(updateDatasetInputs)).toHaveBeenCalled());
    const [jobId, body] = vi.mocked(updateDatasetInputs).mock.calls[0];
    expect(jobId).toBe('job-1');
    expect(body.causal_question).toBe('Does training raise earnings?');
  });

  it('can clear the time column to none', async () => {
    vi.mocked(updateDatasetInputs).mockResolvedValue({
      causal_question: 'Does training raise income?',
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
