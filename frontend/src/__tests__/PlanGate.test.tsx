import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { PlanGate } from '../components/job/analysis/PlanGate';
import type {
  AnalysisPlanGate,
  MethodPlan,
  SelectedDesign,
} from '../services/api';

const gate: AnalysisPlanGate = {
  status: 'needs_user_confirmation',
  reasons: ['cutoff value was inferred, not stated in the question'],
  missing_required: [],
  summary: 'RDD plan needs a confirmed cutoff.',
  confirmation_card: {
    headline: 'Confirm the regression discontinuity plan',
    plan_summary: 'Sharp RDD on test_score with a cutoff at 50.',
    items: [
      {
        field: 'cutoff_value',
        label: 'cutoff value',
        current_value: '50',
        options: [],
        required: true,
        why: 'The threshold that assigns treatment.',
      },
      {
        field: 'outcome',
        label: 'outcome column',
        current_value: 'earnings',
        options: ['earnings', 'employment'],
        required: true,
        why: 'The measured result.',
      },
      {
        field: 'intervention_date',
        label: 'intervention date',
        current_value: null,
        options: [],
        required: false,
        why: 'When the policy started, if known.',
      },
    ],
  },
};

const design: SelectedDesign = {
  lane: 'rdd',
  design_label: 'sharp regression discontinuity',
  confidence: 'high',
  rationale: 'Assignment jumps deterministically at the cutoff.',
  required_fields: { cutoff_value: '50' },
  missing_requirements: [],
  warnings: [],
};

const methodPlan: MethodPlan = {
  lane: 'rdd',
  estimator: 'local_linear_rdd',
  estimand: 'late',
  outcome: 'earnings',
  treatment: null,
  covariates: [],
  settings: { bandwidth: 'ik' },
  rationale: 'Local linear fit on both sides of the cutoff.',
};

function renderGate(overrides: Partial<Parameters<typeof PlanGate>[0]> = {}) {
  const onConfirm = vi.fn();
  const onReject = vi.fn();
  render(
    <PlanGate
      gate={gate}
      design={design}
      methodPlan={methodPlan}
      onConfirm={onConfirm}
      onReject={onReject}
      submitting={false}
      error={null}
      {...overrides}
    />,
  );
  return { onConfirm, onReject };
}

describe('PlanGate', () => {
  it('renders the headline, plan summary, design label with rationale, and reasons', () => {
    renderGate();
    expect(screen.getByText('Confirm the regression discontinuity plan')).toBeTruthy();
    expect(screen.getByText('Sharp RDD on test_score with a cutoff at 50.')).toBeTruthy();
    expect(screen.getByText('sharp regression discontinuity')).toBeTruthy();
    expect(
      screen.getByText('Assignment jumps deterministically at the cutoff.'),
    ).toBeTruthy();
    expect(
      screen.getByText(/cutoff value was inferred, not stated/),
    ).toBeTruthy();
  });

  it('renders a select for items with options and a text input otherwise, prefilled', () => {
    renderGate();
    const cutoff = screen.getByLabelText(/cutoff value/i) as HTMLInputElement;
    expect(cutoff.tagName).toBe('INPUT');
    expect(cutoff.value).toBe('50');

    const outcome = screen.getByLabelText(/outcome column/i) as HTMLSelectElement;
    expect(outcome.tagName).toBe('SELECT');
    expect(outcome.value).toBe('earnings');
    expect(screen.getByRole('option', { name: 'employment' })).toBeTruthy();

    const date = screen.getByLabelText(/intervention date/i) as HTMLInputElement;
    expect(date.tagName).toBe('INPUT');
    expect(date.value).toBe('');

    // each item's why renders as a caption
    expect(screen.getByText('The threshold that assigns treatment.')).toBeTruthy();
    expect(screen.getByText('The measured result.')).toBeTruthy();
    expect(screen.getByText('When the policy started, if known.')).toBeTruthy();

    // required items carry a marker; intervention_date is optional
    expect(screen.getAllByText('*')).toHaveLength(2);
  });

  it('confirm sends only changed or newly filled fields, as strings keyed by field', () => {
    const { onConfirm } = renderGate();

    fireEvent.change(screen.getByLabelText(/cutoff value/i), {
      target: { value: '60' },
    });
    fireEvent.change(screen.getByLabelText(/intervention date/i), {
      target: { value: '2024-01-01' },
    });
    // outcome select left at its prefilled value: must NOT ride along

    fireEvent.click(screen.getByRole('button', { name: /confirm plan/i }));

    expect(onConfirm).toHaveBeenCalledTimes(1);
    expect(onConfirm).toHaveBeenCalledWith({
      cutoff_value: '60',
      intervention_date: '2024-01-01',
    });
  });

  it('confirm with nothing edited sends an empty edits object', () => {
    const { onConfirm } = renderGate();
    fireEvent.click(screen.getByRole('button', { name: /confirm plan/i }));
    expect(onConfirm).toHaveBeenCalledWith({});
  });

  it('reject requires a non-empty reason before it can submit', () => {
    const { onReject } = renderGate();
    fireEvent.click(screen.getByRole('button', { name: /^reject$/i }));

    const submit = screen.getByRole('button', { name: /confirm reject/i });
    expect(submit).toBeDisabled();
    fireEvent.click(submit);
    expect(onReject).not.toHaveBeenCalled();

    fireEvent.change(screen.getByPlaceholderText(/reason for rejecting/i), {
      target: { value: 'wrong outcome' },
    });
    fireEvent.click(submit);
    expect(onReject).toHaveBeenCalledWith('wrong outcome');
  });

  it('disables both actions and shows progress text while submitting', () => {
    const { onConfirm } = renderGate({ submitting: true });
    const confirm = screen.getByRole('button', { name: /submitting/i });
    expect(confirm).toBeDisabled();
    expect(screen.getByRole('button', { name: /^reject$/i })).toBeDisabled();
    fireEvent.click(confirm);
    expect(onConfirm).not.toHaveBeenCalled();
  });

  it('renders the submit error as text', () => {
    renderGate({ error: 'edits.cutoff_value: must be numeric' });
    expect(screen.getByText('edits.cutoff_value: must be numeric')).toBeTruthy();
  });
});
