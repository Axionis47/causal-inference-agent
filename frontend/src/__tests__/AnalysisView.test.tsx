import { describe, it, expect } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import { AnalysisView } from '../components/job/analysis/AnalysisView';
import type {
  AnalysisPlanGate,
  AnalysisViewResponse,
  MethodPlan,
  SelectedDesign,
} from '../services/api';

// A realistic mid-run payload: intake passed (summary + 2 artifacts),
// profiling running, design_detection still waiting.
const fixture: AnalysisViewResponse = {
  job_id: 'job-1',
  status: 'running',
  current_state: 's3_design_candidates_created',
  stage_index: 3,
  total_stages: 13,
  causal_question: 'Does job training raise later earnings?',
  error_message: null,
  spec_summary: {
    question_type: 'effect_of_cause',
    confidence: 'high',
    outcome: 're78',
    treatment: 'treat',
  },
  causal_dag: null,
  agents: [
    {
      agent: 'intake',
      stage: 's1_intake',
      status: 'passed',
      public_summary: 'Mapped the question to treatment=treat and outcome=re78.',
      current_step: null,
      warnings: [],
      artifact_ids: ['intake/causal_spec', 'intake/overlap_plot'],
      tool_call_count: 4,
      tokens: { input_tokens: 1200, output_tokens: 350 },
      cost_usd: 0.0123,
      elapsed_seconds: 42,
      attempt: 1,
    },
    {
      agent: 'profiling',
      stage: 's2_profiling',
      status: 'running',
      public_summary: null,
      current_step: 'profiling 18 columns',
      warnings: [],
      artifact_ids: [],
      tool_call_count: 2,
      tokens: { input_tokens: 800, output_tokens: 100 },
      cost_usd: 0.004,
      elapsed_seconds: 12,
      attempt: 1,
    },
    {
      agent: 'design_detection',
      stage: 's3_design',
      status: 'waiting',
      public_summary: null,
      current_step: null,
      warnings: [],
      artifact_ids: [],
      tool_call_count: 0,
      tokens: { input_tokens: 0, output_tokens: 0 },
      cost_usd: 0,
      elapsed_seconds: null,
      attempt: 1,
    },
  ],
  artifacts: [
    {
      artifact_id: 'intake/causal_spec',
      kind: 'json',
      stage: 's1_intake',
      agent: 'intake',
      title: 'Causal spec',
      summary: 'The parsed question.',
      media_type: 'application/json',
      created_at: '2026-06-11T10:00:00Z',
    },
    {
      artifact_id: 'intake/overlap_plot',
      kind: 'plot',
      stage: 's1_intake',
      agent: 'intake',
      title: 'Overlap plot',
      summary: null,
      media_type: 'image/png',
      created_at: '2026-06-11T10:00:05Z',
    },
  ],
  costs: {
    total_input_tokens: 2000,
    total_output_tokens: 450,
    total_cost_usd: 0.0163,
    total_tool_calls: 6,
  },
  events: [
    {
      sequence: 1,
      from_state: 's0_created',
      to_state: 's1_intake_done',
      agent_name: 'intake',
      timestamp: '2026-06-11T10:00:00Z',
      warnings: [],
    },
  ],
  plan_gate: null,
  method_plan: null,
  selected_design: null,
  notebook: null,
};

const methodPlan: MethodPlan = {
  lane: 'observational',
  estimator: 'aipw',
  estimand: 'ate',
  outcome: 're78',
  treatment: 'treat',
  covariates: ['age', 'educ'],
  settings: { n_folds: 5 },
  rationale: 'Doubly robust under the backdoor set.',
};

const selectedDesign: SelectedDesign = {
  lane: 'observational',
  design_label: 'backdoor adjustment',
  confidence: 'medium',
  rationale: 'No instrument or cutoff found; covariates close the backdoor.',
  required_fields: { outcome: 're78' },
  missing_requirements: [],
  warnings: [],
};

const waitingGate: AnalysisPlanGate = {
  status: 'needs_user_confirmation',
  reasons: ['outcome was inferred from the question, not stated'],
  missing_required: [],
  summary: 'Confirm the outcome before estimation runs.',
  confirmation_card: {
    headline: 'Confirm the analysis plan',
    plan_summary: 'AIPW on re78 with backdoor adjustment.',
    items: [
      {
        field: 'outcome',
        label: 'outcome column',
        current_value: 're78',
        options: ['re78', 're75'],
        required: true,
        why: 'The measured result the effect is estimated on.',
      },
    ],
  },
};

describe('AnalysisView', () => {
  it('renders one panel per agent with its status chip', () => {
    render(<AnalysisView analysis={fixture} />);
    expect(screen.getByTestId('agent-panel-intake')).toBeTruthy();
    expect(screen.getByTestId('agent-panel-profiling')).toBeTruthy();
    expect(screen.getByTestId('agent-panel-design_detection')).toBeTruthy();
    expect(screen.getByText('passed')).toBeTruthy();
    expect(screen.getAllByText('running').length).toBeGreaterThan(0);
    expect(screen.getByText('waiting')).toBeTruthy();
  });

  it('renders each artifact inside the panel of the agent that emitted it', () => {
    render(<AnalysisView analysis={fixture} />);
    const intake = within(screen.getByTestId('agent-panel-intake'));
    expect(intake.getByText('Causal spec')).toBeTruthy();
    expect(intake.getByText('Overlap plot')).toBeTruthy();
    // no standalone artifacts section anymore
    expect(screen.queryByText('[ artifacts ]')).toBeNull();
    // an empty-handed agent panel carries no artifact rows
    const profiling = within(screen.getByTestId('agent-panel-profiling'));
    expect(profiling.queryByRole('link', { name: 'open' })).toBeNull();
  });

  it('lists artifacts from agents without a run entry under other artifacts', () => {
    render(
      <AnalysisView
        analysis={{
          ...fixture,
          artifacts: [
            ...fixture.artifacts,
            {
              artifact_id: 'report/final_report',
              kind: 'markdown',
              stage: 's10_report',
              agent: 'report_notebook',
              title: 'Final report',
              summary: null,
              media_type: 'text/markdown',
              created_at: '2026-06-11T10:05:00Z',
            },
          ],
        }}
      />,
    );
    expect(screen.getByText('[ other artifacts ]')).toBeTruthy();
    expect(screen.getByText('Final report')).toBeTruthy();
  });

  it('shows the passed agent public summary and the running agent current step', () => {
    render(<AnalysisView analysis={fixture} />);
    expect(
      screen.getByText('Mapped the question to treatment=treat and outcome=re78.'),
    ).toBeTruthy();
    expect(screen.getByText('profiling 18 columns')).toBeTruthy();
  });

  it('renders the causal question, current state, and spec summary fields', () => {
    render(<AnalysisView analysis={fixture} />);
    expect(screen.getByText('Does job training raise later earnings?')).toBeTruthy();
    expect(screen.getByText(/s3_design_candidates_created/)).toBeTruthy();
    expect(screen.getByText('effect_of_cause')).toBeTruthy();
    expect(screen.getByText('treat')).toBeTruthy();
    expect(screen.getByText('re78')).toBeTruthy();
  });

  it('links artifacts through analysisArtifactUrl, keeping slashes in the id', () => {
    render(<AnalysisView analysis={fixture} />);
    const links = screen.getAllByRole('link');
    const hrefs = links.map((l) => l.getAttribute('href') ?? '');
    expect(
      hrefs.some((h) => h.endsWith('/jobs/job-1/analysis/artifacts/intake/causal_spec')),
    ).toBe(true);
    expect(
      hrefs.some((h) => h.endsWith('/jobs/job-1/analysis/artifacts/intake/overlap_plot')),
    ).toBe(true);
  });

  it('renders an inline thumbnail for plot artifacts that opens in a new tab', () => {
    render(<AnalysisView analysis={fixture} />);
    const img = screen.getByAltText('Overlap plot');
    expect(img.getAttribute('src')).toContain(
      '/jobs/job-1/analysis/artifacts/intake/overlap_plot',
    );
    const wrapper = img.closest('a');
    expect(wrapper?.getAttribute('target')).toBe('_blank');
  });

  it('renders the run cost line with totals and a 4-decimal cost', () => {
    render(<AnalysisView analysis={fixture} />);
    expect(screen.getByText('2,000')).toBeTruthy();
    expect(screen.getByText('450')).toBeTruthy();
    expect(screen.getByText('$0.0163')).toBeTruthy();
    expect(screen.getByText('6')).toBeTruthy();
  });

  it('shows the plan gate when the run waits for the user and a card exists', () => {
    render(
      <AnalysisView
        analysis={{
          ...fixture,
          status: 'waiting_for_user',
          plan_gate: waitingGate,
          method_plan: methodPlan,
          selected_design: selectedDesign,
        }}
      />,
    );
    expect(screen.getByTestId('plan-gate')).toBeTruthy();
    expect(screen.getByText('Confirm the analysis plan')).toBeTruthy();
    expect(screen.getByRole('button', { name: /confirm plan/i })).toBeTruthy();
  });

  it('does not show the plan gate while waiting if there is no confirmation card', () => {
    render(
      <AnalysisView
        analysis={{
          ...fixture,
          status: 'waiting_for_user',
          plan_gate: { ...waitingGate, confirmation_card: null },
        }}
      />,
    );
    expect(screen.queryByTestId('plan-gate')).toBeNull();
  });

  it('shows the auto-approved note instead of the gate when the plan passed', () => {
    render(
      <AnalysisView
        analysis={{
          ...fixture,
          plan_gate: {
            status: 'pass_auto_approved',
            reasons: [],
            missing_required: [],
            summary: 'All required fields present.',
            confirmation_card: null,
          },
        }}
      />,
    );
    expect(screen.getByText('plan auto-approved')).toBeTruthy();
    expect(screen.queryByTestId('plan-gate')).toBeNull();
  });

  it('renders the compact method-plan line whenever a method plan exists', () => {
    render(<AnalysisView analysis={{ ...fixture, method_plan: methodPlan }} />);
    expect(screen.getByText('observational · aipw · ate · re78')).toBeTruthy();
  });

  it('shows no notebook card while verification has not recorded a result', () => {
    render(<AnalysisView analysis={fixture} />);
    expect(screen.queryByTestId('notebook-download-card')).toBeNull();
  });

  it('offers the verified notebook download and the html preview link', () => {
    render(
      <AnalysisView
        analysis={{
          ...fixture,
          status: 'completed',
          notebook: {
            status: 'verified_running',
            attempts: 2,
            verified_artifact_id: 'notebook/verified',
            html_artifact_id: 'notebook/preview',
          },
        }}
      />,
    );
    expect(screen.getByText('verified: ran top to bottom')).toBeTruthy();
    expect(screen.getByText('2 attempts')).toBeTruthy();
    const download = screen.getByRole('link', { name: /download notebook/i });
    expect(download.getAttribute('href')).toContain(
      '/jobs/job-1/analysis/artifacts/notebook/verified',
    );
    expect(download.hasAttribute('download')).toBe(true);
    const preview = screen.getByRole('link', { name: /preview html/i });
    expect(preview.getAttribute('href')).toContain(
      '/jobs/job-1/analysis/artifacts/notebook/preview',
    );
  });

  it('states the failure instead of offering a file when verification failed', () => {
    render(
      <AnalysisView
        analysis={{
          ...fixture,
          status: 'completed',
          notebook: {
            status: 'failed',
            attempts: 3,
            verified_artifact_id: null,
            html_artifact_id: null,
          },
        }}
      />,
    );
    expect(screen.getByText('verification failed')).toBeTruthy();
    expect(screen.queryByRole('link', { name: /download notebook/i })).toBeNull();
  });

  it('omits the causal model card when no DAG has been built yet', () => {
    render(<AnalysisView analysis={fixture} />);
    expect(screen.queryByText('[ causal model ]')).toBeNull();
  });

  it('renders the causal model DAG with the latent-confounder note once built', () => {
    render(
      <AnalysisView
        analysis={{
          ...fixture,
          causal_dag: {
            nodes: [
              { name: 'treat', observed: true },
              { name: 're78', observed: true },
              { name: 'age', observed: true },
              { name: 'ability', observed: false },
            ],
            edges: [
              { source: 'age', target: 'treat', edge_type: 'directed' },
              { source: 'age', target: 're78', edge_type: 'directed' },
              { source: 'ability', target: 'treat', edge_type: 'directed' },
              { source: 'ability', target: 're78', edge_type: 'directed' },
              { source: 'treat', target: 're78', edge_type: 'directed' },
            ],
            treatment: 'treat',
            outcome: 're78',
            estimand: 'ate',
            adjustment_set: ['age'],
            latent_confounding: true,
          },
        }}
      />,
    );
    expect(screen.getByText('[ causal model ]')).toBeTruthy();
    expect(screen.getByRole('img', { name: 'causal dag' })).toBeTruthy();
    // the unobserved node is drawn and named in the graph
    expect(screen.getByText('ability')).toBeTruthy();
    // the not-identified note appears when a latent confounder breaks the backdoor
    expect(screen.getByText(/not point-identified/i)).toBeTruthy();
  });

  it('renders the error message text when the run failed', () => {
    render(
      <AnalysisView
        analysis={{
          ...fixture,
          status: 'failed',
          error_message: 'effect estimation crashed: singular matrix',
        }}
      />,
    );
    expect(
      screen.getByText('effect estimation crashed: singular matrix'),
    ).toBeTruthy();
  });
});
