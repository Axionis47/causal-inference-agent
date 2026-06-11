import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { AnalysisView } from '../components/job/analysis/AnalysisView';
import type { AnalysisViewResponse } from '../services/api';

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
};

describe('AnalysisView', () => {
  it('renders one tile per agent with its status chip', () => {
    render(<AnalysisView analysis={fixture} />);
    expect(screen.getByTestId('agent-tile-intake')).toBeTruthy();
    expect(screen.getByTestId('agent-tile-profiling')).toBeTruthy();
    expect(screen.getByTestId('agent-tile-design_detection')).toBeTruthy();
    expect(screen.getByText('passed')).toBeTruthy();
    expect(screen.getAllByText('running').length).toBeGreaterThan(0);
    expect(screen.getByText('waiting')).toBeTruthy();
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
