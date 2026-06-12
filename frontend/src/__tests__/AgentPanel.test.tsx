import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { AgentPanel } from '../components/job/analysis/panels/AgentPanel';
import type { AnalysisAgentView, AnalysisArtifactView } from '../services/api';

const base: AnalysisAgentView = {
  agent: 'targeted_eda',
  stage: 's4_targeted_eda_complete',
  status: 'passed',
  public_summary: 'Overlap looks healthy across the propensity range.',
  current_step: null,
  warnings: [],
  artifact_ids: ['eda/covariate_balance_plot'],
  tool_call_count: 3,
  tokens: { input_tokens: 900, output_tokens: 240 },
  cost_usd: 0.0087,
  elapsed_seconds: 75,
  attempt: 1,
};

const balancePlot: AnalysisArtifactView = {
  artifact_id: 'eda/covariate_balance_plot',
  kind: 'plot',
  stage: 's4_targeted_eda_complete',
  agent: 'targeted_eda',
  title: 'Covariate balance plot',
  summary: 'SMD per covariate before adjustment.',
  media_type: 'image/png',
  created_at: '2026-06-11T10:00:00Z',
};

function renderPanel(
  agent: AnalysisAgentView = base,
  artifacts: AnalysisArtifactView[] = [],
) {
  return render(<AgentPanel jobId="job-1" agent={agent} artifacts={artifacts} />);
}

describe('AgentPanel', () => {
  it('shows the agent name with the backend stage it owns', () => {
    renderPanel();
    expect(screen.getByText('targeted_eda')).toBeTruthy();
    expect(screen.getByText('s4_targeted_eda_complete')).toBeTruthy();
  });

  it('shows the warning state with its warnings listed in amber', () => {
    renderPanel({
      ...base,
      status: 'warning',
      warnings: ['weak overlap below ps=0.1', '3 covariates imbalanced'],
    });
    const chip = screen.getByText('warning');
    expect(chip.className).toContain('text-amber');
    expect(screen.getByText('! weak overlap below ps=0.1')).toBeTruthy();
    expect(screen.getByText('! 3 covariates imbalanced')).toBeTruthy();
  });

  it('shows the failed state with the rose status chip', () => {
    renderPanel({ ...base, status: 'failed', attempt: 2 });
    const chip = screen.getByText('failed');
    expect(chip.className).toContain('text-rose');
    expect(screen.getByText('attempt 2')).toBeTruthy();
  });

  it('clamps the summary and expands it with the more/less toggle', () => {
    renderPanel();
    const summary = screen.getByText(
      'Overlap looks healthy across the propensity range.',
    );
    expect(summary.className).toContain('line-clamp-4');
    fireEvent.click(screen.getByText('more'));
    expect(summary.className).not.toContain('line-clamp-4');
    fireEvent.click(screen.getByText('less'));
    expect(summary.className).toContain('line-clamp-4');
  });

  it('renders its own artifact lines, with plots as inline thumbnails', () => {
    renderPanel(base, [balancePlot]);
    expect(screen.getByText('Covariate balance plot')).toBeTruthy();
    expect(screen.getByText('SMD per covariate before adjustment.')).toBeTruthy();
    const img = screen.getByAltText('Covariate balance plot');
    expect(img.getAttribute('src')).toContain(
      '/jobs/job-1/analysis/artifacts/eda/covariate_balance_plot',
    );
    const raw = screen.getByRole('link', { name: 'raw' });
    expect(raw.getAttribute('target')).toBe('_blank');
  });

  it('renders no artifact section when the agent has emitted nothing', () => {
    renderPanel(base, []);
    expect(screen.queryByRole('link', { name: 'raw' })).toBeNull();
  });

  it('renders the tokens, cost, elapsed, and tools footer values', () => {
    renderPanel();
    expect(screen.getByText('tok 900/240')).toBeTruthy();
    expect(screen.getByText('$0.0087')).toBeTruthy();
    expect(screen.getByText('01:15')).toBeTruthy();
    expect(screen.getByText('3 tools')).toBeTruthy();
  });
});
