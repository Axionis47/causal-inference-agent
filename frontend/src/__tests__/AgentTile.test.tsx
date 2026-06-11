import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { AgentTile } from '../components/job/analysis/AgentTile';
import type { AnalysisAgentView } from '../services/api';

const base: AnalysisAgentView = {
  agent: 'ps_diagnostics',
  stage: 's7_diagnostics',
  status: 'passed',
  public_summary: 'Overlap looks healthy across the propensity range.',
  current_step: null,
  warnings: [],
  artifact_ids: ['ps_diagnostics/love_plot'],
  tool_call_count: 3,
  tokens: { input_tokens: 900, output_tokens: 240 },
  cost_usd: 0.0087,
  elapsed_seconds: 75,
  attempt: 1,
};

describe('AgentTile', () => {
  it('shows the warning state with its warnings listed in amber', () => {
    render(
      <AgentTile
        agent={{
          ...base,
          status: 'warning',
          warnings: ['weak overlap below ps=0.1', '3 covariates imbalanced'],
        }}
      />,
    );
    const chip = screen.getByText('warning');
    expect(chip.className).toContain('text-amber');
    expect(screen.getByText('! weak overlap below ps=0.1')).toBeTruthy();
    expect(screen.getByText('! 3 covariates imbalanced')).toBeTruthy();
  });

  it('shows the failed state with the rose status chip', () => {
    render(<AgentTile agent={{ ...base, status: 'failed', attempt: 2 }} />);
    const chip = screen.getByText('failed');
    expect(chip.className).toContain('text-rose');
    expect(screen.getByText('attempt 2')).toBeTruthy();
  });

  it('clamps the summary and expands it with the more/less toggle', () => {
    render(<AgentTile agent={base} />);
    const summary = screen.getByText(
      'Overlap looks healthy across the propensity range.',
    );
    expect(summary.className).toContain('line-clamp-4');
    fireEvent.click(screen.getByText('more'));
    expect(summary.className).not.toContain('line-clamp-4');
    fireEvent.click(screen.getByText('less'));
    expect(summary.className).toContain('line-clamp-4');
  });

  it('renders the tokens, cost, elapsed, tools, and artifact footer values', () => {
    render(<AgentTile agent={base} />);
    expect(screen.getByText('tok 900/240')).toBeTruthy();
    expect(screen.getByText('$0.0087')).toBeTruthy();
    expect(screen.getByText('01:15')).toBeTruthy();
    expect(screen.getByText('3 tools')).toBeTruthy();
    expect(screen.getByText('1 artifacts')).toBeTruthy();
  });
});
