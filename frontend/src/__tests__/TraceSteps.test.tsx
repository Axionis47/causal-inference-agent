import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { TraceSteps } from '../components/job/terminal/TraceSteps';
import type { AgentTrace } from '../services/api';

const base: AgentTrace = {
  agent_name: 'dag_expert',
  timestamp: '2026-06-08T00:00:00Z',
  action: 'step_1 · get_dag_adjustment_set',
  reasoning: 'block the back-door paths via age and education',
  duration_ms: 1840,
  inputs: { treatment: 'treatment' },
  outputs: { status: 'ok', output: 'adjustment set: {age, educ}' },
  tools_called: ['get_dag_adjustment_set'],
  token_usage: { input_tokens: 220, output_tokens: 120 },
};

describe('TraceSteps', () => {
  it('renders the reasoning, tool chip, and string output preview of a step', () => {
    render(<TraceSteps traces={[base]} />);
    expect(screen.getByText(/block the back-door paths/)).toBeTruthy();
    expect(screen.getByText('get_dag_adjustment_set')).toBeTruthy();
    expect(screen.getByText(/adjustment set: \{age, educ\}/)).toBeTruthy();
  });

  it('shows duration in seconds and a token count', () => {
    render(<TraceSteps traces={[base]} />);
    expect(screen.getByText(/1\.8s · 340tok/)).toBeTruthy();
  });

  it('falls back to compact JSON when outputs carry no string output field', () => {
    const noStringOut: AgentTrace = {
      ...base,
      action: 'execute',
      outputs: { rows_dropped: 4, imputed: true },
    };
    render(<TraceSteps traces={[noStringOut]} />);
    expect(screen.getByText(/"rows_dropped":4/)).toBeTruthy();
  });

  it('renders one entry per trace', () => {
    render(<TraceSteps traces={[base, { ...base, action: 'step_2 · check_identifiability' }]} />);
    expect(screen.getByText('step_1 · get_dag_adjustment_set')).toBeTruthy();
    expect(screen.getByText('step_2 · check_identifiability')).toBeTruthy();
  });
});
