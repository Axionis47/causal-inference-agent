import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { FocusPane } from '../components/job/terminal/FocusPane';
import type { AgentEvent, AgentTrace, JobDetail } from '../services/api';

const job = {
  status: 'running',
  progress_percentage: 40,
  iteration_count: 1,
  treatment_variable: 'treat',
  outcome_variable: 'y',
  kaggle_url: 'https://kaggle.com/x',
} as JobDetail;

// A completed event with no headline, so the finding headline only appears in
// the dedicated finding section (not also in the generic "last event" row).
const lastEvent: AgentEvent = {
  timestamp: 't',
  agent_name: 'eda_agent',
  event_type: 'agent_completed',
  data: {},
};

const finding: AgentEvent = {
  timestamp: 't',
  agent_name: 'eda_agent',
  event_type: 'agent_finding',
  data: { headline: 'quality 82/100', status: 'done', flags: ['outcome_skewed'] },
};

const challenge: AgentEvent = {
  timestamp: 't',
  agent_name: 'eda_agent',
  event_type: 'agent_challenge',
  data: {
    headline: 'quality 82/100',
    status: 'done',
    flags: ['outcome_skewed'],
    issues: ['outcome is right-skewed'],
  },
};

const traces: AgentTrace[] = [
  {
    agent_name: 'eda_agent',
    timestamp: 't',
    action: 'step_1 · profile_columns',
    reasoning: 'outcome re78 is right-skewed; log-transform before correlation',
    duration_ms: 1200,
    inputs: { outcome: 're78' },
    outputs: { status: 'ok', output: 'skew 2.1, recommend log' },
    tools_called: ['profile_columns'],
    token_usage: { input_tokens: 100, output_tokens: 50 },
  },
];

function renderPane(overrides: Partial<React.ComponentProps<typeof FocusPane>> = {}) {
  render(
    <FocusPane
      job={job}
      focusAgent="eda_agent"
      focusLatest={lastEvent}
      focusFinding={finding}
      focusChallenge={undefined}
      focusTraces={[]}
      focusTone="ok"
      failed={false}
      datasetView={null}
      onOpenData={() => {}}
      {...overrides}
    />,
  );
}

describe('FocusPane findings and challenges', () => {
  it('renders the finding headline and its flag chip', () => {
    renderPane();
    expect(screen.getByText('quality 82/100')).toBeTruthy();
    expect(screen.getByText('outcome_skewed')).toBeTruthy();
  });

  it('renders the challenge issues when a challenge is present', () => {
    renderPane({ focusChallenge: challenge });
    expect(screen.getByText(/outcome is right-skewed/)).toBeTruthy();
  });

  it('falls back to the dataset dock when no agent is focused', () => {
    renderPane({ focusAgent: null, focusFinding: undefined, focusChallenge: undefined });
    expect(screen.getByText('dataset')).toBeTruthy();
    expect(screen.queryByText('quality 82/100')).toBeNull();
  });

  it('renders the reasoning steps when the focused agent has traces', () => {
    renderPane({ focusTraces: traces });
    expect(screen.getByText('reasoning')).toBeTruthy();
    expect(screen.getByText(/log-transform before correlation/)).toBeTruthy();
    expect(screen.getByText('profile_columns')).toBeTruthy();
    expect(screen.getByText(/skew 2\.1/)).toBeTruthy();
  });

  it('omits the reasoning section when the focused agent has no traces', () => {
    renderPane();
    expect(screen.queryByText('reasoning')).toBeNull();
  });
});
