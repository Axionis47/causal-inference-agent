import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { AgentsRail } from '../components/job/terminal/AgentsRail';
import { SPECIALIST_ROSTER, type AgentStatusMap } from '../components/job/terminal/agents';
import type { AgentEvent } from '../services/api';

const tones: AgentStatusMap = Object.fromEntries(
  SPECIALIST_ROSTER.map((r) => [r.key, 'pending' as const]),
);

function renderRail(overrides: Partial<React.ComponentProps<typeof AgentsRail>> = {}) {
  const onSelect = vi.fn();
  render(
    <AgentsRail
      tones={tones}
      latestByAgent={new Map()}
      findingByAgent={new Map()}
      challengedAgents={new Set()}
      selected={null}
      onSelect={onSelect}
      {...overrides}
    />,
  );
  return { onSelect };
}

describe('AgentsRail', () => {
  it('calls onSelect with the agent key when a row is clicked', () => {
    const { onSelect } = renderRail();
    fireEvent.click(screen.getByText('eda').closest('button')!);
    expect(onSelect).toHaveBeenCalledWith('eda_agent');
  });

  it('marks the selected row with aria-pressed', () => {
    renderRail({ selected: 'eda_agent' });
    expect(screen.getByText('eda').closest('button')!.getAttribute('aria-pressed')).toBe('true');
  });

  it('shows a challenge marker for an agent that raised one', () => {
    renderRail({ challengedAgents: new Set(['dag_expert']) });
    expect(screen.getByTitle('raised a challenge')).toBeTruthy();
  });

  it('shows no challenge marker when no agent raised one', () => {
    renderRail();
    expect(screen.queryByTitle('raised a challenge')).toBeNull();
  });

  it('prefers the finding headline over the generic latest event', () => {
    // agent_completed (no headline) is the latest event but the finding carries
    // the real headline; the subtitle must show the headline, not "eda done".
    const completed: AgentEvent = {
      timestamp: 't', agent_name: 'eda_agent', event_type: 'agent_completed', data: {},
    };
    const finding: AgentEvent = {
      timestamp: 't', agent_name: 'eda_agent', event_type: 'agent_finding',
      data: { headline: 'quality 82/100', status: 'done', flags: [] },
    };
    renderRail({
      latestByAgent: new Map([['eda_agent', completed]]),
      findingByAgent: new Map([['eda_agent', finding]]),
    });
    expect(screen.getByText('quality 82/100')).toBeTruthy();
  });
});
