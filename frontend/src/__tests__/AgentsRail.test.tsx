import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { AgentsRail } from '../components/job/terminal/AgentsRail';
import { SPECIALIST_ROSTER, type AgentStatusMap } from '../components/job/terminal/agents';

const tones: AgentStatusMap = Object.fromEntries(
  SPECIALIST_ROSTER.map((r) => [r.key, 'pending' as const]),
);

function renderRail(overrides: Partial<React.ComponentProps<typeof AgentsRail>> = {}) {
  const onSelect = vi.fn();
  render(
    <AgentsRail
      tones={tones}
      latestByAgent={new Map()}
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
});
