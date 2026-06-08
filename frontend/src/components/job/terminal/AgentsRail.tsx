// Left pane: roster of 13 specialists with status dot, label, and latest-event headline.
// Live row gets a faint amber wash; pending rows fade to tertiary; settled rows sit secondary.

import type { AgentEvent } from '../../../services/api';
import { SPECIALIST_ROSTER, type AgentStatusMap } from './agents';
import { StatusDot, Caption } from './atoms';
import { describeEvent } from './describe';

export interface AgentsRailProps {
  tones: AgentStatusMap;
  latestByAgent: Map<string, AgentEvent>;
  /** Latest finding per agent; its headline is preferred for the subtitle. */
  findingByAgent: Map<string, AgentEvent>;
  /** Agents that raised a challenge; rendered with a rose marker. */
  challengedAgents: Set<string>;
  /** The agent currently pinned in the focus pane, or null. */
  selected: string | null;
  /** Click a row to inspect that agent (toggles off when re-clicked). */
  onSelect: (key: string) => void;
}

export function AgentsRail({
  tones,
  latestByAgent,
  findingByAgent,
  challengedAgents,
  selected,
  onSelect,
}: AgentsRailProps) {
  return (
    <aside className="w-[260px] shrink-0 bg-canvas-raised border-r border-edge-subtle flex flex-col overflow-hidden">
      <Caption>[ agents ]</Caption>
      <div className="flex-1 overflow-y-auto">
        {SPECIALIST_ROSTER.map(row => {
          const tone = tones[row.key];
          const latest = latestByAgent.get(row.key);
          const finding = findingByAgent.get(row.key);
          // Prefer the agent's finding headline; the generic latest event
          // (agent_completed) would otherwise read as a bare "<agent> done".
          const headline =
            finding && typeof finding.data?.headline === 'string'
              ? finding.data.headline
              : latest
                ? typeof latest.data?.headline === 'string'
                  ? latest.data.headline
                  : describeEvent(latest)
                : row.answers;
          const isSelected = selected === row.key;
          const hasChallenge = challengedAgents.has(row.key);
          return (
            <button
              key={row.key}
              type="button"
              onClick={() => onSelect(row.key)}
              aria-pressed={isSelected}
              className={`w-full text-left flex items-center gap-2 px-3 py-2 border-b border-edge-subtle/40 transition-colors hover:bg-canvas-overlay ${
                isSelected ? 'bg-amber/10 border-l-2 border-l-amber' : tone === 'live' ? 'bg-amber/5' : ''
              }`}
            >
              <StatusDot tone={tone} />
              <div className="min-w-0 flex-1">
                <div className={`font-mono text-xs truncate ${
                  tone === 'live' ? 'text-ink' :
                  tone === 'pending' ? 'text-ink-tertiary' :
                  'text-ink-secondary'
                }`}>{row.label}</div>
                <div className="font-mono text-2xs text-ink-tertiary truncate">{headline}</div>
              </div>
              {hasChallenge && (
                <span
                  title="raised a challenge"
                  className="inline-block w-1.5 h-1.5 rounded-full bg-rose shrink-0"
                />
              )}
            </button>
          );
        })}
      </div>
    </aside>
  );
}
