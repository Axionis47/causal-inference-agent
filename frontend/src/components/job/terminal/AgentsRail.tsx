// Left pane: roster of 13 specialists with status dot, label, and latest-event headline.
// Live row gets a faint amber wash; pending rows fade to tertiary; settled rows sit secondary.

import type { AgentEvent } from '../../../services/api';
import { SPECIALIST_ROSTER, type AgentStatusMap } from './agents';
import { StatusDot, Caption } from './atoms';
import { describeEvent } from './describe';

export interface AgentsRailProps {
  tones: AgentStatusMap;
  latestByAgent: Map<string, AgentEvent>;
}

export function AgentsRail({ tones, latestByAgent }: AgentsRailProps) {
  return (
    <aside className="w-[260px] shrink-0 bg-canvas-raised border-r border-edge-subtle flex flex-col overflow-hidden">
      <Caption>[ agents ]</Caption>
      <div className="flex-1 overflow-y-auto">
        {SPECIALIST_ROSTER.map(row => {
          const tone = tones[row.key];
          const latest = latestByAgent.get(row.key);
          const headline = latest
            ? (typeof latest.data?.headline === 'string' ? latest.data.headline : describeEvent(latest))
            : row.answers;
          return (
            <div
              key={row.key}
              className={`flex items-center gap-2 px-3 py-2 border-b border-edge-subtle/40 ${
                tone === 'live' ? 'bg-amber/5' : ''
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
            </div>
          );
        })}
      </div>
    </aside>
  );
}
