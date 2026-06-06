// Middle pane: streaming event tape, newest at the top.
// Each row arrival fades from amber-tinted to transparent (animate-tape-arrival).
// Right-aligned delta column shows time gap to the previous event.

import type { AgentEvent } from '../../../services/api';
import { StatusDot, Caption } from './atoms';
import { describeEvent } from './describe';
import { formatHMS, formatDelta } from './format';

export interface TapeProps {
  /** Events in newest-first order. */
  reverseEvents: AgentEvent[];
  /** Currently dispatched agent. Rows for this agent get live tone instead of ok. */
  currentAgent: string | null | undefined;
}

export function Tape({ reverseEvents, currentAgent }: TapeProps) {
  return (
    <section className="flex-1 bg-canvas flex flex-col overflow-hidden min-w-0">
      <Caption>[ tape ]</Caption>
      <div className="flex-1 overflow-y-auto">
        {reverseEvents.length === 0 && (
          <div className="px-3 py-6 text-2xs font-mono text-ink-tertiary uppercase tracking-[0.15em]">
            awaiting first event…
          </div>
        )}
        {reverseEvents.map((ev, i) => {
          const key = `${ev.timestamp}-${ev.agent_name ?? '_'}-${ev.event_type}-${i}`;
          const isCurrent = ev.agent_name === currentAgent;
          // reverseEvents[i+1] is the chronologically prior event.
          const priorISO = reverseEvents[i + 1]?.timestamp;
          const delta = formatDelta(ev.timestamp, priorISO);
          return (
            <div
              key={key}
              className="flex items-baseline gap-3 px-3 py-1.5 border-b border-edge-subtle/40 animate-tape-arrival"
            >
              <span className="text-2xs font-mono text-ink-tertiary tabular shrink-0 w-16">
                {formatHMS(ev.timestamp)}
              </span>
              <span className="shrink-0">
                <StatusDot tone={isCurrent ? 'live' : 'ok'} />
              </span>
              <span className="font-mono text-xs shrink-0 w-44 truncate text-ink-secondary">
                {ev.agent_name ?? 'orchestrator'}
              </span>
              <span className="font-mono text-xs text-bone flex-1 truncate min-w-0">
                {describeEvent(ev)}
              </span>
              <span className="font-mono text-2xs text-ink-tertiary tabular shrink-0 w-16 text-right">
                {delta}
              </span>
            </div>
          );
        })}
      </div>
    </section>
  );
}
