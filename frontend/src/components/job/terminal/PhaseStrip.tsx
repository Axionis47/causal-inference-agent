// 13-cell strip across the top-bar middle. One cell per specialist, lit by current tone.
// Live cell pulses amber; completed cells stay mint; pending sit edge-subtle; failed flips rose.

import { SPECIALIST_ROSTER, type AgentStatusMap } from './agents';

export function PhaseStrip({ tones }: { tones: AgentStatusMap }) {
  return (
    <div className="flex-1 flex items-center justify-center gap-0.5 min-w-0">
      {SPECIALIST_ROSTER.map(row => {
        const tone = tones[row.key];
        const bg =
          tone === 'live' ? 'bg-amber animate-pulse-live' :
          tone === 'ok' ? 'bg-mint' :
          tone === 'failed' ? 'bg-rose' :
          'bg-edge-subtle';
        return (
          <div
            key={row.key}
            className={`h-1.5 w-6 ${bg}`}
            title={`${row.label} · ${tone}`}
          />
        );
      })}
    </div>
  );
}
