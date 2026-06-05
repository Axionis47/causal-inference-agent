// Reusable visual atoms for the terminal layout. Each is a sub-10-line presentational component.

import type { AgentTone } from './agents';

/** 6px status dot, tone-coloured. Live tone pulses amber. */
export function StatusDot({ tone }: { tone: AgentTone }) {
  const cls: Record<AgentTone, string> = {
    pending: 'bg-edge-subtle',
    live: 'bg-amber animate-pulse-live',
    ok: 'bg-mint',
    failed: 'bg-rose',
  };
  return <span className={`inline-block w-1.5 h-1.5 rounded-full ${cls[tone]}`} />;
}

/** Section label rendered above each pane. Mono, tertiary, uppercase, tracked. */
export function Caption({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-2xs font-mono text-ink-tertiary uppercase tracking-[0.15em] px-3 py-2 border-b border-edge-subtle">
      {children}
    </div>
  );
}

/** Label/value pair used in FocusPane. `mono` switches the value to JetBrains Mono; `truncate` clips long values. */
export function FocusRow({
  label,
  value,
  mono,
  truncate,
}: { label: string; value: string; mono?: boolean; truncate?: boolean }) {
  return (
    <div className="flex items-baseline gap-3">
      <dt className="text-2xs font-mono text-ink-tertiary uppercase tracking-[0.15em] w-20 shrink-0">
        {label}
      </dt>
      <dd className={`text-xs ${mono ? 'font-mono' : ''} text-ink ${truncate ? 'truncate' : ''} min-w-0 flex-1`}>
        {value}
      </dd>
    </div>
  );
}

/** Function-key button used in the bottom bar. Disabled state dims and blocks the click. */
export function FKey({
  n, label, enabled, onClick,
}: { n: string; label: string; enabled: boolean; onClick?: () => void }) {
  return (
    <button
      type="button"
      onClick={enabled ? onClick : undefined}
      disabled={!enabled}
      className={`flex items-center gap-1.5 px-3 py-1 border-r border-edge-subtle uppercase tracking-[0.15em] ${
        enabled ? 'text-ink-secondary hover:text-ink hover:bg-canvas-overlay' : 'text-ink-tertiary cursor-not-allowed'
      } transition-colors`}
    >
      <span className="text-ink-tertiary">{n}</span>
      <span>{label}</span>
    </button>
  );
}
