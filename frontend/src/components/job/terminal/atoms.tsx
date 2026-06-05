// Reusable visual atoms for the terminal layout. Each is a sub-10-line presentational component.

import type { AgentTone } from './agents';
import type { BlockStatus } from '../../../services/api';

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

/** Map a dataset-block status to a status-dot tone. */
export function blockTone(status: BlockStatus): AgentTone {
  if (status === 'downloading') return 'live';
  if (status === 'downloaded' || status === 'loaded') return 'ok';
  if (status === 'failed' || status === 'error') return 'failed';
  return 'pending';
}

/** dot + label + status word, the header line of every dataset block. */
export function StatusLine({ label, status }: { label: string; status: BlockStatus }) {
  return (
    <div className="flex items-center gap-2">
      <StatusDot tone={blockTone(status)} />
      <span className="text-2xs font-mono uppercase tracking-[0.15em] text-ink-secondary">{label}</span>
      <span className="text-2xs font-mono uppercase tracking-[0.15em] text-ink-tertiary">{status}</span>
    </div>
  );
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
