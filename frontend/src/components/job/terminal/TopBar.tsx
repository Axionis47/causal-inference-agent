// Top bar: job id + orchestrator mode (left), phase strip (middle),
// status + elapsed + tokens + cancel (right).

import type { JobDetail } from '../../../services/api';
import type { AgentStatusMap } from './agents';
import { StatusDot } from './atoms';
import { PhaseStrip } from './PhaseStrip';
import { statusTone, statusPillLabel } from './format';

export interface TopBarProps {
  job: JobDetail;
  agentTones: AgentStatusMap;
  elapsed: string;
  /** Total tokens (input + output) summed across traces; null while unknown. */
  tokens: number | null;
  isPreview: boolean;
  onCancel: () => void;
  cancelPending: boolean;
}

export function TopBar({ job, agentTones, elapsed, tokens, isPreview, onCancel, cancelPending }: TopBarProps) {
  const canCancel = !isPreview && job.status !== 'completed' && job.status !== 'failed';

  return (
    <header className="flex items-center h-10 px-3 bg-canvas-raised border-b border-edge-subtle shrink-0 gap-6">
      <div className="flex items-center gap-4 min-w-0 shrink-0">
        <span className="font-mono text-xs text-ink tabular">
          [ job <span className="text-ink-secondary">{job.id.slice(0, 8)}</span> ]
        </span>
        <span className="text-2xs font-mono uppercase tracking-[0.15em] text-indigo">
          standard
        </span>
      </div>

      <PhaseStrip tones={agentTones} />

      <div className="flex items-center gap-4 text-xs font-mono tabular shrink-0">
        <span className="inline-flex items-center gap-1.5">
          <StatusDot tone={statusTone(job.status)} />
          <span className="text-2xs uppercase tracking-[0.15em] text-ink-secondary">
            {statusPillLabel(job.status)}
          </span>
        </span>
        <span className="text-ink-secondary">
          <span className="text-ink-tertiary mr-1">elapsed</span>
          <span className="text-ink">{elapsed}</span>
        </span>
        <span className="text-ink-secondary">
          <span className="text-ink-tertiary mr-1">tokens</span>
          <span className="text-ink">{tokens != null && tokens > 0 ? tokens.toLocaleString() : '—'}</span>
        </span>
        {canCancel && (
          <button
            onClick={onCancel}
            disabled={cancelPending}
            className="text-2xs font-mono uppercase tracking-[0.15em] text-rose hover:text-ink border border-rose/40 hover:border-rose px-2 py-1 transition-colors"
          >
            {cancelPending ? 'stopping…' : 'cancel'}
          </button>
        )}
      </div>
    </header>
  );
}
