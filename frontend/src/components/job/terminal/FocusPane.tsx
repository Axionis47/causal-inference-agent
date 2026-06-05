// Right pane: zoomed-in view of the currently dispatched agent and the job parameters.
// Empty state when no agent is dispatched. Error block appears at the bottom when present.

import type { AgentEvent, JobDetail } from '../../../services/api';
import { StatusDot, Caption, FocusRow } from './atoms';
import { describeEvent } from './describe';
import { statusPillLabel } from './format';

export interface FocusPaneProps {
  job: JobDetail;
  focusAgent: string | null;
  focusLatest: AgentEvent | undefined;
  failed: boolean;
}

export function FocusPane({ job, focusAgent, focusLatest, failed }: FocusPaneProps) {
  return (
    <aside className="w-[300px] shrink-0 bg-canvas-raised border-l border-edge-subtle flex flex-col overflow-hidden">
      <Caption>[ focus ]</Caption>
      <div className="flex-1 overflow-y-auto p-3">
        {focusAgent ? (
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <StatusDot tone={failed ? 'failed' : 'live'} />
              <span className="font-mono text-base text-ink">{focusAgent}</span>
            </div>

            <dl className="space-y-1.5">
              <FocusRow label="status" value={statusPillLabel(job.status).toLowerCase()} />
              <FocusRow label="progress" value={`${job.progress_percentage ?? 0}%`} mono />
              <FocusRow
                label="last event"
                value={focusLatest ? describeEvent(focusLatest) : 'awaiting…'}
              />
              <FocusRow label="iteration" value={String(job.iteration_count ?? 1)} mono />
            </dl>

            <div className="pt-2 border-t border-edge-subtle">
              <div className="text-2xs font-mono text-ink-tertiary uppercase tracking-[0.15em] mb-2">
                parameters
              </div>
              <dl className="space-y-1.5">
                <FocusRow label="treatment" value={job.treatment_variable || 'auto'} mono />
                <FocusRow label="outcome" value={job.outcome_variable || 'auto'} mono />
                <FocusRow label="dataset" value={job.kaggle_url.replace(/^https?:\/\//, '')} mono truncate />
              </dl>
            </div>

            {job.error_message && (
              <div className="pt-2 border-t border-edge-subtle">
                <div className="text-2xs font-mono text-rose uppercase tracking-[0.15em] mb-2">
                  error
                </div>
                <p className="text-xs font-mono text-bone">{job.error_message}</p>
              </div>
            )}
          </div>
        ) : (
          <p className="text-2xs font-mono text-ink-tertiary uppercase tracking-[0.15em]">
            no agent currently dispatched
          </p>
        )}
      </div>
    </aside>
  );
}
