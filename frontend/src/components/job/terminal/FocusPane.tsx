// Right pane: zoomed-in view of the currently dispatched agent and the job parameters.
// When no agent is dispatched (e.g. the fetch stage), it shows the download/dataset
// dock instead, so the download status is visible without reaching for a key.

import type { AgentEvent, DatasetView, JobDetail } from '../../../services/api';
import type { AgentTone } from './agents';
import { StatusDot, Caption, FocusRow } from './atoms';
import { describeEvent } from './describe';
import { statusPillLabel } from './format';

export interface FocusPaneProps {
  job: JobDetail;
  focusAgent: string | null;
  focusLatest: AgentEvent | undefined;
  failed: boolean;
  datasetView: DatasetView | null;
  onOpenData: () => void;
}

const DL_TONE: Record<string, AgentTone> = {
  downloading: 'live',
  downloaded: 'ok',
  loaded: 'ok',
  failed: 'failed',
  error: 'failed',
  pending: 'pending',
};

/** Download/dataset dock shown in the right pane while no agent is dispatched. */
function DatasetDock({ view, onOpenData }: { view: DatasetView | null; onOpenData: () => void }) {
  const dl = view?.download;
  const shape = view?.profile.data;
  const primary = dl?.files.find((f) => f.used) || dl?.files[0];
  return (
    <div className="space-y-3">
      <div className="text-2xs font-mono text-ink-tertiary uppercase tracking-[0.15em]">dataset</div>
      {!dl || dl.status === 'pending' ? (
        <p className="text-2xs font-mono text-ink-tertiary uppercase tracking-[0.15em]">waiting for download…</p>
      ) : (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <StatusDot tone={DL_TONE[dl.status] || 'pending'} />
            <span className="text-2xs font-mono uppercase tracking-[0.15em] text-ink-secondary">download</span>
            <span className="text-2xs font-mono uppercase tracking-[0.15em] text-ink-tertiary">{dl.status}</span>
          </div>
          {primary && <p className="font-mono text-xs text-ink truncate">{primary.name}</p>}
          {shape && (
            <p className="font-mono text-xs text-ink-secondary tabular">
              {shape.n_samples.toLocaleString()} rows × {shape.n_features} cols
            </p>
          )}
          {dl.error && <p className="text-xs text-rose">{dl.error}</p>}
        </div>
      )}
      <button
        type="button"
        onClick={onOpenData}
        className="text-2xs font-mono uppercase tracking-[0.15em] text-indigo hover:text-ink border border-edge-subtle hover:border-edge-strong px-2 py-1 transition-colors"
      >
        open full dataset · F1
      </button>
    </div>
  );
}

export function FocusPane({ job, focusAgent, focusLatest, failed, datasetView, onOpenData }: FocusPaneProps) {
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
          <DatasetDock view={datasetView} onOpenData={onOpenData} />
        )}
      </div>
    </aside>
  );
}
