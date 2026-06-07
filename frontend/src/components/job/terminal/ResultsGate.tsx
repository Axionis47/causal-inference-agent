// Results-review gate (checkpoint B). Shown when a job parks at the results gate:
// the analyst reviews the effect estimates (forest plot), robustness, and
// covariate balance (Love plot), then approves (finalize), revises (re-estimate
// with a note), or rejects. Posts to POST /jobs/:id/approval, same as DagGate.

import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { submitApproval, ApprovalRequestBody, ResultsGatePayload } from '../../../services/api';
import { ForestPlot } from './ForestPlot';
import { BalancePlot } from './BalancePlot';

const labelCls = 'text-2xs font-mono uppercase tracking-[0.15em]';
const inputCls =
  'bg-canvas-inset border border-edge-subtle text-ink font-mono text-xs px-2 py-1 ' +
  'focus:outline-none focus:border-amber';

function _num(d: Record<string, unknown> | null, key: string): string | null {
  const v = d?.[key];
  return typeof v === 'number' ? v.toFixed(2) : null;
}

export function ResultsGate({ jobId, payload }: { jobId: string; payload: ResultsGatePayload }) {
  const queryClient = useQueryClient();
  const [rejecting, setRejecting] = useState(false);
  const [context, setContext] = useState('');
  const [reason, setReason] = useState('');

  const mutation = useMutation({
    mutationFn: (body: ApprovalRequestBody) => submitApproval(jobId, body),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['job', jobId] });
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
    },
  });

  const approve = () => mutation.mutate({ decision: 'approved' });
  const revise = () => {
    if (context.trim()) mutation.mutate({ decision: 'revise', appended_context: context.trim() });
  };
  const reject = () => {
    if (reason.trim()) mutation.mutate({ decision: 'rejected', reason: reason.trim() });
  };

  const { effects, sensitivity, balance, ps_diagnostics: ps, treatment_variable: t, outcome_variable: y } = payload;
  const overlap = _num(ps, 'overlap_pct');
  const maxSmd = _num(ps, 'max_smd_weighted');

  return (
    <div className="fixed bottom-0 left-0 right-0 z-[60] border-t border-amber/40 bg-canvas-raised max-h-[60vh] overflow-y-auto">
      <div className="px-4 py-3">
        <div className="flex items-center gap-2 mb-3">
          <span className="h-1.5 w-1.5 rounded-full bg-amber animate-pulse" />
          <span className={`${labelCls} text-amber`}>review results</span>
          {t && y && (
            <span className="text-xs font-mono text-ink-tertiary">
              {t} <span className="text-ink-secondary">&rarr;</span> {y}
            </span>
          )}
        </div>

        <div className="flex gap-4 flex-wrap md:flex-nowrap">
          <div className="w-full md:w-1/2 min-w-0">
            <p className={`${labelCls} text-ink-tertiary mb-1`}>effect estimates · 95% ci</p>
            {effects.length > 0 ? (
              <ForestPlot effects={effects} />
            ) : (
              <p className="text-2xs font-mono text-ink-tertiary">no estimates</p>
            )}
          </div>

          <div className="w-full md:w-1/2 min-w-0 space-y-2 text-xs font-mono text-ink-secondary">
            {balance.length > 0 && (
              <div>
                <p className={`${labelCls} text-ink-tertiary mb-1`}>covariate balance · smd</p>
                <BalancePlot balance={balance} />
              </div>
            )}
            {(overlap || maxSmd) && (
              <p>
                <span className={`${labelCls} text-ink-tertiary`}>overlap</span>{' '}
                {overlap ? `${overlap}%` : 'n/a'}
                {maxSmd && <span className="text-ink-tertiary"> · max weighted smd {maxSmd}</span>}
              </p>
            )}
            {sensitivity.length > 0 && (
              <div>
                <p className={`${labelCls} text-ink-tertiary`}>robustness</p>
                <ul className="list-disc list-inside text-ink-tertiary">
                  {sensitivity.map((s, i) => (
                    <li key={i}>
                      {s.method}: {s.robustness_value.toFixed(2)} — {s.interpretation}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>

        {mutation.isError && (
          <span className="block mt-2 text-2xs font-mono text-rose">submit failed, retry</span>
        )}

        {!rejecting ? (
          <div className="mt-3 flex items-center gap-3 flex-wrap">
            <input
              value={context}
              onChange={(e) => setContext(e.target.value)}
              placeholder="add context to re-estimate (e.g. exclude the 2009 cohort)"
              className={`${inputCls} flex-1 min-w-[16rem]`}
            />
            <button
              onClick={() => setRejecting(true)}
              disabled={mutation.isPending}
              className={`${labelCls} border border-rose/50 px-3 py-1.5 text-rose hover:bg-rose/10 disabled:opacity-50`}
            >
              reject
            </button>
            <button
              onClick={revise}
              disabled={mutation.isPending || !context.trim()}
              className={`${labelCls} border border-amber/50 px-3 py-1.5 text-amber hover:bg-amber/10 disabled:opacity-40`}
            >
              revise · re-estimate
            </button>
            <button
              onClick={approve}
              disabled={mutation.isPending}
              className={`${labelCls} bg-amber px-4 py-1.5 text-canvas hover:bg-amber/90 disabled:opacity-50`}
            >
              {mutation.isPending ? 'submitting…' : 'approve · finalize'}
            </button>
          </div>
        ) : (
          <div className="mt-3 flex items-center gap-3 flex-wrap">
            <input
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="reason for rejecting the results (required)"
              autoFocus
              className={`${inputCls} flex-1 min-w-[18rem] focus:border-rose`}
            />
            <button
              onClick={() => {
                setRejecting(false);
                setReason('');
              }}
              className={`${labelCls} px-3 py-1.5 text-ink-tertiary`}
            >
              cancel
            </button>
            <button
              onClick={reject}
              disabled={mutation.isPending || !reason.trim()}
              className={`${labelCls} bg-rose px-4 py-1.5 text-canvas hover:bg-rose/90 disabled:opacity-50`}
            >
              confirm reject
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
