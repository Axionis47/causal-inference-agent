// DAG-review gate (checkpoint A). Shown when a job parks at the DAG gate: the
// analyst reviews the causal graph and its justification, then approves (run
// estimation), revises (add context and redo the DAG), or rejects (fail). Posts
// to POST /jobs/:id/approval and invalidates the job query so SSE picks up the
// resume / redo / failure. Matches the terminal aesthetic.

import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { submitApproval, ApprovalRequestBody, DagGatePayload } from '../../../services/api';
import { DagSvg } from './DagSvg';

const labelCls = 'text-2xs font-mono uppercase tracking-[0.15em]';
const inputCls =
  'bg-canvas-inset border border-edge-subtle text-ink font-mono text-xs px-2 py-1 ' +
  'focus:outline-none focus:border-amber';

export function DagGate({ jobId, payload }: { jobId: string; payload: DagGatePayload }) {
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

  const dag = payload.dag;
  const just = payload.justification;
  const t = payload.treatment_variable;
  const y = payload.outcome_variable;

  return (
    <div className="fixed bottom-0 left-0 right-0 z-[60] border-t border-amber/40 bg-canvas-raised max-h-[60vh] overflow-y-auto">
      <div className="px-4 py-3">
        <div className="flex items-center gap-2 mb-3">
          <span className="h-1.5 w-1.5 rounded-full bg-amber animate-pulse" />
          <span className={`${labelCls} text-amber`}>review causal dag</span>
          {t && y && (
            <span className="text-xs font-mono text-ink-tertiary">
              {t} <span className="text-ink-secondary">&rarr;</span> {y}
            </span>
          )}
        </div>

        <div className="flex gap-4 flex-wrap md:flex-nowrap">
          <div className="w-full md:w-[340px] shrink-0 border border-edge-subtle bg-canvas h-[260px]">
            {dag && dag.nodes.length > 0 ? (
              <DagSvg
                nodes={dag.nodes}
                edges={dag.edges}
                treatment={t}
                outcome={y}
                adjustmentSet={dag.adjustment_set}
              />
            ) : (
              <div className="flex items-center justify-center h-full text-2xs font-mono text-ink-tertiary uppercase tracking-[0.15em]">
                no dag produced
              </div>
            )}
          </div>

          <div className="flex-1 min-w-0 space-y-2 text-xs font-mono text-ink-secondary">
            {just && (
              <>
                <p className="text-ink">{just.identification}</p>
                {just.adjustment_set.length > 0 && (
                  <p>
                    <span className={`${labelCls} text-ink-tertiary`}>adjust for</span>{' '}
                    <span className="text-indigo">{just.adjustment_set.join(', ')}</span>
                  </p>
                )}
                {just.flags.length > 0 && (
                  <div className="flex flex-wrap gap-1.5">
                    {just.flags.map((f) => (
                      <span key={f} className="text-2xs font-mono text-amber border border-amber/40 px-1.5 py-0.5">
                        {f}
                      </span>
                    ))}
                  </div>
                )}
                {just.concerns.length > 0 && (
                  <ul className="list-disc list-inside text-ink-tertiary">
                    {just.concerns.map((c, i) => (
                      <li key={i}>{c}</li>
                    ))}
                  </ul>
                )}
                {just.interpretation && (
                  <p className="text-ink-tertiary border-l border-edge-subtle pl-2">{just.interpretation}</p>
                )}
              </>
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
              placeholder="add context to redo the dag (e.g. age is a mediator, not a confounder)"
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
              revise · redo dag
            </button>
            <button
              onClick={approve}
              disabled={mutation.isPending}
              className={`${labelCls} bg-amber px-4 py-1.5 text-canvas hover:bg-amber/90 disabled:opacity-50`}
            >
              {mutation.isPending ? 'submitting…' : 'approve · estimate'}
            </button>
          </div>
        ) : (
          <div className="mt-3 flex items-center gap-3 flex-wrap">
            <input
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="reason for rejecting the dag (required)"
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
