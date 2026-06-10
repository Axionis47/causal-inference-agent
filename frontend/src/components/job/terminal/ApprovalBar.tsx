// Data-review gate action bar. Shown over the dataset view while a job sits at
// AWAITING_APPROVAL at the data gate: the analyst reviews the data + inputs, then
// confirms (storing the confirmed dataset and ending the input flow) or rejects
// (failing the job). Confirm posts to POST /jobs/:id/confirm; reject posts to
// POST /jobs/:id/approval. Both invalidate the job query so polling picks up the
// change.

import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { confirmDataset, submitApproval } from '../../../services/api';

const labelCls = 'text-2xs font-mono uppercase tracking-[0.15em]';
const inputCls =
  'bg-canvas-inset border border-edge-subtle text-ink font-mono text-xs px-2 py-1 ' +
  'focus:outline-none focus:border-rose';

export function ApprovalBar({
  jobId,
  onOpenData,
  canConfirm = true,
}: {
  jobId: string;
  onOpenData?: () => void;
  // Advisory gate: false when the persisted treatment/outcome are not yet valid
  // columns, so confirm is blocked before it can 422 server-side. The backend
  // confirm guard remains authoritative.
  canConfirm?: boolean;
}) {
  const queryClient = useQueryClient();
  const [rejecting, setRejecting] = useState(false);
  const [reason, setReason] = useState('');

  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: ['job', jobId] });
    queryClient.invalidateQueries({ queryKey: ['jobs'] });
  };

  const confirm = useMutation({
    mutationFn: () => confirmDataset(jobId),
    onSuccess: invalidate,
  });

  const reject = useMutation({
    mutationFn: () =>
      submitApproval(jobId, { decision: 'rejected', reason: reason.trim() }),
    onSuccess: invalidate,
  });

  const pending = confirm.isPending || reject.isPending;

  return (
    <div className="fixed bottom-0 left-0 right-0 z-[60] border-t border-amber/40 bg-canvas-raised px-4 py-3">
      <div className="flex items-center gap-4 flex-wrap">
        <div className="flex items-center gap-2">
          <span className="h-1.5 w-1.5 rounded-full bg-amber animate-pulse" />
          <span className={`${labelCls} text-amber`}>review data</span>
          <span className="text-xs font-mono text-ink-tertiary">
            confirm to store this dataset and its inputs
          </span>
        </div>

        {onOpenData && (
          <button
            onClick={onOpenData}
            className={`${labelCls} border border-edge-subtle px-3 py-1.5 text-ink-secondary hover:text-ink hover:border-edge-strong`}
          >
            view data
          </button>
        )}

        {(confirm.isError || reject.isError) && (
          <span className="text-2xs font-mono text-rose">submit failed, retry</span>
        )}

        {!canConfirm && !rejecting && (
          <span className="text-2xs font-mono text-amber">
            set a valid treatment &amp; outcome in the data view to confirm
          </span>
        )}

        {!rejecting ? (
          <div className="ml-auto flex items-center gap-3">
            <button
              onClick={() => setRejecting(true)}
              disabled={pending}
              className={`${labelCls} border border-rose/50 px-3 py-1.5 text-rose hover:bg-rose/10 disabled:opacity-50`}
            >
              reject
            </button>
            <button
              onClick={() => confirm.mutate()}
              disabled={pending || !canConfirm}
              className={`${labelCls} bg-mint px-4 py-1.5 text-ink-inverse hover:bg-mint-bright disabled:opacity-50`}
            >
              {confirm.isPending ? 'confirming…' : 'confirm dataset'}
            </button>
          </div>
        ) : (
          <div className="ml-auto flex items-center gap-3">
            <input
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="reason for rejecting (required)"
              autoFocus
              className={`${inputCls} w-72 max-w-[45vw]`}
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
              onClick={() => reject.mutate()}
              disabled={pending || !reason.trim()}
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
