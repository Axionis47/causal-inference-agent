// Data-review gate action bar. Shown over the dataset view while a job sits
// at AWAITING_APPROVAL: the analyst reviews the downloaded data, then approves
// to run the analysis or rejects (with a reason) to fail the job. Posts to
// POST /jobs/:id/approval and invalidates the job query so polling/SSE picks
// up the resume or failure.

import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { submitApproval, ApprovalRequestBody } from '../../../services/api';

const labelCls = 'text-2xs font-mono uppercase tracking-[0.15em]';
const inputCls =
  'bg-canvas-inset border border-edge-subtle text-ink font-mono text-xs px-2 py-1 ' +
  'focus:outline-none focus:border-amber';

export function ApprovalBar({
  jobId,
  onOpenData,
}: {
  jobId: string;
  onOpenData?: () => void;
}) {
  const queryClient = useQueryClient();
  const [rejecting, setRejecting] = useState(false);
  const [reason, setReason] = useState('');
  const [notes, setNotes] = useState('');

  const mutation = useMutation({
    mutationFn: (body: ApprovalRequestBody) => submitApproval(jobId, body),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['job', jobId] });
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
    },
  });

  const approve = () =>
    mutation.mutate({
      decision: 'approved',
      appended_context: notes.trim() || undefined,
    });

  const reject = () => {
    if (!reason.trim()) return;
    mutation.mutate({ decision: 'rejected', reason: reason.trim() });
  };

  return (
    <div className="fixed bottom-0 left-0 right-0 z-[60] border-t border-amber/40 bg-canvas-raised px-4 py-3">
      <div className="flex items-center gap-4 flex-wrap">
        <div className="flex items-center gap-2">
          <span className="h-1.5 w-1.5 rounded-full bg-amber animate-pulse" />
          <span className={`${labelCls} text-amber`}>review data</span>
          <span className="text-xs font-mono text-ink-tertiary">
            approve to start the analysis on this data
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

        {mutation.isError && (
          <span className="text-2xs font-mono text-rose">submit failed, retry</span>
        )}

        {!rejecting ? (
          <div className="ml-auto flex items-center gap-3">
            <input
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="optional notes for the analysis"
              className={`${inputCls} w-64 max-w-[40vw]`}
            />
            <button
              onClick={() => setRejecting(true)}
              disabled={mutation.isPending}
              className={`${labelCls} border border-rose/50 px-3 py-1.5 text-rose hover:bg-rose/10 disabled:opacity-50`}
            >
              reject
            </button>
            <button
              onClick={approve}
              disabled={mutation.isPending}
              className={`${labelCls} bg-amber px-4 py-1.5 text-canvas hover:bg-amber/90 disabled:opacity-50`}
            >
              {mutation.isPending ? 'submitting…' : 'approve · run analysis'}
            </button>
          </div>
        ) : (
          <div className="ml-auto flex items-center gap-3">
            <input
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="reason for rejecting (required)"
              autoFocus
              className={`${inputCls} w-72 max-w-[45vw] focus:border-rose`}
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
