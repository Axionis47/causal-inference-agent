// Shown over a CONFIRMED job: the dataset + inputs are stored and the input flow
// has ended. The analyst launches the analysis pipeline here (POST /jobs/:id/run),
// the separate step after the input slice.

import { useMutation, useQueryClient } from '@tanstack/react-query';
import { runAnalysis } from '../../../services/api';

const labelCls = 'text-2xs font-mono uppercase tracking-[0.15em]';

export function RunAnalysisBar({
  jobId,
  onOpenData,
}: {
  jobId: string;
  onOpenData?: () => void;
}) {
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: () => runAnalysis(jobId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['job', jobId] });
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
    },
  });

  return (
    <div className="fixed bottom-0 left-0 right-0 z-[60] border-t border-mint/40 bg-canvas-raised px-4 py-3">
      <div className="flex items-center gap-4 flex-wrap">
        <div className="flex items-center gap-2">
          <span className="h-1.5 w-1.5 rounded-full bg-mint" />
          <span className={`${labelCls} text-mint`}>dataset confirmed</span>
          <span className="text-xs font-mono text-ink-tertiary">
            stored and ready; run the analysis when you want
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
          <span className="text-2xs font-mono text-rose">run failed, retry</span>
        )}

        <button
          onClick={() => mutation.mutate()}
          disabled={mutation.isPending}
          className={`${labelCls} ml-auto bg-mint px-4 py-1.5 text-ink-inverse hover:bg-mint-bright disabled:opacity-50`}
        >
          {mutation.isPending ? 'starting…' : 'run analysis'}
        </button>
      </div>
    </div>
  );
}
