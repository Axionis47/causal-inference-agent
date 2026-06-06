// Bottom bar: function-key chips + a right-aligned connection-mode caption.

import type { JobDetail } from '../../../services/api';
import { getNotebookUrl } from '../../../services/api';
import { FKey } from './atoms';

export interface FKeyBarProps {
  job: JobDetail;
  isPreview: boolean;
  onCancel: () => void;
  onData: () => void;
  onResults: () => void;
}

export function FKeyBar({ job, isPreview, onCancel, onData, onResults }: FKeyBarProps) {
  const canCancel = !isPreview && job.status !== 'completed' && job.status !== 'failed';
  const isDone = job.status === 'completed';

  return (
    <footer className="flex items-center h-9 bg-canvas-raised border-t border-edge-subtle shrink-0 px-3 text-2xs font-mono">
      <FKey n="F1" label="data" enabled onClick={onData} />
      <FKey n="F2" label="cancel" enabled={canCancel} onClick={onCancel} />
      <FKey n="F3" label="notebook" enabled={isDone} onClick={() => { window.location.href = getNotebookUrl(job.id); }} />
      <FKey n="F4" label="results" enabled={isDone} onClick={onResults} />
      <div className="ml-auto flex items-center gap-2 text-ink-tertiary">
        <span>{isPreview ? 'preview · synthetic data' : 'live · sse connected'}</span>
      </div>
    </footer>
  );
}
