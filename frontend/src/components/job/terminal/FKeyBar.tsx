// Bottom bar: five function-key chips + a right-aligned connection-mode caption.

import type { JobDetail } from '../../../services/api';
import { getNotebookUrl } from '../../../services/api';
import { FKey } from './atoms';

export interface FKeyBarProps {
  job: JobDetail;
  isPreview: boolean;
  onCancel: () => void;
}

export function FKeyBar({ job, isPreview, onCancel }: FKeyBarProps) {
  const canCancel = !isPreview && job.status !== 'completed' && job.status !== 'failed';
  const isDone = job.status === 'completed';

  return (
    <footer className="flex items-center h-9 bg-canvas-raised border-t border-edge-subtle shrink-0 px-3 text-2xs font-mono">
      <FKey n="F1" label="cancel" enabled={canCancel} onClick={onCancel} />
      <FKey n="F2" label="approve" enabled={false} />
      <FKey n="F3" label="notebook" enabled={isDone} onClick={() => { window.location.href = getNotebookUrl(job.id); }} />
      <FKey n="F4" label="traces" enabled />
      <FKey n="F5" label="results" enabled={isDone} />
      <div className="ml-auto flex items-center gap-2 text-ink-tertiary">
        <span>{isPreview ? 'preview · synthetic data' : 'live · sse connected'}</span>
      </div>
    </footer>
  );
}
