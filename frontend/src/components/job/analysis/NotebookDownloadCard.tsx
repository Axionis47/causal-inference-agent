// The verified-notebook card: the run's headline deliverable. A notebook that
// executed top to bottom gets a download button (the executed .ipynb) and an
// html preview link; a failed verification says so instead of offering a file.

import type { AnalysisNotebookView } from '../../../services/api';
import { analysisArtifactUrl } from '../../../services/api';

const labelCls = 'text-2xs font-mono uppercase tracking-[0.15em]';

export function NotebookDownloadCard({
  jobId,
  notebook,
}: {
  jobId: string;
  notebook: AnalysisNotebookView;
}) {
  const verified =
    notebook.status === 'verified_running' && !!notebook.verified_artifact_id;

  return (
    <div
      data-testid="notebook-download-card"
      className="bg-canvas-raised border border-edge-subtle rounded-md p-3 space-y-2"
    >
      <div className="flex items-center gap-2 min-w-0">
        <span className={`${labelCls} text-ink-tertiary`}>[ notebook ]</span>
        <span className={`${labelCls} ${verified ? 'text-mint' : 'text-rose'}`}>
          {verified ? 'verified: ran top to bottom' : 'verification failed'}
        </span>
        <span className="ml-auto font-mono text-2xs text-ink-tertiary tabular shrink-0">
          {notebook.attempts} attempt{notebook.attempts === 1 ? '' : 's'}
        </span>
      </div>

      {verified ? (
        <div className="flex items-center gap-3">
          <a
            href={analysisArtifactUrl(jobId, notebook.verified_artifact_id!)}
            download
            className="inline-block px-3 py-2 bg-mint text-ink-inverse text-sm font-medium rounded-md hover:bg-mint-bright transition-colors"
          >
            Download notebook
          </a>
          {notebook.html_artifact_id && (
            <a
              href={analysisArtifactUrl(jobId, notebook.html_artifact_id)}
              target="_blank"
              rel="noreferrer"
              className="text-2xs font-mono text-indigo hover:text-ink underline underline-offset-2"
            >
              preview html
            </a>
          )}
        </div>
      ) : (
        <p className="text-2xs font-mono text-ink-tertiary">
          The notebook did not execute cleanly, so there is no download. The
          artifacts below remain available.
        </p>
      )}
    </div>
  );
}
