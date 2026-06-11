// Artifacts emitted by the analysis run, grouped by agent. Each row links to
// the raw-bytes endpoint via analysisArtifactUrl; plots additionally render an
// inline thumbnail that opens the full image in a new tab.

import type { AnalysisArtifactView } from '../../../services/api';
import { analysisArtifactUrl } from '../../../services/api';

const labelCls = 'text-2xs font-mono uppercase tracking-[0.15em]';

function ArtifactRow({
  jobId,
  artifact,
}: {
  jobId: string;
  artifact: AnalysisArtifactView;
}) {
  const url = analysisArtifactUrl(jobId, artifact.artifact_id);
  return (
    <div className="border-b border-edge-subtle/40 py-1.5 min-w-0">
      <div className="flex items-center gap-2 min-w-0">
        <span
          className={`${labelCls} shrink-0 px-1.5 py-0.5 border border-edge-subtle text-ink-tertiary`}
        >
          {artifact.kind}
        </span>
        <span className="text-xs text-ink truncate" title={artifact.artifact_id}>
          {artifact.title}
        </span>
        <a
          href={url}
          target="_blank"
          rel="noreferrer"
          className="ml-auto shrink-0 text-2xs font-mono text-indigo hover:text-ink underline underline-offset-2"
        >
          open
        </a>
      </div>
      {artifact.summary && (
        <p className="text-2xs text-ink-tertiary mt-0.5 truncate">{artifact.summary}</p>
      )}
      {artifact.kind === 'plot' && (
        <a href={url} target="_blank" rel="noreferrer" className="block mt-1.5 w-fit">
          <img
            src={url}
            alt={artifact.title}
            className="max-h-40 border border-edge-subtle"
            loading="lazy"
          />
        </a>
      )}
    </div>
  );
}

export function ArtifactList({
  jobId,
  artifacts,
}: {
  jobId: string;
  artifacts: AnalysisArtifactView[];
}) {
  if (artifacts.length === 0) {
    return <p className="text-2xs font-mono text-ink-tertiary">no artifacts yet</p>;
  }

  // Group by agent, preserving first-appearance order.
  const groups = new Map<string, AnalysisArtifactView[]>();
  for (const a of artifacts) {
    const list = groups.get(a.agent);
    if (list) list.push(a);
    else groups.set(a.agent, [a]);
  }

  return (
    <div className="space-y-3">
      {[...groups.entries()].map(([agent, items]) => (
        <div key={agent}>
          <div className={`${labelCls} text-ink-tertiary mb-1`}>{agent}</div>
          <div>
            {items.map((a) => (
              <ArtifactRow key={a.artifact_id} jobId={jobId} artifact={a} />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
