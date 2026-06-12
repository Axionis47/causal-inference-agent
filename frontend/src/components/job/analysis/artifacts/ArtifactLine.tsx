// One artifact emitted by an agent, rendered inside that agent's panel.
// "view" expands the presentable rendering (via the artifact directory)
// inline; "raw" keeps the raw-bytes endpoint a click away. Plots show a
// thumbnail even while collapsed.

import { useState } from 'react';
import type { AnalysisArtifactView } from '../../../../services/api';
import { analysisArtifactUrl } from '../../../../services/api';
import { ArtifactInline } from './ArtifactInline';

const labelCls = 'text-2xs font-mono uppercase tracking-[0.15em]';

export function ArtifactLine({
  jobId,
  artifact,
}: {
  jobId: string;
  artifact: AnalysisArtifactView;
}) {
  const [expanded, setExpanded] = useState(false);
  const url = analysisArtifactUrl(jobId, artifact.artifact_id);

  return (
    <div className="border-b border-edge-subtle/40 py-1.5 min-w-0 last:border-b-0">
      <div className="flex items-center gap-2 min-w-0">
        <span
          className={`${labelCls} shrink-0 px-1.5 py-0.5 border border-edge-subtle text-ink-tertiary`}
        >
          {artifact.kind}
        </span>
        <span className="text-xs text-ink truncate" title={artifact.artifact_id}>
          {artifact.title}
        </span>
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="ml-auto shrink-0 text-2xs font-mono text-indigo hover:text-ink underline underline-offset-2"
        >
          {expanded ? 'hide' : 'view'}
        </button>
        <a
          href={url}
          target="_blank"
          rel="noreferrer"
          className="shrink-0 text-2xs font-mono text-ink-tertiary hover:text-ink underline underline-offset-2"
        >
          raw
        </a>
      </div>
      {artifact.summary && (
        <p className="text-2xs text-ink-tertiary mt-0.5 truncate">{artifact.summary}</p>
      )}
      {!expanded && artifact.kind === 'plot' && (
        <a href={url} target="_blank" rel="noreferrer" className="block mt-1.5 w-fit">
          <img
            src={url}
            alt={artifact.title}
            className="max-h-40 border border-edge-subtle"
            loading="lazy"
          />
        </a>
      )}
      {expanded && <ArtifactInline artifact={artifact} url={url} />}
    </div>
  );
}
