// The expanded body of an artifact line: resolves the artifact through the
// directory, fetches the bytes when the renderer needs them (artifacts are
// immutable once written, so the body caches forever), and renders the
// presentable view.

import { useQuery } from '@tanstack/react-query';
import type { AnalysisArtifactView } from '../../../../services/api';
import { resolveArtifact } from './directory';
import { JsonView } from './renderers/JsonView';
import { MarkdownView } from './renderers/MarkdownView';
import { TableView } from './renderers/TableView';

function ArtifactBody({
  url,
  renderer,
}: {
  url: string;
  renderer: 'json' | 'table' | 'markdown';
}) {
  const { data, isLoading, error } = useQuery({
    queryKey: ['analysis-artifact-body', url],
    queryFn: async () => {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`artifact fetch failed: ${res.status}`);
      return res.text();
    },
    staleTime: Infinity,
    // artifacts are immutable files; a failed read will not heal on retry
    retry: false,
  });

  if (isLoading) {
    return <p className="text-2xs font-mono text-ink-tertiary">loading…</p>;
  }
  if (error || data == null) {
    return (
      <p className="text-2xs font-mono text-rose">
        could not load artifact{error ? `: ${(error as Error).message}` : ''}
      </p>
    );
  }
  if (renderer === 'markdown') return <MarkdownView text={data} />;
  if (renderer === 'table') return <TableView csv={data} />;
  return <JsonView text={data} />;
}

export function ArtifactInline({
  artifact,
  url,
}: {
  artifact: AnalysisArtifactView;
  url: string;
}) {
  const { description, renderer } = resolveArtifact(artifact);

  return (
    <div className="mt-1.5 bg-canvas border border-edge-subtle/60 rounded-sm p-2 space-y-1.5 min-w-0">
      {description && (
        <p className="text-2xs text-ink-tertiary border-b border-edge-subtle/40 pb-1">
          {description}
        </p>
      )}
      {renderer === 'plot' ? (
        <img src={url} alt={artifact.title} className="max-w-full border border-edge-subtle" />
      ) : renderer === 'link' ? (
        <p className="text-2xs font-mono text-ink-tertiary">
          this artifact opens outside the panel:{' '}
          <a
            href={url}
            target="_blank"
            rel="noreferrer"
            className="text-indigo underline underline-offset-2"
          >
            open {artifact.kind}
          </a>
        </p>
      ) : (
        <ArtifactBody url={url} renderer={renderer} />
      )}
    </div>
  );
}
