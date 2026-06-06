// Dedicated dataset overlay for the live job view (the F1 "data" key in
// JobPage, and the default view on arrival). This is the data-review surface:
// it shows ONLY the raw rows and what files were downloaded. No inference, no
// labels of any kind here (no schema, no roles, no domain, no Kaggle metadata)
// — all of that is profiling that belongs after the data is approved.
// useDatasetView polls /jobs/:id/dataset until the blocks settle.

import { useEffect } from 'react';
import type {
  DatasetView as DatasetViewData,
  FileEntry,
} from '../../../services/api';
import { Caption, StatusLine } from './atoms';
import { SampleRowsView } from './SampleRowsView';

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function DownloadBlockView({ view }: { view: DatasetViewData }) {
  const d = view.download;
  return (
    <section>
      <StatusLine label="download" status={d.status} />
      {d.url && <p className="mt-2 font-mono text-2xs text-ink-tertiary break-all">{d.url}</p>}
      {d.error && <p className="mt-2 text-xs text-rose">{d.error}</p>}
      {d.files.length > 0 && (
        <table className="mt-3 w-full text-xs">
          <thead>
            <tr className="border-b border-edge-subtle text-2xs font-mono uppercase tracking-[0.12em] text-ink-tertiary">
              <th className="text-left py-1.5 pr-3">File</th>
              <th className="text-left py-1.5 pr-3">Format</th>
              <th className="text-right py-1.5 pr-3">Size</th>
              <th className="text-left py-1.5">Used</th>
            </tr>
          </thead>
          <tbody className="font-mono text-ink-secondary">
            {d.files.map((f: FileEntry) => (
              <tr key={f.name} className="border-b border-edge-subtle/50">
                <td className={`py-1.5 pr-3 ${f.used ? 'text-ink' : ''}`}>{f.name}</td>
                <td className="py-1.5 pr-3">{f.format}</td>
                <td className="py-1.5 pr-3 text-right tabular">{fmtBytes(f.size_bytes)}</td>
                <td className="py-1.5">
                  {f.used ? <span className="text-mint">yes</span> : <span className="text-ink-tertiary">no</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}

export function DatasetView({
  view,
  jobId,
  onClose,
}: {
  view: DatasetViewData | null;
  jobId: string | null;
  onClose: () => void;
}) {
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, [onClose]);

  return (
    <div className="terminal fixed inset-0 z-40 bg-canvas flex flex-col">
      <div className="flex items-center justify-between h-10 px-3 bg-canvas-raised border-b border-edge-subtle shrink-0">
        <span className="text-2xs font-mono uppercase tracking-[0.15em] text-ink-tertiary">[ dataset ]</span>
        <button
          onClick={onClose}
          className="text-2xs font-mono uppercase tracking-[0.15em] text-ink-secondary hover:text-ink"
        >
          esc · close
        </button>
      </div>
      <div className="flex-1 overflow-y-auto">
        {!view ? (
          <p className="p-4 text-xs text-ink-tertiary">No dataset information yet.</p>
        ) : (
          <div className="max-w-5xl p-4 space-y-8">
            <div>
              <Caption>[ raw data ]</Caption>
              <div className="pt-3"><SampleRowsView view={view} jobId={jobId} /></div>
            </div>
            <div>
              <Caption>[ download ]</Caption>
              <div className="pt-3"><DownloadBlockView view={view} /></div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
